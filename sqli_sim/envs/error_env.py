import gym
from gym import spaces
import random
import numpy as np
from gym.utils import seeding

from sqli_sim.envs._helper.ErrorEncoder import ErrorEncoder
from sqli_sim.envs._helper.error_message import ERROR_MESSAGES, ESCAPE_CHAR, DATABASES, TABLES, FLAGS, \
    TOTAL_ESCAPE_CHAR, Attack, error_messages
from sqli_sim.envs._helper.reward import ErrorReward


class SQLInjectionEnv(gym.Env):
    def __init__(self, step_limit=100, reward_limit=10, history_length=4, attack_values=100, verbose=False):
        self.attack_values = attack_values
        self.gen_actions()

        # Define the action space as a discrete set of SQL injection commands
        self.action_space = spaces.Discrete(self.actions)
        # Define the observation space as a 2D array with shape (n_actions, history_length+1)
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.actions, history_length + 1), dtype=int)

        self.reward = ErrorReward()
        self.step_limit = step_limit
        self.reward_limit = reward_limit
        self.history_length = history_length

        self.done = False
        self.verbose = verbose

        self.seed()
        self.reset()

        # Define our error message one-hot encoder
        self.error_encoder = ErrorEncoder(state=self.state, history_length=history_length)

    def gen_actions(self):
        self.error_messages = error_messages()
        self.gen_values = Attack(self.attack_values)
        self.ACTIONS = self.gen_values.ACTIONS
        self.ATTACK_ACTION = self.gen_values.ATTACK_ACTION

        self.take_actions = [(escape, action) for escape in TOTAL_ESCAPE_CHAR for action in self.ACTIONS]
        self.actions = len(self.take_actions)
        self.command_dict = {i: self.take_actions[i] for i in range(self.actions)}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # Take a step
        self.steps += 1

        if self.verbose:
            print(f"========== STEP: {self.steps:2d} ===========")
            print(f"TAKING-ACTION: {action}")

        # Convert the action index to a SQL injection command
        (escape, sql_action) = self._index_to_command(action)
        # Simulate the execution of the SQL command and get the resulting error message
        self.current_error = self._execute_sql(escape, sql_action)
        self.error_list.append(self.current_error)
        # Calculate the reward based on the error message
        self.current_reward = self._calculate_reward(self.current_error)
        self.total_reward += self.current_reward

        # Update observation state
        self.state = np.roll(self.state, 1, axis=1)  # shift history to the right
        self.state[action, 0] = 1  # set current action to 1
        # use error_encoder to get error encoding
        self.state[:, -self.history_length:] = self.error_encoder.get_error_encoding(self.error_list
                                                                                     )[:, :self.history_length]

        # Set done flag to False until enough information is received to inject the database
        self.done = (self.total_reward >= self.reward_limit) or (self.steps >= self.step_limit)

        # Initialize info as an empty dictionary
        info = {'msg': f'Server response is {self.state[action]}'}

        if self.verbose:
            print(f"ESCAPE: {escape}")
            print(f"SQLI: {sql_action}")
            print(f"ERROR: {self.current_error}")
            print(f"REWARD: {self.current_reward}")
            print(f"===============================")

        # Return the observation, reward, done flag, and any additional info
        return self._get_observation(), self.current_reward, self.done, info

    def reset(self):
        # Set new attacks actions
        self.gen_actions()

        # Initialize state
        self.state = np.zeros((self.actions, self.history_length + 1), dtype=np.float32)
        self.error_list = []

        # Reset the environment by setting the current error message to None and the current reward to 0
        self.current_error = None
        self.current_reward = 0
        self.steps = 0
        self.total_reward = 0

        # Choose our database type to use in this trial
        self.db_type = np.random.choice(DATABASES)

        # Get the possible escape chars that will work
        self.escapes = np.random.choice(ESCAPE_CHAR, np.random.randint(1, len(ESCAPE_CHAR)))

        # Get the injection command
        self.injection_command = np.random.choice(self.ATTACK_ACTION, np.random.randint(0, 2))

        # Return the initial observation
        return self._get_observation()

    def _index_to_command(self, action_index):
        return self.command_dict[action_index]

    def _execute_sql(self, escape, command):
        if command in self.ATTACK_ACTION:
            if escape in self.escapes:
                if command in self.injection_command:
                    return np.random.choice(FLAGS)
                else:
                    return escape
            else:
                return ""
        elif escape in self.escapes:
            # Get possible error messages for the given command and database type
            error = self.error_messages[command][self.db_type]

            # Generate a random subset of the possible error messages
            # num_errors = random.randint(1, len(possible_errors))
            # error_subset = random.sample(possible_errors, num_errors)

            # Concatenate the error messages and return as a single string
            return ("\n".join(error)).lower()
        else:
            return ""

    def _calculate_reward(self, error):
        if error == "":
            return self.reward.nothing

        for flag in FLAGS:
            if flag in error:
                return self.reward.error_values[self.reward.action_type(flag)]
        for table in TABLES:
            if table in error:
                if self.verbose: print(f"TABLES:  {table}  FOUND!")
                return self.reward.error_values[self.reward.table_type(table)]
        for action in self.ACTIONS:
            if action in error:
                if self.verbose: print(f"ACTIONS:  {action}  FOUND!")
                return self.reward.error_values[self.reward.action_type(action)]
        for db_type in DATABASES:
            if db_type.lower() in error:
                if self.verbose: print(f"DATABASES:  {db_type}  FOUND!")
                return self.reward.db_type
        for escape in ESCAPE_CHAR:
            if escape in error:
                return self.reward.error_values[self.reward.escape_type(escape)]

        return self.reward.nothing

    def _get_observation(self):
        # Return the current error message and reward as the observation
        return self.state
