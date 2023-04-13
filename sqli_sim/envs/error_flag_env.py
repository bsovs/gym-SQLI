from copy import deepcopy

import gym
from gym import spaces
import random
import numpy as np
from gym.utils import seeding

from sqli_sim.envs._helper.error_message import ERROR_MESSAGES, ESCAPE_CHAR, DATABASES, TABLES, FLAGS, \
    TOTAL_ESCAPE_CHAR, Attack, error_messages, table_error_messages
from sqli_sim.envs._helper.reward import ErrorReward


class SQLInjectionFlagEnv(gym.Env):
    def __init__(self, step_limit=100, reward_threshold=1, reward_limit=10, history_length=4, attack_values=100,
                 rewards=ErrorReward(),
                 verbose=False):
        self.attack_values = attack_values
        self.rewards_base = rewards
        self.gen_actions()

        # Define the action space as a discrete set of SQL injection commands
        self.action_space = spaces.Discrete(self.actions)
        # Define the observation space as a 2D array with shape (n_actions, history_length+1)
        self.obs_shape = 4
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.actions, self.obs_shape), dtype=int)

        self.reward_range = (rewards.nothing * step_limit, rewards.flag)
        self.reward_threshold = reward_threshold

        self.reward = ErrorReward()
        self.step_limit = step_limit
        self.reward_limit = reward_limit
        self.history_length = history_length

        self.done = False
        self.verbose = verbose

        self.seed()
        self.reset()

    def gen_actions(self):
        self.error_messages = table_error_messages()
        self.gen_values = Attack(self.attack_values)
        self.ACTIONS = self.gen_values.ACTIONS
        self.TABLE_ACTION = self.gen_values.TABLE_ACTION
        self.ATTACK_ACTION = self.gen_values.FLAG_ATTACK

        self.take_actions = self.gen_values.escape_list

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
        error_value, self.current_error = self._execute_sql(escape, sql_action)
        self.error_list.append(self.current_error)
        # Calculate the reward based on the error message
        self.current_reward = self._calculate_reward(action, self.current_error)
        self.total_reward += self.current_reward

        # Update observation state
        # self.state[action, 0] = self.prev_value
        # self.state[action, 1] = self.current_reward
        if escape in self.escapes:
            self.state[action, 0] = 2
        # self.state[action, -1] = error_value
        self.prev_value = error_value

        # self.update_observation(action, self.current_error)

        # self.state = np.roll(self.state, 1, axis=1)  # shift history to the right
        # use error_encoder to get error encoding
        # self.state[:, -self.history_length:] = self.error_encoder.get_error_encoding(self.error_list)[:, :self.history_length]

        # Set done flag to False until enough information is received to inject the database
        self.done = self.flag_found is True or (self.steps >= self.step_limit)
        # (self.total_reward >= self.reward_limit) or (self.steps >= self.step_limit)

        # Initialize info as an empty dictionary
        info = {'msg': f'Server response is {self.state[action]}'}

        if self.done:
            if self.total_reward >= self.reward_threshold:
                info['is_success'] = True
            else:
                info['is_success'] = False

            info['episode_reward'] = self.total_reward
            self.total_reward = 0

        if self.verbose:
            print(f"ESCAPE: {escape}")
            print(f"SQLI: {sql_action}")
            print(f"ERROR: {self.current_error}")
            print(f"REWARD: {self.current_reward}")
            print(f"===============================")

        # Return the observation, reward, done flag, and any additional info
        return self._get_observation(), self.current_reward, self.done, info

    def reset(self):
        self.reward = deepcopy(self.rewards_base)
        self.flag_found = False

        # Set new attacks actions
        self.gen_actions()
        self.prev_value = 0

        # Initialize state
        self.state = np.ones((self.actions, self.obs_shape), dtype=np.float32)
        self.error_list = []

        # Reset the environment by setting the current error message to None and the current reward to 0
        self.current_error = None
        self.current_reward = 0
        self.steps = 0
        self.total_reward = 0

        # Choose our database type to use in this trial
        self.db_type = np.random.choice(DATABASES)

        # Get the possible escape chars that will work
        e = np.random.choice(len(ESCAPE_CHAR))
        self.escapes = [ESCAPE_CHAR[e]]

        t = np.random.choice(len(TABLES))
        self.table = TABLES[t]
        # allowed table action
        a = np.random.choice(len(self.gen_values.TABLE_ACTION))
        self.table_action = self.gen_values.TABLE_ACTION[a]

        # Get the injection command with proper escape included
        escape = np.random.choice(self.escapes)
        i = np.random.choice(self.gen_values.num_actions)

        # self.injection_command = self.gen_values.flag_map[(escape, self.table, self.table_action, i)]

        f = ((a + 1) * 4 + 1) + t
        self.injection_command = self.command_dict[self.gen_values.flag_map[(f, e)]][1]

        if self.verbose:
            print(f"===============================")
            print(f"TOTAL ACTION SPACE: {self.actions}")
            print(f"ESCAPE: {escape}")
            print(f"TABLE: {self.table}")
            print(f"ACTON: {self.table_action}")
            print(f"INDEX: {i}")
            for i, (escape, sql_action) in enumerate(self.take_actions):
                if sql_action == self.injection_command:
                    self.injection_action = i
                    print(f"FLAG CMD: {i}")
            print(f"ACTION INDEX: {self.injection_command}")
            print(f"===============================")

        # Return the initial observation
        return self._get_observation()

    def _index_to_command(self, action_index):
        return self.command_dict[action_index]

    def _execute_sql(self, escape, command):
        if command in self.ATTACK_ACTION:
            if escape in self.escapes:
                if command == self.injection_command:
                    return 2, np.random.choice(FLAGS)
                else:
                    return 0, escape
            else:
                return 0, ""
        elif escape in self.escapes:
            if command is None:
                return 2, escape

            table_action = None
            table = None
            for _action in self.TABLE_ACTION:
                if _action in command:
                    table_action = _action
            for _table in TABLES:
                if _table in command:
                    table = _table

            if table and table != self.table:
                return 0, escape

            if table_action in self.injection_command:
                # Get possible error messages for the given command and database type
                error = self.error_messages[table_action][self.db_type][table]
                return 2, error.lower()

            return 0, escape
        else:
            return 0, ""

    def _calculate_reward(self, action, error):
        reward = self.reward.nothing
        self.state[action, :] = np.zeros(self.obs_shape)

        if error == "":
            return reward

        for flag in FLAGS:
            if flag in error:
                if self.verbose:
                    print("FLAG FOUND!")
                self.state[action, :] = [2, 2, 2, 2]
                self.flag_found = True
                return self.reward.error_values[self.reward.action_type(flag)]

        for table in TABLES:
            if table in error and table == self.table:
                if self.verbose: print(f"TABLES:  {table}  FOUND!")

                reward = max(reward, self.reward.error_values[self.reward.table_type(table)])
                self.reward.error_values[self.reward.table_type(table)] = -1

                self.state[action, 2] = 2

        for _action in self.TABLE_ACTION:
            if _action.lower() in error:
                if self.verbose: print(f"ACTIONS:  {_action}  FOUND!")

                reward = max(reward, self.reward.error_values[self.reward.action_type(_action)])
                self.state[action, 1] = 2

        for escape in ESCAPE_CHAR:
            if escape in error[:5]:
                if self.verbose: print(f"ESCAPE FOUND!")

                reward = max(reward, self.reward.error_values[self.reward.escape_type(escape)])
                self.state[action, 0] = 2

        return reward

    def _get_observation(self):
        # Return the current error message and reward as the observation
        return self.state
