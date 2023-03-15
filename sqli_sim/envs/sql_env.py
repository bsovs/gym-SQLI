import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from sqli_sim.envs.reward import Reward


class ActionSpace:
    def __init__(self, escapes=3, columns=5):
        self.escapes = escapes
        self.columns = columns

        self.escape_space = escapes * 2
        self.column_space = escapes * columns * 2
        self.capture_space = escapes * columns

        self.actions = self.escape_space + self.column_space + self.capture_space

        self.offset = self.actions // escapes

    def set_sequence(self, r, f):
        """
        Set the correct escape and column sequence based on the action space
        """

        # Get the set of actions that are syntactically correct
        self.syntaxmin = 0 + (r * self.offset)
        self.syntaxmax = self.offset + self.syntaxmin

        # self.flag = ((self.columns * 2) + f + 2) + r * self.offset
        self.sequence = [0 + r * self.offset, 2 + r * self.offset, ((self.columns * 2) + f + 2) + r * self.offset]

        assert self.syntaxmin + self.escape_space // self.escapes + self.column_space // self.escapes <= \
               self.sequence[2] <= self.syntaxmax

        return self.sequence


class CTFSQLEnv0(gym.Env):
    """
    Description:
        A webserver exposing a query with a potential SQL injection vulnerability. Behind the vulnerability lies a flag.
    Observation:
        Type: MiltiDiscrete(3)
        Num    Observation
        0   action tried and returned a negative answer
        1   action never tried
        2   action tried and returned a positive answer
    Actions:
        Type: Discrete(n)
        Num    Action
        n    SQL statement n
    Reward:
        +10 for capturing the flag, -1 in all the other cases.
    Starting State:
        Webserver initialized with a random query. No action tested.
    Episode Termination:
        Capture the flag.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, escapes=3, columns=5, rewards=None):
        # Init action space
        self.space = ActionSpace(escapes, columns)
        self.actions = self.space.actions

        # Reward values
        self.rewards = rewards if rewards is not None else Reward(capture=10, escape=-1, rows=-1, wrong=-1)

        # Action space
        self.action_space = spaces.Discrete(self.actions)

        # Observation space
        self.observation_space = spaces.MultiDiscrete(np.ones(self.actions) * 3)

        self.set_matrix()

        self.done = False
        self.verbose = False
        if (self.verbose): print('Game setup with a random query')

        self.seed()
        self.viewer = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        """
        0neg
        1neut
        2pos
        """
        # Process action
        if (action == self.setup[2]):
            if (self.verbose): print('Flag captured. I return 2')
            self.done = True
            self.state[action] = 2
            return self.state, self.rewards.capture, self.done, {'msg': 'FLAG Server response is 2'}
        elif (action == self.setup[0]):
            if (self.verbose): print('Escape found')
            self.state[action] = 2
            return self.state, self.rewards.escape, self.done, {'msg': 'FLAG Server response is 2'}
        elif (action >= self.space.syntaxmin and action <= self.space.syntaxmax):
            """ 
            Action is within the correct escape sequence range.
            See if the action contains the correct queried row given it is within the space of correct escape sequence
            this row can. The following checks if it contains correct Nc with the SQL commands LIMIT or OFFSET appended.
            """
            if (action == self.flag_cols * 2 + self.setup[1] + 0 or action == self.flag_cols * 2 + self.setup[1] + 1):
                if (self.verbose):
                    print('Query with correct number of rows')
                self.state[action] = 2
                return self.state, self.rewards.rows, self.done, {'msg': 'Server response is 2'}

            if (self.verbose):
                print('Query has the correct escape, but contains the wrong number of rows. I return 2')
            self.state[action] = 0
            return self.state, self.rewards.escape, self.done, {'msg': 'Server response is 2 wrong number of rows'}
        else:
            if (self.verbose): print('Query is syntactically wrong. I return 0')
            self.state[action] = 0
            return self.state, self.rewards.wrong, self.done, {'msg': 'Server response is 0'}

    def set_matrix(self):
        # State
        self.state = np.ones(self.actions)

        # Random integers to setup the server
        r = np.random.randint(self.space.escapes)  # escapes
        f = np.random.randint(self.space.columns)  # column actions
        self.flag_cols = f

        # The random setup contains the correct escape sequences and the correct SQL injection
        self.setup = self.space.set_sequence(r, f)  # [0 + r * 17, 1 + r * 17, (12 + f) + r * 17]

    def reset(self):
        self.done = False

        # Reinitializing the random query
        self.set_matrix()

        if (self.verbose): print('Game reset (with a new random query!)')
        return self.state  # ,0,self.done,{'msg':'Game reset'}

    def render(self, mode='human'):
        return None

    def close(self):
        return
