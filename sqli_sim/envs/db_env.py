import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

from sqli_sim.envs._helper.reward import Reward


class DBType:
    def __init__(self, instance, errors=True, codes=4):
        self.instance = instance
        self.errors = errors
        self.codes = codes


class ActionSpace:
    def __init__(self, escapes=3, columns=5, db_types=3):
        self.escapes = escapes
        self.columns = columns

        # supply the type of database to provide error messages
        self.db_types = db_types
        self.db = [DBType(i) for i in range(db_types)]

        self.escape_space = escapes * 2
        self.column_space = escapes * columns * 2
        self.capture_space = escapes * columns
        self.db_space = escapes * db_types * self.db[0].codes

        self.actions = self.escape_space + self.column_space + self.db_space + self.capture_space

        self.offset = self.actions // escapes

    def set_sequence(self, r, f, d):
        """
        Set the correct escape and column sequence based on the action space
        """

        # Get the set of actions that are syntactically correct
        self.syntaxmin = 0 + (r * self.offset)
        self.syntaxmax = self.offset + self.syntaxmin

        assert self.syntaxmin == r * (
                self.escape_space // self.escapes + self.column_space // self.escapes + self.db_space //
                self.escapes + self.capture_space // self.escapes)

        # self.sequence = [0 + r * self.offset, 1 + r * self.offset, ((self.columns * 2) + f + 2) + r * self.offset]

        self.escape = r * self.offset
        self.column = self.escape + (self.escape_space // self.escapes) + f
        self.db_type = self.escape + (self.escape_space // self.escapes) + (self.column_space // self.escapes) + d
        self.flag = self.escape + (self.escape_space // self.escapes) + (self.column_space // self.escapes) + (
                self.db_space // self.escapes) + f

        assert self.column > self.escape >= self.syntaxmin
        assert self.db_type > self.column < self.syntaxmin + (self.escape_space // self.escapes) + (
                self.column_space // self.escapes)
        assert self.db_type >= self.syntaxmax - (self.capture_space // self.escapes) - (
                self.db_space // self.escapes)
        assert self.syntaxmax >= self.flag >= self.syntaxmax - (self.capture_space // self.escapes)


class CTFSQLEnv1(gym.Env):
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

    def __init__(self, escapes=3, columns=5, db_types=3, rewards=None):
        # Init action space
        self.space = ActionSpace(escapes, columns, db_types)
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
        if (action == self.space.flag):
            if (self.verbose): print('Flag captured. I return 2')
            self.done = True
            self.state[action] = 2
            return self.state, self.rewards.capture, self.done, {'msg': 'FLAG Server response is 2'}
        elif (action == self.space.escape):
            if (self.verbose): print('Query has the correct escape')
            self.state[action] = 2
            return self.state, self.rewards.escape, self.done, {'msg': 'Server response is 2'}
        elif (action >= self.space.syntaxmin and action < self.space.syntaxmax):
            """ 
            Action is within the correct escape sequence range.
            See if the action contains the correct queried row given it is within the space of correct escape sequence
            this row can. The following checks if it contains correct Nc with the SQL commands LIMIT or OFFSET appended.
            Check if an error code was emitted by the db that contains the db type.
            """
            if (action == self.space.column or action == self.space.column + 1):
                if (self.verbose):
                    print('Query with correct number of rows')
                self.state[action] = 2
                return self.state, self.rewards.rows, self.done, {'msg': 'Server response is 2'}
            elif (action == self.space.db_type + 1 or
                  action == self.space.db_type + 2 or
                  action == self.space.db_type + 2):
                if (self.verbose):
                    print('Query contains error msg')
                self.state[action] = 2
                return self.state, self.rewards.error, self.done, {'msg': 'Server response is 2'}

            if (self.verbose):
                print('Query has the correct escape, but contains the wrong number of rows. I return 0')
            self.state[action] = 0
            return self.state, self.rewards.wrong, self.done, {'msg': 'Server response is 0 wrong number of rows'}
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
        d = np.random.randint(self.space.db_types)  # db type to use

        # The random setup contains the correct escape sequences and the correct SQL injection
        self.space.set_sequence(r, f, d)  # [0 + r * 17, 1 + r * 17, (12 + f) + r * 17]

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
