import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from math import comb

from sqli_sim.envs._helper.reward import Reward


class Math:
    @classmethod
    def combinations_wo_order_wo_repetition(cls, n, k):
        return comb(n, k)

    @classmethod
    def combinations_w_order_wo_repetition(cls, n, k):
        return

    @classmethod
    def combinations_w_order_w_repetition(cls, n, k):
        return n ** k


class Hint:
    columns = "columns"
    table = "table"
    db_type = "db_type"
    value = "value"


class DBType:
    def __init__(self, instance, actions, errors=1, codes=4):
        self.instance = instance
        self.errors = errors
        self.codes = codes
        self.actions = actions
        self.reset()

    def reset(self):
        self.query_length = np.random.randint(10) + 5
        self.error_occurs_at = np.random.randint(self.query_length)
        self.escapes = [np.random.randint(self.actions) for _ in range(np.random.randint(3) + 1)]
        self.messages = [np.random.randint(self.codes) for _ in self.escapes]

    def attack(self, action):
        if action in self.escapes:
            return self.messages[self.escapes.index(action)]
        return None

    def get_hint(self, code):
        if code == 0:
            return Hint.db_type
        if code == 1:
            return Hint.table
        if code == 2:
            return Hint.value
        if code == 3:
            return Hint.columns


class ActionSpace:
    def __init__(self, escapes=3, links=5, db_types=3, codes=4, attack_length=5):
        assert db_types > 0

        self.escapes = escapes
        self.links = links

        self.escape_space = escapes * 2
        self.link_space = Math.combinations_w_order_w_repetition(links, attack_length)

        self.actions = self.escape_space + self.link_space

        # supply the type of database to provide error messages
        self.db_types = db_types
        self.codes = codes
        self.db = [DBType(i, self.actions, codes=codes) for i in range(db_types)]
        self.db_space = db_types * self.db[0].codes

    def set_sequence(self, db_type):
        """
        Set the correct db error msg
        """
        self.target = self.db[db_type]
        self.target.reset()


class CTFSQLEnv1(gym.Env):
    """
    Description:
        TODO
    Observation:
        TODO
        Type: MiltiDiscrete(3)
        Num    Observation
        0   action tried and returned a negative answer
        1   action never tried
        2   action tried and returned a positive answer
    Actions:
        TODO
        Type: Discrete(n)
        Num    Action
        n    SQL statement n
    Reward:
        TODO
        +10 for capturing the flag, -1 in all the other cases.
    Starting State:
        Webserver initialized with a random query. No action tested.
    Episode Termination:
        Capture the flag.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, escapes=3, links=5, db_types=3, rewards=None):
        # Init action space
        self.space = ActionSpace(escapes, links, db_types)
        self.actions = self.space.actions

        # Reward values
        self.rewards = rewards if rewards is not None else Reward(capture=10, escape=-1, rows=-1, wrong=-1)

        # Action space
        self.action_space = spaces.Discrete(self.actions)

        # Observation space
        self.observation_space = spaces.MultiDiscrete(np.ones(self.actions) * 3)

        self.done = False
        self.verbose = False
        if (self.verbose): print('Game setup with a random query')

        self.seed()
        self.viewer = None
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        message = self.space.target.attack(action=action)
        if message is not None:
            hint = self.space.target.get_hint(code=message)

    def set_matrix(self):
        # State
        self.state = np.ones(self.actions)

        # Random integers to setup the server
        # r = np.random.randint(self.space.escapes)  # escapes
        # f = np.random.randint(self.space.columns)  # column actions
        db_type = np.random.randint(self.space.db_types)  # db type to use

        # The random setup contains the correct escape sequences and the correct SQL injection
        self.space.set_sequence(db_type=db_type)  # [0 + r * 17, 1 + r * 17, (12 + f) + r * 17]

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
