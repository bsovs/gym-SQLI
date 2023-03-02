from gym.envs.registration import register

register(
    id='sqli_sim-v0',
    entry_point='sqli_sim.envs:CTFSQLEnv0',
)
