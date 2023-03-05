import sys
import numpy as np
from utils import evaluate as ev
from tqdm import tqdm
from stable_baselines3 import DQN
import gym
import sqli_sim


# Evaluate the mean reward given from the simulations
# Expected values should be between 5-9 with a length of 4-7 steps to the goal
def evaluate(model):
    for i in tqdm(range(n_simulations)):
        mean_reward, max_reward, episode_length = ev.evaluate(
            model[i].load('tests/out/sim_{0}_for_escapes_{1}_columns_{2}'.format(str(i), escapes, columns)),
            env,
            num_steps=10 ** 6)
        print('dqn: mean reward = {0}, max eps reward = {1}, episode_length = {2}'.format(mean_reward,
                                                                                          max_reward,
                                                                                          episode_length))


# Running the simulations
def run():
    model = [DQN('MlpPolicy', env, verbose=1) for i in range(n_simulations)]
    print(model)

    for i in tqdm(range(n_simulations)):
        model[i].learn(total_timesteps=10 ** 6)
        model[i].save('tests/out/sim_{0}_for_escapes_{1}_columns_{2}'.format(str(i), escapes, columns))

    evaluate(model)


if __name__ == '__main__':
    env_name = 'sqli_sim-v0'
    escapes = 5
    columns = 5

    env = gym.make(env_name, escapes=escapes, columns=columns)
    n_simulations = 1

    run()
