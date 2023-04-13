from sqli_sim.envs._helper.error_message import ESCAPE_CHAR
from sqli_sim.envs._helper.reward import Reward, ErrorReward
from utils import evaluate as ev
from tqdm import tqdm
from stable_baselines3 import DQN, PPO, SAC, A2C
import gym


# Evaluate the mean reward given from the simulations
# Expected values should be a positive reward (ie. anything > 0)
def evaluate(sim_model, n_simulations, total_timesteps):
    for i in tqdm(range(n_simulations)):
        mean_reward, max_rewards, episode_length_mean = ev.evaluate(
            sim_model[i].load('tests/out/sim_db_error_dqn_{0}'.format(str(i))),
            env,
            num_steps=total_timesteps)
        print(f'PPO-Model[{i}]:  mean reward = {mean_reward},  '
              f'max-100 eps reward = {max_rewards},  '
              f'mean episode length = {episode_length_mean}')


# Running the simulations
def run(sim_model, n_simulations=10, total_timesteps=10 ** 6):
    print(sim_model)

    for i in tqdm(range(n_simulations)):
        sim_model[i].learn(total_timesteps=total_timesteps, log_interval=1)
        sim_model[i].save('tests/out/sim_db_error_dqn_{0}'.format(str(i)))

    return sim_model


if __name__ == '__main__':
    env_name = 'db_sim-v1'
    step_limit = 10
    reward_limit = 10
    history_length = 4
    attack_values = 100

    env = gym.make(env_name, verbose=False,
                   step_limit=step_limit,
                   reward_limit=reward_limit,
                   history_length=history_length,
                   attack_values=attack_values)

    n_simulations = 10
    total_timesteps = (10 ** 5) * 6  # amount of time needed for convergence

    model = [
        DQN(env=env,
            policy='MlpPolicy',
            verbose=1
            )
        for i in range(n_simulations)
    ]
    model = run(model, n_simulations=n_simulations, total_timesteps=total_timesteps)
    evaluate(model, n_simulations=n_simulations, total_timesteps=(10 ** 5))

    print(f"Action-Space num_actions={env.actions}")
