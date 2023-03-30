from sqli_sim.envs._helper.reward import Reward
from utils import evaluate as ev
from tqdm import tqdm
from stable_baselines3 import DQN
import gym


# Evaluate the mean reward given from the simulations
# Expected values should be between 5-9 with a length of 4-7 steps to the goal
def evaluate(sim_model, n_simulations, total_timesteps):
    for i in tqdm(range(n_simulations)):
        mean_reward, max_rewards, episode_length = ev.evaluate(
            sim_model[i].load(
                'tests/out/sim_db_{0}_for_escapes_{1}_columns_{2}_db_{1}'.format(str(i), escapes, columns, db_types)),
            env,
            num_steps=total_timesteps)
        print(f'DQN-Model[{i}]: mean reward = {mean_reward}, '
              f'max-100 eps reward = {max_rewards}, '
              f'episode_length = {episode_length}')


# Running the simulations
def run(sim_model, n_simulations=10, total_timesteps=10 ** 6):
    print(sim_model)

    for i in tqdm(range(n_simulations)):
        sim_model[i].learn(total_timesteps=total_timesteps)
        sim_model[i].save(
            'tests/out/sim_db_{0}_for_escapes_{1}_columns_{2}_db_{1}'.format(str(i), escapes, columns, db_types))

    return sim_model


if __name__ == '__main__':
    env_name = 'db_sim-v0'
    escapes = 3
    columns = 5
    db_types = 3

    env = gym.make(env_name,
                   escapes=escapes,
                   columns=columns,
                   db_types=db_types,
                   rewards=Reward(capture=10, escape=-1, rows=-1, error=-1, wrong=-1))

    n_simulations = 1
    total_timesteps = (10 ** 6)

    model = [DQN('MlpPolicy', env, verbose=1) for i in range(n_simulations)]
    model = run(model, n_simulations=n_simulations, total_timesteps=total_timesteps)
    evaluate(model, n_simulations=n_simulations, total_timesteps=total_timesteps)

    print(f"Action-Space num_actions={env.space.actions}")
    print(
        f"Spaces: escape_space={env.space.escape_space}, "
        f"column_space={env.space.column_space},  "
        f"db_space={env.space.db_space}, "
        f"capture_space={env.space.capture_space}"
    )
