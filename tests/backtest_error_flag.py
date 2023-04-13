import time

import numpy as np
import tensorboardX
from stable_baselines3.common.callbacks import EvalCallback
from sqli_sim.envs._helper.error_message import ESCAPE_CHAR
from sqli_sim.envs._helper.reward import Reward, ErrorReward

from utils import evaluate as ev
from tqdm import tqdm
from stable_baselines3 import DQN, PPO
import gym


# Evaluate the mean reward given from the simulations
# Expected values should be a positive reward (ie. anything > 0)
def evaluate(sim_model, n_simulations, total_timesteps):
    for i in tqdm(range(n_simulations)):
        mean_reward, max_rewards, episode_length_mean, flags_found = ev.evaluate_db_error(
            sim_model[i].load('tests/out/sim_db_flag_{0}'.format(str(i))),
            env,
            num_steps=total_timesteps)
        print(f'PPO-Model[{i}]:  mean reward = {mean_reward},  '
              f'max-10% eps reward = {max_rewards},  '
              f'mean episode length = {round(episode_length_mean, 3)}  '
              f'flags found = {len(flags_found)} avg. {round(np.mean(flags_found), 3)} steps')


# Running the simulations
def run(sim_model, n_simulations=10, total_timesteps=10 ** 6):
    print(sim_model)

    for i in tqdm(range(n_simulations)):
        sim_model[i].learn(total_timesteps=total_timesteps, log_interval=1, callback=eval_callback,
                           tb_log_name="PPO", reset_num_timesteps=False)
        sim_model[i].save('tests/out/sim_db_flag_{0}'.format(str(i)))

    return sim_model


def select_model(models):
    # keys to upper
    models = {k.upper(): v for k, v in models.items()}

    print()
    print(f"Models: ")
    for i, option in enumerate(models.keys()):
        print(f"- {option}")
    print()

    selected_option = input("Select a model: ")
    selected_option = selected_option.upper()

    while not selected_option or selected_option not in models:
        selected_option = input("Invalid input. Select a model: ")

    selected_model = models[selected_option]
    print()

    return selected_model


if __name__ == '__main__':
    env_name = 'db_sim-v2'
    step_limit = 184
    reward_limit = 10
    history_length = 4
    attack_values = 13
    reward_threshold = 1

    # specify where to save monitor results
    monitor_dir = 'tests/out/monitor_results'
    log_dir = "./logs/flag/"

    env = gym.make(env_name, verbose=False,
                   step_limit=step_limit,
                   reward_threshold=reward_threshold,
                   reward_limit=reward_limit,
                   history_length=history_length,
                   attack_values=attack_values,
                   rewards=ErrorReward(nothing=-1, db_type=-1, flag=10, escape=-1, table=-1))

    # create a tensorboard writer
    tb_writer = tensorboardX.SummaryWriter(log_dir, max_queue=1, flush_secs=0)

    # define the evaluation callback
    eval_callback = EvalCallback(
        eval_env=env,
        callback_on_new_best=None,
        eval_freq=5000,
        n_eval_episodes=100,
        log_path=log_dir,
        best_model_save_path=f"{log_dir}best_model",
        verbose=1,
    )

    n_simulations = 1
    total_timesteps = (10 ** 6)  # amount of time needed for convergence

    dqn_model = [
        DQN(env=env,
            policy='MlpPolicy',
            verbose=1,
            _init_setup_model=True,
            tensorboard_log=f"{log_dir}dqn-{i}-{int(time.time())}")
        for i in range(n_simulations)
    ]
    ppo_model = [
        PPO(env=env,
            policy='MlpPolicy',
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=32,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            clip_range_vf=None,
            normalize_advantage=True,
            ent_coef=0.0,
            vf_coef=0.5,
            max_grad_norm=0.5,
            use_sde=False,
            sde_sample_freq=-1,
            _init_setup_model=True,
            tensorboard_log=f"{log_dir}ppo-{i}-{int(time.time())}")
        for i in range(n_simulations)
    ]

    # choose model
    model = select_model({"PPO": ppo_model,
                          "DQN": dqn_model})
    model = run(model, n_simulations=n_simulations, total_timesteps=total_timesteps)
    # evaluate(model, n_simulations=n_simulations, total_timesteps=10 ** 5)

    print(f"Action-Space num_actions={env.actions}")
    print()
    print(f"Run tensorboard logs cmd:  tensorboard --logdir={log_dir} --port=6006")

    env.close()
    tb_writer.close()
