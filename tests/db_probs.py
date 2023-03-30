import random

import gym

from sqli_sim.envs._helper.error_message import ESCAPE_CHAR
from sqli_sim.envs._helper.reward import ErrorReward


def simulate(steps, attack_values, num_actions, prob_escape, prob_flag, num_simulations=10000):
    """
    Monte Carlo simulation:
    expected value of a function which gets a reward of -10 if wrong and +1 if right
    where there are num_actions total cases and attack_values cases where we can get the reward,
    but it has probability 0.5 where we can take 100 steps.
    """
    estimate = 0
    for i in range(num_simulations):
        total_reward = 0
        reward = 0
        for j in range(steps):
            if reward <= 0:
                action = random.randint(1, num_actions)
                rand = random.random()
                if rand <= prob_flag:
                    reward = flag_reward
                elif action >= attack_values and rand <= prob_escape:
                    reward = table_reward
                elif action <= attack_values and rand <= prob_escape:
                    reward = escape_reward
                else:
                    reward = wrong_reward
            total_reward += reward
            if total_reward >= reward_limit:
                break
        estimate += total_reward
    return estimate / num_simulations


def expected_value(attack_values, num_actions):
    p_wrong = 0.5 * (1 - attack_values / num_actions)
    p_right = 0.5 * (attack_values / num_actions)
    return (p_wrong * (-10) + p_right * (+1))


def expected_reward():
    num_actions = env.actions
    prob_flag = 0.5 / num_actions  # flag exists in (0,1) state
    prob_escape = 0.5 / (len(ESCAPE_CHAR) / (len(ESCAPE_CHAR) - 2)) - prob_flag  # (1, len(ESCAPE_CHAR)) minus empty str
    prob_table = prob_escape * ((num_actions - attack_values) / num_actions) * 0.5
    prob_wrong = (1.0 - prob_escape)

    print()
    print("REWARD PROBABILITIES")
    print(f"flag   = {prob_flag:3f}  r = {prob_flag * flag_reward:1f}")
    print(f"escape = {(prob_escape - prob_table):3f}  r = {(prob_escape - prob_table) * escape_reward:1f}")
    print(f"table  = {prob_table:3f}  r = {prob_table * table_reward:1f}")
    print(f"wrong  = {prob_wrong:3f}  r = {prob_wrong * wrong_reward:1f}")

    # RANDOM
    random_reward = 0
    random_reward += prob_flag * flag_reward  # flag exists in (0,1) state, reward = 100
    random_reward += (prob_escape - prob_table) * escape_reward  # (1, len(ESCAPE_CHAR)) == 0.5, reward = 1
    random_reward += prob_table * table_reward
    random_reward += (1.0 - prob_escape) * wrong_reward

    max_reward = flag_reward
    min_reward = wrong_reward

    # simulate a smart action scheme where we exploit positive rewards using monte carlo
    # just take one of each escape char until it hits then keep using it to get reward=1
    monte_carlo = simulate(steps=step_limit, attack_values=attack_values, num_actions=num_actions,
                           prob_escape=prob_escape,
                           prob_flag=prob_flag)

    print()
    print("REWARD SYSTEM")
    print()
    print(f"Max Trial    = {max_reward}")
    print(f"Min Trial    = {min_reward * step_limit}")
    print()
    print(f"For each step:")
    print(f"Random Reward = {random_reward}")
    print(f"Smart Reward   = {monte_carlo / step_limit}")
    print()
    print(f"For each trial:")
    print(f"Random Reward = {random_reward * step_limit}")
    print(f"Smart Reward   = {monte_carlo}")
    print()


if __name__ == '__main__':
    env_name = 'db_sim-v1'
    step_limit = 100
    reward_limit = 10
    history_length = 4
    attack_values = 100

    rewards = ErrorReward()
    wrong_reward = rewards.nothing
    escape_reward = rewards.escape
    flag_reward = rewards.flag
    table_reward = rewards.table

    env = gym.make(env_name, verbose=False,
                   step_limit=step_limit,
                   reward_limit=reward_limit,
                   history_length=history_length,
                   attack_values=attack_values)

    expected_reward()
