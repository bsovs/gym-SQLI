import gym
import numpy as np

from sqli_sim.envs._helper.error_message import ESCAPE_CHAR
from sqli_sim.envs._helper.reward import ErrorReward


def run():
    obs = env.reset()
    done = False

    print()
    for action in env.command_dict:
        print(f"[{action}]  {env.command_dict[action]}")
    print()

    print(env.table)
    print(env.table_action)
    print(env.injection_command)
    print()

    action_list = [[[] for _ in ESCAPE_CHAR], [[] for _ in ESCAPE_CHAR], [[] for _ in ESCAPE_CHAR]]
    for action in env.command_dict:
        if 21 > action % 37 > 0:
            f = action % 37
            e = action // 37
            if 5 > f > 0:
                action_list[0][e].append(action)
            else:
                action_list[1][e].append(action)
                action_list[2][e].append(action + 16)

    i = 0
    j = 0
    prev_total = 0
    while not done:
        # action = input(f"Enter action (0 to {env.actions}): ")
        # action = int(action)

        if i == 0:
            action = action_list[i][j].pop()
        elif prev_total == 6:
            action = action + 16
        else:
            action = action_list[i][e].pop()

        # Take a step
        obs, reward, done, info = env.step(action)

        if sum(obs[action]) > prev_total:
            i += 1
            e = action // 37
        prev_total = sum(obs[action])

        if i == 0 and prev_total == 0:
            j += 1

        # Display the reward and done flag
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Observations: [TABLE, TABLE_ACTION, ESCAPE_CHAR, VALUE]")
        print(f"Observation: action={action}  {obs[action]}")
        print(f"{env.command_dict[action]}")

    print()
    print(f"VICTORY!  TOOK [{env.steps}] STEPS TO FIND FLAG!")
    return env.steps


if __name__ == '__main__':
    env_name = 'db_sim-v2'
    step_limit = 184
    reward_limit = 10
    history_length = 4
    attack_values = 13

    env = gym.make(env_name, verbose=False,
                   step_limit=step_limit,
                   reward_limit=reward_limit,
                   history_length=history_length,
                   attack_values=attack_values,
                   rewards=ErrorReward(nothing=-1, db_type=-1, flag=10, escape=-1, table=-1))

    trials = 1000
    steps_taken = [run() for _ in range(trials)]

    print()
    print(f"Max Steps Taken: {np.max(steps_taken)}")
    print(f"Min Steps Taken: {np.min(steps_taken)}")
    print(f"Mean Steps Taken in {trials} trials: {np.mean(steps_taken)}")

    env.close()
