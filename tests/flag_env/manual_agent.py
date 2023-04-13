import gym

from sqli_sim.envs._helper.error_message import ESCAPE_CHAR
from sqli_sim.envs._helper.reward import ErrorReward


def run():
    env.reset()
    done = False

    print()
    for action in env.command_dict:
        print(f"[{action}]  {env.command_dict[action]}")
    print()

    print(env.table)
    print(env.table_action)
    print(env.injection_command)
    print()

    while not done:
        action = input(f"Enter action (0 to {env.actions}): ")
        action = int(action)

        # Take a step
        obs, reward, done, info = env.step(action)

        # Display the reward and done flag
        print(f"Reward: {reward}")
        print(f"Done: {done}")
        print(f"Observations: [TABLE, TABLE_ACTION, ESCAPE_CHAR, VALUE]")
        print(f"Observation: action={action}  {obs[action]}")
        print(f"{env.command_dict[action]}")

    print()
    print(f"VICTORY!  TOOK [{env.steps}] STEPS TO FIND FLAG!")


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

    run()
    env.close()
