import gym

from sqli_sim.envs._helper.reward import ErrorReward
import sqli_sim.envs

if __name__ == '__main__':
    env_name = 'db_sim-v2'
    step_limit = 50
    reward_limit = 10
    history_length = 4
    attack_values = 13

    env = gym.make(env_name, verbose=False,
                   step_limit=step_limit,
                   reward_limit=reward_limit,
                   history_length=history_length,
                   attack_values=attack_values,
                   rewards=ErrorReward(nothing=-1, db_type=-1, flag=100, escape=-1, table=-1))

    # Check if the environment follows the MDP properties
    if not isinstance(env.observation_space, gym.spaces.Discrete):
        print("observation_space not Discrete")
    if not isinstance(env.action_space, gym.spaces.Discrete):
        print("action_space not Discrete")
    if env.reward_threshold is None:
        print("reward_threshold is None")

    if (
            isinstance(env.observation_space, gym.spaces.Discrete) or isinstance(env.observation_space, gym.spaces.Box)
    ) and isinstance(env.action_space,
                     gym.spaces.Discrete) and env.reward_threshold is not None:
        print("The environment is a Markov Decision Process")
    else:
        print("The environment is not a Markov Decision Process")

    # Steps to check if a gym environment with Box observation space is an MDP:
    # 1. Verify that the environment satisfies the Markov property by:
    #    - Setting the environment to a specific state s.
    #    - Taking an action a and observing the next state s'.
    #    - Resetting the environment to the same state s.
    #    - Taking the same action a again and observing the next state s''.
    #    - Comparing s' and s'' to verify that they are the same.
    #    - If s' and s'' are the same, then the environment satisfies the Markov property.
    # 2. Verify that the environment satisfies the state-transition probabilities property by:
    #    - Setting the environment to a specific state s.
    #    - Taking an action a and observing the next state s'.
    #    - Repeating this process multiple times to obtain a sample of state transitions.
    #    - Computing the empirical probabilities of transitioning from state s to state s' given action a.
    #    - Verifying that the probabilities sum to 1.
    #    - If the probabilities sum to 1, then the environment satisfies the state-transition probabilities property.
    # 3. Verify that the environment satisfies the reward function property by:
    #    - Setting the environment to a specific state s.
    #    - Taking an action a and observing the next state s'.
    #    - Computing the reward obtained by the agent for taking action a in state s.
    #    - Changing the action taken to a different action a' and computing the reward obtained by the agent for taking action a' in state s.
    #    - Verifying that the rewards obtained are different for different actions.
    #    - If the rewards obtained are different for different actions, then the environment satisfies the reward function property.
