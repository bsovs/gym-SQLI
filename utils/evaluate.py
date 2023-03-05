import numpy as np


def evaluate(model, env, num_steps=1000, verbose=False):
    episode_rewards = [0.0]
    episode_length = [0]
    obs = env.reset()
    for i in range(num_steps):
        action, _ = model.predict(obs)
        # Need to take the first element in action, as sometimes it is a vector of length n.
        if (np.shape(action) != ()):
            action = action[0]
            if (verbose): print(action)
        obs, reward, done, _ = env.step(action)
        episode_rewards[-1] += reward
        episode_length[-1] += 1
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            episode_length.append(0)
            if (verbose): print("vicory")

    mean_reward = round(np.mean(episode_rewards[:-1]), 3)
    max_reward = round(np.max(episode_rewards), 3)

    return mean_reward, max_reward, np.mean(episode_length[:-1])


def evaluate_model(model, env, num_steps=1000, verbose=False):
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        action, _ = model.predict(obs)
        # Need to take the first element in action, as sometimes it is a vector of length n.
        if (np.shape(action) != ()):
            action = action[0]
            if (verbose): print(action)
        obs, reward, done, _ = env.step(action)
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)
            if (verbose): print("vicory")

    mean_reward = round(np.mean(episode_rewards), 3)
    return mean_reward, len(episode_rewards) - 1


def evaluate_random(env, num_steps=1000):
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        obs, reward, done, _ = env.step(env.action_space.sample())
        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    mean_reward = round(np.mean(episode_rewards), 3)
    return mean_reward, len(episode_rewards) - 1
