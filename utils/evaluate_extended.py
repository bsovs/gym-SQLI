import numpy as np


def evaluate_model_nondeter(model, env, num_steps=1000):
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        if (isinstance(action, np.ndarray)):
            obs, reward, done, _ = env.step(action[0])
        else:
            obs, reward, done, _ = env.step(action)

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    mean_reward = round(np.mean(episode_rewards), 3)
    median_reward = np.median(episode_rewards)
    return mean_reward, len(episode_rewards) - 1, median_reward


def evaluate_model_deterministic(model, env, num_steps):
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    mean_reward = round(np.mean(episode_rewards), 3)
    median_reward = np.median(episode_rewards)
    return mean_reward, len(episode_rewards) - 1, median_reward


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


def test_episodes(model, env, num_episodes, max_steps=1000):
    episode_rewards = []
    for j in range(num_episodes):
        done = False
        steps = 0
        obs = env.reset()
        episode_rewards.append(0.0)
        while (not done and steps < max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, _ = env.step(action)

            episode_rewards[-1] += reward
            steps += 1
        # print("steps", steps, end = ",")
        print("j", j, end=" ")

    return (episode_rewards)


def plot_evaluation_determistic(model, env, num_steps):
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    mean_reward = round(np.mean(episode_rewards), 3)
    median_reward = np.median(episode_rewards)

    return (episode_rewards)


def plot_evaluation(model, env, num_steps):
    episode_rewards = [0.0]
    obs = env.reset()
    for i in range(num_steps):
        action, _states = model.predict(obs)
        if (isinstance(action, np.ndarray)):
            obs, reward, done, _ = env.step(action[0])
        else:
            obs, reward, done, _ = env.step(action)

        episode_rewards[-1] += reward
        if done:
            obs = env.reset()
            episode_rewards.append(0.0)

    return (episode_rewards)
