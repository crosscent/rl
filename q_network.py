from collections import deque

import numpy as np
import gym

from agents.QNetwork import QNetwork

EPISODE_COUNT = 200


def q_network():
    env = gym.make('CartPole-v1')
    agent = QNetwork(env.action_space.n, env.observation_space.shape[0])
    train(agent, env)


def train(agent, env):
    past_rewards = deque(maxlen=20)
    for i in range(EPISODE_COUNT):
        state = env.reset()
        rewards = 0

        for _ in range(300):
            action = agent.get_action(state).item()
            next_state, reward, done, _ = env.step(action)
            rewards += reward

            # post processing
            if done:
                agent.add_history(state, action, reward, None)
            else:
                agent.add_history(state, action, reward, next_state)
            agent.learn()

            # prepare for next iteration
            if done:
                break
            else:
                state = next_state

        if i % 10 == 0:
            agent.update_target()

        past_rewards.append(rewards)
        if np.mean(past_rewards) > 200:
            return
        print("Episode: ", i, "Rewards:", np.mean(past_rewards))


if __name__ == '__main__':
    q_network()
