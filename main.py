import gym
from gym.wrappers import Monitor

from agents.DQN import DeepQNetwork


def train(agent: DeepQNetwork, env):
    step = 0
    for episode in range(1000):
        observation = env.reset()
        while True:
            # RL choose action based on observation
            action = agent.choose_action(observation)

            # RL take action and get next observation and reward
            observation_, reward, done, info = env.step(action)

            agent.store_transition(observation, action, reward, observation_)

            if step % 5 == 0:
                agent.learn()

            # swap observation
            observation = observation_

            if done:
                break
            step += 1


def main():
    env = Monitor(gym.make("CartPole-v1"), 'cartpole', force=True)
    agent = DeepQNetwork(dim_actions=env.action_space.n,
                         dim_states=env.observation_space.shape[0],
                         learning_rate=0.0000001,
                         batch_size=5000,
                         replace_target_iter=200)
    train(agent, env)


if __name__ == '__main__':
    main()
