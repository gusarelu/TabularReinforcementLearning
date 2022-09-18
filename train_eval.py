import gym
import numpy as np
from time import time
from q_agent import Agent as q_agent
from dq_agent import Agent as dq_agent
from sarsa_agent import Agent as sarsa_agent
from contextlib import contextmanager

N_RUNS = 5  # number of independent runs per agent type
LABELS = ('Q-learning', 'Double Q-learning', 'SARSA')  # for plots
NAMES = ('q_agent', 'dq_agent', 'sarsa_agent')  # for save files
COLORS = ('b', 'r', 'g')
AGENTS = (q_agent, dq_agent, sarsa_agent)
N_AGENTS = len(AGENTS)


@contextmanager
def timer(name):
    start = time()
    yield
    end = time()
    print(f'{name}: {end - start:.2f} seconds.')


def load_env(env_name='FrozenLake-v1'):
    try:
        env = gym.make(env_name)
        print('Loaded', env_name)
    except gym.error.NameNotFound:
        print(env_name + ':Env')
        gym.envs.register(
            id=env_name + '-v0',
            entry_point=env_name + ':Env',
        )
        env = gym.make(env_name + '-v0')
        print('Loaded', env_name)
    return env


def train_agent(env, agent, N=1000, episodic_task=True):
    rewards = np.zeros(N)
    if episodic_task:
        for ep in range(N):
            observation = env.reset()
            done = False
            while not done:
                action = agent.act(observation)
                observation, reward, done, _ = env.step(action)
                agent.observe(observation, reward, done)
            rewards[ep] = reward
    else:
        observation = env.reset()
        for step in range(N):
            action = agent.act(observation)
            observation, reward, done, _ = env.step(action)
            agent.observe(observation, reward, done)
            rewards[step] = reward
            if done:
                observation = env.reset()
    return rewards


def eval_agent(env, agent, N=1000, episodic_task=True):
    rewards = np.zeros(N)
    with agent.greedy():  # temporarily sets agent's epsilon to zero
        if episodic_task:
            for ep in range(N):
                observation = env.reset()
                done = False
                while not done:
                    action = agent.act(observation)
                    observation, reward, done, _ = env.step(action)
                rewards[ep] = reward
        else:
            observation = env.reset()
            for step in range(N):
                action = agent.act(observation)
                observation, reward, done, _ = env.step(action)
                rewards[step] = reward
                if done:
                    observation = env.reset()
    return rewards


def main():
    N = 10_000  # number of training episodes (or steps, depending on whether task is episodic or not)

    for env_name in ('FrozenLake-v1', 'riverswim'):
        episodic_task = True if env_name == 'FrozenLake-v1' else False  # riverswim env is not episodic

        # loading environment
        env = load_env(env_name)
        action_dim = env.action_space.n
        state_dim = env.observation_space.n

        # instantiating agents
        agents = [[agent(state_dim, action_dim) for _ in range(N_RUNS)] for agent in AGENTS]

        # training loop
        with timer('Training loop'):
            rewards_train = np.zeros((N_AGENTS, N_RUNS, N))
            for i_agent, agent_type in enumerate(agents):
                for i_run, agent in enumerate(agent_type):
                    rewards_train[i_agent, i_run] = train_agent(env, agent, N, episodic_task)
        path = f'./Rewards/{env_name}_train'
        np.save(path, rewards_train)

        # evaluation loop
        with timer('Evaluation loop'):
            rewards_eval = np.zeros((N_AGENTS, N_RUNS, N))
            for i_agent, agent_type in enumerate(agents):
                for i_run, agent in enumerate(agent_type):
                    rewards_eval[i_agent, i_run] = eval_agent(env, agent, N, episodic_task)
        path = f'./Rewards/{env_name}_eval'
        np.save(path, rewards_eval)

        # saving Q-table of best agent for each type
        sum_rewards = np.sum(rewards_eval, axis=-1)  # (N_AGENTS, N_RUNS)
        indices = np.argmax(sum_rewards, axis=-1)  # (N_AGENTS,) index of best agent per agent_type
        for agent_type, index, name in zip(agents, indices, NAMES):
            agent = agent_type[index]
            path = f'./Q_tables/{env_name}_{name}'
            agent.save_Q_table(path)


if __name__ == '__main__':
    with timer('\nTotal'):
        main()
