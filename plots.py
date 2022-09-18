import warnings
import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from train_eval import N_RUNS, LABELS, NAMES, COLORS, N_AGENTS, timer

sns.set_theme()


def moving_average(x, size=None, zero_start=True):
    if size is None:
        size = len(x) // 10
    assert size <= len(x), 'Moving average size larger than length of array'
    ma = np.zeros_like(x, dtype=float)
    if not zero_start:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for i in range(size - 1):
                ma[i] = np.mean(x[max(0, i - size + 1):i + 1])
    ma[size - 1:] = np.convolve(x, np.ones(size) / size, mode='valid')
    return ma


def plot_individual_runs(moving_averages):
    for agent, color, label in zip(moving_averages, COLORS, LABELS):
        for i, run in enumerate(agent):
            sns.lineplot(data=run, color=color, label=f'Run {i + 1}')
        plt.title(label)
        plt.xlabel('Step')
        plt.ylabel('Reward')
        plt.show()


def plots(moving_averages, title, save_file):
    n_episodes = moving_averages.shape[-1]
    mean_rewards = np.mean(moving_averages, axis=1)
    crit_value = st.t.ppf(q=0.975, df=N_RUNS - 1)  # small sample size -> t-distribution
    ci = crit_value * np.std(moving_averages, axis=1, ddof=1) / (N_RUNS ** 0.5)
    for agent_means, agent_ci, label, color in zip(mean_rewards, ci, LABELS, COLORS):
        sns.lineplot(data=agent_means, label=label, color=color)
        lcb, ucb = agent_means - agent_ci, agent_means + agent_ci  # upper/lower confidence bounds
        plt.fill_between(range(n_episodes), lcb, ucb, color=color, alpha=.1)
    plt.title(title)
    xlabel = 'Episode' if 'FrozenLake' in title else 'Step'
    plt.xlabel(xlabel)
    plt.ylabel('Reward')
    plt.legend(title='Agents', facecolor='white')
    plt.savefig(fname=save_file)
    plt.clf()


def visualize_Q_values(path, agent, env_name, label):
    q_table = np.load(path)
    sns.heatmap(q_table, cmap='Blues', fmt='.2f', annot=True)
    plt.title(f'Q-values\n{env_name} - {label}')
    plt.xlabel('Actions')
    xticks = np.arange(0.5, 4.5) if env_name == 'FrozenLake-v1' else np.arange(0.5, 2.5)
    xticklabels = ('Left', 'Down', 'Right', 'Up') if env_name == 'FrozenLake-v1' else ('Left', 'Right')
    plt.xticks(xticks, xticklabels)
    plt.ylabel('States')
    plt.savefig(f'./Plots/Q_values_{env_name}_{agent}.png')
    plt.clf()


def visualize_state_values(path, agent, env_name, label):
    q_table = np.load(path)
    state_values = 0.05 * np.mean(q_table, axis=-1) + 0.95 * np.max(q_table, axis=-1)
    shape = (4, 4) if env_name == 'FrozenLake-v1' else (1, 6)
    state_values = np.reshape(state_values, shape)
    sns.heatmap(state_values, cmap='Blues', fmt='.2f', annot=True)
    plt.title(f'State values\n{env_name} - {label}')
    plt.savefig(f'./Plots/state_values_{env_name}_{agent}.png')
    plt.clf()


def visualize_greedy_policy(path, env_name, label):
    if 'FrozenLake' in env_name:
        action_map = (u"\u2190", u"\u2193", u"\u2192", u"\u2191")  # LEFT, DOWN, RIGHT, UP
    else:
        action_map = (u"\u2190", u"\u2192")  # LEFT, RIGHT
    q_table = np.load(path)
    actions = np.argmax(q_table, axis=-1)
    greedy_policy = np.array(action_map)[actions]
    shape = (4, 4) if env_name == 'FrozenLake-v1' else (6,)
    greedy_policy = np.reshape(greedy_policy, shape)
    print(f'\n{env_name} - {label} greedy policy:')
    print(greedy_policy)


def main():
    ma_size = None  # moving average window size. default if None: N // 10

    for env_name in ('FrozenLake-v1', 'riverswim'):

        print(f'\n{env_name}')

        # training moving averages
        rewards_train = np.load(f'./Rewards/{env_name}_train.npy')
        ma_train = np.zeros_like(rewards_train)
        for i_agent in range(N_AGENTS):
            for i_run in range(N_RUNS):
                ma_train[i_agent, i_run] = moving_average(rewards_train[i_agent, i_run], ma_size)

        # training plots
        with timer(f'Training plot'):
            save_file = f'./Plots/{env_name}_train.png'
            title = f'{env_name}\nMoving average reward during training'
            plots(ma_train, title, save_file)

        # evaluation moving averages
        rewards_eval = np.load(f'./Rewards/{env_name}_eval.npy')
        ma_eval = np.zeros_like(rewards_eval)
        for i_agent in range(N_AGENTS):
            for i_run in range(N_RUNS):
                ma_eval[i_agent, i_run] = moving_average(rewards_eval[i_agent, i_run], ma_size)

        # evaluation plot function
        with timer(f'Evaluation plot'):
            save_file = f'./Plots/{env_name}_eval.png'
            title = f'{env_name}\nMoving average reward during evaluation'
            plots(ma_eval, title, save_file)

        # visualizing q-values and state values
        with timer('Visualizations'):
            for agent, label in zip(NAMES, LABELS):
                path = f'Q_tables/{env_name}_{agent}.npy'
                visualize_Q_values(path, agent, env_name, label)
                visualize_state_values(path, agent, env_name, label)

        # visualizing greedy policy
        with timer('Greedy policy'):
            for agent, label in zip(NAMES, LABELS):
                visualize_greedy_policy(path, env_name, label)


if __name__ == '__main__':
    with timer('\nPlots'):
        main()
