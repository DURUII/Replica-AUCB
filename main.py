from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from arms import StrategicArm  # Ensure this module exists and is correctly implemented.
from algorithms import *  # Ensure these are implemented.

plt.style.use(['science', 'grid'])


class Emulator:
    def __init__(self, n_arms=60, n_selected=20, budget=5e5):
        self.N = n_arms
        self.K = n_selected
        self.B = budget
        self.arms = [StrategicArm() for _ in range(self.N)]

    def simulate(self, solver, epsilon=None):
        if epsilon is not None:
            algo = solver(self.arms, self.N, self.K, self.B, epsilon=epsilon)
        else:
            algo = solver(self.arms, self.N, self.K, self.B)
        algo.initialize()
        return algo.loop()


if __name__ == '__main__':
    """
    Authorship:
    Primary Development: ChatGPT
    Guidance and Conceptualization: DURUII
    """

    # Define colors and markers for each algorithm
    algo_styles = {
        'AUCB': {'color': 'black', 'marker': 's', 'label': 'AUCB'},  # Square marker
        'optimal': {'color': 'red', 'marker': 'o', 'label': 'optimal'},  # Circle marker
        'separated': {'color': 'blue', 'marker': '^', 'label': 'separated'},  # Triangle up marker
        '0.1-first': {'color': 'green', 'marker': 'v', 'label': '0.1-first'},  # Triangle down marker
        '0.5-first': {'color': 'purple', 'marker': '<', 'label': '0.5-first'},  # Diamond marker
    }

    # Correct the keys in the results dictionary to match those used in the plotting loop
    results = {
        'AUCB': {'rewards': [], 'rounds': []},
        'optimal': {'rewards': [], 'rounds': []},  # Change 'OptimalAlgorithm' to 'optimal'
        'separated': {'rewards': [], 'rounds': []},  # Change 'SeparatedAlgorithm' to 'separated'
        '0.1-first': {'rewards': [], 'rounds': []},  # Change 'EpsilonFirstAlgorithm_0.1' to '0.1-first'
        '0.5-first': {'rewards': [], 'rounds': []},  # Change 'EpsilonFirstAlgorithm_0.5' to '0.5-first'
    }

    # Simulation parameters
    N_values = [50, 60, 70, 80, 90, 100]
    K_values = [10, 20, 30, 40, 50]
    budget_values = np.logspace(4, 6, num=20)

    # Run simulations for different algorithms
    for B in tqdm(budget_values, desc="Budgets"):
        emu = Emulator(budget=B)
        for algo_label in algo_styles:
            if algo_label == '0.1-first':
                algo = lambda arms, N, K, B: EpsilonFirstAlgorithm(arms, N, K, B, epsilon=0.1)
            elif algo_label == '0.5-first':
                algo = lambda arms, N, K, B: EpsilonFirstAlgorithm(arms, N, K, B, epsilon=0.5)
            elif algo_label == 'optimal':
                algo = OptimalAlgorithm
            elif algo_label == 'separated':
                algo = SeparatedAlgorithm
            elif algo_label == 'AUCB':
                algo = AUCB

            total_reward, rounds = emu.simulate(algo)
            results[algo_styles[algo_label]['label']]['rewards'].append(total_reward)
            results[algo_styles[algo_label]['label']]['rounds'].append(rounds)

    # Plotting the results
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 10.5))

    # Line plots for budget vs rewards and rounds
    for algo_label, style in algo_styles.items():
        axs[0, 0].plot(budget_values, results[style['label']]['rewards'], label=style['label'],
                       color=style['color'], marker=style['marker'])
        axs[0, 1].plot(budget_values, results[style['label']]['rounds'], label=style['label'],
                       color=style['color'], marker=style['marker'], )

    # Reset results for varying K
    for key in results:
        results[key]['rewards'] = []
        results[key]['rounds'] = []

    # Simulate over different N (number of arms)
    for N in tqdm(N_values, desc="Number of Arms"):
        emu = Emulator(n_arms=N, n_selected=20, budget=5e5)  # Use a fixed budget and K for N simulations
        for algo_label in algo_styles:
            if algo_label == '0.1-first':
                algo = lambda arms, N, K, B: EpsilonFirstAlgorithm(arms, N, K, B, epsilon=0.1)
            elif algo_label == '0.5-first':
                algo = lambda arms, N, K, B: EpsilonFirstAlgorithm(arms, N, K, B, epsilon=0.5)
            elif algo_label == 'optimal':
                algo = OptimalAlgorithm
            elif algo_label == 'separated':
                algo = SeparatedAlgorithm
            elif algo_label == 'AUCB':
                algo = AUCB
            total_reward, rounds = emu.simulate(algo)
            results[algo_styles[algo_label]['label']]['rewards'].append(total_reward)
            results[algo_styles[algo_label]['label']]['rounds'].append(rounds)

    # Line plots for N vs rewards and rounds
    for algo_label, style in algo_styles.items():
        axs[1, 0].plot(N_values, results[style['label']]['rewards'], label=style['label'],
                       color=style['color'], marker=style['marker'])
        axs[1, 1].plot(N_values, results[style['label']]['rounds'], label=style['label'],
                       color=style['color'], marker=style['marker'], )

    # Reset results for varying K
    for key in results:
        results[key]['rewards'] = []
        results[key]['rounds'] = []

    # Simulate over different K (number of selected arms)
    for K in tqdm(K_values, desc="Selected Arms"):
        emu = Emulator(n_arms=60, n_selected=K, budget=5e5)  # Use a fixed budget and N for K simulations
        for algo_label in algo_styles:
            if algo_label == '0.1-first':
                algo = lambda arms, N, K, B: EpsilonFirstAlgorithm(arms, N, K, B, epsilon=0.1)
            elif algo_label == '0.5-first':
                algo = lambda arms, N, K, B: EpsilonFirstAlgorithm(arms, N, K, B, epsilon=0.5)
            elif algo_label == 'optimal':
                algo = OptimalAlgorithm
            elif algo_label == 'separated':
                algo = SeparatedAlgorithm
            elif algo_label == 'AUCB':
                algo = AUCB
            total_reward, rounds = emu.simulate(algo)
            results[algo_styles[algo_label]['label']]['rewards'].append(total_reward)
            results[algo_styles[algo_label]['label']]['rounds'].append(rounds)

    # Line plots for K vs rewards and rounds
    for algo_label, style in algo_styles.items():
        axs[2, 0].plot(K_values, results[style['label']]['rewards'], label=style['label'],
                       color=style['color'], marker=style['marker'])
        axs[2, 1].plot(K_values, results[style['label']]['rounds'], label=style['label'],
                       color=style['color'], marker=style['marker'], )

    # Configure axes and titles
    for i, ax in enumerate(axs.flat):
        ax.set_xlabel('Budget ($10^4$)' if i % 2 == 0 else 'Budget ($10^3$)')
        ax.set_ylabel('Total Rewards' if i % 2 == 0 else 'Total Rounds')
        ax.set_title(
            f'Fig. {i + 1}. Total {("rewards" if i % 2 == 0 else "rounds")} vs. {"budget" if i < 2 else ("N" if i < 4 else "K")}')
        ax.legend(bbox_to_anchor=(0, 1.02, 1, 0.2), loc="upper left", mode="expand", ncol=5)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('fig.pdf')
