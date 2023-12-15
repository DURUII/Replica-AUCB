from matplotlib import pyplot as plt
import scienceplots

plt.style.use(['science', 'grid'])

from tqdm import tqdm

from arms import StrategicArm
from algorithms import *


class Emulator:
    def __init__(self, n_arms=60, n_selected=20, budget=5e5):
        self.N = n_arms
        self.K = n_selected
        self.B = budget
        self.arms = [StrategicArm() for _ in range(self.N)]

    def simulate(self, solver=AUCB):
        algo = solver(self.arms, self.N, self.K, self.B)
        algo.initialize()
        return algo.loop()


if __name__ == '__main__':
    # Initialize the range of parameters based on the requirements
    N_values = [50, 60, 70, 80, 90, 100]
    K_values = [10, 20, 30, 40, 50]
    budget_values = np.logspace(4, 6, num=20)

    # Prepare the data for the plots
    results_rewards_vs_budget = []
    results_rounds_vs_budget = []
    results_rewards_vs_N = []
    results_rounds_vs_N = []
    results_rewards_vs_K = []
    results_rounds_vs_K = []

    # Simulate for different budgets
    for B in tqdm(budget_values):
        emu = Emulator(budget=B)
        total_reward, rounds = emu.simulate()
        results_rewards_vs_budget.append(total_reward / 1e4)
        results_rounds_vs_budget.append(rounds / 1e3)

    # Simulate for different N (number of arms)
    for N in tqdm(N_values):
        emu = Emulator(n_arms=N)
        total_reward, rounds = emu.simulate()
        results_rewards_vs_N.append(total_reward / 1e4)
        results_rounds_vs_N.append(rounds / 1e4)

    # Simulate for different K (number of arms selected in each round)
    for K in tqdm(K_values):
        emu = Emulator(n_selected=K)
        total_reward, rounds = emu.simulate()
        results_rewards_vs_K.append(total_reward / 1e4)
        results_rounds_vs_K.append(rounds / 1e4)

    # Plotting
    fig, axs = plt.subplots(3, 2, figsize=(12.5, 8))

    # Figure 1: Total Achieved Rewards vs. Budget
    axs[0, 0].plot(budget_values, results_rewards_vs_budget)
    axs[0, 0].set_xscale('log')
    axs[0, 0].set_xlabel('Budget')
    axs[0, 0].set_ylabel('Total Rewards (x1e4)')
    axs[0, 0].set_title('Figure 1: Total Achieved Rewards vs. Budget')

    # Figure 2: Total Rounds vs. Budget
    axs[0, 1].plot(budget_values, results_rounds_vs_budget)
    axs[0, 1].set_xscale('log')
    axs[0, 1].set_xlabel('Budget')
    axs[0, 1].set_ylabel('Total Rounds (x1e3)')
    axs[0, 1].set_title('Figure 2: Total Rounds vs. Budget')

    # Figure 3: Rewards vs. N (Number of Arms)
    axs[1, 0].bar(N_values, results_rewards_vs_N)
    axs[1, 0].set_xlabel('Number of Arms (N)')
    axs[1, 0].set_ylabel('Total Rewards (x1e4)')
    axs[1, 0].set_title('Figure 3: Rewards vs. N (Number of Arms)')

    # Figure 4: Rounds vs. N (Number of Arms)
    axs[1, 1].plot(N_values, results_rounds_vs_N)
    axs[1, 1].set_xlabel('Number of Arms (N)')
    axs[1, 1].set_ylabel('Total Rounds (x1e4)')
    axs[1, 1].set_title('Figure 4: Rounds vs. N (Number of Arms)')

    # Figure 5: Rewards vs. K (Number of Arms Selected in Each Round)
    axs[2, 0].bar(K_values, results_rewards_vs_K)
    axs[2, 0].set_xlabel('K (Number of Arms Selected in Each Round)')
    axs[2, 0].set_ylabel('Total Rewards (x1e4)')
    axs[2, 0].set_title('Figure 5: Rewards vs. K (Number of Arms Selected in Each Round)')

    # Figure 6: Rounds vs. K (Number of Arms Selected in Each Round)
    axs[2, 1].plot(K_values, results_rounds_vs_K)
    axs[2, 1].set_xlabel('K (Number of Arms Selected in Each Round)')
    axs[2, 1].set_ylabel('Total Rounds (x1e4)')
    axs[2, 1].set_title('Figure 6: Rounds vs. K (Number of Arms Selected in Each Round)')

    plt.tight_layout()
    plt.show()
