# Author: DURUII
# Date: 2023.12.16
# This Python file implements the Auction-based UCB (AUCB) algorithm,
# as described in the paper. The AUCB algorithm selects a subset of arms
# to play in each round within a budget constraint, considering both the rewards and bids
# of the arms, using the Upper Confidence Bound strategy.

import math
import numpy as np

from arms import StrategicArm, NormalArm  # Assumed to be provided modules


class OptimalAlgorithm:
    def __init__(self, arms: list, n_arms: int, n_selected: int, budget: float):
        self.arms = arms
        self.N = n_arms
        self.K = n_selected
        self.B = budget
        self.t = 0
        self.R = 0.0

    def initialize(self):
        pass

    def loop(self):
        # while True:
        #     self.t += 1
        #     pseudo_observed = np.array([arm.draw() for arm in self.arms])
        #
        #     rules = -1 * (pseudo_observed - np.array([arm.b for arm in self.arms]))
        #     pivot = np.argsort(rules)[self.K]
        #     mask = rules < pseudo_observed[pivot]
        #     self.B += sum(rules * mask)
        #     if self.B < 0:
        #         break
        #     self.R += sum(pseudo_observed * mask)

        return 0, 0


class AUCB:
    # Initialization of the AUCB class with the required parameters.
    def __init__(self, arms: list[StrategicArm], n_arms: int, n_selected: int, budget: float, n_rounds=0):
        self.arms = arms  # The set of arms N.
        self.N = n_arms  # Total number of arms.
        self.K = n_selected  # Number of arms selected in a round.
        self.t = n_rounds  # Current round, starts from 0.
        self.B = budget  # Budget B.

        # Initialize the count for how many times each arm is selected (β_i(t)).
        counts = np.zeros(n_arms)
        self.beta = counts

        # Initialize the empirical mean reward for each arm (r̄_i(t)).
        empirical = np.zeros(n_arms)
        self.r_bar = empirical

        # Total reward (R).
        self.R = 0.

    # Function to initialize the algorithm by selecting all arms once.
    def initialize(self):
        # Mask to select all arms in the first round (ϕ^1 = N).
        mask = np.ones(self.N)

        # Obtain the reward values r^1_i for all arms (Line 2).
        pseudo_observed = np.array([arm.draw() for arm in self.arms])
        # Update empirical mean reward (r̄_i(t)) and count of selections (β_i(t)) for each arm (Line 4).
        self.r_bar = ((self.r_bar * self.beta + pseudo_observed) / (self.beta + np.ones(self.N))) * mask \
                     + self.r_bar * (1 - mask)
        self.beta += mask
        # Update total reward (R) (Line 4).
        self.R += np.sum(pseudo_observed * mask)

        # Determine the payments for selected arms (p^t_i) assuming maximum cost (c_max) (Line 3).
        p = mask * StrategicArm.c_max
        # Update budget (B) after payments (Line 4).
        self.B -= np.sum(p)

    # The main loop of the AUCB algorithm that runs until the budget is depleted.
    def loop(self):
        while True:
            # Increment the round counter (t).
            self.t += 1

            # Calculate UCB values for each arm (r̂_i(t)).
            ucb_values = [0.0 for i in range(self.N)]
            for i in range(self.N):
                # Calculate the bonus term (u_i(t)) using UCB formula (Line 8).
                bonus = math.sqrt(((self.K + 1) * math.log(np.sum(self.beta))) / float(self.beta[i]))
                # UCB value is the sum of empirical reward and bonus (r̂_i(t)).
                ucb_values[i] = self.r_bar[i] + bonus
            # Bids of each arm (b_i).
            bids = [arm.b for arm in self.arms]

            # Sort and select the top K arms based on the rules (Line 7-9).
            ucb_values, bids = map(np.array, (ucb_values, bids))
            rules = -1 * ucb_values / bids
            pivot = np.argsort(rules)[self.K]
            r_hat_norm, r_hat_K_1 = rules[pivot], ucb_values[pivot]

            # Mask to identify the top K arms.
            mask = rules < r_hat_norm

            # Compute the payments for each selected arm (p^t_i) (Line 10).
            p = np.minimum(StrategicArm.c_max, ucb_values / r_hat_K_1) * mask
            # If the sum of payments exceeds the budget, terminate (Line 11).
            if np.sum(p) >= self.B:
                break

            # Update states with the new rewards obtained (Lines 13-14).
            pseudo_observed = np.array([arm.draw() for arm in self.arms])
            self.r_bar = ((self.r_bar * self.beta + pseudo_observed) / (self.beta + np.ones(self.N))) * mask \
                         + self.r_bar * (1 - mask)
            self.beta += mask
            # Update total reward (R) (Line 14).
            self.R += np.sum(pseudo_observed * mask)

            # Update budget (B) after payments (Line 14).
            self.B -= np.sum(p)

        # Return the total reward (R) and rounds (t) after the algorithm terminates.
        return self.R, self.t


"""
Implementation of the Separated Algorithm and the ϵ-First Algorithm as described in the paper:
"Auction-based combinatorial multi-armed bandit mechanisms with strategic arms" by Gao et al. (2021).
These algorithms are auction-based strategies for multi-armed bandit problems where arms can bid, 
and there is a budget constraint.

The Separated Algorithm divides the budget between exploration and exploitation phases, while the 
ϵ-First Algorithm uses a fraction of the budget for exploration and the rest for exploitation.

Author: ChatGPT
Guidance: DURUII
"""

import numpy as np
import random


class SeparatedAlgorithm:
    def __init__(self, arms: list, n_arms: int, n_selected: int, budget: float):
        self.arms = arms
        self.N = n_arms
        self.K = n_selected
        self.B = budget
        self.t = 0
        self.R = 0.0
        self.mean_rewards = [0] * n_arms  # Initialize mean rewards for all arms
        self.selection_counts = [0] * n_arms  # Initialize selection counts for all arms
        self.c_max = max(arm.c for arm in arms)  # Assuming arm.c is the cost attribute of the arm
        self.exploration_budget = (self.c_max * self.N * np.log(self.N * self.B)) / (2 * (self.B ** (2 / 3)))

    def update_mean_reward(self, arm_index, reward):
        # Update the mean reward and selection count for the given arm
        self.selection_counts[arm_index] += 1
        self.mean_rewards[arm_index] = ((self.mean_rewards[arm_index] * (
                self.selection_counts[arm_index] - 1)) + reward) / self.selection_counts[arm_index]

    def initialize(self):
        pass

    def loop(self):
        # Exploration phase
        while self.t < (self.N / self.K) and self.B > self.exploration_budget:
            for arm_index, arm in enumerate(self.arms):
                reward = arm.draw()
                self.update_mean_reward(arm_index, reward)  # Update mean reward for the arm
                self.R += reward
                self.B -= self.c_max  # Payment is the maximum cost c_max.
                self.t += 1

        # Exploitation phase
        while self.B > 0:
            # Sort arms based on the calculated mean rewards
            sorted_arm_indices = sorted(range(self.N), key=lambda i: self.mean_rewards[i], reverse=True)
            for arm_index in sorted_arm_indices[:self.K]:
                arm = self.arms[arm_index]
                reward = arm.draw()
                # Do not update mean reward during exploitation for SeparatedAlgorithm
                self.R += reward
                payment = min(reward / arm.b, self.c_max)
                self.B -= payment  # Update the budget based on the payment.
                if self.B <= 0:  # Check if the budget is exhausted.
                    break
            self.t += 1

        return self.R, self.t


class EpsilonFirstAlgorithm:
    def __init__(self, arms: list, n_arms: int, n_selected: int, budget: float, epsilon=0.1):
        self.arms = arms
        self.N = n_arms
        self.K = n_selected
        self.B = budget
        self.epsilon = epsilon
        self.t = 0
        self.R = 0.0
        self.mean_rewards = [0] * n_arms  # Initialize mean rewards for all arms
        self.selection_counts = [0] * n_arms  # Initialize selection counts for all arms
        self.c_max = StrategicArm.c_max

    def update_mean_reward(self, arm_index, reward):
        # Update the mean reward and selection count for the given arm
        self.selection_counts[arm_index] += 1
        self.mean_rewards[arm_index] = ((self.mean_rewards[arm_index] * (
                self.selection_counts[arm_index] - 1)) + reward) / self.selection_counts[arm_index]

    def initialize(self):
        pass

    def loop(self):
        exploration_budget = self.epsilon * self.B

        # Exploration phase
        while self.B > exploration_budget:
            for arm_index in random.sample(range(self.N), self.K):
                arm = self.arms[arm_index]
                reward = arm.draw()
                self.update_mean_reward(arm_index, reward)  # Update mean reward for the arm
                self.R += reward
                self.B -= self.c_max  # Payment is the maximum cost c_max.
                self.t += 1

        # Exploitation phase
        while self.B > 0:
            # Sort arms based on the calculated mean rewards
            sorted_arm_indices = sorted(range(self.N), key=lambda i: self.mean_rewards[i], reverse=True)
            for arm_index in sorted_arm_indices[:self.K]:
                arm = self.arms[arm_index]
                reward = arm.draw()
                # No need to update mean reward during exploitation
                self.R += reward
                payment = min(reward / arm.b, self.c_max)
                self.B -= payment  # Update the budget based on the payment.
                if self.B <= 0:  # Check if the budget is exhausted.
                    break
            self.t += 1

        return self.R, self.t
