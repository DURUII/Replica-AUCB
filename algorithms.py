# Author: DURUII
# Date: 2023.12.16
# This Python file implements the Auction-based UCB (AUCB) algorithm,
# as described in the paper. The AUCB algorithm selects a subset of arms
# to play in each round within a budget constraint, considering both the rewards and bids
# of the arms, using the Upper Confidence Bound strategy.

import math
import numpy as np

from arms import StrategicArm, NormalArm  # Assumed to be provided modules


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
            rules = ucb_values / bids
            pivot = np.argsort(rules)[self.K]
            r_hat_norm, r_hat_K_1 = rules[pivot], ucb_values[pivot]

            # Mask to identify the top K arms.
            mask = rules > r_hat_norm

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
