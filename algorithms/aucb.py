"""
Author: DURUII
Date: 2023/12/16

Ref:
1. pseudocode given in the paper and the corresponding equations
2. https://github.com/johnmyleswhite/BanditsBook/blob/master/python/algorithms/ucb/ucb1.py
"""
import numpy as np

from algorithms.base import BaseAlgorithm
from arms import StrategicArm


class AUCB(BaseAlgorithm):
    def __init__(self, arms: list[StrategicArm], n_arms: int, n_selected: int, budget: float):
        super().__init__(arms, n_arms, n_selected, budget)
        # β_i(t), count for how many times each arm is selected
        # r̄_i(t),empirical mean reward for each arm
        self.beta, self.r_bar = np.zeros(n_arms), np.zeros(n_arms)

    def initialize(self):
        """ Select all arms at the very beginning. """

        # Mask to select all arms in the first round (ϕ^1 = N, Line 1)
        mask = np.ones(self.N)

        # Observe the reward values r^1_i for all arms, just like a God (Line 2)
        omni = self.omniscience()

        # Determine the payments for selected arms (p^t_i) assuming maximum cost (c_max) (Line 3)
        p = mask * StrategicArm.c_max

        # Update empirical mean reward (r̄_i(t)) and count of selections (β_i(t)) for each arm (Line 4)
        self.r_bar = ((self.r_bar * self.beta + omni) / (self.beta + np.ones(self.N))) * mask + self.r_bar * (1 - mask)
        self.beta += mask

        # Update total reward (R) (Line 4)
        self.R += np.sum(omni * mask)

        # Update budget (B) after payments (Line 4)
        self.B -= np.sum(p)

    def run(self):
        # Bids of each arm (b_i)
        b = np.array([arm.b for arm in self.arms])

        while True:
            # Increment the round counter (t) (Line 5)
            self.t += 1

            # Calculate UCB values for each arm (r̂_i(t-1)) (Equation 8)
            # TODO Review this Equation, because it is incompatible with code in BanditsBook
            u = np.sqrt((self.K + 1) * (np.log(self.t - 1)) / self.beta)
            r_hat = self.r_bar + u

            # Sort and select the top K arms based on the rules (Line 7-9)
            criteria, mask = r_hat / b, np.zeros(self.N)
            arrange = np.argsort(criteria)[::-1]
            mask[arrange[:self.K]] = 1

            # Compute the payments for each selected arm (p^t_i) (Line 10)
            deno = r_hat[arrange[self.K]]  # \hat{r_i}_{K+1}(t-1)
            p = np.minimum(r_hat / deno * b[arrange[self.K]], StrategicArm.c_max) * mask

            # If the sum of payments exceeds the budget, terminate (Line 11)
            if np.sum(p) >= self.B:
                break

            # Update states with the new rewards obtained (Lines 13-14)
            omni = self.omniscience()
            self.r_bar = (self.r_bar * self.beta + omni) / (self.beta + np.ones(self.N)) * mask + self.r_bar * (
                    1 - mask)
            self.beta += mask

            # Update total reward (R) (Line 14)
            self.R += np.sum(omni * mask)

            # Update budget (B) after payments (Line 14)
            self.B -= np.sum(p)

        # Return the total reward (R) and rounds (t) after the algorithm terminates.
        return self.R, self.t
