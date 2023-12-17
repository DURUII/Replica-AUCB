"""
Author: DURUII
Date: 2023/12/16

Ref:
1. https://github.com/johnmyleswhite/BanditsBook#adding-new-algorithms-api-expectations
"""

from abc import ABCMeta, abstractmethod

import numpy as np

from arms import StrategicArm


class BaseAlgorithm(metaclass=ABCMeta):
    def __init__(self, arms: list[StrategicArm], n_arms: int, n_selected: int, budget: float):
        super().__init__()
        self.arms = arms  # \mathcal{N}, the set of arms
        self.N = n_arms  # N, the number of total arms
        self.K = n_selected  # K, the number of arms selected in a round
        self.B = budget  # B, budget
        self.t = 1  # t, the indexes for round
        self.R = 0.0  # r, total reward

    @abstractmethod
    def initialize(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass

    def omniscience(self):
        return np.array([arm.draw() for arm in self.arms])
