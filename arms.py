"""
Author: DURUII
Date: 2023/12/16

Ref:
1. https://github.com/johnmyleswhite/BanditsBook/blob/master/python/arms/normal.py
2. simulation settings in the paper
"""
import random


class NormalArm:
    def __init__(self, mu: float, sigma: float):
        """ Mean and standard deviation for the normal distribution."""
        self.mu = mu
        self.sigma = sigma

    def draw(self):
        """ Returns the achieved reward of the arm at this round. """
        return random.gauss(self.mu, self.sigma)


class StrategicArm(NormalArm):
    c_min, c_max = 0.1, 1

    def __init__(self):
        # in the paper, r is expected reward
        r = random.uniform(0.1, 1)
        # to make that sample value is within 0~1 with 97%
        sigma = random.uniform(0, min(r / 3, (1 - r) / 3))
        super().__init__(r, sigma)

        # c for cost, b for bid, c_i = b_i according to the theorem 2
        self.c = random.uniform(0.1, 1)
        self.b = self.c
