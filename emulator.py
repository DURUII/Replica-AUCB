"""
Author: DURUII
Date: 2023/12/17

Ref:
1. https://github.com/johnmyleswhite/BanditsBook/blob/master/python/testing_framework/tests.py
2. default simulation settings in the paper
"""
from algorithms.aucb import AUCB
from algorithms.eps import EpsilonFirst
from algorithms.opt import Opt
from algorithms.separated import Separated
from arms import StrategicArm
import pickle


class Emulator:
    algorithms = ['AUCB', 'optimal', 'separated', '0.1-first', '0.5-first']

    def __init__(self, arms: list[StrategicArm] = None, n_arms: int = 60, n_selected: int = 20, budget: float = 5e5):
        self.N = n_arms
        self.K = n_selected
        self.B = budget

        self.arms = arms
        if arms is None:
            self.arms = [StrategicArm() for _ in range(self.N)]

        self.name2sol = {}

    def build(self):
        for algo in Emulator.algorithms:
            if algo == 'AUCB':
                self.name2sol[algo] = AUCB(self.arms, self.N, self.K, self.B)
            elif algo == 'optimal':
                self.name2sol[algo] = Opt(self.arms, self.N, self.K, self.B)
            elif algo == 'separated':
                self.name2sol[algo] = Separated(self.arms, self.N, self.K, self.B)
            elif algo.endswith('-first'):
                self.name2sol[algo] = EpsilonFirst(self.arms, self.N, self.K, self.B, float(algo[:-6]))

    def simulate(self):
        self.build()
        name2res = {name: None for name in self.name2sol.keys()}
        for name in name2res.keys():
            # instance of an algorithm
            solver = self.name2sol[name]
            solver.initialize()
            name2res[name] = solver.run()
        return name2res
