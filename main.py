"""
Author: DURUII
Date: 2023/12/17
"""
import os

import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from arms import StrategicArm
from config import Config
from emulator import Emulator
import pickle

plt.style.use(['science', 'grid'])
config = Config

# data preparation
if not os.path.exists('./runs.pkl'):
    data = []
    for X in ['N', 'K', 'B']:
        for x in tqdm(eval(f'config.{X}_range'), desc=X):
            if X == 'N':
                name2res = Emulator(n_arms=x).simulate()
            elif X == 'K':
                name2res = Emulator(n_selected=x).simulate()
            else:
                name2res = Emulator(budget=x).simulate()

            for key in name2res.keys():
                data.append([X, x, key, name2res[key][0], name2res[key][1]])

    df = pd.DataFrame(np.array(data), columns=['X', 'Val', 'Algorithm', 'Reward', 'Round'])

    with open('./runs.pkl', 'wb') as fout:
        pickle.dump(df, fout)

else:
    with open('./runs.pkl', 'rb') as fin:
        df = pickle.load(fin)

# result visualization
df['Val'] = df['Val'].astype(float)
df['Reward'] = df['Reward'].astype(float)
df['Round'] = df['Round'].astype(float)
fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.5))

# line charts
for algo in Emulator.algorithms:
    data = df[(df.X == 'B') & (df.Algorithm == algo)]
    ax = axes[0, 0]
    ax.plot(data['Val'], data['Reward'], **config.line_styles[algo])
    ax.set_xlabel('Budget')
    ax.set_ylabel('Total rewards')

    ax = axes[0, 1]
    ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
    ax.set_xlabel('Budget')
    ax.set_ylabel('Total rounds')

    data = df[(df.X == 'N') & (df.Algorithm == algo)]
    # axes[1, 0].plot(data['Val'], data['Reward'], **config.line_styles[algo])
    ax = axes[1, 1]
    ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
    ax.set_xlabel('Number of arms (N)')
    ax.set_ylabel('Total rounds')

    data = df[(df.X == 'K') & (df.Algorithm == algo)]
    # axes[2, 0].plot(data['Val'], data['Reward'], **config.line_styles[algo])
    ax = axes[2, 1]
    ax.plot(data['Val'], data['Round'], **config.line_styles[algo])
    ax.set_xlabel('Parameter (K)')
    ax.set_ylabel('Total rounds')

# bar plots
n_algos = len(Emulator.algorithms)

for X, ax in zip(['N', 'K'], [axes[1, 0], axes[2, 0]]):
    data = df[df.X == X].pivot(index='Val', columns='Algorithm', values='Reward')
    for i, algo in enumerate(Emulator.algorithms):
        xpos = np.arange(len(data.index)) + (i - n_algos / 2) * config.bar_width
        ax.bar(xpos, data[algo], width=config.bar_width, **config.bar_styles[algo])

        ax.set_xlabel('Total rewards')
        ax.set_xticks(range(len(data.index)))
        ax.set_xticklabels(data.index)

for i, ax in enumerate(axes.flat):
    ax.legend()

plt.savefig('fig.pdf')
