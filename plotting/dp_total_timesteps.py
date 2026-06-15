import glob
import json
from pathlib import Path

import pandas as pd

from .preamble import *

FILE_NAME = Path(__file__).stem
DIR_NAME = Path(__file__).parent


df = pd.read_csv("plotting/data/toal-timesteps-reset.csv")

keys = list(df.keys())

dp_cut = 47
dp_cut = -1

# global_step_dp = df[keys[0]].to_numpy()
# reward = df[keys[1]].to_numpy()
global_step_dp = df[keys[0]].to_numpy()[:dp_cut]
reward = df[keys[1]].to_numpy()[:dp_cut]

print(np.mean(reward))
# print(np.mean(reward2))

fig, ax = plt.subplots(figsize=(5, 5))


ax.set_xlabel("Steps")
ax.set_ylabel("Reward")
fig.tight_layout()

ax.plot(global_step_dp, reward)
# ax.plot(global_step2, reward2, label="Training loss")
ax.legend(frameon=False)
fig.savefig(DIR_NAME / "plots" / (FILE_NAME + "_full.png"))
