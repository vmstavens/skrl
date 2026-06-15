import glob
import json
from pathlib import Path
import pandas as pd
from .preamble import *

FILE_NAME = Path(__file__).name
DIR_NAME = Path(__file__).parent


def _get_loss_data() -> dict:
    file = glob.glob(
        "testing/experiments/pipe_insert/.runs/data_mocap_100_20260228_124142/media/*.json"
    )[0]
    with open(file, "r") as f:
        return json.load(f)


df = pd.read_csv("plotting/data/wandb_export_2026-03-01T20_50_19.325+01_00.csv")

columns = df.columns

steps = df[columns[0]]

reward = df[columns[2]]

steps = steps.to_numpy()
reward = reward.to_numpy()

fig, ax = plt.subplots()

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
fig.tight_layout()


ax.plot(steps, reward, label="Reward")
ax.legend(frameon=False)
fig.savefig(DIR_NAME / "plots" / (FILE_NAME + "_reward_no_reset.png"))

