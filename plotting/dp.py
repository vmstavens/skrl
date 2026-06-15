import glob
import json
from pathlib import Path

from .preamble import *

FILE_NAME = Path(__file__).name
DIR_NAME = Path(__file__).parent


def _get_loss_data() -> dict:
    file = glob.glob(
        "testing/experiments/pipe_insert/.runs/data_mocap_100_20260228_124142/media/*.json"
    )[0]
    with open(file, "r") as f:
        return json.load(f)


loss_data = _get_loss_data()

epochs = loss_data["epochs"]
train_losses = loss_data["train_losses"]
val_losses = loss_data["val_losses"]

fig, ax = plt.subplots()

ax.set_xlabel("Epochs")
ax.set_ylabel("Loss")
fig.tight_layout()

ax.plot(epochs, train_losses, label="Training loss")
ax.plot(epochs, val_losses, label="Validation loss")
ax.legend(frameon=False)
fig.savefig(DIR_NAME / "plots" / (FILE_NAME + "_losses.png"))
