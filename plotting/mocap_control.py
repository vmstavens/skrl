import json

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------
# Load data
# ----------------------------
with open("sim/data/test_new.json", "r") as f:
    data = json.load(f)

weld_pos_err = np.asarray(data["weld_pos_err"], dtype=float)
weld_rot_err = np.asarray(data["weld_rot_err"], dtype=float)

weld_pos_err = weld_pos_err[:, 0, :][25:]
weld_rot_err = weld_rot_err[:, 0, :][25:]

t = np.arange(len(weld_pos_err))

# ----------------------------
# Styling (clean & modern)
# ----------------------------
plt.style.use("seaborn-v0_8-whitegrid")

colors = {
    "x": "#1f77b4",
    "y": "#ff7f0e",
    "z": "#2ca02c",
    "roll": "#1f77b4",
    "pitch": "#ff7f0e",
    "yaw": "#2ca02c",
}

lw = 2.2

fig, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

# ----------------------------
# Position error
# ----------------------------
ax[0].plot(t, weld_pos_err[:, 0], color=colors["x"], lw=lw, label="$e_x$")
ax[0].plot(t, weld_pos_err[:, 1], color=colors["y"], lw=lw, label="$e_y$")
ax[0].plot(t, weld_pos_err[:, 2], color=colors["z"], lw=lw, label="$e_z$")

ax[0].set_ylabel("Position Error [m]")
ax[0].set_title("Position Error")
ax[0].legend(loc="upper right", frameon=True)
ax[0].grid(True, alpha=0.3)

# ----------------------------
# Rotation error
# ----------------------------
ax[1].plot(t, weld_rot_err[:, 0], color=colors["roll"], lw=lw, label="$e_{roll}$")
ax[1].plot(t, weld_rot_err[:, 1], color=colors["pitch"], lw=lw, label="$e_{pitch}$")
ax[1].plot(t, weld_rot_err[:, 2], color=colors["yaw"], lw=lw, label="$e_{yaw}$")

ax[1].set_ylabel("Rotation Error [rad]")
ax[1].set_xlabel("Timestep")
ax[1].set_title("Rotation Error")
ax[1].legend(loc="upper right", frameon=True)
ax[1].grid(True, alpha=0.3)

plt.tight_layout()

fig.savefig("plotting/plots/mocap_control.png")

print(np.max(weld_pos_err))
print(np.max(weld_rot_err))
print(np.rad2deg(np.max(weld_rot_err)))


# plt.show()
