import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- config ---
schemes = ["linearT", "linearBeta", "logT", "Cv_beta"]
scheme_names = {
    "linearT":    r"Lin. in $T$",
    "linearBeta": r"Lin. in $\beta$",
    "logT":       r"Log. in $T$",
    "Cv_beta":    r"$C_V$-based"
}
markers = {"linearT": "o", "linearBeta": "h", "logT": "^", "Cv_beta": "*"}

ticksize   = 22
fontsize   = 24
markersize = 15

seed = 310411727          # Hard instance
target_global_steps = 5   # only choose global_steps = 5
target_num_temps = 20     # only consider 20 temperatures

# plasma colors for the 4 schemes
cmap = plt.get_cmap("plasma")
plasma_colors = [cmap(v) for v in np.linspace(0.1, 0.9, 4)]
scheme_to_color = dict(zip(schemes, plasma_colors))

# --- load GA data ---
path = f'../../Data/Omega/Success_rates/GA/results_{seed}.txt'
names = ["global_steps", "steps", "num_temps", "scheme", "minimum", "mean", "time"]
data = pd.read_csv(path, sep=' ', header=None, names=names)

# filters: global_steps = 5 AND num_temps = 20
data = data[(data["global_steps"] == target_global_steps) &
            (data["num_temps"] == target_num_temps)].copy()

# success rate per (steps, scheme) at the best (global) minimum
GS_energy = data["minimum"].min()
group_cols = ["steps", "scheme"]
successes = (
    data.groupby(group_cols)["minimum"]
        .apply(lambda col: (col == GS_energy).mean())
        .reset_index(name="success_rate")
)

# mean runtime per (steps, scheme)
times = (
    data.groupby(group_cols)["time"]
        .mean()
        .reset_index(name="mean_time")
)

# merge and compute success per second
df_all = pd.merge(successes, times, on=group_cols, how="inner")
df_all["success_per_sec"] = df_all["success_rate"] / df_all["mean_time"].replace(0, np.nan)

# --- plot (all schedules together) ---
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

for scheme in schemes:
    df = df_all[df_all["scheme"] == scheme].copy()
    if df.empty:
        continue

    df_plot = df.sort_values("steps")

    ax.plot(
        df_plot["steps"], df_plot["success_per_sec"],
        marker=markers.get(scheme, "o"),
        linestyle="--",
        linewidth=2.5,
        markersize=markersize,
        color=scheme_to_color[scheme],
        label=scheme_names.get(scheme, scheme)
    )
ax.grid(True)
ax.set_xlabel(r"$k$", fontsize=fontsize)
ax.set_ylabel(r"Success Prob./Time ($s^{-1}$)", fontsize=fontsize)
ax.tick_params(axis="both", labelsize=ticksize)
ax.legend(fontsize=18, loc="lower right")
ax.set_ylim(-0.0001)
ax.set_xlim(0)


plt.tight_layout()
plt.savefig("../Figures_paper/best_k.png",
            dpi=300, bbox_inches="tight")
plt.savefig("../Figures_paper/best_k.pdf",
            dpi=300, bbox_inches="tight")
plt.show()
