import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import expit  # numerically stable logistic

is_systematic = ""  

schemes = ["logT"]
scheme_names = {
    "linearT":    r"Lin. in $T$",
    "linearBeta": r"Lin. in $\beta$",
    "logT":       r"Log. in $T$",
    "Cv_beta":    r"$c_V$-based"
}

label = {1736329224: "Easy instance", 310411727: "Hard instance"}
markers = {"linearT": "o", "linearBeta": "x", "logT": "^", "Cv_beta": "*"}
colors = ["forestgreen", "firebrick", "royalblue", "goldenrod"]
all_algorithms = ["SA", "PA", "GA", "GA_global"]
zordering = {"GA_global":4, "SA": 3, "PA": 2, "GA": 1}
names_algorithms = {"GA_global":r"$\mathrm{GA}_0$", "SA": "SA", "PA": "PA", "GA": r"$\mathrm{GA}_{15}$"}



ticksize   = 22
fontsize   = 24
markersize = 15

seeds = [1736329224, 310411727]

# --- normalized logistic in linear time: S(0)=0, S(+inf)=1 ---
# Base logistic: L(x) = expit(A*(x - B)) = 1 / (1 + exp(-A*(x-B)))
# Normalize: S(x) = (L(x) - L(0)) / (1 - L(0))
def logistic_linear_normalized(x, A, B):
    x = np.asarray(x, dtype=float)
    Lx = expit(A * (x - B))
    L0 = expit(-A * B)          # L(0)
    denom = (1.0 - L0)
    # guard against numerical issues if denom ~ 0
    out = (Lx - L0) / np.maximum(denom, 1e-15)
    return np.clip(out, 0.0, 1.0)

fig, axes = plt.subplots(1, 2, figsize=(20, 6), sharey=True)

for ax, seed in zip(axes, seeds):
    for n, algorithm in enumerate(all_algorithms):

        if algorithm in ["GA", "GA_global"]:
            path = f'../../Data/Omega/Success_rates/{algorithm}/results_{seed}.txt'
            names = ["global_steps","steps","num_temps","scheme","minimum","mean","time"]
            data = pd.read_csv(path, sep=' ', header=None, names=names)
            if algorithm == "GA":
                data = data[data["steps"] == 15].copy()
            group_cols = ["global_steps","steps","num_temps","scheme"]

            GS_energy = data["minimum"].min()
            successes = (
                data.groupby(group_cols)["minimum"]
                    .apply(lambda col: (col == GS_energy).mean())
                    .reset_index(name="success_rate")
            )
            stats = data.groupby(group_cols)["time"].agg(["mean","std"]).reset_index()

        elif algorithm == "PA":
            base_path = f'../../Data/Omega/Success_rates/PA/results_{seed}{is_systematic}.txt'
            names = ["steps","num_temps","scheme","minimum","mean","time"]
            data_time = pd.read_csv(base_path, sep=' ', header=None, names=names)

            dt_path = f'../../Data/Omega/Success_rates/PA/results_{seed}{is_systematic}_dt.txt'
            data_dt = pd.read_csv(dt_path, sep=' ', header=None, names=names)

            data_combined = pd.concat([data_time, data_dt], ignore_index=True)
            group_cols = ["steps","num_temps","scheme"]

            GS_energy = data_combined["minimum"].min()
            successes = (
                data_combined
                    .groupby(group_cols)["minimum"]
                    .apply(lambda col: (col == GS_energy).mean())
                    .reset_index(name="success_rate")
            )
            stats = (
                data_time
                    .groupby(group_cols)["time"]
                    .agg(["mean","std"])
                    .reset_index()
            )

        else:  # SA
            path = f'../../Data/Omega/Success_rates/{algorithm}/results_{seed}.txt'
            names = ["steps","num_temps","scheme","minimum","mean","time"]
            data = pd.read_csv(path, sep=' ', header=None, names=names)
            group_cols = ["steps","num_temps","scheme"]

            GS_energy = data["minimum"].min()
            successes = (
                data.groupby(group_cols)["minimum"]
                    .apply(lambda col: (col == GS_energy).mean())
                    .reset_index(name="success_rate")
            )
            stats = data.groupby(group_cols)["time"].agg(["mean","std"]).reset_index()

        # plot each scheme’s points + fit
        for scheme in schemes:
            st = stats[stats["scheme"] == scheme]
            sc = successes[successes["scheme"] == scheme]
            merge_keys = [c for c in group_cols if c in st.columns and c in sc.columns]
            df = pd.merge(st, sc, on=merge_keys)

            # scatter points
            ax.scatter(
                df["mean"], df["success_rate"],
                marker=markers[scheme], color=colors[n],
                s=markersize**2, zorder=zordering[algorithm]
            )

            # fit normalized logistic (linear time)
            x = df["mean"].to_numpy(dtype=float)
            y = df["success_rate"].to_numpy(dtype=float)

            if np.isfinite(x).sum() >= 3 and np.isfinite(y).sum() >= 3:
                # Initial guesses:
                # - B ~ median time (midpoint)
                # - A ~ slope chosen so transition spans roughly the IQR
                x_pos = x[np.isfinite(x)]
                B0 = float(np.median(x_pos))
                iqr = np.subtract(*np.percentile(x_pos, [75, 25])) if x_pos.size >= 2 else max(B0, 1.0)
                A0 = 4.0 / max(iqr, 1e-6)  # heuristic: ~ logistic span
                try:
                    popt, _ = curve_fit(
                        logistic_linear_normalized, x, y,
                        p0=(A0, B0),
                        bounds=([1e-8, 0.0], [10.0, np.inf]),
                        maxfev=20000
                    )
                    xfit = np.linspace(0.5, 3000, 5000)
                    yfit = logistic_linear_normalized(xfit, *popt)
                    ax.plot(
                        xfit, yfit,
                        linestyle="-", linewidth=6.0,
                        color=colors[n], alpha=0.7,
                        zorder=zordering[algorithm] + 0.1
                    )
                except Exception:
                    pass

    ax.grid(True)
    ax.set_xlim(0.8, 2000)
    ax.set_xlabel("Mean Time (s)", fontsize=fontsize)
    ax.set_title(label[seed], fontsize=fontsize)
    ax.tick_params(axis="both", labelsize=ticksize)
    ax.set_xscale("log")   # plotting on log-x is fine; the fit is in linear time

axes[0].set_ylabel("Success Probability", fontsize=fontsize)

legend_colors = [
    Line2D([0],[0], color=c, lw=6, label=names_algorithms[alg])
    for c, alg in zip(colors, all_algorithms)
]
for ax in axes:
    ax.legend(handles=legend_colors, fontsize=20, loc="upper left")

plt.tight_layout()
plt.savefig(f"../Figures_paper/success_vs_time_comparison{is_systematic}_wglobal.png",
            dpi=300, bbox_inches="tight")
plt.savefig(f"../Figures_paper/success_vs_time_comparison{is_systematic}_wglobal.pdf",
            dpi=300, bbox_inches="tight")
