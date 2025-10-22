import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.special import expit  # numerically stable logistic

is_systematic = ""  

# ── Schedules & names ─────────────────────────────────────────────────────────
schemes = ["linearT", "linearBeta", "logT", "Cv_beta"]
scheme_names = {
    "linearT":    r"Lin. in $T$",
    "linearBeta": r"Lin. in $\beta$",
    "logT":       r"Log. in $T$",
    "Cv_beta":    r"$C_V$-based"
}

# ── Plot cosmetics ────────────────────────────────────────────────────────────
label = {1736329224: "Easy instance", 310411727: "Hard instance"}
markers = {"linearT": "o", "linearBeta": "p", "logT": "^", "Cv_beta": "*"}
colors = ["forestgreen", "firebrick", "royalblue", "goldenrod"]
all_algorithms = ["SA", "PA", "GA", "GA_global"]
zordering = {"GA_global":4, "SA": 3, "PA": 2, "GA": 1}
names_algorithms = {"GA_global":r"$\mathrm{GA}_0$", "SA": "SA", "PA": "PA", "GA": r"$\mathrm{GA}_{15}$"}

ticksize   = 22
fontsize   = 24
markersize = 15

seeds = [1736329224, 310411727]

# --- normalized logistic in linear time: S(0)=0, S(+inf)=1 ---
def logistic_linear_normalized(x, A, B):
    x = np.asarray(x, dtype=float)
    Lx = expit(A * (x - B))
    L0 = expit(-A * B)          # L(0)
    denom = (1.0 - L0)
    out = (Lx - L0) / np.maximum(denom, 1e-15)
    return np.clip(out, 0.0, 1.0)

# ── Grid: 4 schedules × 2 seeds ───────────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=len(schemes), ncols=len(seeds),
    figsize=(20, 24), sharex=True, sharey=True
)

for i, scheme in enumerate(schemes):
    for j, seed in enumerate(seeds):
        ax = axes[i, j]

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

            # ── Only the current scheme for this panel ────────────────────────
            st = stats[stats["scheme"] == scheme]
            sc = successes[successes["scheme"] == scheme]
            merge_keys = [c for c in group_cols if c in st.columns and c in sc.columns]
            df = pd.merge(st, sc, on=merge_keys)

            if df.empty:
                continue

            # scatter points
            ax.scatter(
                df["mean"], df["success_rate"],
                marker=markers[scheme], color=colors[n],
                s=markersize**2, zorder=zordering[algorithm]
            )

            # fit normalized logistic (linear time), plotted over log-x
            x = df["mean"].to_numpy(dtype=float)
            y = df["success_rate"].to_numpy(dtype=float)

            if np.isfinite(x).sum() >= 3 and np.isfinite(y).sum() >= 3:
                x_pos = x[np.isfinite(x)]
                B0 = float(np.median(x_pos)) if x_pos.size else 1.0
                iqr = np.subtract(*np.percentile(x_pos, [75, 25])) if x_pos.size >= 2 else max(B0, 1.0)
                A0 = 4.0 / max(iqr, 1e-6)
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

        # ── axes cosmetics per panel ─────────────────────────────────────────
        ax.grid(True, linestyle=':', alpha=0.6)
        ax.set_xlim(0.8, 2000)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xscale("log")   # fit in linear time; plot on log-x
        if j == 0:
            ax.set_ylabel("Success Probability", fontsize=fontsize)
        if i == len(schemes) - 1:
            ax.set_xlabel("Mean Time (s)", fontsize=fontsize)

        # Titles: left = scheme name; right = scheme + seed label
        ax.set_title(f"{label[seed]} — {scheme_names[scheme]}",
                     fontsize=fontsize)

        ax.tick_params(axis="both", labelsize=ticksize)

# ── single, figure-level legend ───────────────────────────────────────────────
legend_colors = [
    Line2D([0],[0], color=c, lw=6, label=names_algorithms[alg])
    for c, alg in zip(colors, all_algorithms)
]
fig.legend(handles=legend_colors, fontsize=20, loc="upper center",
           ncol=len(all_algorithms), bbox_to_anchor=(0.5, 0.995))

plt.tight_layout(rect=[0, 0, 0.98, 0.96])  # leave space for legend
plt.savefig(f"../Figures_paper/success_vs_time_comparison{is_systematic}_wglobal_4x2.png",
            dpi=300, bbox_inches="tight")
plt.savefig(f"../Figures_paper/success_vs_time_comparison{is_systematic}_wglobal_4x2.pdf",
            dpi=300, bbox_inches="tight")
plt.show()
