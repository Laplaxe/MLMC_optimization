import matplotlib.pyplot as plt 
import numpy as np

seed = 310411727  # Example seed value
algorithms = ["SA", "PA", "GA"]

# Load the data
tempPT = ["1.92", "0.77", "0.29", "0.10"]
qPT = [np.load(f'../../Data/Alpha/Pq/Pq_{T}.npy') for T in tempPT]

q = {alg: np.load(f'../../Data/Omega/Pq/Pq_{alg}_{seed}_v2.npy') for alg in algorithms}
temps = {alg: np.load(f'../../Data/Omega/Pq/temperatures_{alg}_{seed}_v2.npy') for alg in algorithms}
for alg in algorithms:
    indices = [np.where(temps[alg] == float(value))[0].item() for value in tempPT]
    print(f"{alg}, {len(temps[alg])}, {indices}")

# Pre-defined indices into each algorithm's temperature series
#shorter
#indices = {
#    "SA": [0, 34, 65, 95],
#    "PA": [0, 18, 33, 45],
#    "GA": [0, 15, 27, 35],
#}
#longer
indices = {
    "SA": [0, 34, 65, 95],
    "PA": [0, 18, 33, 45],
    "GA": [0, 18, 33, 45],
}
colors = {"SA": 'forestgreen', "PA": 'firebrick', "GA": 'royalblue'}

# Create 4×3 grid
fig, axs = plt.subplots(
    nrows=4, ncols=3,
    figsize=(15, 10),
    sharey='row',
    sharex='all'
)

# Pull panels in closer
fig.subplots_adjust(
    left=0.12,   # narrower left margin
    right=0.98,  # use more of the right side
    top=0.90,    # raise top so titles sit nearer the edge
    wspace=0.15, # less horizontal gap
    hspace=0.25  # less vertical gap
)

# 1) Column titles: algorithm names
for ax, alg in zip(axs[0], algorithms):
    ax.set_title(alg, fontsize=22, pad=12)

# 2) Row y-labels: vertical P(q) and T = …
for ax, T in zip(axs[:, 0], tempPT):
    ax.set_ylabel(
        r'$P(q)$' + '\n' + rf'$T = {T}$',
        rotation=90,
        fontsize=22,
        labelpad=15
    )

# 3) Plotting histograms with colored contours for both datasets
for n in range(4):
    for i, alg in enumerate(algorithms):
        ax = axs[n, i]
        ax.grid(True, ls="--", alpha=0.5)

        # PT distribution: filled grey + black contour
        counts_pt, bins, _ = ax.hist(
            np.concatenate((qPT[n], -qPT[n])),
            bins=50,
            alpha=0.5,
            density=True,
            color="darkgrey",
            zorder=3
        )
        ax.hist(
            np.concatenate((qPT[n], -qPT[n])),
            bins=bins,
            density=True,
            histtype='step',
            color='black',
            linewidth=2,
            zorder=4
        )

        # Omega algorithm distribution: filled color + same-color contour
        counts_alg, _, _ = ax.hist(
            q[alg][indices[alg][n]],
            bins=bins,
            alpha=0.7,
            density=True,
            color=colors[alg],
            zorder=5
        )
        ax.hist(
            q[alg][indices[alg][n]],
            bins=bins,
            density=True,
            histtype='step',
            color=colors[alg],
            linewidth=3,
            zorder=6
        )

        # x-limits and tick styling
        ax.set_xlim(-1, 1)
        ax.tick_params(
            axis='both',
            which='major',
            labelsize=20
        )
        if n == 3:
            ax.set_xlabel(r'$q$', fontsize=22)

# Save the figure
plt.savefig('../Figures_paper/histogram_Pq_plot.png', dpi=300, bbox_inches='tight')
plt.savefig('../Figures_paper/histogram_Pq_plot.pdf', bbox_inches='tight')

