import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ─── DEFINE A CONTINUOUS SUNSET COLORMAP AND REVERSE IT ────────────────────────
sunset = [
    "#a50026", "#d73027", "#f46d43", "#fdae61", "#fee090",
    "#e0f3f8", "#abd9e9", "#74add1", "#4575b4", "#313695"
]
# continuous colormap
base_cmap = mcolors.LinearSegmentedColormap.from_list("sunset", sunset, N=256)
cmap = base_cmap.reversed()  # so smaller ratios map toward blue

def read_and_average_last_column_of_last_10_lines(file_path):
    try:
        df = pd.read_csv(file_path, header=None, dtype=str, sep=r"\s+")
        last_column = pd.to_numeric(df.iloc[-10:, -1], errors='coerce')
        if last_column.dropna().mean() > 150:
            print(f"File {file_path} has a high average: {last_column.dropna().mean()}")
        return last_column.dropna().mean()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def main():
    pa_dir = "../../Data/Omega/multiple_instances_L10/PA_alternative"
    st_dir = "../../Data/Omega/multiple_instances_L10/GA_alternative"
    if not os.path.exists(pa_dir) or not os.path.exists(st_dir):
        print("One or both directories do not exist.")
        return

    pa_files = [f for f in os.listdir(pa_dir)
                if f.startswith("results_") and f.endswith(".txt")]
    pa_averages, st_averages = [], []
    print(f"Total files = {len(pa_files)}")
    
    for file_name in pa_files:
        seed = file_name.split("_")[1].replace(".txt", "")
        pa_path = os.path.join(pa_dir, file_name)
        st_path = os.path.join(st_dir, f"results_{seed}.txt")
        if not os.path.exists(st_path):
            continue
        pa_avg = read_and_average_last_column_of_last_10_lines(pa_path)
        st_avg = read_and_average_last_column_of_last_10_lines(st_path)
        if pa_avg is not None and st_avg is not None:
            pa_averages.append(pa_avg)
            st_averages.append(st_avg)

    if not pa_averages or not st_averages:
        print("No valid data to plot.")
        return

    # ─── COMPUTE RATIOS & CLIP TO [inf, sup] ──────────────────────────────────────
    inf = 0.2
    sup = 5
    raw_ratios     = np.array(st_averages) / np.array(pa_averages)
    ratios_clipped = np.clip(raw_ratios, inf, sup)

    # continuous normalization with center at 1
    norm = mcolors.TwoSlopeNorm(vmin=inf, vcenter=1.0, vmax=sup)

    # ─── FIGURE & GRID SETUP ─────────────────────────────────────────────────────
    fig = plt.figure(figsize=(8, 8))
    scatter_ax = plt.subplot2grid((4, 4), (1, 0), rowspan=3, colspan=3)
    hist_x_ax  = plt.subplot2grid((4, 4), (0, 0), colspan=3, sharex=scatter_ax)
    hist_y_ax  = plt.subplot2grid((4, 4), (1, 3), rowspan=3, sharey=scatter_ax)

    # ─── SET LOG SCALES FOR ALL AXES ────────────────────────────────────────────
    scatter_ax.set_xscale('log')
    scatter_ax.set_yscale('log')
    scatter_ax.set_xlim(10, 600)
    scatter_ax.set_ylim(10, 600)

    # ─── MAIN SCATTER WITH DIFFERENT MARKERS (like in the old inset) ────────────
    mask = np.array(pa_averages) > 552  # identify PA times > 552
    
    # Plot circles for PA ≤ 552
    scatter_ax.scatter(
        np.array(pa_averages)[~mask],
        np.array(st_averages)[~mask],
        c=ratios_clipped[~mask],
        cmap=cmap, norm=norm,
        alpha=1.0, s=80, zorder=3,
        marker='o',
        edgecolors='black', linewidths=0.25
    )
    
    # Plot crosses for PA > 552
    scatter_ax.scatter(
        np.array(pa_averages)[mask],
        np.array(st_averages)[mask],
        c=ratios_clipped[mask],
        cmap=cmap, norm=norm,
        alpha=1.0, s=100, zorder=3,  # Increased marker size
        marker='x', linewidths=3.0   # Increased linewidth
    )

    # Set log-scale ticks
    scatter_ax.set_xticks([10, 20, 50, 100, 200, 500])
    scatter_ax.set_yticks([10, 20, 50, 100, 200, 500])
    scatter_ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    scatter_ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    
    scatter_ax.tick_params(axis='both', labelsize=24)
    scatter_ax.plot([10, 600], [10, 600], color='black', ls='--', linewidth=4)
    scatter_ax.set_xlabel("PA times (s)", fontsize=28)
    scatter_ax.set_ylabel(r"$\mathrm{GA}_{15}$ times (s)", fontsize=28)
    scatter_ax.grid(True, which='both', linestyle='--', alpha=0.7)

    # ─── COLORBAR (CONTINUOUS, inf→sup) ──────────────────────────────────────────
    cbar_ax = inset_axes(
        scatter_ax,
        width="100%", height="5%",
        bbox_to_anchor=(0.4, -0.35, 0.5, 0.6),
        bbox_transform=scatter_ax.transAxes
    )
    cbar = plt.colorbar(
        scatter_ax.collections[0], cax=cbar_ax,  # Use first scatter collection for colorbar
        orientation='horizontal',
        extend='neither'
    )
    cbar.set_ticks([inf, 1.0, sup])
    cbar.set_label(r"$\mathrm{GA}_{15}$/PA time ratio", fontsize=24)
    cbar.ax.xaxis.set_ticks_position('bottom')
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.ax.tick_params(labelsize=24)

    # ─── TOP HISTOGRAM (PA) WITH LOG SCALE ──────────────────────────────────────
    # Use log-spaced bins for log-scale histogram
    log_bins = np.logspace(np.log10(10), np.log10(1000), 30)
    hist_x_ax.hist(
        pa_averages, bins=log_bins,
        color='firebrick', alpha=0.5, edgecolor='none', zorder=3
    )
    hist_x_ax.hist(
        pa_averages, bins=log_bins,
        histtype='step', edgecolor='firebrick',
        lw=4, alpha=1, zorder=5
    )
    
    hist_x_ax.set_ylabel("Counts", fontsize=22)
    hist_x_ax.grid(True, ls="--", alpha=0.5)
    hist_x_ax.tick_params(labelbottom=False)
    hist_x_ax.tick_params(axis='y', labelsize=22)
    hist_x_ax.set_xscale('log')  # Ensure x-axis is log scale

    # ─── RIGHT HISTOGRAM (GA) WITH LOG SCALE ────────────────────────────────────
    hist_y_ax.hist(
        st_averages, bins=log_bins,
        orientation='horizontal',
        color='royalblue', alpha=0.5, edgecolor='none', zorder=3
    )
    hist_y_ax.hist(
        st_averages, bins=log_bins,
        orientation='horizontal',
        histtype='step', edgecolor='royalblue',
        lw=4, alpha=1, zorder=5
    )
    hist_y_ax.set_xlabel("Counts", fontsize=22)
    hist_y_ax.grid(True, ls="--", alpha=0.5)
    hist_y_ax.tick_params(labelleft=False)
    hist_y_ax.tick_params(axis='x', labelsize=22)
    hist_y_ax.set_yscale('log')  # Ensure y-axis is log scale

    # ─── SAVE FIGURE ────────────────────────────────────────────────────────────
    os.makedirs("../Figures_paper", exist_ok=True)
    fig.tight_layout()
    fig.savefig("../Figures_paper/scatter_plot_with_histograms.png", dpi=300)
    fig.savefig("../Figures_paper/scatter_plot_with_histograms.pdf")

if __name__ == "__main__":
    main()