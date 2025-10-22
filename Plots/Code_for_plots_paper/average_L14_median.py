import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
base_dir     = '../../Data/Omega/multiple_instances_L14'
st_dir       = os.path.join(base_dir, 'GA')
st_alt_dir   = os.path.join(base_dir, 'GA_alternative')
pa_alt_dir   = os.path.join(base_dir, 'PA_alternative')

all_files = sorted(f for f in os.listdir(pa_alt_dir) if f.startswith('results_'))
filenames = [f for f in all_files
             if int(f.split('_')[1].split('.')[0]) not in []] #To exclude seeds
total_seeds = len(filenames)
print(f"Total files = {total_seeds}")

# ─── 1) COMPUTE GLOBAL-SEARCH GROUND-TRUTH FOR EACH SEED (intensive energies) ─
gs_values = {}
for fname in filenames:
    df = pd.read_csv(
        os.path.join(st_dir, fname),
        sep=r'\s+',
        header=None,
        comment="#",
        names=['gsteps','lsteps','nTemps','schedule','minEnergy','AvgEnergy','Runtime']
    )
    gs_values[fname] = df['minEnergy'].min()

# Print average intensive GS energy (± SEM) across reference files
gs_arr = np.array(list(gs_values.values()), dtype=float)
gs_arr = np.sort(gs_arr)[::-1]
if gs_arr.size > 0:
    gs_mean = gs_arr.mean()
    gs_std  = gs_arr.std(ddof=1) if gs_arr.size > 1 else 0.0
    gs_sem  = gs_std / np.sqrt(gs_arr.size) if gs_arr.size > 1 else 0.0
    print(f"Reference GS energy (intensive) across {len(gs_arr)} files: "
          f"mean = {gs_mean:.6f}, SEM = {gs_sem:.6f} (std = {gs_std:.6f})")
else:
    print("No reference GS energies found to average.")

# ─── 2) GATHER succ_prob & avg_time FOR EACH SEED ─────────────────────────────
def gather_stats(dirpath, cols):
    dfs = []
    for fname in filenames:
        df = pd.read_csv(
            os.path.join(dirpath, fname),
            sep=r'\s+',
            comment="#",
            header=None,
            names=cols
        )
        GS = gs_values[fname]
        grouped = (
            df
            .groupby('nTemps')
            .agg(
                succ_prob=('minEnergy', lambda x: (x == GS).mean()),
                avg_time =('Runtime',  'mean')
            )
            .reset_index()
        )
        grouped['seed'] = int(fname.split('_')[1].split('.')[0])
        dfs.append(grouped)
    return pd.concat(dfs, ignore_index=True)

st_stats = gather_stats(
    st_alt_dir,
    cols=['gsteps','lsteps','nTemps','schedule','minEnergy','AvgEnergy','Runtime']
)
pa_stats = gather_stats(
    pa_alt_dir,
    cols=['steps','nTemps','schedule','minEnergy','AvgEnergy','Runtime']
)

# ─── 3) COMPUTE IMPUTED MEANS ─────────────────────────────────────────────────
def compute_imputed_mean(df):
    agg = (
        df
        .groupby('nTemps')
        .agg(
            succ_mean=('succ_prob','mean'),
            avg_time =('avg_time','mean'),
            count    =('succ_prob','count')
        )
        .reset_index()
    )
    agg['succ_imputed'] = (
        agg['succ_mean'] * agg['count'] + (total_seeds - agg['count']) * 1.0
    ) / total_seeds
    return agg[['nTemps','succ_imputed','avg_time','count']]

st_imputed = compute_imputed_mean(st_stats)
pa_imputed = compute_imputed_mean(pa_stats)

# ─── 4) PLOT with imputed ±1σ bands + best/worst seed curves ──────────────────
plt.figure(figsize=(4,3))

def plot_imputed_band(stats, imp, color, alg_name):
    lower, upper, medians = [], [], []
    for _, row in imp.iterrows():
        nT = row['nTemps']
        cnt = int(row['count'])
        actual = stats.loc[stats['nTemps'] == nT, 'succ_prob'].values
        arr = np.concatenate([actual, np.ones(total_seeds - cnt)])  # pad with ones

        q25 = np.percentile(arr, 25)
        q75 = np.percentile(arr, 75)
        median = np.median(arr)

        lower.append(q25)
        upper.append(q75)
        medians.append(median)

    x = imp['avg_time']
    lo = np.array(lower)
    hi = np.array(upper)
    mid = np.array(medians)

    # band + percentile edges
    plt.fill_between(x, lo, hi, color=color, alpha=0.2)
    plt.plot(x, lo, linestyle=':', color=color, lw=3, label='_nolegend_', zorder=4)
    plt.plot(x, hi, linestyle=':', color=color, lw=3, label='_nolegend_', zorder=4)
    plt.plot(x, mid, linestyle='-', color=color, lw=3, label=alg_name, zorder=4)

def find_best_worst_curves(stats, threshold=0.9, tail_n=10):
    per_seed = []
    for seed, sdf in stats.groupby('seed'):
        sdf = sdf.sort_values('avg_time')
        times = sdf['avg_time'].to_numpy()
        succs = sdf['succ_prob'].to_numpy()
        hit_idx = np.where(succs >= threshold)[0]
        t90 = times[hit_idx[0]] if len(hit_idx) else None
        end_mean = succs[-tail_n:].mean() if len(succs) else np.nan
        per_seed.append({
            'seed': int(seed), 'times': times, 'succs': succs,
            't90': t90, 'end_mean': end_mean
        })

    with_t90 = [p for p in per_seed if p['t90'] is not None]
    none_t90 = [p for p in per_seed if p['t90'] is None]

    best = (min(with_t90, key=lambda p: p['t90'])
            if with_t90 else max(per_seed, key=lambda p: p['end_mean']))
    worst = (min(none_t90, key=lambda p: p['end_mean'])
             if none_t90 else max(with_t90, key=lambda p: p['t90']))

    return best, worst

# Bands + medians
plot_imputed_band(pa_stats, pa_imputed, 'firebrick', "PA")
plot_imputed_band(st_stats, st_imputed, 'royalblue', r"$\mathrm{GA}_{15}$")

# Best/worst curves (dotted, no legend)
st_best, st_worst = find_best_worst_curves(st_stats, threshold=0.9, tail_n=10)
pa_best, pa_worst = find_best_worst_curves(pa_stats, threshold=0.9, tail_n=10)

plt.plot(st_best['times'],  st_best['succs'],  color='royalblue',
         linestyle='-.', linewidth=1.0, alpha=0.7, label='_nolegend_')
plt.plot(st_worst['times'], st_worst['succs'], color='royalblue',
         linestyle='--', linewidth=1.0, alpha=0.7, label='_nolegend_')

plt.plot(pa_best['times'],  pa_best['succs'],  color='firebrick',
         linestyle='-.', linewidth=1.0, alpha=0.7, label='_nolegend_')
plt.plot(pa_worst['times'], pa_worst['succs'], color='firebrick',
         linestyle='--', linewidth=1.0, alpha=0.7, label='_nolegend_')

# ─── Print best/worst seeds ───────────────────────────────────────────────────
print(f"[GA] Best seed:  {st_best['seed']}  | Worst seed: {st_worst['seed']}")
print(f"[PA] Best seed:  {pa_best['seed']}  | Worst seed: {pa_worst['seed']}")

# Threshold, axes, legend, save
plt.axhline(0.9, linestyle='--', color='black', linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Success Probability', fontsize=12)
plt.ylim(-0.005, 1.005)
plt.xlim(500, 3125)   # L=14 time scale
plt.legend(loc="center left", fontsize=10)
plt.grid(True, linestyle=':', alpha=0.5)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()

# ─── SAVE TO PAPER DIRECTORY ──────────────────────────────────────────────────
plt.savefig("../Figures_paper/avg_with_runs_success_vs_time_L14.png", dpi=300)
plt.savefig("../Figures_paper/avg_with_runs_success_vs_time_L14.pdf")
