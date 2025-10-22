import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ─── CONFIG ───────────────────────────────────────────────────────────────────
base_dir     = '../../Data/Omega/multiple_instances_L10'
st_dir       = os.path.join(base_dir, 'GA')
pa_alt_dir   = os.path.join(base_dir, 'PA_alternative')  # used only to list seeds

# (Optional) infer L from base_dir (not used for energies; kept for context)
m = re.search(r'L(\d+)', base_dir)
L = int(m.group(1)) if m else None

# ─── LIST OF SEED FILES (exclude any seeds by putting ids in the list) ────────
all_files = sorted(f for f in os.listdir(pa_alt_dir) if f.startswith('results_'))
filenames = [f for f in all_files
             if int(f.split('_')[1].split('.')[0]) not in []]
total_seeds = len(filenames)
print(f"Total files = {total_seeds}")

# ─── COLLECT INTENSIVE GS ENERGIES FROM REFERENCE FILES ───────────────────────
gs_values = {}  # intensive GS energy per file
for fname in filenames:
    df = pd.read_csv(
        os.path.join(st_dir, fname),
        sep=r'\s+',
        header=None,
        names=['gsteps','lsteps','nTemps','schedule','minEnergy','AvgEnergy','Runtime']
    )
    gs_values[fname] = df['minEnergy'].min()  # already intensive (per-spin)

gs_arr = np.array(list(gs_values.values()), dtype=float)
gs_arr = np.sort(gs_arr)[::-1]
gs_arr = gs_arr[:]

# Print quick stats
if gs_arr.size > 0:
    gs_mean = gs_arr.mean()
    gs_std  = gs_arr.std(ddof=1) if gs_arr.size > 1 else 0.0
    gs_sem  = gs_std / np.sqrt(gs_arr.size) if gs_arr.size > 1 else 0.0
    if L is not None:
        print(f"Reference GS energy (intensive) across {len(gs_arr)} files (L={L}): "
              f"mean = {gs_mean:.6f}, SEM = {gs_sem:.6f} (std = {gs_std:.6f})")
    else:
        print(f"Reference GS energy (intensive) across {len(gs_arr)} files: "
              f"mean = {gs_mean:.6f}, SEM = {gs_sem:.6f} (std = {gs_std:.6f})")
else:
    print("No reference GS energies found to plot.")

# ─── PLOT HISTOGRAM ───────────────────────────────────────────────────────────
plt.figure(figsize=(5,3.5))
bins = min(30, max(5, len(gs_arr)//2)) if len(gs_arr) > 0 else 10

plt.hist(gs_arr, bins=bins, edgecolor='black', alpha=0.75)
if gs_arr.size > 0:
    plt.axvline(gs_mean, linestyle='--', linewidth=2, color='red',
                label=f"mean = {gs_mean:.6f}")
    plt.legend(fontsize=9)

plt.xlabel('Ground-state energy (intensive)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.5)
plt.tight_layout()

# ─── SAVE ─────────────────────────────────────────────────────────────────────
os.makedirs("../Figures_paper", exist_ok=True)
plt.savefig("../Figures_paper/hist_GS.png", dpi=300)
