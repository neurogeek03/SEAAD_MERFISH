"""
Quick exploratory script: correlation between CPS and Braak score per donor.
Output: out/cps_vs_braak.png
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path

ROOT = Path(__file__).parents[1]
obs_path = ROOT / "data" / "SEAAD_obs.csv"
out_dir = ROOT / "out"
out_dir.mkdir(exist_ok=True)

# Load and deduplicate per donor
df = pd.read_csv(obs_path, low_memory=False)
donor_df = df[["Donor ID", "Braak", "Continuous Pseudo-progression Score"]].drop_duplicates(subset="Donor ID")

# Map Braak string → integer
braak_map = {
    "Braak 0": 0, "Braak I": 1, "Braak II": 2,
    "Braak III": 3, "Braak IV": 4, "Braak V": 5, "Braak VI": 6,
}
donor_df = donor_df.copy()
donor_df["Braak_int"] = donor_df["Braak"].map(braak_map)
donor_df = donor_df.dropna(subset=["Braak_int", "Continuous Pseudo-progression Score"])
donor_df["Braak_int"] = donor_df["Braak_int"].astype(int)

x = donor_df["Braak_int"].values
y = donor_df["Continuous Pseudo-progression Score"].values

# Spearman + Pearson
pearson_r, pearson_p = stats.pearsonr(x, y)
spearman_r, spearman_p = stats.spearmanr(x, y)

# Jitter for overlapping points
rng = np.random.default_rng(42)
x_jit = x + rng.uniform(-0.15, 0.15, size=len(x))

# Plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(x_jit, y, alpha=0.7, edgecolors="steelblue", facecolors="lightskyblue", linewidths=0.6, s=60)

# Regression line
m, b = np.polyfit(x, y, 1)
x_line = np.linspace(x.min(), x.max(), 100)
ax.plot(x_line, m * x_line + b, color="steelblue", linewidth=1.5, linestyle="--")

ax.set_xticks(range(7))
ax.set_xticklabels([f"Braak {v}" for v in range(7)], rotation=30, ha="right")
ax.set_xlabel("Braak Score", fontsize=12)
ax.set_ylabel("Continuous Pseudo-progression Score (CPS)", fontsize=12)
ax.set_title("CPS vs Braak Score (per donor)", fontsize=13)

stats_text = (
    f"Pearson r = {pearson_r:.3f} (p={pearson_p:.2e})\n"
    f"Spearman ρ = {spearman_r:.3f} (p={spearman_p:.2e})\n"
    f"n = {len(donor_df)} donors"
)
ax.text(0.03, 0.97, stats_text, transform=ax.transAxes,
        verticalalignment="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8))

plt.tight_layout()
out_path = out_dir / "cps_vs_braak.png"
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
print(f"Pearson r={pearson_r:.3f}, p={pearson_p:.2e}")
print(f"Spearman ρ={spearman_r:.3f}, p={spearman_p:.2e}")