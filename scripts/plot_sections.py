import scanpy as sc
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

project_path = Path.cwd().parents[1]
print(f"current working directory: {project_path}")
output_base = project_path / 'out'
output_base.mkdir(exist_ok=True, parents=True)
sc_ad = sc.read_h5ad("/scratch/mfafouti/SEA-AD/SEAAD_MTG_MERFISH.2024-12-11.h5ad")
color_csv = pd.read_csv('/scratch/mfafouti/SEA-AD/data/cluster_order_and_colors.csv')

column = 'X_spatial_raw'
donor_id = 'H20.33.004'
section = 'H20.33.035.Cx26.MTG.02.007.1.01.03'

spatial_df = pd.DataFrame(sc_ad.obsm[column], 
                          index=sc_ad.obs.index, 
                          columns=['x', 'y'])

# Add Donor ID to the spatial dataframe
spatial_df['Section'] = sc_ad.obs['Section'].values
spatial_df['Donor ID'] = sc_ad.obs['Donor ID'].values
spatial_df['ct'] = sc_ad.obs['Subclass']
spatial_df['depth'] = sc_ad.obs['Depth from pia']

# obtaining taxonomy colors 
subclass_colors = color_csv[['subclass_label', 'subclass_color']].drop_duplicates()
spatial_df = spatial_df.merge(subclass_colors, left_on='ct', right_on='subclass_label', how='left')

# Subset for a specific donor, e.g., "H20.33.035"
subset_df = spatial_df[spatial_df['Section'] == section]
# subset_df = subset_df[subset_df['depth'].notna()]

plt.figure(figsize=(12, 9))
scatter = plt.scatter(
    subset_df["x"], subset_df["y"],
    c=subset_df["subclass_color"],
    s=3
)
plt.axis("equal")
plt.title(f"Spatial plot {section}")

unique_cts = subset_df[['ct', 'subclass_color']].drop_duplicates().dropna().sort_values('ct')
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
            for label, color in zip(unique_cts['ct'], unique_cts['subclass_color'])]
plt.legend(handles=handles, title="Subclass", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
fig_path = output_base / f'whole_colored_depth_{column}_id_{donor_id}.png'
plt.savefig(fig_path, dpi=300)