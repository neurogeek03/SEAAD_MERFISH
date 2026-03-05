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

#########################################################

# PARAMS
RECTANGLE = False
celltype_taxonomy = 'Class' #options: 'Supertype' , 'Subclass' , 'Class'
celltype_color = f'{celltype_taxonomy}_color'
column = 'X_spatial_raw'
donor_id = 'H20.33.004'
section = 'H20.33.035.Cx26.MTG.02.007.1.01.03'

#########################################################
spatial_df = pd.DataFrame(sc_ad.obsm[column], 
                          index=sc_ad.obs.index, 
                          columns=['x', 'y'])

# Add Donor ID to the spatial dataframe
spatial_df['Section'] = sc_ad.obs['Section'].values
spatial_df['Donor ID'] = sc_ad.obs['Donor ID'].values
# spatial_df['ct'] = sc_ad.obs[celltype_taxonomy]
spatial_df[celltype_taxonomy] = sc_ad.obs[celltype_taxonomy]
spatial_df['depth'] = sc_ad.obs['Depth from pia']

# obtaining taxonomy colors 
# subclass_colors = color_csv[['subclass_label', 'subclass_color']].drop_duplicates()
spatial_df = spatial_df.merge(color_csv, left_on=celltype_taxonomy, right_on=f'{celltype_taxonomy}_label', how='left')

# Subset for a specific donor, e.g., "H20.33.035"
subset_df = spatial_df[spatial_df['Section'] == section]

if RECTANGLE:
    subset_df = subset_df[subset_df['depth'].notna()]

plt.figure(figsize=(12, 9))
scatter = plt.scatter(
    subset_df["x"], subset_df["y"],
    c=subset_df[celltype_color],
    s=3
)
plt.axis("equal")
plt.title(f"Spatial plot {section}")

unique_cts = subset_df[[celltype_color, celltype_taxonomy]].drop_duplicates().dropna().sort_values(celltype_taxonomy)
handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=label)
            for label, color in zip(unique_cts[celltype_taxonomy], unique_cts[celltype_color])]
plt.legend(handles=handles, title=celltype_taxonomy, bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
fig_path = output_base / f'whole_{celltype_taxonomy}_colored_depth_{column}_section_{section}.png'
plt.savefig(fig_path, dpi=300)