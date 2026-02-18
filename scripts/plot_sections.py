import scanpy as sc
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

project_path = Path.cwd().parents[1]
print(f"current working directory: {project_path}")
output_base = project_path / 'out'
output_base.mkdir(exist_ok=True, parents=True)
sc_ad = sc.read_h5ad("/scratch/mfafouti/SEA-AD/SEAAD_MTG_MERFISH.2024-12-11.h5ad")

column = 'X_spatial_raw'
donor_id = 'H20.33.004'
section = 'H20.33.035.Cx26.MTG.02.007.1.01.03'

spatial_df = pd.DataFrame(sc_ad.obsm[column], 
                          index=sc_ad.obs.index, 
                          columns=['x', 'y'])

# Add Donor ID to the spatial dataframe
spatial_df['Section'] = sc_ad.obs['Section'].values
spatial_df['Donor ID'] = sc_ad.obs['Donor ID'].values

# Subset for a specific donor, e.g., "H20.33.035"
subset_df = spatial_df[spatial_df['Section'] == section]

coords = pd.DataFrame(sc_ad.obsm[column], columns=["x", "y"])

plt.figure(figsize=(12, 9))
sc = plt.scatter(
    coords["x"], coords["y"],
    s=1
)
plt.axis("equal")
plt.title(f"Spatial plot {section}")

fig_path = output_base / f'{column}_id_{donor_id}.png'
plt.savefig(fig_path, dpi=300)