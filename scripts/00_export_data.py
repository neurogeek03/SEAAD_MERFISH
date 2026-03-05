import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt

sc_ad = sc.read_h5ad("/scratch/mfafouti/SEA-AD/SEAAD_MTG_MERFISH.2024-12-11.h5ad")

# --- Export obs ---
obs_df = sc_ad.obs.copy()
obs_df.to_csv("/scratch/mfafouti/SEA-AD/out/SEAAD_obs.csv")
print("obs exported to SEAAD_obs.csv")

# --- Export obsm ---
# Flatten each obsm matrix into a DataFrame with column names like key_0, key_1, ...
obsm_dict = {}
for key in sc_ad.obsm.keys():
    arr = sc_ad.obsm[key]
    col_names = [f"{key}_{i}" for i in range(arr.shape[1])]
    obsm_dict[key] = pd.DataFrame(arr, index=sc_ad.obs_names, columns=col_names)

# Concatenate all obsm DataFrames horizontally
obsm_df = pd.concat(obsm_dict.values(), axis=1)
obsm_df.to_csv("/scratch/mfafouti/SEA-AD/out/SEAAD_obsm.csv")
print("obsm exported to SEAAD_obsm.csv")

# Loop through them and print some info
for key in sc_ad.obsm.keys():
    arr = sc_ad.obsm[key]
    print(f"\n=== {key} ===")
    print(f"Type: {type(arr)}")
    print(f"Shape: {arr.shape}")
    print("First 5 rows:")
    print(arr[:5])
