# SEAAD_MERFISH
How does spatial proximity of specific cell types predict Alzheimer's score?

Data in `h5ad` format available at: https://sea-ad-spatial-transcriptomics.s3.amazonaws.com/index.html#middle-temporal-gyrus/all_donors-h5ad/

## Crucial `.obs` columns: 
- `Depth from pia`: has a value only for cells that are part of a smaller rechtangular area on the tissue which contains cell types at the required proportions. This type of filtering was used for downstream analyses in the paper. 

## Crucial `.obsm` columns: 
- `X_spatial_raw_0` , `X_spatial_raw_1`: used to plot 1 section
- `X_spatial_tiled_0`, `X_spatial_tiled_1`: used to plot all sections together