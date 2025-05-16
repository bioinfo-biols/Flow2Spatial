## Flow2Spatial reconstructs spatial proteomics through transfer learning 

Flow2Spatial is the computational part of PLATO (parallel flow projection and transfer learning across omics data). 

It aims to reconstruct spatial proteomics from the values of parallel-flow projections in PLATO. Leveraging transfer learning, Flow2Spatial can restore fine structure of protein spatial distribution in different tissue types. 


<p align="center">
  <img src='./docs/Flow2Spatial.png'>
</p>
<p align="center">
  Overview of Flow2Spatial.
</p>

### Prerequisites 
    "torch", "shapely", "cvxpy", "anndata",
     "scipy", "numpy", "pandas"

Further tutorials please refer to  https://Flow2Spatial.readthedocs.io/. 


### Multi-omics datasets 
The multi-omics cerebellum datasets can be found at https://github.com/bioinfo-biols/Flow2Spatial/tree/main/datasets. These data are in the Anndata format, which is compatible with the Python package `anndata`. The corresponding spatial domain can be accessed via adata.obs['cluster'].
Specifically, 'Cerebellum-PLATO.h5ad' contains the spatial proteomics of the cerebellum, which was reconstructed based on the H&E staining reference. The highly variable proteins after sc.pp.highly_variable_genes() selection are saved in `adata.X`, and the raw intensity of all reconstructed proteins can be accessed via `adata.raw.X`. 
In addition, 'Cerebellum-MAGIC-seq.h5ad' and 'Cerebellum-MALDI-MSI.h5ad' correspond to the spatial transcriptomics and spatial metabolomics datasets, respectively. The normalized data after log-transformation is saved in `adata.X`, and the raw data can be accessed via `adata.layers['raw']`.


### Citation 

Beiyu Hu, Ruiqiao He, Kun Pang, Guibin Wang, et al. High-resolution spatially resolved proteomics of complex tissues based on microfluidics and transfer learning. Cell 188, 1-15 (2025). 
