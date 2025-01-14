Generators
----------------------------

Generator.omics
````````````````````

.. code-block:: python

    Flow2Spatial.generator.omics(adata, mask, dir_run='./save_environ') 

Transfer molecular distribution from spatial omics as training data. 

The first input is the spatial distribution of moleculars from spatial omics (in the format of anndata). And the second inputs a bool matrix, showing whether tissue slice is palced in certain pixel. In default, output file will locate in the directory "./save_environ". 

Corresponding adata and mask locate at https://github.com/bioinfo-biols/Flow2Spatial/tree/main/tests under the names *Gut_reference.h5ad* and *mask*. 


Generator.histology
`````````````````````

.. code-block:: python

    Flow2Spatial.generator.histology(line_row, line_col, mask, segments, channel_intensity, radius=[0.5, 0.5], dir_run='./save_environ') 

Transfer histological information as training data. 

The first two inputs (*line_row*, *line_col*) are the parameters of the line equation in the parallel-flow projection of the first and last slice, which is placed by microfluidics chip. The input format should be dict, with the key as 'line1' 'line2' ... 'lineN' and the value as the (a, b, c) of the line equation ax + by + c = 0, hence {'line1': (a1, b1, c1), ... 'lineN': (aN, bN, cN)}. The third inputs a bool matrix, showing whether tissue slice is palced in certain pixel. After that, a list of histological clusters *segments* and MS intensity in each channel are needed for the function. 

Corresponding mask and channel_intensity locate at https://github.com/bioinfo-biols/Flow2Spatial/tree/main/tests under the names *mask* and *df_pro_gut.csv*. And *segments* can be generated from adata with following code: 

.. code-block:: python

    from anndata import read_h5ad
    adata = read_h5ad('./Gut_reference.h5ad')
    import pickle
    with open('./mask', 'rb') as handle:
        mask = pickle.load(handle)
    segments = F2S.transfer_masks(adata, mask, list_s=['Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6', 'Cluster7', 'Cluster8', 'Cluster9']) 

In default, output file will locate in the directory "./save_environ". 


Generator.random
````````````````````

.. code-block:: python

    Flow2Spatial.generator.random(input_type=['omics', 'histology'], times=20000, dir_run='./save_environ') 

Imporve the diversity of spatial distribution in the training set.

The first parameter represents which reference data will be used as the source for the random generator. We can specify *['histology']* as the only source or both *['omics', 'histology']*, which is the default. We can set parameter *times* to control the number of samples generated. Default is 20,000. In default, output file will locate in the directory "./save_environ". 
