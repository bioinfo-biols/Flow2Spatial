Reconstructing the spatial proteomics of villus
----------------------------------------------------

To present the pipeline of Flow2Spatial more clearly, we show the villus spatial proteomics reconstruction as an instance. 

Here, we applied PLATO to an adult rat large intestinal villus tissue using a 25 µm chip. A total of 2,451 proteins groups in ~100 channels (two angle) was successfully detected using LC-MS/MS, and needed to be reconstructed by Flow2Spatial. 

Reconstruction 
````````````````

With the trained model weights (located at https://github.com/bioinfo-biols/Flow2Spatial/tree/main/tests), we can use ``F2S.model.reconstruction()`` to reconstruct spatial proteomics with the real MS values. 

.. code-block:: python 

    import Flow2Spatial as F2S
    import pandas as pd
    ## Read protein abundance table
    channel_intensity_list_dat = pd.read_csv('./df_pro_gut.csv')
    channel_intensity_list_dat.loc[channel_intensity_list_dat['PG.Genes'].isna(), 'PG.Genes'] = channel_intensity_list_dat.loc[channel_intensity_list_dat['PG.Genes'].isna(), 'PG.ProteinAccessions'] + '_gene'
    channel_intensity_list_dat_dropna = channel_intensity_list_dat.dropna()

    ## Prediction
    F2S.model.reconstruction(channel_intensity_list_dat_dropna, DNN_model='./Recontruct_weights_gut.pkl', Xchannels=57, mask='./mask')

Corresponding channel_intensity, DNN_model and mask locate at https://github.com/bioinfo-biols/Flow2Spatial/tree/main/tests. We can reconstruct protein spatial distribution with these files. 

In default, an AnnData object would be saved at "./save_environ/adata.h5ad". You are welcome to use h5ad readers, such as ``sc.read_h5ad('save_environ/adata.h5ad')`` in scanpy, for further spatial proteomics analysis. 
