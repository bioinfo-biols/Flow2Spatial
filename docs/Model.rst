Model
-----------------

Model.preparation
````````````````````

.. code-block:: python

    Flow2Spatial.model.preparation(input_type=['omics', 'histology', 'random'], dir_run='./save_environ', testing=0.1, mask='mask', design_row_file='design_row', design_col_file='design_col') 

Prepare the generated data into the format needed for the DNN model with default parameters. 

The first parameter represents which reference data will be used as the source for the training data. The default is all *['omics', 'histology', 'random']*. By default, the output files will locate in the directory "./save_environ" directory. The training and testing sets are randomly divided, with fraction controlled by *testing*. By default, the training and testing sets are set as 9:1. The following parameters uses the output of *Flow2Spatial.generator* by default. 


Model.training
````````````````````

.. code-block:: python

    Flow2Spatial.model.training(DNN_para=[12, 10, 8], batch_size=32, learning_rate=1e-3, epochs = 100, save_epoch=2, y_flag = 0, dir_run='./save_environ')

Train the reconstruction model. 

The first parameter represents the dimension of the DNN model. The default model parameter is *[12, 10, 8]*, which outputs a matrix with dimension 70x70. You can change it to meet custom needs. The following parameters represent the batch size, learning rate and  epochs of the model training.  The parameter *save_epoch* means that the trained model weights would be saved every 2 epochs. Finally, the loss from the taining and testing sets will locate at ``save_environ/loss.csv`` by default. Based on the loss, you can choose a better model (suitable epoch) for later reconstruction. 


Model.reconstruction
``````````````````````

.. code-block:: python

    Flow2Spatial.model.reconstruction(channel_intensity, DNN_model, Xchannels, mask='/DNN_data/mask', dir_run='./save_environ', out_adata=True, DNN_para=[12, 10, 8])

Reconstruct spatial proteomics with the real MS values. 

The first parameter is the MS intensity in each channel. The second is the path for the best model weights we select. The third means the number of channels in the first angle. The parameter *mask* inputs a bool matrix, showing whether tissue slice is palced in certain pixel. And the parameters *dir_run* and *out_adata* represent the name of the output file of reconstructed spatial proteomics, which will be in the format of h5ad. It locates at ``save_environ/adata.h5ad`` by default. If you change the DNN_para in ``F2S.model.training()`` , you will also need to pass it in ``F2S.model.reconstruction()``. 
