Functions
---------

The main functions in Flow2Spatial are listed below:


.. code-block:: python

    Flow2Spatial.generator.omics(adata, mask, dir_run='./save_environ') 


.. code-block:: python

    Flow2Spatial.generator.histology(line_row, line_col, mask, segments, channel_intensity, radius=[0.5, 0.5], dir_run='./save_environ') 


.. code-block:: python

    Flow2Spatial.generator.random(input_type=['omics', 'histology'], times=20000, dir_run='./save_environ')


.. code-block:: python

    Flow2Spatial.transfer_masks(adata, mask, list_s=['Cluster1', 'Cluster2','Cluster3', 'Cluster4'])


.. code-block:: python

    Flow2Spatial.model.preparation(input_type=['omics', 'histology', 'random'], dir_run='./save_environ', testing=0.1, mask='mask', design_row_file='design_row', design_col_file='design_col') 


.. code-block:: python

    Flow2Spatial.model.training(DNN_para=[12, 10, 8], batch_size=32, learning_rate=1e-3, epochs = 100, save_epoch=2, y_flag = 0, dir_run='./save_environ')


.. code-block:: python

    Flow2Spatial.model.reconstruction(channel_intensity, DNN_model, Xchannels, mask='/DNN_data/mask', dir_run='./save_environ', out_adata=True, DNN_para=[12, 10, 8])

