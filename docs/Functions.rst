Functions
---------

The main functions in Flow2Spatial are listed below:


.. code-block:: python

    Flow2Spatial.generator.omics(adata, mask, dir_run='./save_environ') 


.. code-block:: python

    Flow2Spatial.generator.histology(line_row, line_col, mask, segments, channel_intensity, radius=[0.5, 0.5], dir_run='./save_environ') 


.. code-block:: python

    Flow2Spatial.generator.random(input_type=['omics', 'histology'], dir_run='./save_environ', times=20000)


.. code-block:: python

    Flow2Spatial.model.preparation(input_type=['omics', 'histology', 'random'], dir_run='./save_environ', testing=0.1, mask='mask', design_row_file='design_row', design_col_file='design_col') 


.. code-block:: python

    Flow2Spatial.model.training(DNN_para=[12, 10, 8], batch_size = 32, learning_rate=1e-3, epochs = 100, save_epoch=2, y_flag = 0, dir_run='./save_environ')


.. code-block:: python

    Flow2Spatial.model.reconstruction(select_epoch, channel_intensity, out_adata='adata', DNN_para=[12, 10, 8], dir_run='./save_environ')

