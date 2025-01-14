Prerequisites & Installation
-------------

Prerequisites
````````````

Flow2Spatial requires ``python >= 3.7.9``\.

::

    "torch", "shapely", "scikit-image", "cvxpy", 
    "scanpy", "anndata", "scipy", "numpy", "pandas" 


Installation
````````````

.. code-block:: bash

    pip install Flow2Spatial

Because Flow2Spatial requires ``Pytorch`` to train deep learning model, an additional installation step is needed. Please refer to https://pytorch.org/get-started/locally/ for customized installation. 

We also suggest to use a separate conda environment for installing Flow2Spatial. 

.. code-block:: bash

    conda create -y -n Flow2Spatial_env python=3.8.10
    conda activate Flow2Spatial_env
    pip install Flow2Spatial

Similarly, ``Pytorch`` also requires a additional customized installation (https://pytorch.org/get-started/locally/).

