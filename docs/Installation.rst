Prerequisites & Installation
-------------

Prerequisites
````````````

Flow2Spatial requires ``python >= 3.8``\.

::

    "torch", "shapely", "scikit-image", "cvxpy", 
    "scanpy", "anndata", "scipy", "numpy", "pandas" 


Installation
````````````

.. code-block:: bash

    pip install Flow2Spatial 

We also suggest to use a separate conda environment for installing Flow2Spatial. 

.. code-block:: bash

    conda create -y -n Flow2Spatial_env python=3.8
    conda activate Flow2Spatial_env
    pip install Flow2Spatial

