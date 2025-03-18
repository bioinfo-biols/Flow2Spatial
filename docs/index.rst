--------------------------------------------------------------------------------------------
Flow2Spatial reconstructs spatial proteomics through transfer learning
--------------------------------------------------------------------------------------------

**Flow2Spatial** is the computational part of PLATO (parallel flow projection and transfer learning across omics data).

It aims to reconstruct spatial proteomics from the values of parallel-flow projections in POTTLE. Leveraging transfer learning, Flow2Spatial can restore fine structure of protein spatial distribution in different tissue types.

.. image:: ./Flow2Spatial.png
    :width: 800px


Key functions of Flow2Spatial
----------------------------

+ Three data generators: generator.omics(), generator.histology() and generator.random(). 
+ Reconstruction model training: model.preparation(), model.training() and model.reconstruction(). 


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   Overview
   Installation
   Functions
   Generators
   Model

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   Reconstructing the spatial proteomics of rat large intestinal
