--------------------------------------------------------------------------------------------
Flow2Spatial reconstructs spatial proteomics through transfer learning
--------------------------------------------------------------------------------------------

**Flow2Spatial** is the computational part of SPRING (global spatial proteomics with thousands of high-resolution pixels by microfluidics and transfer learning).

It aims to reconstruct spatial proteomics from the values of parallel-flow projections in SPRING. Leveraging transfer learning, Flow2Spatial can restore fine structure of protein spatial distribution in different tissue types.

.. image:: ./Flow2Spatial.png
    :width: 800px


Key functions of Flow2Spatial
----------------------------

+ three data generators: generator.omics(), generator.histology() and generator.random(). 
+ reconstruction model training: model.preparation(), model.training() and model.reconstruction(). 


.. toctree::
   :caption: Main
   :maxdepth: 1
   :hidden:

   Overview
   Installation
   Functions

.. toctree::
   :caption: Tutorials
   :maxdepth: 1
   :hidden:

   Reconstructing the spatial proteomics of villus
