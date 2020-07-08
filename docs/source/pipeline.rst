WormPose pipeline
-----------------

There are several commands to run one at a time, please run all the commands from the same folder.

1 - Assemble a dataset
~~~~~~~~~~~~~~~~~~~~~~

|image0|

WormPose analyzes single worm videos where the simple non-coiled
centerlines have already been calculated with another worm tracker, for
example the `Tierpsy
tracker <https://github.com/ver228/tierpsy-tracker>`__. The tierpsy
format is supported out of the box for a single worm per video,
check the :ref:`tierpsy` guide to setup a Tierpsy dataset.

There is also a simple dataset used in the tutorials, which is a folder of images
associated with a HDF5 file for the features. You can also assemble your
own dataset composed of worm videos associated with their features
(`instructions <https://github.com/iteal/wormpose/tree/master/examples/toy_dataset>`__).

2 - Generate training data
~~~~~~~~~~~~~~~~~~~~~~~~~~

|image1|

The datagen command will create binary files \*.tfrecord containing
labeled images for training and evaluation, as well as a configuration
file. This step can take some time, a system with several CPU cores is recommended.

.. code:: bash

   wormpose datagen DATASET_LOADER_NAME DATASET_PATH

where DATASET_LOADER_NAME is “tierpsy” or “sample_data” or your custom
loader name.

**Warning**

This command generates a file "config.json" storing the information about the data: image size, dataset loader, resize factor etc.
This file needs to exist for the following steps to work. If you did not generate the training dataset and just want to predict new data for example, you need to obtain the configuration file (or regenerate a new one with the same parameters) and use the option --config to pass the path of the configuration file to the next scripts.

**Troubleshooting**

For the error *dataset_loader "tierpsy" (or other) not found in the package entry points*, please reinstall WormPose. If using Google Colab, please restart the runtime. If running WormPose source code directly, please run `pip install -e .` at the root level.

3 - Train the neural network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The neural network will be trained on the data generated from the
previous command. This step can take some time, a GPU is recommended.

.. code:: bash

   wormpose train DATASET_PATH

4 - Predict
~~~~~~~~~~~

|image2|

The network predicts the videos of the dataset, then there is a
processing step to discard wrong results and obtain a final time series
with a consistent head-tail orientation.

.. code:: bash

   wormpose predict DATASET_PATH

Extra optional commands
~~~~~~~~~~~~~~~~~~~~~~~

- Postprocess the results with interpolation and smoothing

.. code:: bash

   wormpose postprocess DATASET_PATH

-  Export the results in another format (only WCON format supported for
   Tierpsy data)

.. code:: bash

   wormpose export DATASET_PATH

-  Visualize the predictions as images

.. code:: bash

   wormpose viz DATASET_PATH

-  Calibrate to fine tune the image score threshold used during the
   prediction step

.. code:: bash

   wormpose calibrate DATASET_LOADER_NAME DATASET_PATH

-  Evaluate the trained model on new synthetic images

.. code:: bash

   wormpose evaluate DATASET_PATH


Outputs
~~~~~~~

All output files will be stored in a “experiments” folder in the
directory where the scripts are ran, or use the ``--work_dir`` option to
specify a custom directory.

::

   +-- experiments (or the value of work_dir)
   |   +-- dataset_name
   |   |   +-- config.json 
   |   |   +-- training_data  // contains the generated files used for training the network
   |   |   |   +-- *.tfrecord 
   |   |   +-- models  // contains the trained models
   |   |   +-- results  // contains the video analysis results
   |   |   |   +-- video_name0 
   |   |   |   |   +-- results.h5  // original and new angles and skeletons, image scores
   |   |   |   |   +-- images_results.zip  // Optional, when the script visualize_results is run. Visualize predicted skeletons overlaid on top of the original frame.
   |   |   |   +-- video_name1  
   |   |   |   |    ...

.. |image0| image:: https://i.imgur.com/rwZvTB9.png
.. |image1| image:: https://i.imgur.com/czrSMCY.png
.. |image2| image:: https://i.imgur.com/KBhnG1s.png

