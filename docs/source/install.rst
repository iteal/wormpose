Installation
------------

**Requirements**

Install Python 3.6 or higher.

This project make extensive use of multi processing, so we
recommend you use a system with as many CPU cores as possible.

A GPU is recommended to train the network, please check Tensorflow
`documentation <https://www.tensorflow.org/install/gpu>`__ to install
Tensorflow with GPU support.

**Install with pip**

You can install WormPose directly from the Python package index with:

.. code:: bash

   pip install --upgrade wormpose

This will install a command line command called `wormpose`, that you can call
directly in the terminal. Alternatively, you can use Jupyter notebooks
or directly call the Python functions.

.. code:: bash

   # How to run wormpose from the terminal:
   wormpose COMMAND ARGUMENTS

   # The available commands are:
   # datagen, train, predict, postprocess, viz,
   # calibrate, evaluate, export

   # Get information on the arguments for each command by calling:
   wormpose COMMAND -h


**Troubleshooting**

If the `wormpose` command is not found, maybe you installed WormPose with the pip option ``--user``. Please check
that the local user folder where pip installs the script is in your PATH
variable.


**Limitations**

WormPose doesn't support at the moment multiple worms in one image.

Only grayscale images (1 channel) are supported.

The resolution of the worm should not be too low, we have tested it successfully with images where the worm length was 90 pixels, but having a smaller resolution will degrade the accuracy rapidly.
