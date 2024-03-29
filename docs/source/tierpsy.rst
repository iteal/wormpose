.. _tierpsy:

Use with Tierpsy tracker
------------------------

WormPose can load use Tierpsy tracker data, by specifying:

.. code:: python

    dataset_loader="tierpsy"

Create a root folder for an experiment, containing videos of similar recording conditions if possible.
Each subdirectory contains one video, the file tree should look like below (note the names of the subfolder match the names of the files):

.. code:: bash

    +-- dataset_path
    |   +-- video_name0
    |   |   +-- video_name0.hdf5
    |   |   +-- video_name0_features.hdf5  or video_name0_featuresN.hdf5
    |   +-- video_name1
    |   |   +-- video_name1.hdf5
    |   |   +-- video_name1_features.hdf5 or video_name1_featuresN.hdf5
    |   ...

For using `Open Worm Movement Database <http://movement.openworm.org/>`__ videos from the `Zenodo <https://zenodo.org/>`__ website, we provide a `download script <https://github.com/iteal/wormpose_data/tree/main/datasets/tierpsy>`__ that will create the file tree described above.

**Limitations and troubleshooting**

Please use the option "extract timestamp" in Tierpsy so that the field "timestamp/raw" is set in the hdf5 file. The configuration without the timestamp is not currently supported in WormPose.

For Tierpsy files with several worm indexes per file, WormPose will only load one worm, the one with the smallest index.

The Tierpsy coordinates loader uses the 'skeleton' key of the file {video_name}_features.hdf5 or {video_name}_featuresN.hdf5. Please verify that the coordinates of this table are in pixels and not microns or another unit. You may need to leave the default value of Microns Per Pixel = -1.00 in Tierpsy so that the skeletons values stay in image pixel coordinates.

Please contact the authors for any problems loading Tierpsy tracker files. Some features may not be implemented, the notebook check_dataset.ipynb is a good starting point to troubleshoot Tierpsy dataset loading issues.

**Advanced use**

The Tierpsy Dataset loader uses the default `SimpleFrameProcessing` class to segment the worms from the background. If this doesn't work for your images and you have a custom function to segment the worms from the background, you should implement a custom Dataset loader.
You can still use the tierpsy `FramesDataset` and `FeaturesDataset` provided, but you will need to reimplement `FramePreprocessing`.

Follow the example of toy_dataset to add a custom Dataset loader, which will look like below:

.. code:: python

    from wormpose import BaseFramePreprocessing

    # We use the Tierpsy loaders from WormPose
    from wormpose.dataset.loaders.tierpsy import FramesDataset, FeaturesDataset

    # But we redefine a custom frame preprocessing function
    class FramePreprocessing(BaseFramePreprocessing):

        def process(self, frame)

            segmented_frame = #TODO segment the frame, or load if precalculated
            background_color = #TODO find the background color
            return segmented_frame, background_color