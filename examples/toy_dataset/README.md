## Load custom data

### Toy dataset example
An example of implementing a custom dataset loader is available [here](https://github.com/iteal/wormpose/tree/main/examples/toy_dataset) (using toy data).

First install the WormPose package.
```bash
pip install --upgrade wormpose
```
To install the toy dataset loader, run the following command in the folder `toy_dataset`:
```bash
pip install -e .
```
Now you can use the dataset loader named "toy" :
```bash
wormpose COMMAND toy "toy_dataset_path"
```
Toy data will be generated at runtime and saved to toy_dataset_path.

### Use your own data

Follow the toy dataset example to add your custom dataset loader.

First, choose a name for your dataset loader (`cool_worms` for example).

Create a python module (`cool_worms_loader.py` for example), containing three classes, such as below:

```python
from wormpose import BaseFramesDataset, BaseFeaturesDataset, BaseFramePreprocessing

class FramesDataset(BaseFramesDataset):
    """
    Reader for the images of the dataset
    """
    raise NotImplementedError("Here, implement interface BaseFramesDataset") 

class FeaturesDataset(BaseFeaturesDataset):
    """
    Features corresponding to the images (skeleton, width)
    """
    raise NotImplementedError("Here, implement interface BaseFeaturesDataset")

class FramePreprocessing(BaseFramePreprocessing):
    """
    Contains a function to segment the worm in images and return the background color
    """
    raise NotImplementedError("Here, implement interface BaseFramePreprocessing")
```

The API documentation can be found [here](https://github.com/iteal/wormpose/blob/main/wormpose/dataset/base_dataset.py). 

Create a file `setup.py` containing the entry point `worm_dataset_loaders` refering to your python module, with its loader name, such as below:
```python
from setuptools import setup

setup(name = "cool_worms_dataset",
      entry_points = {"worm_dataset_loaders": ["cool_worms=cool_worms_loader"]})
```

After installing wormpose, install your module with:
```bash
pip install -e .
```

Now you can use the dataset loader named "cool_worms" :
```bash
wormpose COMMAND cool_worms "path/to/cool_worms_dataset000XX"
```

Use the notebook [check_dataset](https://github.com/iteal/wormpose/blob/main/examples/check_dataset.ipynb) to validate your custom dataset loader. You can view if the frames load correctly, and if the features are accurate.