#!/usr/bin/env python

import setuptools

if __name__ == "__main__":
    with open("README.md") as f:
        long_description = f.read()

    version = {}
    with open("wormpose/version.py") as fp:
        exec(fp.read(), version)

    setuptools.setup(
        version=version["__version__"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        include_package_data=True,
        package_data={"wormpose": ["resources/postures_model.json.gz"]},
        entry_points={
            "console_scripts": ["wormpose = wormpose.__main__:main"],
            "worm_dataset_loaders": [
                "tierpsy = wormpose.dataset.loaders.tierpsy",
                "sample_data = wormpose.dataset.loaders.sample_data",
            ],
        },
    )
