#!/usr/bin/env python

"""
Entry point for all WormPose commands
"""

import importlib
import sys
import argparse

from wormpose import __version__


def main():

    entry_points = {
        "datagen": "generate_training_data",
        "train": "train_model",
        "predict": "predict_dataset",
        "postprocess": "postprocess_results",
        "viz": "visualize_results",
        "calibrate": "calibrate_dataset",
        "evaluate": "evaluate_model",
        "export": "export_results",
    }

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "command", type=str, choices=entry_points.keys(), help="Choose which command to run",
    )
    parser.add_argument("-v", "--version", action="version", version=__version__)
    parser.parse_known_args()

    command = sys.argv.pop(1)
    module_name = entry_points[command]
    module = importlib.import_module(".".join(["wormpose", "commands", module_name]))
    module.main()


if __name__ == "__main__":
    main()
