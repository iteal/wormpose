"""
Contains constant values for default paths and folder names
"""

# root directory for all experiments, results etc
WORK_DIR = "experiments"

# Root folder to store the trained models
MODELS_DIRS = "models"

# Root folder to store the tfrecord files of training and evaluation data
TRAINING_DATA_DIR = "training_data"

# Path of the json file to store configuration about the experiment (which dataset loader, which image size etc)
CONFIG_FILENAME = "config.json"

# Root folder of the prediction results and the image visualization
RESULTS_DIR = "results"

# Name of the file containing the prediction results
RESULTS_FILENAME = "results.h5"

# Interpolation and smoothed results in this file
POSTPROCESSED_RESULTS_FILENAME = "processed_results.h5"

# Root folder for the results of the calibrate script
CALIBRATION_RESULTS_DIR = "calibration"

# Filename pattern of the dataset files
SYNTH_TRAIN_DATASET_NAMES = "synthetic_{index}.tfrecord"
REAL_EVAL_DATASET_NAMES = "real_{index}.tfrecord"
