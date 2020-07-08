from typing import Callable


def _log_parameters(logger: Callable, params: dict):
    """
    Utility function to log script parameters with a more readable formatting
    """
    logger("\n\t" + ", \n\t".join([f"{x[0]}: {x[1]}" for x in params.items()]))


from wormpose.commands.calibrate_dataset import calibrate
from wormpose.commands.evaluate_model import evaluate
from wormpose.commands.export_results import export
from wormpose.commands.generate_training_data import generate
from wormpose.commands.postprocess_results import post_process
from wormpose.commands.predict_dataset import predict
from wormpose.commands.train_model import train
from wormpose.commands.visualize_results import visualize
