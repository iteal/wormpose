"""
Implements a Keras callback to save the top N best models on evaluation data,
"""

import heapq
import json
import logging
import os

import tensorflow as tf

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class _BestModelsHeap(object):
    """
    Keeps track of the top N named objects with the smallest value
    uses a max-heap (heapq and inverting the sign of the value)
    """

    def __init__(self, values, names, maxlen):
        self.maxlen = maxlen
        self._queue = []

        for val, name in zip(values, names):
            heapq.heappush(self._queue, (-val, name))

    def append(self, value, name):

        if len(self._queue) < self.maxlen:
            heapq.heappush(self._queue, (-value, name))
        else:
            heapq.heappushpop(self._queue, (-value, name))

    def get_sorted(self):
        sorted_queue = list(reversed(sorted(self._queue)))
        return [-x[0] for x in sorted_queue], [x[1] for x in sorted_queue]


_NAME_PATTERN = "model.{epoch:02d}-{val_loss:.2f}.hdf5"


class BestModels(tf.keras.callbacks.Callback):
    def __init__(self, models_dir, models_to_keep=5):

        self.models_name_pattern = os.path.join(models_dir, _NAME_PATTERN)
        self.models_dir = models_dir
        self.backup_filepath = os.path.join(self.models_dir, "models_info.json")

        try:
            with open(self.backup_filepath, "r") as f:
                backup = json.load(f)
                val_loss = backup["best_val_loss"]
                names = backup["best_models"]
                epoch = backup["epoch"]
                last = backup["last_model"]
        except Exception:
            val_loss = []
            names = []
            epoch = 0
            last = ""

        self.epoch = epoch
        self.last_model_path = os.path.join(models_dir, last)
        self.best_model_path = os.path.join(models_dir, names[0]) if len(names) > 0 else self.last_model_path
        self.models_heap = _BestModelsHeap(values=val_loss, names=names, maxlen=models_to_keep)

        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        """
        At the end of each epoch, update the best models and save to json
        """

        val_loss = logs["val_loss"]

        last_model_path = os.path.basename(self.models_name_pattern.format(epoch=epoch + 1, val_loss=val_loss))
        self.models_heap.append(val_loss, last_model_path)

        best_val_loss, best_model_names = self.models_heap.get_sorted()

        for f in os.listdir(self.models_dir):
            if f not in best_model_names and f != os.path.basename(self.backup_filepath) and f != last_model_path:
                os.remove(os.path.join(self.models_dir, f))

        with open(self.backup_filepath, "w") as f:
            json.dump(
                {
                    "epoch": epoch + 1,
                    "best_val_loss": best_val_loss,
                    "best_models": best_model_names,
                    "last_model": last_model_path,
                },
                f,
                indent=4,
            )
