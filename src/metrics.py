import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
import tensorflow as tf
import tensorflow_addons as tfa


class CustomF1Score(tfa.metrics.F1Score):
    __name__ = "CustomF1Score"

    def __init__(self, num_classes, average=None, threshold=0.5, **kwargs) -> None:
        self.num_classes = num_classes
        super().__init__(
            self.num_classes, average=average, threshold=threshold, **kwargs
        )

    def __call__(self, y, pred):
        pred = tf.convert_to_tensor(pred)
        y = tf.cast(y, pred.dtype)
        result = super().__call__(y, pred)
        return result


loss = tf.keras.losses.binary_crossentropy
accuracy = tf.keras.metrics.binary_accuracy
