import tensorflow as tf
import numpy as np


def schedule(epoch, lr):
    if epoch <= 1:
        return lr
    else:
        return lr * np.sqrt(3)**(1-epoch)


def get_scheduler():
    return tf.keras.callbacks.LearningRateScheduler(schedule=schedule)
