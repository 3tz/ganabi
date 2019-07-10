import os
import sys
import gin
from tensorflow.keras.layers import ReLU, PReLU
from tensorflow.keras.activations import selu

# Set path
cur_dir = os.path.dirname(os.path.abspath(__file__))
par_dir = os.path.dirname(cur_dir) # parent directory
sys.path.insert(0, par_dir + '/baseline')
from mlp import *

# Include activations used in random search
ACTIVATIONS = [LeakyReLU, ELU, Softmax, ReLU, PReLU, selu]
for a in ACTIVATIONS:
    gin.external_configurable(a)

@gin.configurable
def build_model(hypers):
    """
    Arguments:
        - hypers: dict
            Dictionary of hyperparameters to be used. See details in
            `naive_mlp.config.gin`.
            Note: since the output is a pre-trained model, batch size must be
            manually passed in during actual training.
    Returns:
        - a compiled Keras model with specified hyperparameters besides
          training batch size.
    """
    m = Mlp(None, None, None, None, io_sizes=(SIZE_OBS_VEC, SIZE_ACT_VEC),
            out_activation=Softmax, loss='categorical_crossentropy',
            metrics=['accuracy'], **hypers, verbose=0)
    m.construct_model()
    m.model.compile(optimizer=Adam(lr=m.lr, decay=m.decay),
                    loss=m.loss, metrics=m.metrics)
    print('Fit with a batch size of', m.batch_size)
    return m.model
