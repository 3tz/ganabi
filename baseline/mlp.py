import os
import numpy as np
import pickle
import logging
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Embedding, Input, Flatten, Dropout
from tensorflow.keras.layers import BatchNormalization, LeakyReLU, ELU, Softmax
from tensorflow.keras.activations import sigmoid, tanh
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.utils import plot_model
from cross_validation import *

# suppress TF warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Set path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

class Mlp(object):
    def __init__(self, X_tr, Y_tr, X_va, Y_va, lr, batch_size, decay=.001,
                 activations=[LeakyReLU, LeakyReLU, Softmax],
                 layer_sizes=[SIZE_OBS_VEC, 256, 128, SIZE_ACT_VEC],
                 bNorm=True, dropout=False, regularizer=None,
                 loss='categorical_crossentropy', metrics=['accuracy'],
                 verbose=1):
        """ Initialize parameters required for training and testing the network.
            Structure:

                          [Input Layer]
                      length: @layer_sizes[0]
                                |
                                V
                         [Hidden Layer 1]
                      length: @layer_sizes[1]
                                |
                                V
                               ...
                         [Hidden Layer n]
                      length: @layer_sizes[n]
                                |
                                V
                             [Output]
                      length: @layer_sizes[-1]

        Arguments:
            - X_tr: np.matrix
                Training matrix that contains the observations.
            - Y_tr: np.matrix
                Training matrix that contains the actions.
            - X_va: np.matrix
                Validation matrix that contains the observations.
            - Y_va: np.matrix
                Validation matrix that contains the actions.
            - lr: int
                Learning rate of the network.
            - decay: float, default 0.001
                Learning rate decay.
            - batch_size: int
                Size of each mini-batch.
            - activations: func, default [LeakyReLU, LeakyReLU, Softmax]
                Activation functions that connect the layers.
            - layer_sizes: list, default [SIZE_OBS_VEC, 256, 128, SIZE_ACT_VEC]
                List of the sizes of each layer.
                Note: the 1st element is a tuple that contains the sizes of the
                 embedding layers for user IDs and joke IDs correspondingly.
             - bNorm: boolean, default True
                 Indicates whether to use Batch Normalization on all hiddens.
            - dropout: boolean, default False
                Incidates whether to use dropout for hidden layers.
            - regularizer: func, default None
                Regularizer to use.
            - loss: str, default 'categorical_crossentropy'
                Name of the loss function to be used.
            - metrics: list, default ['accuracy']
                List of metrics to be used for training.
            - verbose: int, default 1
                Value for `verbose` in keras fit() function.
        Returns:
            - None
        """

        assert(len(layer_sizes) == len(activations) + 1)

        self.X_tr = X_tr
        self.Y_tr = Y_tr
        self.X_va = X_va
        self.Y_va = Y_va
        self.lr = lr
        self.decay = decay
        self.batch_size = batch_size
        self.activations = activations
        self.layer_sizes = layer_sizes
        self.bNorm = bNorm
        self.dropout = dropout
        self.regularizer = regularizer
        self.loss = loss
        self.metrics = metrics
        self.verbose = verbose
        self.model = None # Model will be stored here after construct_model()
        self.hist = None  # History will be stored here after train_model()

    def construct_model(self):
        """ Construct model based on attributes
        """

        input = Input(shape=self.layer_sizes[0], name='input')

        layer = input
        for i in range(1, len(self.layer_sizes)-1):
            n = self.layer_sizes[i]
            a = self.activations[i-1]
            # Connect layers
            z = Dense(n, kernel_regularizer=self.regularizer,
                      name='hidden_%d' % i)(layer)

            if a._keras_api_names[0].split('.')[1] == 'activations':
                z = a(z)
            else: # Keras advance functions
                z = a()(z)

            if self.bNorm:
                z = BatchNormalization()(z)
            if self.dropout:
                z = Dropout(0.5)(z)
            layer = z

        out = Dense(self.layer_sizes[-1], name='output')(layer)
        out = self.activations[-1]()(out)

        self.model =  Model(inputs=input, outputs=out)

    def train_model(self, n_epoch=100, callbacks=None, verbose=True):
        """
        Train self.model with dataset stored in attributes.
        Arguments:
            - n_epoch: int, default 150
                Number of epochs to train.
            - callbacks: list, default None
                List of keras.callbacks.Callback objects to run
            - verbose: boolean, default True
                If true, model info will be displayed.
        """
        if verbose:
            print("Learning Rate:\t", self.lr)
            print("Batch Size:\t", self.batch_size)
            print("Activation:\t", self.activations)
            print("Layer Sizes:\t", self.layer_sizes)
            print("Dropout:\t", self.dropout)
            print("Batch Norm:\t", self.bNorm)
            print("Regularizer:\t", self.regularizer)
            print("Loss function:\t", self.loss)
            print("Callbacks:\t", callbacks)
            print()

        self.model.compile(optimizer=Adam(lr=self.lr, decay=self.decay),
                           loss=self.loss, metrics=self.metrics)

        self.hist = self.model.fit(self.X_tr, self.Y_tr, epochs=n_epoch,
                                   verbose=self.verbose,
                                   validation_data=(self.X_va, self.Y_va),
                                   batch_size=self.batch_size)
