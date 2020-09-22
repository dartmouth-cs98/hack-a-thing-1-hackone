import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.layers import Dense, Dropout, GRU, LSTM
from tensorflow.keras import Sequential
import numpy as np

"""
Here I wanted to practice building a bigger custom model that can be used for a specific purpose. That purpose I never came up with.

The general idea is to take a video, feed each frame through a CNN to extract features, and then feed each frames features through an RNN. 
I used the prebuilt InceptionV3 CNN (seemed quite powerful and popular) and then either a GRU or LSTM RNN.

The output is a single value (between 0 and 1). I got this style of architecture from a few papers that were trying to determine 
if a video was a deepfake or not. I feel like this could be extended to alot of cool scenerios

I used 'Hands-On Machine Learning with Skikit-Learn, Keras and TensorFlow' by Aurelien Geron 
This book had a lot of cool information about building custom models, CNNs and RNNs.

If I had more time I would try to train this on real video data to try to predict some real quality of a video. The inception 
model has ~2.1x10^7 trainable parameters and the RNN had ~5.4x10^5, so would probably need alot of computing power to train this 
for real.

Running this will give a model summary and then a test forward pass just to make sure it works
"""


class InceptionV3toRNN(tf.keras.Model):
    def get_config(self):
        pass

    def __init__(self, rnn_type='GRU', input_shape=None, *args, **kwargs):
        """
        Defines the various layers and modules that will be used in a forward pass of the model.
        This model preprocess the data, uses the InceptionV3 CNN, and then either a GRU or LSTM RNN.

        :param rnn_type: Defines what type of RNN the inception model will feed into. Either GRU or LSTM
        :param input_shape: Tuple to represent the shape of the input to the network. Should be in the form
                            (frames, height, width, channels), and by default (300,299,299,3)
        """

        # Build the inpection model. Input shape: [input_shape]. Output shape: [frames, 8,8,2048]
        super().__init__(*args, **kwargs)
        self.inception_model = self.build_inception(input_shape)

        # Add pooling layers to normalize the input for the
        self.pooling = tf.keras.layers.AveragePooling2D(pool_size=(8, 8), data_format='channels_last')
        self.reshaping = tf.keras.layers.Reshape([2048])

        self.RNN = self.build_rnn(input_shape=[300, 2048],
                                  output_shape=1,
                                  rnn_type=rnn_type,
                                  layers=[60, 80, 80, 80, 100],
                                  activation_function='tanh')

    def summary(self):
        """
        Simply prints out the summaries of all the components of the model
        :return:
        """
        print('#################### Inception Summary ##########################')
        self.inception_model.summary()

        print('#################### RNN Summary ##########################')
        self.RNN.summary()

    def call(self, inputs, **kwargs):
        """
        Takes the given input, runs a single forward pass, then return the output

        :param inputs: Tensor representing a single video. Should be of the shape [300, 299, 299, 3]
        :return: A tensor representing the probability that the video is fake at any one frame.
        """

        # Preprocess input
        processed_input = preprocess_input(inputs)

        # Run the inputs through the inception model. Output is of size [frames, 8, 8, 2048]
        out = self.inception_model(processed_input)

        # Normalize the input to the RNN. Output is shape [frames, 2048]
        out = self.pooling(out)
        out = self.reshaping(out)

        # Run the inputs through the RNN. Adds a dummy batch dimension, which is 1. Output is of shape [1,300,1]
        out = tf.expand_dims(out, axis=0)
        out = self.RNN(out)

        # Squeeze the output down to a single column vector of shape [frames]
        out = tf.squeeze(out)

        return out

    def build_inception(self, input_shape) -> tf.keras.Model:
        """
        Builds the inceptionV3 model.

        :param input_shape: Shape of the video input
        :return: The model
        """
        super(InceptionV3toRNN, self).__init__()

        # Define the inception model
        base_model = InceptionV3(include_top=False, weights='imagenet')
        hidden_layer = base_model.layers[-1].output

        # Create the feature extraction model
        # Set the input shape of the data
        if not input_shape:
            inputs = base_model.input
        else:
            inputs = tf.keras.Input(input_shape)

        return tf.keras.Model(inputs=inputs, outputs=hidden_layer)

    def build_rnn(self, input_shape, output_shape, rnn_type, layers, activation_function) -> tf.keras.Model:
        """
        Builds a LSTM or GRU model.

        :param input_shape: Shape of the input layer of the model
        :param output_shape: Shape of the output layer of the model
        :param rnn_type: Either 'GRU' or 'LSTM'
        :param layers: List of ints. Describes the depth of each layer of the model
        :param activation_function: Activation function for each RNN layer
        :return: The model
        """

        num_layers = len(layers)

        # Get the appropriate keras layer class
        rnn_class = {'GRU': GRU, 'LSTM': LSTM}[rnn_type]

        # Determine the dropout probabilities for each layer. Spaced out evenly between 20% and 60%
        dropouts = np.linspace(0.2, 0.6, num_layers)

        model = Sequential()

        # Add the input layer
        model.add(rnn_class(layers[0],
                            activation=activation_function,
                            return_sequences=True,
                            input_shape=input_shape))
        model.add(Dropout(dropouts[0]))

        # Add the hidden layers except the last RNN layer
        for layer_i in range(1, num_layers - 1):
            model.add(rnn_class(layers[layer_i], activation=activation_function, return_sequences=True))
            model.add(Dropout(dropouts[layer_i]))

        # Add the final layer
        model.add(rnn_class(layers[-1], activation=activation_function, return_sequences=False))
        model.add(Dropout(dropouts[-1]))

        # Add the final dense layer
        model.add(Dense(output_shape, activation='sigmoid'))

        return model

    def loss_fn(self, ground_truth, calculated):
        """
        This loss function simply takes the difference in the magnitude of the two values. Simple stuff

        :param ground_truth: This is the ground truth label. [0,1]
        :param calculated: This is the predicted value by a single forward pass of the network
        :return: A value for loss given between -1 and 1
        """

        return abs(ground_truth - calculated)


if __name__ == '__main__':

    # Create some test output. This is just 300 frames of 299x299 video with RGB all being 1. This is just to see if
    # forward passes work
    test_input = tf.ones([300, 299, 299, 3], dtype=tf.float32)

    # Make then model
    model = InceptionV3toRNN('GRU')

    model.compile()
    model.summary()
    test_output = model.call(test_input)

    print(f'Test forward pass output: {test_output}')
