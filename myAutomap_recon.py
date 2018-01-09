import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from matplotlib import pyplot as plt
from generate_input_motion import load_images_from_folder


# Load development/test data:
dir_dev = "path to the folder with dev/test images"
n_im_dev = 60  # How many images to load
# Load images and create motion-corrupted frequency space
# No normalization or rotations:
X_dev, Y_dev = load_images_from_folder(  # Load images for evaluating model
    dir_dev,
    n_im_dev,
    normalize=False,
    imrotate=False)
print('X_dev.shape at input = ', X_dev.shape)
print('Y_dev.shape at input = ', Y_dev.shape)


def create_placeholders(n_H0, n_W0):
    """ Creates placeholders for x and y for tf.session
    :param n_H0: image height
    :param n_W0: image width
    :return: x and y - tf placeholders
    """

    x = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0, 2], name='x')
    y = tf.placeholder(tf.float32, shape=[None, n_H0, n_W0], name='y')

    return x, y


def initialize_parameters():
    """ Initializes filters for the convolutional and de-convolutional layers
    :return: parameters - a dictionary of filters (W1 - first convolutional
    layer, W2 - second convolutional layer, W3 - de-convolutional layer
    """

    W1 = tf.get_variable("W1", [5, 5, 1, 64],  # 64 filters of size 5x5
                         initializer=tf.contrib.layers.xavier_initializer
                         (seed=0))
    W2 = tf.get_variable("W2", [5, 5, 64, 64],  # 64 filters of size 5x5
                         initializer=tf.contrib.layers.xavier_initializer
                         (seed=0))
    W3 = tf.get_variable("W3", [7, 7, 1, 64],  # 64 filters of size 7x7
                         initializer=tf.contrib.layers.xavier_initializer
                         (seed=0))  # conv2d_transpose

    parameters = {"W1": W1,
                  "W2": W2,
                  "W3": W3}

    return parameters


def forward_propagation(x, parameters):
    """ Defines all layers for forward propagation:
    Fully connected (FC1) -> tanh activation: size (n_im, n_H0 * n_W0)
    -> Fully connected (FC2) -> tanh activation:  size (n_im, n_H0 * n_W0)
    -> Convolutional -> ReLU activation: size (n_im, n_H0, n_W0, 64)
    -> Convolutional -> ReLU activation: size (n_im, n_H0, n_W0, 64)
    -> De-convolutional: size (n_im, n_H0, n_W0)
    :param x: Input - images in frequency space, size (n_im, n_H0, n_W0, 2)
    :param parameters: parameters of the layers (e.g. filters)
    :return: output of the last layer of the neural network
    """

    x_temp = tf.contrib.layers.flatten(x)  # size (n_im, n_H0 * n_W0 * 2)
    n_out = np.int(x.shape[1] * x.shape[2])  # size (n_im, n_H0 * n_W0)

    # FC: input size (n_im, n_H0 * n_W0 * 2), output size (n_im, n_H0 * n_W0)
    FC1 = tf.contrib.layers.fully_connected(
        x_temp,
        n_out,
        activation_fn=tf.tanh,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=None,
        biases_regularizer=None,
        reuse=True,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope='fc1')

    # FC: input size (n_im, n_H0 * n_W0), output size (n_im, n_H0 * n_W0)
    FC2 = tf.contrib.layers.fully_connected(
        FC1,
        n_out,
        activation_fn=tf.tanh,
        normalizer_fn=None,
        normalizer_params=None,
        weights_initializer=tf.contrib.layers.xavier_initializer(),
        weights_regularizer=None,
        biases_initializer=None,
        biases_regularizer=None,
        reuse=True,
        variables_collections=None,
        outputs_collections=None,
        trainable=True,
        scope='fc2')

    # Reshape output from FC layers into array of size (n_im, n_H0, n_W0, 1):
    FC_M = tf.reshape(FC2, [tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1])

    # Retrieve the parameters from the dictionary "parameters":
    W1 = parameters['W1']
    W2 = parameters['W2']
    W3 = parameters['W3']

    # CONV2D: filters W1, stride of 1, padding 'SAME'
    # Input size (n_im, n_H0, n_W0, 1), output size (n_im, n_H0, n_W0, 64)
    Z1 = tf.nn.conv2d(FC_M, W1, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    CONV1 = tf.nn.relu(Z1)

    # CONV2D: filters W2, stride 1, padding 'SAME'
    # Input size (n_im, n_H0, n_W0, 64), output size (n_im, n_H0, n_W0, 64)
    Z2 = tf.nn.conv2d(CONV1, W2, strides=[1, 1, 1, 1], padding='SAME')
    # RELU
    CONV2 = tf.nn.relu(Z2)

    # DE-CONV2D: filters W3, stride 1, padding 'SAME'
    # Input size (n_im, n_H0, n_W0, 64), output size (n_im, n_H0, n_W0, 1)
    batch_size = tf.shape(x)[0]
    deconv_shape = tf.stack([batch_size, x.shape[1], x.shape[2], 1])
    DECONV = tf.nn.conv2d_transpose(CONV2, W3, output_shape=deconv_shape,
                                    strides=[1, 1, 1, 1], padding='SAME')
    DECONV = tf.squeeze(DECONV)

    return DECONV


def model(X_dev):
    """ Runs the forward propagation to reconstruct images using trained model
    :param X_dev: input development frequency-space data
    :return: returns the image, reconstructed using a trained model
    """

    ops.reset_default_graph()  # to not overwrite tf variables
    (_, n_H0, n_W0, _) = X_dev.shape

    # Create Placeholders
    X, Y = create_placeholders(n_H0, n_W0)

    # Initialize parameters
    parameters = initialize_parameters()

    # Build the forward propagation in the tf graph
    forward_propagation(X, parameters)

    # Add ops to save and restore all the variables
    saver = tf.train.Saver()

    # Start the session to compute the tf graph
    with tf.Session() as sess:

        saver.restore(sess, "path to saved model/model_name.ckpt")

        print("Model restored")

        Y_recon_temp = forward_propagation(X, parameters)
        Y_recon = Y_recon_temp.eval({X: X_dev})

    return parameters, Y_recon


# Reconstruct the image using trained model
_, Y_recon = model(X_dev)
print('Y_recon.shape = ', Y_recon.shape)
print('Y_dev.shape = ', Y_dev.shape)


# Visualize the images, their reconstruction using iFFT and using trained model
# 4 images to visualize:
im1 = 32
im2 = 33
im3 = 34
im4 = 35

# iFFT back to image from corrupted frequency space
# Complex image from real and imaginary part
X_dev_compl = X_dev[:, :, :, 0] + X_dev[:, :, :, 1] * 1j

#iFFT
X_iFFT0 = np.fft.ifft2(X_dev_compl[im1, :, :])
X_iFFT1 = np.fft.ifft2(X_dev_compl[im2, :, :])
X_iFFT2 = np.fft.ifft2(X_dev_compl[im3, :, :])
X_iFFT3 = np.fft.ifft2(X_dev_compl[im4, :, :])

# Magnitude of complex image
X_iFFT_M1 = np.sqrt(np.power(X_iFFT0.real, 2)
                    + np.power(X_iFFT0.imag, 2))
X_iFFT_M2 = np.sqrt(np.power(X_iFFT1.real, 2)
                    + np.power(X_iFFT1.imag, 2))
X_iFFT_M3 = np.sqrt(np.power(X_iFFT2.real, 2)
                    + np.power(X_iFFT2.imag, 2))
X_iFFT_M4 = np.sqrt(np.power(X_iFFT3.real, 2)
                    + np.power(X_iFFT3.imag, 2))

# SHOW
# Show Y - input images
plt.subplot(341), plt.imshow(Y_dev[im1, :, :], cmap='gray')
plt.title('Y_dev1'), plt.xticks([]), plt.yticks([])
plt.subplot(342), plt.imshow(Y_dev[im2, :, :], cmap='gray')
plt.title('Y_dev2'), plt.xticks([]), plt.yticks([])
plt.subplot(343), plt.imshow(Y_dev[im3, :, :], cmap='gray')
plt.title('Y_dev3'), plt.xticks([]), plt.yticks([])
plt.subplot(344), plt.imshow(Y_dev[im4, :, :], cmap='gray')
plt.title('Y_dev4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using iFFT
plt.subplot(345), plt.imshow(X_iFFT_M1, cmap='gray')
plt.title('X_iFFT1'), plt.xticks([]), plt.yticks([])
plt.subplot(346), plt.imshow(X_iFFT_M2, cmap='gray')
plt.title('X_iFFT2'), plt.xticks([]), plt.yticks([])
plt.subplot(347), plt.imshow(X_iFFT_M3, cmap='gray')
plt.title('X_iFFT3'), plt.xticks([]), plt.yticks([])
plt.subplot(348), plt.imshow(X_iFFT_M4, cmap='gray')
plt.title('X_iFFT4'), plt.xticks([]), plt.yticks([])

# Show images reconstructed using model
plt.subplot(349), plt.imshow(Y_recon[im1, :, :], cmap='gray')
plt.title('Y_recon1'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 10), plt.imshow(Y_recon[im2, :, :], cmap='gray')
plt.title('Y_recon2'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 11), plt.imshow(Y_recon[im3, :, :], cmap='gray')
plt.title('Y_recon3'), plt.xticks([]), plt.yticks([])
plt.subplot(3, 4, 12), plt.imshow(Y_recon[im4, :, :], cmap='gray')
plt.title('Y_recon4'), plt.xticks([]), plt.yticks([])
plt.show()



