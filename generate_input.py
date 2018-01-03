import numpy as np
import cv2
import os
# from matplotlib import pyplot as plt


def load_images_from_folder(folder, n_im, normalize=False, imrotate=False):
    """ Loads n_im images from the folder and puts them in an array bigy of
    size (n_im, im_size1, im_size2), where (im_size1, im_size2) is an image
    size.
    Performs FFT of every input image and puts it in an array bigx of size
    (n_im, im_size1, im_size2, 2), where "2" represents real and imaginary
    dimensions
    :param folder: path to the folder, which contains images
    :param n_im: number of images to load from the folder
    :param normalize: if True - the xbig data will be normalized
    :param imrotate: if True - the each input image will be rotated by 90, 180,
    and 270 degrees
    :return:
    bigx: 4D array of frequency data of size (n_im, im_size1, im_size2, 2)
    bigy: 3D array of images of size (n_im, im_size1, im_size2)
    """

    # Initialize the arrays:
    if imrotate:  # number of images is 4 * n_im
        bigy = np.empty((n_im * 4, 64, 64))
        bigx = np.empty((n_im * 4, 64, 64, 2))
    else:
        bigy = np.empty((n_im, 64, 64))
        bigx = np.empty((n_im, 64, 64, 2))

    im = 0  # image counter
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
            bigy[im, :, :] = bigy_temp
            bigx[im, :, :, :] = create_x(bigy_temp, normalize)
            im += 1
            if imrotate:
                for angle in [90, 180, 270]:
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigx_rot = create_x(bigy_rot, normalize)
                    bigy[im, :, :] = bigy_rot
                    bigx[im, :, :, :] = bigx_rot
                    im += 1

        if imrotate:
            if im > (n_im * 4 - 1):  # how many images to load
                break
        else:
            if im > (n_im - 1):  # how many images to load
                break

    if normalize:
        bigx = (bigx - np.amin(bigx)) / (np.amax(bigx) - np.amin(bigx))

    return bigx, bigy


def create_x(y, normalize=False):
    """
    Prepares frequency data from image data: applies to_freq_space,
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: frequency data 4D array of size (1, im_size1, im_size2, 2)
    """
    x = to_freq_space(y)  # FFT: (128, 128, 2)
    x = np.expand_dims(x, axis=0)  # (1, 128, 128, 2)
    if normalize:
        x = x - np.mean(x)

    return x


def to_freq_space(img):
    """ Performs FFT of an image
    :param img: input 2D image
    :return: Frequency-space data of the input image, third dimension (size: 2)
    contains real ans imaginary part
    """

    img_f = np.fft.fft2(img)  # FFT
    img_fshift = np.fft.fftshift(img_f)  # FFT shift
    img_real = img_fshift.real  # Real part: (im_size1, im_size2)
    img_imag = img_fshift.imag  # Imaginary part: (im_size1, im_size2)
    img_real_imag = np.dstack((img_real, img_imag))  # (im_size1, im_size2, 2)

    return img_real_imag


def im_rotate(img, angle):
    """ Rotates an image by angle degrees
    :param img: input image
    :param angle: angle by which the image is rotated, in degrees
    :return: rotated image
    """
    rows, cols = img.shape
    rotM = cv2.getRotationMatrix2D((cols/2-0.5, rows/2-0.5), angle, 1)
    imrotated = cv2.warpAffine(img, rotM, (cols, rows))

    return imrotated


'''
# For debugging: show the images and their frequency space

dir_temp = 'path to folder with images'
X, Y = load_images_from_folder(dir_temp, 5, normalize=False, imrotate=True)

print(Y.shape)
print(X.shape)


plt.subplot(221), plt.imshow(Y[12, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(Y[13, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(Y[14, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(Y[15, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()

X_m = 20*np.log(np.sqrt(np.power(X[:, :, :, 0], 2) +
                        np.power(X[:, :, :, 1], 2)))  # Magnitude
plt.subplot(221), plt.imshow(X_m[12, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(X_m[13, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(X_m[14, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(X_m[15, :, :], cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
'''
