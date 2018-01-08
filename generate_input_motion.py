import numpy as np
import cv2
import os
#from matplotlib import pyplot as plt


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
        bigy = np.empty((n_im * 4, 80, 80))
        bigx = np.empty((n_im * 4, 80, 80, 2))
    else:
        bigy = np.empty((n_im, 80, 80))
        bigx = np.empty((n_im, 80, 80, 2))

    im = 0  # image counter
    for filename in os.listdir(folder):
        if not filename.startswith('.'):
            bigy_temp = cv2.imread(os.path.join(folder, filename),
                                   cv2.IMREAD_GRAYSCALE)
            bigy_padded = np.zeros((80, 80))
            bigy_padded[8:72, 8:72] = bigy_temp
            bigy[im, :, :] = bigy_padded
            bigx[im, :, :, :] = create_x(bigy_temp, normalize)
            im += 1
            if imrotate:
                for angle in [90, 180, 270]:
                    bigy_rot = im_rotate(bigy_temp, angle)
                    bigx_rot = create_x(bigy_rot, normalize)

                    bigy_rot_padded = np.zeros((80, 80))
                    bigy_rot_padded[8:72, 8:72] = bigy_rot

                    bigy[im, :, :] = bigy_rot_padded
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
    Prepares frequency data from image data: first image y is padded by 8
    pixels of value zero from each side (y_pad_loc1), then second image is
    created by moving the input image (64x64) 8 pixels down -> two same images
    at different locations are created; then both images are transformed to
    frequency space and their frequency space is combined as if the image
    moved half-way through the acquisition (upper part of freq space from one
    image and lower part of freq space from another image)
    expands the dimensions from 3D to 4D, and normalizes if normalize=True
    :param y: input image
    :param normalize: if True - the frequency data will be normalized
    :return: "Motion corrupted" frequency-space data of the input image,
    4D array of size (1, im_size1, im_size2, 2), third dimension (size: 2)
    contains real and imaginary part
    """

    # Pad y and move 8 pixels
    y_pad_loc1 = np.zeros((80, 80))
    y_pad_loc2 = np.zeros((80, 80))
    y_pad_loc1[8:72, 8:72] = y
    y_pad_loc2[0:64, 8:72] = y

    # FFT of both images
    img_f1 = np.fft.fft2(y_pad_loc1)  # FFT
    img_fshift1 = np.fft.fftshift(img_f1)  # FFT shift
    img_f2 = np.fft.fft2(y_pad_loc2)  # FFT
    img_fshift2 = np.fft.fftshift(img_f2)  # FFT shift

    # Combine halfs of both k-space - as if subject moved 8 pixels in the
    # middle of acquisition
    x_compl = np.zeros((80, 80), dtype=np.complex_)
    x_compl[0:41, :] = img_fshift1[0:41, :]
    x_compl[41:81, :] = img_fshift2[41:81, :]

    # Finally, separate into real and imaginary channels
    x_real = x_compl.real
    x_imag = x_compl.imag
    x = np.dstack((x_real, x_imag))

    x = np.expand_dims(x, axis=0)

    if normalize:
        x = x - np.mean(x)

    return x


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

# Image
plt.subplot(221), plt.imshow(Y[8, :, :], cmap='gray')
plt.title('Y_rot0'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(Y[9, :, :], cmap='gray')
plt.title('Y_rot90'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(Y[10, :, :], cmap='gray')
plt.title('Y_rot180'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(Y[11, :, :], cmap='gray')
plt.title('Y_rot270'), plt.xticks([]), plt.yticks([])
plt.show()

# Corresponding frequency space (magnitude)
X_m = np.sqrt(np.power(X[:, :, :, 0], 2)
              + np.power(X[:, :, :, 1], 2))
plt.subplot(221), plt.imshow(X_m[8, :, :], cmap='gray')
plt.title('X_freq_rot0'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(X_m[9, :, :], cmap='gray')
plt.title('X_freq_rot90'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(X_m[10, :, :], cmap='gray')
plt.title('X_freq_rot180'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(X_m[11, :, :], cmap='gray')
plt.title('X_freq_rot270'), plt.xticks([]), plt.yticks([])
plt.show()


# iFFT back to image from corrupted frequency space
X_compl = X[:, :, :, 0] + X[:, :, :, 1] * 1j

im_artif0 = np.fft.ifft2(X_compl[8, :, :])
im_artif1 = np.fft.ifft2(X_compl[9, :, :])
im_artif2 = np.fft.ifft2(X_compl[10, :, :])
im_artif3 = np.fft.ifft2(X_compl[11, :, :])

img_artif_M0 = np.sqrt(np.power(im_artif0.real, 2)
                       + np.power(im_artif0.imag, 2))
img_artif_M1 = np.sqrt(np.power(im_artif1.real, 2)
                       + np.power(im_artif1.imag, 2))
img_artif_M2 = np.sqrt(np.power(im_artif2.real, 2)
                       + np.power(im_artif2.imag, 2))
img_artif_M3 = np.sqrt(np.power(im_artif3.real, 2)
                       + np.power(im_artif3.imag, 2))

plt.subplot(221), plt.imshow(img_artif_M0, cmap='gray')
plt.title('X_rot0'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(img_artif_M1, cmap='gray')
plt.title('X_rot1'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.imshow(img_artif_M2, cmap='gray')
plt.title('X_rot2'), plt.xticks([]), plt.yticks([])
plt.subplot(224), plt.imshow(img_artif_M3, cmap='gray')
plt.title('X_rot3'), plt.xticks([]), plt.yticks([])
plt.show()
'''
