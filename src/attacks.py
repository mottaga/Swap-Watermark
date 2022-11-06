from numpy import random as np_random, clip as np_clip, float32 as np_float32
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter
from cv2 import imread as cv2_imread, resize as cv2_resize, IMWRITE_JPEG_QUALITY as cv2_IMWRITE_JPEG_QUALITY, imwrite as cv2_imwrite
from os import remove as os_remove


def awgn(img, std):
    mean = 0.0
    attacked = img + np_random.normal(mean, std, img.shape)
    attacked = np_clip(attacked, 0, 255)
    return attacked


def blur(img, sigma):
    attacked = gaussian_filter(img, sigma)
    return attacked


def sharpening(img, sigma, alpha):
    filter_blurred_f = gaussian_filter(img, sigma)
    attacked = img + alpha * (img - filter_blurred_f)
    return attacked


def median(img, kernel_size):
    attacked = medfilt(img, kernel_size)
    return attacked


def resizing(img, scale):
    x, y = img.shape
    new_x = int(x*scale)
    new_y = int(y*scale)
    attacked = cv2_resize(img, (new_x, new_y))
    attacked = cv2_resize(attacked, (x, y))
    return attacked


def jpeg_compression(img, QF):
    cv2_imwrite('tmp.jpg', img, [int(cv2_IMWRITE_JPEG_QUALITY), QF])
    attacked = cv2_imread('tmp.jpg', 0)
    attacked = np_float32(attacked)
    os_remove('tmp.jpg')
    return attacked