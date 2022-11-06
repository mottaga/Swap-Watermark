from embedding import embedding
from detection import detection
import attacks
from cv2 import imread as cv2_imread
from numpy import load as np_load
from copy import deepcopy as copy_deepcopy


def main():
    # Load the mark and the image
    mark_path = '../mark/mark.npy'
    original_image_path = '../images/lena.bmp'
    mark_array = np_load(mark_path)
    original_image = cv2_imread(original_image_path, 0) # 0 for grayscale

    # Embed the image
    watermarked_image = embedding(original_image, mark_array)

    # Attack the image
    #attacked_image = attacks.awgn(copy_deepcopy(watermarked_image), 100)
    attacked_image = attacks.blur(copy_deepcopy(watermarked_image), 3)

    # Check if the attack was successful
    # wpsnr_wa is the wpsrn of the attacked image compared to the watermarked one
    # if the wpsnr is less than 35 the image is destroyed and finding the mark is not important
    result, wpsnr_wa = detection(original_image, watermarked_image, attacked_image)

    if result:
        print(f'Mark found with wpsnr of {wpsnr_wa}')
    else:
        print(f'Mark not found with a wpsnr of {wpsnr_wa}')
    
    return


if __name__ == '__main__':
    main()