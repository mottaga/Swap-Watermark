from scipy.fft import dct
from scipy.signal import convolve2d as ss_convolve2d
from copy import deepcopy as copy_deepcopy
from math import sqrt as math_sqrt
import numpy as np


def wpsnr(original_matrix, modified_matrix):
    original_matrix_copy = (copy_deepcopy(original_matrix) / 255.0)
    modified_matrix_copy = (copy_deepcopy(modified_matrix) / 255.0)
    difference = original_matrix_copy - modified_matrix_copy
    if not np.any(difference):
        return 255
    csf = np.genfromtxt('../wpsnr_weights/csf.csv', delimiter = ',')
    ew = ss_convolve2d(difference, np.rot90(csf, 2), mode = 'valid')
    quality = 20.0 * np.log10(1.0 / math_sqrt(np.mean(np.mean(ew ** 2))))
    return round(quality, 5)


def similarity(original_mark_array, new_mark_array):
    difference = original_mark_array - new_mark_array
    if not np.any(difference):
        return 24
    difference = np.zeros(len(new_mark_array)) - new_mark_array
    if not np.any(difference):
        return 0
    sim = np.sum(np.multiply(original_mark_array, new_mark_array)) / np.sqrt(np.sum(np.multiply(new_mark_array, new_mark_array)))
    return sim


def detection(original_image, watermarked_image, attacked_image):
    # Settings
    chunk_size = (8, 8)
    spots_to_swap = [[(2, 6), (3, 5)], [(5, 3), (6, 2)], [(0, 7), (7, 0)]]
    mark_size = 1024
    attacked_mark_extracted = np.zeros(mark_size, dtype = np.float64)
    original_mark_extracted = np.zeros(mark_size, dtype = np.float64)
    optimal_average = 64
    inverting_limit = 44.5
    mark_partitioning = [325, 650]
    threshold = 12.6

    # Sort the chunk based on their average in the spatial domain (ascending order)
    chunks = []
    for i in range(0, original_image.shape[0], chunk_size[0]):
        for j in range(0, original_image.shape[1], chunk_size[1]):
            chunk = original_image[i : i + chunk_size[0], j : j + chunk_size[1]]
            temp = {'coordinates' : (i, i + chunk_size[0], j, j + chunk_size[1]), 'average' : np.average(chunk)}
            chunks.append(temp)
    chunks_sorted = sorted(chunks, key = lambda d : d['average'], reverse = False)
    
    # Find the index where the average is > optimal_average
    index_average = 0
    for i in range(len(chunks_sorted)):
        if chunks_sorted[i]['average'] > optimal_average:
            index_average = i
            break
    if len(chunks_sorted) - index_average < mark_size:
        index_average -= mark_size # A possible improvement could be index_average -= (mark_size - len(chunk_sorted) - index_average)
        if index_average < 0:
            index_average = 0

    # Walk the image chunk by chunk
    portion_index = 0
    for i in range(mark_size):
        # Change the spots to swap
        if i >= mark_partitioning[0] and i < mark_partitioning[1]:
            portion_index = 1
        elif i >= mark_partitioning[1]:
            portion_index = 2
        # Apply the DCT to the chunk
        original_chunk = original_image[chunks_sorted[i + index_average]['coordinates'][0] : chunks_sorted[i + index_average]['coordinates'][1], chunks_sorted[i + index_average]['coordinates'][2] : chunks_sorted[i + index_average]['coordinates'][3]]
        original_chunk_dct = dct(dct(original_chunk, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
        attacked_chunk = attacked_image[chunks_sorted[i + index_average]['coordinates'][0] : chunks_sorted[i + index_average]['coordinates'][1], chunks_sorted[i + index_average]['coordinates'][2] : chunks_sorted[i + index_average]['coordinates'][3]]
        attacked_chunk_dct = dct(dct(attacked_chunk, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
        watermarked_chunk = watermarked_image[chunks_sorted[i + index_average]['coordinates'][0] : chunks_sorted[i + index_average]['coordinates'][1], chunks_sorted[i + index_average]['coordinates'][2] : chunks_sorted[i + index_average]['coordinates'][3]]
        watermarked_chunk_dct = dct(dct(watermarked_chunk, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
        # Check the mark value of the attacked image
        if (((original_chunk_dct[spots_to_swap[portion_index][0]] > original_chunk_dct[spots_to_swap[portion_index][1]]) and (attacked_chunk_dct[spots_to_swap[portion_index][0]] > attacked_chunk_dct[spots_to_swap[portion_index][1]])) 
            or  ((original_chunk_dct[spots_to_swap[portion_index][0]] < original_chunk_dct[spots_to_swap[portion_index][1]]) and (attacked_chunk_dct[spots_to_swap[portion_index][0]] < attacked_chunk_dct[spots_to_swap[portion_index][1]]))):
            attacked_mark_extracted[i] = 0
        else:
            attacked_mark_extracted[i] = 1
        # Check the mark value of the watermarked image
        if (((original_chunk_dct[spots_to_swap[portion_index][0]] > original_chunk_dct[spots_to_swap[portion_index][1]]) and (watermarked_chunk_dct[spots_to_swap[portion_index][0]] > watermarked_chunk_dct[spots_to_swap[portion_index][1]]))
            or ((original_chunk_dct[spots_to_swap[portion_index][0]] < original_chunk_dct[spots_to_swap[portion_index][1]]) and (watermarked_chunk_dct[spots_to_swap[portion_index][0]] < watermarked_chunk_dct[spots_to_swap[portion_index][1]]))):
            original_mark_extracted[i] = 0
        else:
            original_mark_extracted[i] = 1
        
    # Calculate the wpsnr between the watermarked image and the attacked image
    wpsnr_wa = wpsnr(watermarked_image, attacked_image)

    # Check if we have to invert a part of the mark
    temp = [0, mark_partitioning[0], mark_partitioning[1], mark_size]
    for r in range(len(temp) - 1):
        accuracy = 0
        for k in range(temp[0 + r], temp[1 + r]):
            if original_mark_extracted[k] == attacked_mark_extracted[k]:
                accuracy += 1
        if ((accuracy / (temp[1 + r] - temp[0 + r])) * 100) < inverting_limit:
            attacked_mark_extracted[temp[0 + r] : temp[1 + r]] = np.array([0 if attacked_mark_extracted[k] == 1 else 1 for k in range(temp[0 + r], temp[1 + r])])
        
    # Calculate the similarity and return
    sim = similarity(original_mark_extracted, attacked_mark_extracted)

    if sim >= threshold:
        return (True, wpsnr_wa) # Mark found
    else:
        return (False, wpsnr_wa) # Mark not found
