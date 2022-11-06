import numpy as np
from copy import deepcopy as copy_deepcopy
from scipy.fft import dct, idct


def embedding(original_image, mark_array):
    # Settings
    chunk_size = (8, 8)
    spots_to_swap = [[(2, 6), (3, 5)], [(5, 3), (6, 2)], [(0, 7), (7, 0)]]
    watermarked_image = copy_deepcopy(np.float32(original_image))
    optimal_average = 64
    dct_coefficients_difference_limit = 4
    under_limit_boost = 130
    over_limit_boost = 2
    mark_partitioning = [325, 650]
    
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
    if len(chunks_sorted) - index_average < len(mark_array):
        index_average -= len(mark_array)
        if index_average < 0:
            index_average = 0

    # Walk the image chunk by chunk
    portion_index = 0
    for i in range(len(mark_array)):
        # Change the spots to swap
        if i >= mark_partitioning[0] and i < mark_partitioning[1]:
            portion_index = 1
        elif i >= mark_partitioning[1]:
            portion_index = 2
        # Apply the DCT to the chunk
        chunk = watermarked_image[chunks_sorted[i + index_average]['coordinates'][0] : chunks_sorted[i + index_average]['coordinates'][1], chunks_sorted[i + index_average]['coordinates'][2] : chunks_sorted[i + index_average]['coordinates'][3]]
        chunk_dct = dct(dct(chunk, axis = 0, norm = 'ortho'), axis = 1, norm = 'ortho')
        # If we want to embed 1 swap the values
        if mark_array[i] != 0:
            chunk_dct[spots_to_swap[portion_index][0]], chunk_dct[spots_to_swap[portion_index][1]] = chunk_dct[spots_to_swap[portion_index][1]], chunk_dct[spots_to_swap[portion_index][0]]
        # Boost the max value
        if abs(chunk_dct[spots_to_swap[portion_index][0]] - chunk_dct[spots_to_swap[portion_index][1]]) < dct_coefficients_difference_limit:
            if chunk_dct[spots_to_swap[portion_index][0]] > chunk_dct[spots_to_swap[portion_index][1]]:
                chunk_dct[spots_to_swap[portion_index][0]] += under_limit_boost
            else:
                chunk_dct[spots_to_swap[portion_index][1]] += under_limit_boost
        else:
            if chunk_dct[spots_to_swap[portion_index][0]] > chunk_dct[spots_to_swap[portion_index][1]]:
                chunk_dct[spots_to_swap[portion_index][0]] += (over_limit_boost * abs(chunk_dct[spots_to_swap[portion_index][0]] - chunk_dct[spots_to_swap[portion_index][1]])) 
            else:
                chunk_dct[spots_to_swap[portion_index][1]] += (over_limit_boost * abs(chunk_dct[spots_to_swap[portion_index][1]] - chunk_dct[spots_to_swap[portion_index][0]]))
        # Apply the IDCT to the chunk
        chunk_idct = np.rint(idct(idct(chunk_dct, axis = 1, norm = 'ortho'), axis = 0, norm = 'ortho'))
        if np.max(chunk_idct) > 255 or np.min(chunk_idct) < 0:
            chunk_idct =  np.clip(chunk_idct, 0, 255)
        watermarked_image[chunks_sorted[i + index_average]['coordinates'][0] : chunks_sorted[i + index_average]['coordinates'][1], chunks_sorted[i + index_average]['coordinates'][2] : chunks_sorted[i + index_average]['coordinates'][3]] = chunk_idct
    
    # Return the watermarked image
    return watermarked_image


