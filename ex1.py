# Noa Gol
# 208469486

import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from math import sqrt
from init_centroids import init_centroids

from scipy.misc import imread


# the function get 2 pixels and compute the distance between two pixels
def distance(p1, p2):
    # compute the distance between 2 pixels (points)
    dist = sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)
    return dist


# the function get list of centroid and pixel,
# and return the closest centroid
def find_closest_cent(cent, pixel):
    min_centroid = 100
    min_index = 0
    # goes all the centroids
    for i in range(len(cent)):
        # find the distance from current centroid to pixel
        current_centroid = distance(cent[i], pixel)
        # save the closest centroid and his index
        if current_centroid < min_centroid:
            min_centroid = current_centroid
            min_index = i
    return min_index, min_centroid


# the function get current centroid and pixels,
# compute the next centroids by computing average distance
def find_next_cent(cent, pixels):
    sum_centroid = {}
    count_centroid = {}
    # save the len of centroid
    len_center = len(cent)
    distance_sum = 0

    # create a a new dictionary in size of the centroids
    for i in range(len_center):
        # initialize dict of sum and number of new centroid
        sum_centroid[i] = [0, 0, 0]
        count_centroid[i] = 0

    # goes all the pixels
    for i in range(len(pixels)):
        # find the closest centroid to each pixel
        closest_cent, dist = find_closest_cent(cent, pixels[i])
        # add the distance to sum to compute loss
        distance_sum += dist

        # save the sum of each cluster
        sum_centroid[closest_cent][0] += pixels[i][0]
        sum_centroid[closest_cent][1] += pixels[i][1]
        sum_centroid[closest_cent][2] += pixels[i][2]
        # save number of pixels in each cluster
        count_centroid[closest_cent] += 1

    # goes all the centroids
    for i in range(len_center):
        # update the new centroids
        if count_centroid[i] != 0:
            cent[i][0] = sum_centroid[i][0] / count_centroid[i]
            cent[i][1] = sum_centroid[i][1] / count_centroid[i]
            cent[i][2] = sum_centroid[i][2] / count_centroid[i]
    # compute the loss
    loss = distance_sum / len(pixels)

    return cent, loss


# the function get all pixels and change them according to the centroids
def edit_image(X, centroid):
    # goes all the pixel and update the image
    for i in range(len(X)):
        new_pixel, dist = find_closest_cent(centroid, X[i])
        X[i] = centroid[new_pixel]


# function to print centroid after each centroid update
def print_cent(cent):
    if type(cent) == list:
        cent = np.asarray(cent)
    if len(cent.shape) == 1:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', ')
    else:
        return ' '.join(str(np.floor(100 * cent) / 100).split()).replace('[ ', '[').replace('\n', ' ').replace(' ]',
                                                                                                               ']').replace(
            ' ', ', ')[1:-1]


# the function get the pixels and k, and run the k means algorithm
def run_k_means(X, k):
    print("k=" + str(k) + ":")
    # initialize centroids
    centroid = init_centroids(X, k)
    print("iter 0: %s" % (print_cent(centroid)))
    # 10 iteration for update centroids
    for i in range(10):
        # find the next centroids
        centroid, loss = find_next_cent(centroid, X)
        # print number of iteration
        print("iter %d: %s" % (i + 1, print_cent(centroid)))


def main():
    # data preperation (loading, normalizing, reshaping)
    path = 'dog.jpeg'
    A = imread(path)
    A_norm = A.astype(float) / 255.
    img_size = A_norm.shape
    X = A_norm.reshape(img_size[0] * img_size[1], img_size[2])

    # run the k means algorithm 4 times
    run_k_means(X, 2)
    run_k_means(X, 4)
    run_k_means(X, 8)
    run_k_means(X, 16)


if __name__ == "__main__":
    main()

# plot the image
# plt.imshow(A_norm)
# plt.grid(False)
# plt.show()
