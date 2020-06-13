import numpy as np
import cv2
import random


def create_gaussian_noise_images(path="", number_images = 20000, is_rgb = False):
    """
    Create gaussian noise images
    :param path:
    :param number_images:
    :param is_rgb:
    :return:
    """
    dimensions = 224
    mean = 0
    var = 10
    sigma = var ** 0.5

    image_size = (dimensions, dimensions)
    print("Writing images to: ", path )
    print("Number of images: ", number_images)
    if(is_rgb):
        image_size = (dimensions, dimensions, 3)

    for i in range(0, number_images):
        noisy_image = np.random.normal(mean, sigma, image_size)
        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        file_path_name = path + "/gaussian_" + str(i) + ".png"
        cv2.imwrite(file_path_name, noisy_image)
    print("Images written!")

def sp_noise(image, prob = 0.05):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = np.zeros(image.shape, np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    return output

def create_salt_pepper_noise_images(path="", number_images = 20000, is_rgb = False):
    """
    Create salt and pepper noise images
    :param path:
    :param number_images:
    :param is_rgb:
    :return:
    """
    dimensions = 224

    image_size = (dimensions, dimensions)
    print("Writing images with S&P noise to: ", path )
    print("Number of images: ", number_images)
    if(is_rgb):
        image_size = (dimensions, dimensions, 3)

    for i in range(0, number_images):
        black_image = np.zeros(image_size)
        noisy_image = sp_noise(black_image)
        cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
        noisy_image = noisy_image.astype(np.uint8)
        file_path_name = path + "/gaussian_" + str(i) + ".png"
        cv2.imwrite(file_path_name, noisy_image)
    print("Images written!")


#create_gaussian_noise_images(path="/media/Data/user/Datasets/GaussianNoiseImages/", number_images=20000)
create_salt_pepper_noise_images(path="/media/Data/user/Datasets/SaltAndPepper/", number_images=20000)
