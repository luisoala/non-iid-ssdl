import torch
import torchvision.models as models
from fastai.vision import *
import pingouin as pg
from fastai.callbacks import CSVLogger
from numbers import Integral
import torch
import logging
import sys
from torchvision.utils import save_image
import numpy as np
import pandas as pd
import scipy
from PIL import Image
import torchvision.models.vgg as models2
import torchvision.models as models3
from scipy.stats import entropy
from scipy.spatial import distance

import torchvision.transforms as transforms
import torchvision
from scipy.stats import mannwhitneyu



def pytorch_feature_extractor():
    input = torch.rand(1, 3, 50, 50)
    vgg16 = models3.resnet152(pretrained=True)
    print(vgg16)
    output = vgg16[:-1](input)
    print(output)


def calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=100, p=2, num_batches=10):
    run_results = []
    for i in range(0, num_batches):
        dist_i = calculate_Minowski_feature_space(databunch1, databunch2, model, batch_size, p)
        run_results += [dist_i]
    run_results_np = np.array(run_results)
    mean_results = run_results_np.mean()
    std_results = run_results_np.std()
    return (mean_results, std_results)


def calculate_pdf_dist_stats(databunch1, databunch2, model, batch_size=80,
                                        distance_func=distance.jensenshannon, num_batches=10):
    """

    :param databunch1:
    :param databunch2:
    :param model:
    :param batch_size:
    :param distance_func:
    :param num_batches:
    :return:
    """
    run_results = []
    for i in range(0, num_batches):
        dist_i = calculate_pdf_dist(databunch1, databunch2, model, batch_size, distance_func)
        run_results += [dist_i]
    run_results_np = np.array(run_results)
    mean_results = run_results_np.mean()
    std_results = run_results_np.std()
    return (mean_results, std_results)



def calculate_pdf_dist(databunch1, databunch2, model, batch_size=80, distance_func=distance.jensenshannon):
    # just get the number of dimensions

    tensorbunch1 = databunch_to_tensor(databunch1)
    tensorbunch2 = databunch_to_tensor(databunch2)

    feature_extractor = get_feature_extractor(model)
    batch_tensors1 = tensorbunch1[0:batch_size, :, :, :]
    # get number of features
    features_bunch1 = feature_extractor(batch_tensors1)
    num_features = features_bunch1.shape[1]
    print("Calculating pdf distance for for feature space of dimensions: ", num_features)
    js_dist_dims = []
    # calculate distance of histograms for given
    for i in range(0, num_features):
        js_dist_dims += [
            calculate_distance_hists(tensorbunch1, tensorbunch2, feature_extractor, dimension=i, batch_size=batch_size,
                                     distance_func=distance_func)]
    js_dist_sum = sum(js_dist_dims)
    return js_dist_sum


def calculate_distance_hists(tensorbunch1, tensorbunch2, feature_extractor, dimension, batch_size=20,
                             distance_func=distance.jensenshannon):
    # random pick of batch observations
    total_number_obs_1 = tensorbunch1.shape[0]
    total_number_obs_2 = tensorbunch2.shape[0]
    batch_indices_1 = generate_rand_bin_array(batch_size, total_number_obs_1)
    batch_indices_2 = generate_rand_bin_array(batch_size, total_number_obs_2)
    # create the batch of tensors to get its features
    batch_tensors1 = tensorbunch1[batch_indices_1, :, :, :]
    batch_tensors2 = tensorbunch2[batch_indices_2, :, :, :]

    # get the  features from the selected batch
    features_bunch1 = feature_extractor(batch_tensors1)
    features_bunch2 = feature_extractor(batch_tensors2)
    # get the values of a specific dimension
    values_dimension_bunch1 = features_bunch1[:, dimension].cpu().detach().numpy()
    values_dimension_bunch2 = features_bunch2[:, dimension].cpu().detach().numpy()
    # calculate the histograms
    (hist1, bucks1) = np.histogram(values_dimension_bunch1, bins=15, range=None, normed=None, weights=None,
                                   density=None)
    #ensure that the histograms have the same meaning, by using the same buckets
    (hist2, bucks2) = np.histogram(values_dimension_bunch2, bins=bucks1, range=None, normed=None, weights=None,
                                   density=None)
    # normalize the histograms
    hist1 = np.array(hist1) / sum(hist1)
    hist2 = np.array(hist2) / sum(hist2)
    js_dist = distance_func(hist1.tolist(), hist2.tolist())
    return js_dist


# databunch1 is the smallest
def calculate_Minowski_feature_space(databunch1, databunch2, model, batch_size = 100, p = 2):
    print("Calculating Minowski distance of two samples of two datasets, p: ", p)
    feature_extractor = get_feature_extractor(model)
    tensorbunch1 = databunch_to_tensor(databunch1)
    tensorbunch2 = databunch_to_tensor(databunch2)
    # get the randomized batch indices
    total_number_obs_1 = tensorbunch1.shape[0]
    total_number_obs_2 = tensorbunch2.shape[0]
    batch_indices_1 = generate_rand_bin_array(batch_size, total_number_obs_1)
    batch_indices_2 = generate_rand_bin_array(batch_size, total_number_obs_2)

    # total number of observations of the smallest databunch
    total_observations_bunch1 = tensorbunch1.shape[0]
    # pick random observations for the batch
    batch_tensors1 = tensorbunch1[batch_indices_1, :, :, :].to(device="cuda:0")
    batch_tensors2 = tensorbunch2[batch_indices_2, :, :, :].to(device="cuda:0")
    # extract its features
    features_bunch1 = feature_extractor(batch_tensors1)
    features_bunch2 = feature_extractor(batch_tensors2)

    sum_mses = []
    # one to all distance accumulation
    for i in range(0, batch_size):
        mse_i = calculate_Minowski_observation_min(features_bunch1[i], features_bunch2, p)
        sum_mses += [mse_i.item()]
    sum_mses_np = np.array(sum_mses)
    # delete features to prevent gpu memory overflow
    del features_bunch1
    del features_bunch2
    torch.cuda.empty_cache()
    mse_mean_all_batch = sum_mses_np.mean()
    # take one batch
    return mse_mean_all_batch


def calculate_Minowski_observation(observation, tensorbunch, p=2):
    # vectorize all the images in tensorbunch
    # if it receives an image
    # tensorbunch_vec = img[:].view(-1, tensorbunch.shape[1]*tensorbunch.shape[2]*tensorbunch.shape[3])
    observation_vec = observation.view(-1)
    difference_bunch = tensorbunch - observation_vec
    # for all observations in the bunch, calculate its euclidian proximity
    # L2 Norm over columns
    minowski_distances = torch.norm(difference_bunch, p, 1)
    # choose mse or min?
    minowski_distance = minowski_distances.sum() / len(minowski_distances)
    return minowski_distance


def calculate_Minowski_observation_min(observation, tensorbunch, p = 2):
    # vectorize all the images in tensorbunch
    # if it receives an image
    # tensorbunch_vec = img[:].view(-1, tensorbunch.shape[1]*tensorbunch.shape[2]*tensorbunch.shape[3])
    observation_vec = observation.view(-1)
    # difference between bunch of tensors and the current observation to analyze
    difference_bunch = tensorbunch - observation_vec
    # for all observations in the bunch, calculate its euclidian proximity
    # L2 Norm over columns
    minowski_distances = torch.norm(difference_bunch, p, 1)
    # choose mse or min
    min_dist = minowski_distances.min()
    return min_dist


def databunch_to_tensor(databunch1):
    # tensor of tensor
    tensor_bunch = torch.zeros(len(databunch1.train_ds), databunch1.train_ds[0][0].shape[0],
                               databunch1.train_ds[0][0].shape[1], databunch1.train_ds[0][0].shape[2], device="cuda:0")
    for i in range(0, len(databunch1.train_ds)):
        tensor_bunch[i, :, :, :] = databunch1.train_ds[i][0].data.to(device="cuda:0")

    return tensor_bunch


def get_feature_extractor(model):
    global key

    path = untar_data(URLs.MNIST_SAMPLE)
    data = ImageDataBunch.from_folder(path)
    # save learner to reload it as a pytorch model
    learner = Learner(data, model, metrics=[accuracy])
    learner.export('/media/Data/user/Code_Projects/OOD4SSDL/utilities/model/final_model_' + key + ".pk")
    torch_dict = torch.load('/media/Data/user/Code_Projects/OOD4SSDL/utilities/model/final_model_' + key + ".pk")
    # get the model
    model_loaded = torch_dict["model"]
    # put it on gpu!
    model_loaded = model_loaded.to(device="cuda:0")
    # usually the last set of layers act as classifier, therefore we discard it
    feature_extractor = model_loaded.features[:-1]
    return feature_extractor


def dataset_distance_tester_pdf_half(path_bunch1 = "/media/Data/user/Datasets/CIFAR10_HALF/CIFAR10_60_50/", path_bunch2 = "/media/Data/user/Datasets/CIFAR10_HALF/CIFAR10_60_50/",ood_perc = 50, num_unlabeled = 3000, name_ood_dataset = "half", num_batches=1, size_image = 28, distance_func = distance.jensenshannon):
  """
  Testing
  :return:
  """
  global key
  key = "pdf"
  print("Computing distance for dataset: ", name_ood_dataset)
  model = models.WideResNet(num_groups=3, N=4, num_classes=10, k=2, start_nf=64)
  dists_reference = []
  dists_bunch1_bunch2 = []
  dists_substracted = []
  for i in range(0, num_batches):
      path_mnist_half_in_dist = path_bunch1 + "/batch" + str(i) + "/artifacts/inputs/labelled/"
      print("INDIST: ", path_mnist_half_in_dist)
      path_mnist_half_out_dist = path_bunch2 + "/batch" + str(i) + "/artifacts/inputs/unlabelled/"
      print("outDIST: ", path_mnist_half_out_dist)
      databunch1 = (ImageList.from_folder(path_mnist_half_in_dist)
                    .split_none()
                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())
      databunch2 = (ImageList.from_folder(path_mnist_half_out_dist)
                    .split_none()
                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())


      (dist_ref_i, std_ref) = calculate_pdf_dist_stats(databunch1, databunch1, model, batch_size=50,
                                        distance_func=distance_func, num_batches=3)
      dists_reference += [dist_ref_i]
      print("Distance to itself    (reference): ", dist_ref_i, " for batch: ", i)
      (dist_between_bunches_i, dist_between_bunches_std) = calculate_pdf_dist_stats(databunch1, databunch2, model, batch_size=50,
                                                            distance_func=distance_func, num_batches=3)
      dists_bunch1_bunch2 += [dist_between_bunches_i]
      print("Distance between bunches:  ", dist_between_bunches_i, " for batch:", i)
      dists_substracted += [abs(dist_between_bunches_i - dist_ref_i)]


  dist_between_bunches = np.mean(dists_substracted)
  print("Distance  between bunches: ", dist_between_bunches)
  stat, p_value = scipy.stats.wilcoxon(dists_reference, dists_bunch1_bunch2, correction=True)
  # means
  dists_reference += [np.array(dists_reference).mean()]
  dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).mean()]
  dists_substracted += [np.array(dists_substracted).mean()]
  # stds are the last row
  dists_reference += [np.array(dists_reference).std()]
  dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).std()]
  dists_substracted += [np.array(dists_substracted).std()]

  header3 = 'Distance_substracted with p ' + str(p_value)

  dict_csv = {'Reference': dists_reference,
              'Distance': dists_bunch1_bunch2,
              header3: dists_substracted

              }
  dataframe = pd.DataFrame(dict_csv, columns=['Reference', 'Distance', header3])
  dataframe.to_csv(
      '/media/Data/user/Code_Projects/OOD4SSDL/utilities/csv_distances_reports/' + name_ood_dataset + "ood_perc_" + str(
          ood_perc) + '.csv', index=False, header=True)

  return dist_between_bunches


def dataset_distance_tester_pdf(path_bunch1 = "/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_", path_bunch2 = "/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled/batch_",ood_perc = 100, num_unlabeled = 3000, name_ood_dataset = "in_dist", num_batches=10, size_image = 28, distance_func = distance.jensenshannon):
  """
  Testing
  :return:
  """

  global key
  key = "pdf"
  print("Computing distance for dataset: ", name_ood_dataset)
  model = models.WideResNet(num_groups=3, N=4, num_classes=10, k=2, start_nf=64)
  dists_reference = []
  dists_bunch1_bunch2 = []
  dists_substracted = []
  for i in range(0, num_batches):
      path_mnist_half_in_dist = path_bunch1 + "/batch_" + str(i)
      path_mnist_half_out_dist = path_bunch2 + str(i) + "/batch_" + str(i) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc)
      print("IN DIST PATH ", path_mnist_half_in_dist)
      print("OUT DIST PATH ", path_mnist_half_out_dist)



      databunch1 = (ImageList.from_folder(path_mnist_half_in_dist)
                    .split_none()
                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())
      databunch2 = (ImageList.from_folder(path_mnist_half_out_dist)
                    .split_none()

                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())


      (dist_ref_i, std_ref) = calculate_pdf_dist_stats(databunch1, databunch1, model, batch_size=80,
                                        distance_func=distance_func, num_batches=3)
      dists_reference += [dist_ref_i]
      print("Distance to itself    (reference): ", dist_ref_i, " for batch: ", i)
      (dist_between_bunches_i, dist_between_bunches_std) = calculate_pdf_dist_stats(databunch1, databunch2, model, batch_size=80,
                                                            distance_func=distance_func, num_batches=3)
      dists_bunch1_bunch2 += [dist_between_bunches_i]
      print("Distance between bunches:  ", dist_between_bunches_i, " for batch:", i)
      dists_substracted += [abs(dist_between_bunches_i - dist_ref_i)]




  dist_between_bunches = np.mean(dists_substracted)
  print("Distance  between bunches: ", dist_between_bunches)
  stat, p_value = scipy.stats.wilcoxon(dists_reference, dists_bunch1_bunch2, correction = True)
  #means
  dists_reference += [np.array(dists_reference).mean()]
  dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).mean()]
  dists_substracted += [np.array(dists_substracted).mean()]
  # stds are the last row
  dists_reference += [np.array(dists_reference).std()]
  dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).std()]
  dists_substracted += [np.array(dists_substracted).std()]

  header3 = 'Distance_substracted with p ' + str(p_value)

  dict_csv = {'Reference': dists_reference,
              'Distance': dists_bunch1_bunch2,
              header3: dists_substracted

              }
  dataframe = pd.DataFrame(dict_csv, columns=['Reference', 'Distance', header3])
  dataframe.to_csv('/media/Data/user/Code_Projects/OOD4SSDL/utilities/csv_distances_reports/' + name_ood_dataset + "ood_perc_" + str(ood_perc) + '.csv', index=False, header=True)

  return dist_between_bunches



  #calculate distance

  dist2 = calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=80, p=2, num_batches=3)
  print("Distance MNIST in dist to MNIST out dist : ", dist2)
  reference2 = calculate_Minowski_feature_space_stats(databunch1, databunch1, model, batch_size=80, p=2, num_batches=3)
  print("Distance MNIST in dist to MNIST out dist (second): ", reference2)



def dataset_distance_tester(path_bunch1 = "/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_", path_bunch2 = "/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled/batch_",ood_perc = 100, num_unlabeled = 3000, name_ood_dataset = "in_dist", num_batches=10, size_image = 28, p = 2):
  """
  Testing
  :return:
  """
  global key
  key = "minkowski"


  print("Computing distance for dataset: ", name_ood_dataset, " p: ", p, " ood: ", ood_perc)
  model = models.WideResNet(num_groups=3, N=4, num_classes=10, k=2, start_nf=64)
  dists_reference = []
  dists_bunch1_bunch2 = []
  dists_substracted = []
  for i in range(0, num_batches):
      path_mnist_half_in_dist = path_bunch1 + str(i)
      path_mnist_half_out_dist = path_bunch2 + str(i) + "/batch_" + str(i) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc)

      databunch1 = (ImageList.from_folder(path_mnist_half_in_dist)
                    .split_none()
                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())
      databunch2 = (ImageList.from_folder(path_mnist_half_out_dist)
                    .split_none()

                    .label_from_folder()
                    .transform(size=size_image)
                    .databunch())

      #databunch1 = ImageDataBunch.from_folder(path_mnist_half_in_dist, ignore_empty=True)
      #databunch2 = ImageDataBunch.from_folder(path_mnist_half_out_dist, ignore_empty=True)
      (dist_ref_i, std_ref) = calculate_Minowski_feature_space_stats(databunch1, databunch1, model, batch_size=80, p=p, num_batches=3)
      dists_reference += [dist_ref_i]
      print("Distance to itself    (reference): ", dist_ref_i, " for batch: ", i)
      (dist_between_bunches_i, dist_between_bunches_std) = calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=80, p=p, num_batches=3)
      dists_bunch1_bunch2 += [dist_between_bunches_i]
      print("Distance between bunches:  ", dist_between_bunches_i, " for batch:", i)
      dists_substracted += [abs(dist_between_bunches_i - dist_ref_i)]




  dist_between_bunches = np.mean(dists_substracted)
  print("Distance  between bunches: ", dist_between_bunches)
  stat, p_value = scipy.stats.wilcoxon(dists_reference, dists_bunch1_bunch2, correction = True)
  #means
  dists_reference += [np.array(dists_reference).mean()]
  dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).mean()]
  dists_substracted += [np.array(dists_substracted).mean()]
  # stds are the last row
  dists_reference += [np.array(dists_reference).std()]
  dists_bunch1_bunch2 += [np.array(dists_bunch1_bunch2).std()]
  dists_substracted += [np.array(dists_substracted).std()]

  header3 = 'Distance_substracted with p ' + str(p_value)

  dict_csv = {'Reference': dists_reference,
              'Distance': dists_bunch1_bunch2,
              header3: dists_substracted

              }
  dataframe = pd.DataFrame(dict_csv, columns=['Reference', 'Distance', header3])
  dataframe.to_csv('/media/Data/user/Code_Projects/OOD4SSDL/utilities/csv_distances_reports/' + name_ood_dataset + "_p_" + str(p)+ "_OOD_perc_" + str(ood_perc) +'.csv', index=False, header=True)

  return dist_between_bunches



  #calculate distance

  dist2 = calculate_Minowski_feature_space_stats(databunch1, databunch2, model, batch_size=80, p=2, num_batches=3)
  print("Distance MNIST in dist to MNIST out dist : ", dist2)
  reference2 = calculate_Minowski_feature_space_stats(databunch1, databunch1, model, batch_size=80, p=2, num_batches=3)
  print("Distance MNIST in dist to MNIST out dist (second): ", reference2)


def run_tests_minowski(p = 1, ood_perc = 50):
    print("Calculating distance for  IMAGENET TINY dataset")
    dataset_distance_tester(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_IMAGENET/batch_",
        ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="Imagenet" + str(p), num_batches=10, p=p)

    print("Calculating distance for  Gaussian dataset")
    dataset_distance_tester(
    path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
    path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_Gaussian/batch_",
    ood_perc=ood_perc,
    num_unlabeled=3000, name_ood_dataset="Gaussian", num_batches=10, p=p)
    #salt and pepper
    print("Calculating  distance for Salt and pepper dataset")
    dataset_distance_tester(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_Gaussian/batch_",
        ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="SaltAndPepper", num_batches=10, p = p)

    print("Calculating distance for SVHN dataset")
    dataset_distance_tester(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_SVHN/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="SVHN" + str(p), num_batches=10, p=p)







def run_tests_pdf(distance_str = "js", ood_perc = 50):
    """

    :param distance: distance_str
    :return:
    """
    print("DISTANCE STR: ", distance_str)
    if(distance_str == "js"):
        distance_func = distance.jensenshannon
    elif(distance_str == "cosine"):
        distance_func = distance.cosine


    print("Calculating " + distance_str  + " distance for Gaussian dataset")
    dataset_distance_tester_pdf(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_Gaussian/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="Gaussian_pdf_" + distance_str, num_batches=10, distance_func = distance_func)
    print("Calculating  " + distance_str + " distance for Salt and pepper dataset")
    dataset_distance_tester_pdf(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_SaltAndPepper/batch_",
        ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="SaltAndPepper_pdf_" + distance_str, num_batches=10,
        distance_func=distance_func)
    print("Calculating " + distance_str + " distance for SVHN dataset")


    dataset_distance_tester_pdf(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_SVHN/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="SVHN__pdf_" + distance_str, num_batches=10, distance_func=distance_func)

    print("Calculating " + distance_str + " distance for IMAGENET dataset")
    dataset_distance_tester_pdf(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_IMAGENET/batch_",
        ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="IMAGENET_pdf_" + distance_str, num_batches=10,
        distance_func=distance_func)

    print("Calculating " + distance_str + " distance for HALF dataset")
    dataset_distance_tester_pdf(
        path_bunch1="/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_",
        path_bunch2="/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_out_dist/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="half_pdf_" + distance_str, num_batches=10, distance_func = distance_func)











def generate_rand_bin_array(num_ones, length_array):
    arr = np.zeros(length_array)
    arr[:num_ones] = 1
    np.random.shuffle(arr)
    bool_array = torch.tensor(array(arr.tolist(), dtype=bool))
    return bool_array


#run_tests_minowski(p = 1, ood_perc = 50)
