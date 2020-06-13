
import torchvision
from shutil import copy2
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib
import re
import argparse
import logging
matplotlib.use('Agg')
import os
import random
# import copy
import ntpath
#OOD flag
OOD_LABEL = -1
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler, SubsetRandomSampler
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch

def create_parser():
    """
    Parser for the data partitioner
    :return:
    """
    parser = argparse.ArgumentParser(description='Dataset partitioner')
    parser.add_argument('--mode', type=str, default="unlabeled_partitioner",
                        help='Options: 1. train_partitioner 2. unlabeled_partitioner ')
    parser.add_argument('--batch_id_num', default=0, type=int, help='batch id number')

    #Options for the labeled/unlabeled data partitioner
    parser.add_argument('--path_iod', type=str, default="", help='The directory with the IOD  data')
    parser.add_argument('--path_ood', type=str, default="", help='The directory with the OOD data')
    parser.add_argument('--path_dest', type=str, default="", help='The destination directory')
    parser.add_argument('--ood_perc', default=1.0, type=float, help='From 0 to 1')
    parser.add_argument('--num_unlabeled', default=100, type=int, help='Number of unlabeled observations')
    #Options for the train/test/ood partitioner
    parser.add_argument('--path_base', type=str, default="", help='Base directory')
    parser.add_argument('--list_in_dist_classes', type=str, default="", help='The List of in distribution classes')
    parser.add_argument('--eval_perc', default=0.25, type=float, help='From 0 to 1')
    return parser

def parse_commandline_args():
    """
    Create the parser
    :return:
    """
    return create_parser().parse_args()

def create_train_test_folder_partitions_ood_undersampled(datasetpath_base, percentage_evaluation=0.25, random_state=42, batch=0, create_dirs = True, classes_in_dist = []):
    """
    Train and test partitioner
    :param datasetpath_base:
    :param percentage_used_labeled_observations: The percentage of the labeled observations to use from the 1 -  percentage_evaluation
    :param num_batches: total number of batches
    :param create_dirs:
    :param percentage_evaluation: test percentage of data
    :return:
    """
    #for the same batch, same result
    random.seed(random_state + batch)
    datasetpath_test = datasetpath_base + "/batches_labeled_undersampled_in_dist/batch_" + str(batch) + "/test/"
    datasetpath_train = datasetpath_base + "/batches_labeled_undersampled_in_dist/batch_" + str(batch) + "/train/"
    datasetpath_ood = datasetpath_base + "/batches_unlabeled_undersampled_out_dist/batch_" + str(batch) + "/"

    datasetpath_all = datasetpath_base + "/all"
    print("All data path: ", datasetpath_all)
    #read dataset
    dataset = torchvision.datasets.ImageFolder(datasetpath_all)
    #get filenames
    list_file_names_and_labels = dataset.imgs
    labels_temp = dataset.targets
    list_file_names = []
    list_labels = []
    list_files_labels = []
    # list of file names and labels
    for i in range(0, len(list_file_names_and_labels)):
        file_name_path = list_file_names_and_labels[i][0]
        list_file_names += [file_name_path]
        list_labels += [labels_temp[i]]
        #for random swapping
        list_files_labels += [(file_name_path, labels_temp[i])]
    #total number of classes
    num_classes = len(np.unique(np.array(labels_temp)))
    print("Total number of classes detected: ", num_classes)
    #if no custom in distribution classes were chosen, take them all
    if(classes_in_dist == []):
        print("No out of distribution data required")
        for i in range(0, num_classes): classes_in_dist += [i]

    #get the number of observations for the less represented class
    array_labels = np.array(list_labels)
    observations_under_class = len(array_labels[array_labels == 0])
    array_labels = np.array(list_labels)
    #find the under represented class
    for i in range(0, num_classes):
        num_obs_class = len(array_labels[array_labels == i])
        if(num_obs_class < observations_under_class):
            observations_under_class = num_obs_class
    print("Under represented class with the following number of observations ", observations_under_class)


    if (create_dirs):
        # create the directories
        try:
            for i in classes_in_dist:
                print("Creating test path: ", datasetpath_test + "/" + str(i))
                print("Creating training path: ", datasetpath_train + "/" + str(i))
                os.makedirs(datasetpath_test + "/" + str(i))
                os.makedirs(datasetpath_train + "/" + str(i))
        except:
            print("Rewritting directories...")

        try:
            for j in range(0, num_classes):
                if(not j in in_dist_classes_list):
                    print("Creating OOD path: ", datasetpath_ood + "/" + str(j))
                    os.makedirs(datasetpath_ood + "/" + str(j))
        except:
            print("Rewritting directories...")
    # test and train  splitter for unlabeled and labeled data split
    #for the same batch number, same results

    for curr_class in range(0, num_classes):
        #get all the labels of the

        (list_files_class, list_labels_class) = undersample_list(observations_under_class, list_files_labels, curr_class)
        print("Undersampled list of files size ", len(list_files_class), " for class ", curr_class)
        print("Undersampled list of labels size ", len(list_labels_class), " for class ", curr_class)

        X_train, X_test, y_train, y_test = train_test_split(list_files_class, list_labels_class, test_size=percentage_evaluation,
                                                            random_state=random_state + batch)
        print("Creating trainig partitioned folders...", len(X_train))
        for i in range(0, len(X_train)):
            path_src = X_train[i]
            # extract the file name
            file_name = ntpath.basename(path_src)
            label = y_train[i]
            #if the label is among the in distribution selected label, copy it there
            if(label in classes_in_dist):
                #In distribution data
                path_dest = datasetpath_train + str(label) + "/" + file_name
            else:
                #out distribution data
                path_dest = datasetpath_ood + str(label) + "/" + file_name

            copy2(path_src, path_dest)

        print("Creating test partitioned folders...", len(X_test))
        for i in range(0, len(X_test)):
            label = y_test[i]
            # if the label is among the in distribution selected label, copy it there
            if (label in classes_in_dist):
                path_src = X_test[i]
                file_name = ntpath.basename(path_src)
                path_dest = datasetpath_test + str(y_test[i]) + "/" + file_name
                copy2(path_src, path_dest)

def undersample_list(observations_under_class, list_files_labels, curr_class):
    random.shuffle(list_files_labels)
    #load only the labels
    array_labels = np.array([element[1] for element in list_files_labels])


    list_file_indices_class = (array_labels == curr_class)
    list_files_class_selected = []
    list_labels_class_selected = []
    # get the file names of the class
    number_added = 0
    # undersample the folders to the lowest num of observations per class
    for index in range(0, len(list_file_indices_class)):
        if (list_file_indices_class[index] and number_added < observations_under_class):
            list_files_class_selected += [list_files_labels[index][0]]
            list_labels_class_selected += [list_files_labels[index][1]]
            number_added += 1


    print("Number of labels undersampled ", len(list_labels_class_selected))
    print("First file taken: ")
    print(list_files_class_selected[0])
    return (list_files_class_selected, list_labels_class_selected)

def create_train_test_folder_partitions_ood(datasetpath_base, percentage_evaluation=0.25, random_state=42, batch=0, create_dirs = True, classes_in_dist = []):
    """
    Train and test partitioner
    :param datasetpath_base:
    :param percentage_used_labeled_observations: The percentage of the labeled observations to use from the 1 -  percentage_evaluation
    :param num_batches: total number of batches
    :param create_dirs:
    :param percentage_evaluation: test percentage of data
    :return:
    """

    datasetpath_test = datasetpath_base + "/batches_labeled_in_dist/batch_" + str(batch) + "/test/"
    datasetpath_train = datasetpath_base + "/batches_labeled_in_dist/batch_" + str(batch) + "/train/"
    datasetpath_ood = datasetpath_base + "/batches_unlabeled_out_dist/batch_" + str(batch) + "/"

    datasetpath_all = datasetpath_base + "/all/"
    print("All data path: ", datasetpath_all)
    #read dataset
    dataset = torchvision.datasets.ImageFolder(datasetpath_all)
    #get filenames
    list_file_names_and_labels = dataset.imgs
    labels_temp = dataset.targets
    list_file_names = []
    list_labels = []
    # list of file names and labels
    for i in range(0, len(list_file_names_and_labels)):
        file_name_path = list_file_names_and_labels[i][0]
        list_file_names += [file_name_path]
        list_labels += [labels_temp[i]]
    #total number of classes
    num_classes = len(np.unique(np.array(labels_temp)))
    print("Total number of classes detected: ", num_classes)
    #if no custom in distribution classes were chosen, take them all
    if(classes_in_dist == []):
        for i in range(0, num_classes): classes_in_dist += [i]



    if (create_dirs):
        # create the directories


        try:
            for i in classes_in_dist:
                print("Creating test path: ", datasetpath_test + "/" + str(i))
                print("Creating training path: ", datasetpath_train + "/" + str(i))
                os.makedirs(datasetpath_test + "/" + str(i))
                os.makedirs(datasetpath_train + "/" + str(i))
        except:
            print("Rewritting directories...")

        try:
            for j in range(0, num_classes):
                if(not j in in_dist_classes_list):
                    print("Creating OOD path: ", datasetpath_ood + "/" + str(j))
                    os.makedirs(datasetpath_ood + "/" + str(j))
        except:
            print("Rewritting directories...")
    # test and train  splitter for unlabeled and labeled data split
    #for the same batch number, same results
    X_train, X_test, y_train, y_test = train_test_split(list_file_names, list_labels, test_size=percentage_evaluation,
                                                        random_state=random_state + batch)
    print("Creating trainig partitioned folders...", len(X_train))
    for i in range(0, len(X_train)):
        path_src = X_train[i]
        # extract the file name
        file_name = ntpath.basename(path_src)
        label = y_train[i]
        #if the label is among the in distribution selected label, copy it there
        if(label in classes_in_dist):
            #In distribution data
            path_dest = datasetpath_train + str(label) + "/" + file_name
            # print("COPY TO: " + path_dest)
        else:
            #out distribution data
            path_dest = datasetpath_ood + str(label) + "/" + file_name

        copy2(path_src, path_dest)

    print("Creating test partitioned folders...", len(X_test))
    for i in range(0, len(X_test)):
        label = y_test[i]
        # if the label is among the in distribution selected label, copy it there
        if (label in classes_in_dist):
            path_src = X_test[i]
            file_name = ntpath.basename(path_src)
            path_dest = datasetpath_test + str(y_test[i]) + "/" + file_name
            copy2(path_src, path_dest)

def create_train_test_folder_partitions_simple(datasetpath_base, percentage_evaluation=0.25, random_state=42, batch=0,
                                        create_dirs = True):
    """
    Train and test partitioner
    :param datasetpath_base:
    :param percentage_used_labeled_observations: The percentage of the labeled observations to use from the 1 -  percentage_evaluation
    :param num_batches: total number of batches
    :param create_dirs:
    :param percentage_evaluation: test percentage of data
    :return:
    """
    datasetpath_test = datasetpath_base + "/batch_" + str(batch) + "/test/"
    datasetpath_train = datasetpath_base + "/batch_" + str(batch) + "/train/"
    datasetpath_all = datasetpath_base + "/all"
    print("datasetpath_all")
    print(datasetpath_all)
    dataset = torchvision.datasets.ImageFolder(datasetpath_all)
    list_file_names_and_labels = dataset.imgs
    labels_temp = dataset.targets
    list_file_names = []
    list_labels = []
    # list of file names and labels
    for i in range(0, len(list_file_names_and_labels)):
        file_name_path = list_file_names_and_labels[i][0]
        list_file_names += [file_name_path]
        list_labels += [labels_temp[i]]

    if (create_dirs):
        # create the directories
        print("Trying to create dir")
        print(datasetpath_test)
        os.makedirs(datasetpath_test)
        print(datasetpath_test)
        os.makedirs(datasetpath_train)
        for i in range(0, 6):
            os.makedirs(datasetpath_test + "/" + str(i))
            os.makedirs(datasetpath_train + "/" + str(i))

    # test and train  splitter for unlabeled and labeled data split

    X_train, X_test, y_train, y_test = train_test_split(list_file_names, list_labels, test_size=percentage_evaluation,
                                                        random_state=random_state)
    print("Creating trainig partitioned folders...", len(X_train))
    for i in range(0, len(X_train)):
        path_src = X_train[i]
        # extract the file name
        file_name = ntpath.basename(path_src)
        path_dest = datasetpath_train + str(y_train[i]) + "/" + file_name
        copy2(path_src, path_dest)

    print("Creating test partitioned folders...", len(X_test))
    for i in range(0, len(X_test)):
        path_src = X_test[i]
        file_name = ntpath.basename(path_src)
        path_dest = datasetpath_test + str(y_test[i]) + "/" + file_name
        copy2(path_src, path_dest)

def create_folder_partitions_unlabeled_ood(iod_dataset_path, ood_dataset_path, dest_unlabeled_path_base,
                                           total_unlabeled_obs=1000, ood_percentage=0.5, random_state=42, batch=0,
                                           create_dirs=True):
    """
    Create the folder partitions for unlabeled data repository, preserving the folder structure of train data
    This MUST BE EXECUTED AFTER the training batches have been built
    The OOD data is randomly copied among the training subfolders, given the folder structure used in the MixMatch FAST AI implementation
    :param iod_dataset_path:
    :param ood_dataset_path:
    :param dest_unlabeled_path_base: We create the train folder, as the test folder is just copied from IOD folder (test is always In Distribution)
    :param total_unlabeled_obs:
    :param ood_percentage: percentage of out of distribution data
    :param random_state: seed
    :param batch: batch number id for the folder
    :param create_dirs: create the necessary directories
    :return:
    """
    # read the data from the in distribution dataset train batch (the selected observations for unlabeled data will be deleted from there)
    dataset = torchvision.datasets.ImageFolder(iod_dataset_path)
    dataset_ood = torchvision.datasets.ImageFolder(ood_dataset_path)
    # read the data path
    list_file_names_in_dist_data = dataset.imgs
    list_file_names_out_dist_data = dataset_ood.imgs

    in_dist_classes_list_all = os.listdir(iod_dataset_path)
    print("NEW LABELS TEMP ", in_dist_classes_list_all)

    labels_temp_in_dist = dataset.targets
    # init variables
    list_in_dist_data = []
    list_out_dist_data = []
    # total number of iod observations
    number_iod = int((1 - ood_percentage) * total_unlabeled_obs)
    number_ood = int(ood_percentage * total_unlabeled_obs)
    print("Reading and shuffling data...")
    # list of file names and labels of in distribution data
    #in_dist_classes_list_all = list(np.unique(np.array(labels_temp_in_dist)))
    print("Total number of classes detected: ", len(in_dist_classes_list_all))
    print("List of in distribution classes: ", in_dist_classes_list_all)
    #copy file name and labels to list in data
    for i in range(0, len(list_file_names_in_dist_data)):
        file_name_path = list_file_names_in_dist_data[i][0]
        label_index = labels_temp_in_dist[i]
        #we need to use the actual folder name and not the label index reported by pytorch ImageFolder
        list_in_dist_data += [(file_name_path, in_dist_classes_list_all[label_index])]

    # list of files and labeles out distribution data
    for i in range(0, len(list_file_names_out_dist_data)):
        file_name_path = list_file_names_out_dist_data[i][0]
        list_out_dist_data += [(file_name_path, OOD_LABEL)]
    # shuffle the list and select the percentage of ood and iod data
    random.seed(random_state + batch)
    selected_iod_data = random.sample(list_in_dist_data, number_iod)
    selected_ood_data = random.sample(list_out_dist_data, number_ood)
    print("Number of selected iod observations")
    print(len(selected_iod_data))
    print("Number of selected ood observations")
    print(len(selected_ood_data))
    dest_unlabeled_path_batch = dest_unlabeled_path_base + "/batch_" + str(batch) + "_num_unlabeled_" + str(
        total_unlabeled_obs) + "_ood_perc_" + str(int(100 * ood_percentage)) + "/"
    if (create_dirs):
        # create the directories
        try:
            print("Trying to create directories: ")
            print(dest_unlabeled_path_batch)
            os.makedirs(dest_unlabeled_path_batch)
        except:
            print("Could not create dir, already exists")
    # copy the iid observations
    print("Copying IOD data...")
    for file_label in selected_iod_data:
        path_src = file_label[0]
        label = file_label[1]
        final_dest = dest_unlabeled_path_batch + "/train/" + str(label) + "/"
        try:
            os.makedirs(final_dest)
        except:
            a = 0
        copy2(path_src, final_dest)

    # copy the ood observations
    print("Copying OOD data randomly in the training folders...")
    for file_label in selected_ood_data:
        path_src = file_label[0]
        random_label = random.sample(in_dist_classes_list_all, 1)[0]
        file_name = os.path.basename(path_src)
        _, file_extension = os.path.splitext(path_src)
        try:
            os.makedirs(dest_unlabeled_path_batch + "/train/" + str(random_label))
        except:
            a = 0
        final_dest = dest_unlabeled_path_batch + "/train/" + str(random_label) + "/ood_" + file_name + file_extension
        copy2(path_src, final_dest)
    print("Copying the test folder to the unlabeled data destination... ")
    iod_test_path = iod_dataset_path.replace("/train","") + "/test/"
    print("From: ", iod_test_path)
    print("To: ", dest_unlabeled_path_batch + "/test/")
    shutil.copytree(iod_test_path, dest_unlabeled_path_batch + "/test/")
    print("A total of ", len(selected_ood_data), " OOD observations were randomly added to the IOD train subfolders!")
    return (number_iod, number_ood)


def unit_test_data_partitioner():
    """
    Tester of the partitioner
    :return:
    """
    DEFAULT_PATH_MNIST_1_5 = "/media/Data/user/Datasets/MNIST/OOD_datasets/Out_Distribution_Dataset/all"
    DEFAULT_PATH_MNIST_0_4 = "/media/Data/user/Datasets/MNIST/OOD_datasets/In_Distribution_Datasets/In_Distribution_Dataset_1/batch_0"
    ood_dataset_path = DEFAULT_PATH_MNIST_1_5
    in_dist_dataset_path = DEFAULT_PATH_MNIST_0_4 + "/train/"
    dest_unlabeled_path_base = "/media/Data/user/Datasets/MNIST/OOD_datasets/Out_Distribution_Dataset/unlabeled"
    create_folder_partitions_unlabeled_ood(in_dist_dataset_path, ood_dataset_path, dest_unlabeled_path_base, ood_percentage=1, total_unlabeled_obs=10000)

def unit_test_partitioner_training_test():
    """
    Test for training and test partitioner
    :return:
    """
    DEFAULT_PATH_MNIST_0_4 = "/media/Data/user/Datasets/MNIST/OOD_datasets/In_Distribution_Datasets/In_Distribution_Dataset_1/batch_0"

    random_state_base = 42
    datasetpath_base = DEFAULT_PATH_MNIST_0_4
    for i in range(0, 2):
        random_state_base += 1
        create_train_test_folder_partitions(datasetpath_base, percentage_evaluation=0.25,
                                            random_state=random_state_base, batch=i)





def get_mean_and_std(dataset):
    """
    Compute the mean and std value of dataset.
    :param dataset:
    :return:
    """
    data_loader = torch.utils.data.DataLoader(dataset,  num_workers= 5, pin_memory=True, batch_size =1)

    #init the mean and std
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    k = 1
    for inputs, targets in data_loader:
        #mean and std from the image
        #print("Processing image: ", k)
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
        k += 1

    #normalize
    mean.div_(len(dataset))
    std.div_(len(dataset))
    print("mean: " + str(mean))
    print("std: " + str(std))
    return mean, std


if __name__ == '__main__':
    global args, is_colab
    is_colab = False
    args = parse_commandline_args()

    #use the arguments from cli
    #Labeled/unlabeled data partitioner
    ood_dataset_path = args.path_ood
    in_dist_dataset_path = args.path_iod
    dest_unlabeled_path_base = args.path_dest
    if(args.mode.strip() == "unlabeled_partitioner"):
        create_folder_partitions_unlabeled_ood(in_dist_dataset_path, ood_dataset_path, dest_unlabeled_path_base, ood_percentage=args.ood_perc, total_unlabeled_obs=args.num_unlabeled, batch = args.batch_id_num)
    elif(args.mode.strip() == "train_partitioner_balanced"):
        print("Train partitioner balanced")
        create_train_test_folder_partitions_ood_undersampled(args.path_base, percentage_evaluation=args.eval_perc, random_state=42 + args.batch_id_num, batch=args.batch_id_num,  create_dirs=True)
    #Train/Test data partitioner
    elif(args.mode.strip() == "train_partitioner"):
        in_dist_classes_str = args.list_in_dist_classes
        #assumes a string with the format '0, 0, 0, 11, 0, 0, 0, 0, 0, 19, 0, 9, 0, 0, 0, 0, 0, 0, 11'
        if(in_dist_classes_str != ""):
            in_dist_classes_list = [int(s) for s in in_dist_classes_str.split(',')]
        else:
            in_dist_classes_list = []

        print("List in distribution classes ", in_dist_classes_list)
        create_train_test_folder_partitions_ood(args.path_base, percentage_evaluation=args.eval_perc, random_state=42, batch=args.batch_id_num,  create_dirs=True, classes_in_dist=in_dist_classes_list)
