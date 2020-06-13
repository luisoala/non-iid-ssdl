
from shutil import copy2
import matplotlib
import re
import argparse
import logging
import random
import cli

from dataset_partitioner import create_parser


def create_training_test_datasets_shell_script(script_name_path, num_batches, args_parser, name_program_py):
    """
    Generic creator of traing/test/ood partitioner script
    :param dataset_all_path:
    :param dataset_name:
    :param script_name:
    :param num_batches:
    :param args:
    :return:
    """
    file = open(script_name_path, 'w')
    #create a line for excecuting the program with a different batch
    for i in range(0, num_batches):
        #randomly take 5 of the 10 MNIST classes for in dist data (used to train the model)
        list_in_dist_classes = random.sample([0,1,2,3,4,5,6,7,8,9], 5)
        list_in_dist_classes_str = str(list_in_dist_classes).replace("[", "").replace("]", "").replace(",", "$")
        print(list_in_dist_classes_str)
        args_parser.list_in_dist_classes = list_in_dist_classes_str
        #the new batch id
        args_parser.batch_id_num = i
        print("args_parser.batch_id_num  ", args_parser.batch_id_num )
        print(args_parser)
        args_str_no_pp = str(args_parser)
        args_str = create_args_string(args_str_no_pp, name_program_py)
        if(i < num_batches - 1):
            file.write(args_str + "\n")
        else:
            file.write(args_str)
    file.close()

def create_mixmatch_fully_supervised_shell_script(script_name_path, num_batches, args_parser, name_program_py):
    """
    Generic creator of traing/test/ood partitioner script
    :param dataset_all_path:
    :param dataset_name:
    :param script_name:
    :param num_batches:
    :param args:
    :return:
    """
    file = open(script_name_path, 'w')
    name_program_py += " "
    #create a line for excecuting the program with a different batch
    for i in range(0, num_batches):
        #the new batch id
        temp_str = args_parser.path_labeled
        #avoid multiple number concats
        args_parser.path_labeled += str(i)
        args_str_no_pp = str(args_parser)
        args_str = create_args_string(args_str_no_pp, name_program_py)
        print("Adding to script : ", args_str)
        args_parser.path_labeled = temp_str
        if(i < num_batches - 1):
            file.write(args_str + "\n")
        else:
            file.write(args_str)
    file.close()


def create_mixmatch_ssdl_shell_script(script_name_path, num_batches, args_parser, name_program_py, ood_perc, num_unlabeled, use_unlabeled_external):
    """
    Generic creator of traing/test/ood partitioner script
    :param dataset_all_path:
    :param dataset_name:
    :param script_name:
    :param num_batches:
    :param args:
    :return:
    """
    args_parser.mode = "ssdl"
    file = open(script_name_path, 'w')
    name_program_py += " "
    #create a line for excecuting the program with a different batch
    for i in range(0, num_batches):
        #the new batch id
        temp_str = args_parser.path_labeled
        temp_str2 = args_parser.path_unlabeled
        #avoid multiple number concats
        args_parser.path_labeled += str(i)
        #add details of
        if(use_unlabeled_external):
            args_parser.path_unlabeled += str(i) + "/batch_" + str(i) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" +  str(int(100 * ood_perc))
        args_str_no_pp = str(args_parser)
        args_str = create_args_string(args_str_no_pp, name_program_py)
        print("Adding to script : ", args_str)
        args_parser.path_labeled = temp_str
        args_parser.path_unlabeled = temp_str2
        if(i < num_batches - 1):
            file.write(args_str + "\n")
        else:
            file.write(args_str)
    file.close()


def create_partitioner_unlabeled_script_MNIST(num_unlabeled, ood_perc, num_batches):
    script_name_path = "../shell_scripts/unlabeled_ood_partitioner_ood4ssdl_MNIST_" + str(num_batches) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc) + ".sh"
    args_parser = create_parser().parse_args([])
    args_parser.mode = "unlabeled_partitioner"
    name_program_py = "../utilities/dataset_partitioner.py "
    args_parser.path_ood = "/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_out_dist/batch_"
    #base path
    args_parser.path_dest = "/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled/batch_"
    #OOD percentage
    args_parser.ood_perc = ood_perc
    args_parser.path_iod = "/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_"
    #number of unlabeled observations
    args_parser.num_unlabeled = num_unlabeled


    file = open(script_name_path, 'w')

    for curr_batch in range(0, num_batches):
        args_parser.batch_id_num = curr_batch

        temp_str_dest =  args_parser.path_dest
        args_parser.path_dest +=  str(curr_batch)
        temp_str_ood = args_parser.path_ood
        args_parser.path_ood += str(curr_batch)
        temp_str_iod = args_parser.path_iod
        args_parser.path_iod += str(curr_batch) + "/train"
        #print("PATH IOD ", args_parser.path_iod)
        args_str_no_pp = str(args_parser)
        args_str = create_args_string(args_str_no_pp, name_program_py)
        args_parser.path_dest = temp_str_dest
        args_parser.path_ood = temp_str_ood
        args_parser.path_iod = temp_str_iod

        print("Adding to script: ", args_str)
        if (curr_batch < num_batches - 1):
            file.write(args_str + "\n")
        else:
            file.write(args_str)

    file.close()

def create_partitioner_trainer_script_MNIST():
    """
    Partitioner script for MNIST
    :return:
    """
    script_name_path = "../shell_scripts/training_test_ood_partitioner_ood4ssdl_MNIST.sh"
    args_parser = create_parser().parse_args([])
    args_parser.mode = "train_partitioner"
    args_parser.path_base = "/media/Data/user/Datasets/MNIST_medium_complete/"
    args_parser.eval_perc = 0.25
    num_batches = 10
    create_training_test_datasets_shell_script(script_name_path, num_batches, args_parser)

def create_args_string(args_namespace_str, program_name):
    """
    Format string for posting in an sh file
    :param args_namespace_str:
    :param program_name:
    :return:
    """
    #equals must be eliminated
    args_namespace_str = args_namespace_str.replace("=", " ")
    args_namespace_str = args_namespace_str.replace("Namespace(", "--").replace(")", "")
    args_namespace_str = args_namespace_str.replace(", ", " --").replace("$", ",")
    args_str = "python " + program_name + args_namespace_str
    return args_str

def create_test_scripts_MNIST_fully_supervised():
    """

    :return:
    """
    script_name_path = "../shell_scripts/mixmatch_train_fully_supervised_MNIST.sh"
    args_parser = cli.create_parser().parse_args([])
    args_parser.model = "wide_resnet"
    args_parser.path_labeled = "/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_"
    args_parser.results_file_name = "stats_fully_supervised_MNIST.csv"
    args_parser.lr = 2e-4
    args_parser.weight_decay = 1e-4
    args_parser.epochs = 1
    args_parser.mode = "fully_supervised"
    args_parser.num_classes = 5
    args_parser.size_image = 28
    args_parser.batch_size = 32
    args_parser.log_folder = "logs_MNIST_supervised"
    args_parser.norm_stats = "MNIST"
    args_parser.epochs = 30
    num_batches = 10
    name_program_py = "../MixMatch_OOD_main.py"
    create_mixmatch_fully_supervised_shell_script(script_name_path, num_batches, args_parser, name_program_py)


def create_test_scripts_MNIST_semi_supervised(use_unlabeled_external, num_unlabeled, ood_perc):
    """

    :return:
    """
    script_name_path = "../shell_scripts/mixmatch_train_semi_supervised_MNIST_use_external_" + str(use_unlabeled_external) + "_num_unlabeled_" + str(num_unlabeled) + "_ood_perc_" + str(ood_perc) + ".sh"
    args_parser = cli.create_parser().parse_args([])
    args_parser.model = "wide_resnet"
    args_parser.path_labeled = "/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist/batch_"
    args_parser.path_unlabeled = ""
    if(use_unlabeled_external):
        args_parser.path_unlabeled = "/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled/batch_"

    args_parser.results_file_name = "stats_semi_supervised_MNIST.csv"
    args_parser.lr = 2e-4
    args_parser.weight_decay = 1e-4
    args_parser.epochs = 1
    args_parser.mode = "ssdl"
    args_parser.num_classes = 5
    args_parser.size_image = 28
    args_parser.batch_size = 32
    args_parser.log_folder = "logs_MNIST__semi_supervised"
    args_parser.norm_stats = "MNIST"
    args_parser.epochs = 30
    num_batches = 10
    name_program_py = "../MixMatch_OOD_main.py"
    create_mixmatch_ssdl_shell_script(script_name_path, num_batches, args_parser, name_program_py, ood_perc, num_unlabeled, use_unlabeled_external)

def test_create_test_scripts_MNIST_semi_supervised():
    create_test_scripts_MNIST_semi_supervised(use_unlabeled_external = True, num_unlabeled = 15000, ood_perc = 0.5)

def test_create_unlabeled_script_MNIST():
    create_test_scripts_MNIST_semi_supervised(use_unlabeled_external = True, num_unlabeled = 15000, ood_perc = 0.5, num_batches = 10)

test_create_test_scripts_MNIST_semi_supervised()
