from scipy.spatial import distance
from dataset_distance_measurer import dataset_distance_tester_pdf
from dataset_distance_measurer import dataset_distance_tester

#distance measurer for CIFAR 10






def run_tests_pdf_mnist_half(distance_str = "js", ood_perc = 100):
    """
    :param distance: distance_str
    :return:
    """
    path_labelled =  "/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist"
    path_base_unlabelled = "/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_HALF"

    if(distance_str == "js"):
        distance_func = distance.jensenshannon
    elif(distance_str == "cosine"):
        distance_func = distance.cosine

    #HALF
    print("Calculating " + distance_str  + " distance for MNIST HALF dataset")
    dataset_distance_tester_pdf(
        path_bunch1=path_labelled,
        path_bunch2= path_base_unlabelled + "/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="HALF_OOD_pdf_" + distance_str, num_batches=9, distance_func = distance_func)


def run_tests_minkowski_mnist_half(p = 1, ood_perc = 100):
    """
    :param distance: distance_str
    :return:
    """
    path_labelled =  "/media/Data/user/Datasets/MNIST_medium_complete/batches_labeled_in_dist"
    path_base_unlabelled = "/media/Data/user/Datasets/MNIST_medium_complete/batches_unlabeled_HALF"



    #HALF
    print("Calculating p" + str(p)  + " distance for MNIST HALF dataset")

    dataset_distance_tester(
        path_bunch1=path_labelled+ "/batch_",
        path_bunch2=path_base_unlabelled + "/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="HALF_OOD_p_" + str(p), num_batches=9, p=p)

def run_mnist_tests_p(ood_perc = 50):
    run_tests_minkowski_mnist_half(p=1, ood_perc = ood_perc)
    run_tests_minkowski_mnist_half(p=2, ood_perc = ood_perc)

def run_mnist_tests_pdf(ood_perc = 50):
    run_tests_pdf_mnist_half(distance_str="js", ood_perc = ood_perc)
    run_tests_pdf_mnist_half(distance_str="cosine", ood_perc = ood_perc)




run_mnist_tests_p(ood_perc = 50)
run_mnist_tests_pdf(ood_perc = 50)
