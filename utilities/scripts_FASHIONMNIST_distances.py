from scipy.spatial import distance
from dataset_distance_measurer import dataset_distance_tester_pdf
from dataset_distance_measurer import dataset_distance_tester
#distance measurer for CIFAR 10

def run_tests_minkowski_fashionmnist_no_half(p = 1, ood_perc = 50):
    """
    :param distance: distance_str
    :return:
    """
    path_labelled = "/media/Data/user/Datasets/FASHIONMNIST/batches_labeled_in_dist/batch_"
    path_base_unlabelled = "/media/Data/user/Datasets/FASHIONMNIST"
    print("Path labelled: ", path_labelled)
    print("Path unlabelled: ",path_base_unlabelled )



    #Gaussian distance
    print("Calculating p" + str(p) + " distance for Gaussian dataset")
    dataset_distance_tester(
        path_bunch1=path_labelled,
        path_bunch2= path_base_unlabelled + "/batches_unlabeled_Gaussian/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="Gaussian_p_" + str(p), num_batches=10, p = p)

    #Salt and pepper
    print("Calculating  p" + str(p) + " distance for Salt and pepper dataset")
    dataset_distance_tester(
        path_bunch1=path_labelled,
        path_bunch2=path_base_unlabelled +"/batches_unlabeled_SaltAndPepper/batch_",
        ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="SaltAndPepper_p_" + str(p), num_batches=10,
        p = p)

    #FASHION PRODUCT
    print("Calculating p" + str(p) + " distance for FASHION PRODUCT dataset")
    dataset_distance_tester(
        path_bunch1=path_labelled,
        path_bunch2= path_base_unlabelled + "/batches_unlabeled_FASHIONPRODUCT/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="FASHIONPRODUCT__p_" + str(p), num_batches = 10, p = p)

    #Imagenet
    print("Calculating p" + str(p) + " distance for Imagenet dataset")
    dataset_distance_tester(
        path_bunch1=path_labelled,
        path_bunch2=path_base_unlabelled+"/batches_unlabeled_IMAGENET/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="Imagenet_p_" + str(p), num_batches=10, p = p)


def run_tests_pdf_fashionmnist_no_half(distance_str = "js", ood_perc = 50):
    """
    :param distance: distance_str
    :return:
    """
    path_labelled =  "/media/Data/user/Datasets/FASHIONMNIST/batches_labeled_in_dist/batch_"
    path_base_unlabelled = "/media/Data/user/Datasets/FASHIONMNIST"
    print("Path labelled: ", path_labelled)
    print("Path unlabelled: ", path_base_unlabelled)
    if(distance_str == "js"):
        distance_func = distance.jensenshannon
    elif(distance_str == "cosine"):
        distance_func = distance.cosine

    #Gaussian distance
    print("Calculating " + distance_str  + " distance for Gaussian dataset")
    dataset_distance_tester_pdf(
        path_bunch1=path_labelled,
        path_bunch2= path_base_unlabelled + "/batches_unlabeled_Gaussian/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="Gaussian_pdf_" + distance_str, num_batches=10, distance_func = distance_func)

    #Salt and pepper
    print("Calculating  " + distance_str + " distance for Salt and pepper dataset")
    dataset_distance_tester_pdf(
        path_bunch1=path_labelled,
        path_bunch2=path_base_unlabelled +"/batches_unlabeled_SaltAndPepper/batch_",
        ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="SaltAndPepper_pdf_" + distance_str, num_batches=10,
        distance_func=distance_func)

    #SVHN
    print("Calculating " + distance_str + " distance for FASHION PRODUCT dataset")
    dataset_distance_tester_pdf(
        path_bunch1=path_labelled,
        path_bunch2= path_base_unlabelled + "/batches_unlabeled_FASHIONPRODUCT/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="FASHIONPRODUCT__pdf_" + distance_str, num_batches=10, distance_func=distance_func)

    #Imagenet
    print("Calculating " + distance_str + " distance for Imagenet dataset")
    dataset_distance_tester_pdf(
        path_bunch1=path_labelled,
        path_bunch2=path_base_unlabelled+"/batches_unlabeled_IMAGENET/batch_", ood_perc=ood_perc,
        num_unlabeled=3000, name_ood_dataset="Imagenet_pdf_" + distance_str, num_batches=10, distance_func = distance_func)

def run_fashionmnist_tests_pdf():
    run_tests_pdf_fashionmnist_no_half(distance_str="js", ood_perc=50)
    run_tests_pdf_fashionmnist_no_half(distance_str="js", ood_perc=100)
    run_tests_pdf_fashionmnist_no_half(distance_str="cosine", ood_perc=50)
    run_tests_pdf_fashionmnist_no_half(distance_str="cosine", ood_perc=100)

def run_fashionmnist_tests_p():
    run_tests_minkowski_fashionmnist_no_half(p = 1, ood_perc=50)
    run_tests_minkowski_fashionmnist_no_half(p = 1, ood_perc=100)
    run_tests_minkowski_fashionmnist_no_half(p = 2, ood_perc=50)
    run_tests_minkowski_fashionmnist_no_half(p = 2, ood_perc=100)



run_fashionmnist_tests_pdf()
