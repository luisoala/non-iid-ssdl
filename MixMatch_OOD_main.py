from fastai.vision import *
from fastai.callbacks import CSVLogger
from numbers import Integral
import torch
import logging
import sys
from torchvision.utils import save_image
import numpy as np
#from utilities.InBreastDataset import InBreastDataset
from utilities.run_context import RunContext
import utilities.cli as cli
import torchvision
#from utilities.albumentations_manager import get_albumentations


import mlflow
import os
import shutil
import time
import datetime
import matplotlib.pyplot as plt
import imageio
from skimage import transform


class MultiTransformLabelList(LabelList):
    def __getitem__(self, idxs: Union[int, np.ndarray]) -> 'LabelList':
        """
        Create K transformed images for the unlabeled data
        :param idxs:
        :return:
        """
        "return a single (x, y) if `idxs` is an integer or a new `LabelList` object if `idxs` is a range."
        global args
        #print("MULTITRANSFORM LIST")
        idxs = try_int(idxs)
        if isinstance(idxs, Integral):
            if self.item is None:
                #CALLED EVEN FOR UNLABELED DATA, Y IS USED!
                x, y = self.x[idxs], self.y[idxs]
            else:
                x, y = self.item, 0
            if self.tfms or self.tfmargs:
                #THIS IS DONE FOR UNLABELED DATA
                x = [x.apply_tfms(self.tfms, **self.tfmargs) for _ in range(args.K_transforms)]
            if hasattr(self, 'tfms_y') and self.tfm_y and self.item is None:
                #IS NOT CALLED FOR UNLABELED DATA
                y = y.apply_tfms(self.tfms_y, **{**self.tfmargs_y, 'do_resolve': False})
            if y is None: y = 0
            return x, y
        else:
            return self.new(self.x[idxs], self.y[idxs])



def MixmatchCollate(batch):
    """
    # I'll also need to change the default collate function to accomodate multiple augments
    :param batch:
    :return:
    """
    batch = to_data(batch)
    if isinstance(batch[0][0], list):
        batch = [[torch.stack(s[0]), s[1]] for s in batch]
    return torch.utils.data.dataloader.default_collate(batch)





class MixupLoss(nn.Module):
    """
    Implements the mixup loss
    """

    def forward(self, preds, target, unsort=None, ramp=None, bs=None):
        """
        Ramp, unsort and bs is None when doing validation
        :param preds:
        :param target:
        :param unsort:
        :param ramp:
        :param bs:
        :return:
        """
        global args

        if(args.balanced==5):
            return self.forward_balanced_cross_entropy(preds, target, unsort, ramp, bs)
        else:
            return self.forward_original(preds, target, unsort, ramp, bs)

    def forward_cross_entropy(self, preds, target, unsort=None, ramp=None, bs=None):

        global args
        if unsort is None:
            return F.cross_entropy(preds, target)

        calculate_cross_entropy = nn.CrossEntropyLoss()
        preds = preds[unsort]
        preds_l = preds[:bs]
        preds_ul = preds[bs:]
        # calculate log of softmax, to ensure correct usage of cross entropy
        # one column per class, one batch per row

        preds_ul = torch.softmax(preds_ul, dim=1)
        # TARGETS CANNOT BE 1-K ONE HOT VECTOR
        (highest_values, highest_classes) = torch.max(target[:bs], 1)

        highest_classes = highest_classes.long()

        loss_x = calculate_cross_entropy(preds_l, highest_classes)
        # loss_x = -(preds_l * target[:bs]).sum(dim=1).mean()
        loss_u = F.mse_loss(preds_ul, target[bs:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + args.lambda_unsupervised * ramp * loss_u


    def forward_original(self, preds, target, unsort=None, ramp=None, num_labeled=None):
        global args
        """
        Implements the forward pass of the loss function
        :param preds: predictions of the model
        :param target: ground truth targets
        :param unsort: ?
        :param ramp: ramp weight
        :param num_labeled:
        :return:
        """
        if unsort is None:
            #used for evaluation
            return F.cross_entropy(preds,target)
        preds = preds[unsort]
        #labeled and unlabeled observations were packed in the same array
        preds_l = preds[:num_labeled]
        preds_ul = preds[num_labeled:]
        #apply logarithm to softmax of output, to ensure the correct usage of cross entropy
        preds_l = torch.log_softmax(preds_l,dim=1)
        preds_ul = torch.softmax(preds_ul,dim=1)
        loss_x = -(preds_l * target[:num_labeled]).sum(dim=1).mean()
        loss_u = F.mse_loss(preds_ul, target[num_labeled:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + args.lambda_unsupervised * ramp * loss_u

    def forward_balanced(self, preds, target, unsort=None, ramp=None, bs=None):
        """
        Balanced forward implementation
        :param preds:
        :param target:
        :param unsort:
        :param ramp:
        :param bs:
        :return:
        """
        global args
        if unsort is None:
            return F.cross_entropy(preds, target)
        # target contains mixed up targets!! not just 0s and 1s
        preds = preds[unsort]
        preds_l = preds[:bs]
        preds_ul = preds[bs:]
        # calculate log of softmax, to ensure correct usage of cross entropy
        # one column per class, one batch per row
        preds_l = torch.log_softmax(preds_l, dim=1)

        # get the weights for the labeled observations
        weights_labeled = self.get_weights_observations(target[:bs])
        preds_ul = torch.softmax(preds_ul, dim=1)
        # get the weights for the unlabeled observations
        weights_unlabeled = self.get_weights_observations(target[bs:])
        loss_x = -(weights_labeled * preds_l * target[:bs]).sum(dim=1).mean()
        loss_u = F.mse_loss(weights_unlabeled * preds_ul, weights_unlabeled * target[bs:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + args.lambda_unsupervised * ramp * loss_u

    def forward_balanced_cross_entropy(self, preds, target, unsort=None, ramp=None, bs=None):
        global args, class_weights
        if unsort is None:
            return F.cross_entropy(preds, target)
        weights_unlabeled = self.get_weights_observations(target[bs:]).float()
        calculate_cross_entropy = nn.CrossEntropyLoss(weight = class_weights.float())
        preds = preds[unsort]
        preds_l = preds[:bs]
        preds_ul = preds[bs:]
        # calculate log of softmax, to ensure correct usage of cross entropy
        # one column per class, one batch per row
        preds_ul = torch.softmax(preds_ul, dim=1)
        # TARGETS CANNOT BE 1-K ONE HOT VECTOR
        (highest_values, highest_classes) = torch.max(target[:bs], 1)
        highest_classes = highest_classes.long()
        loss_x = calculate_cross_entropy(preds_l, highest_classes)
        loss_u = F.mse_loss(weights_unlabeled * preds_ul, weights_unlabeled * target[bs:])
        self.loss_x = loss_x.item()
        self.loss_u = loss_u.item()
        return loss_x + args.lambda_unsupervised * ramp * loss_u

    def get_weights_observations(self, array_predictions):
        global class_weights
        # class_weights = torch.tensor([0.2, 0.2, 0.2, 0.2, 0.2])
        # each column is a class, each row an observation
        num_classes = array_predictions.shape[1]
        num_observations = array_predictions.shape[0]
        (highest_values, highest_classes) = torch.max(array_predictions, 1)
        # turn the highest_classes array a column vector
        highest_classes_col = highest_classes.view(-1, 1)
        # highest classes for all the observations (rows) and classes (columns)
        highest_classes_all = highest_classes_col.repeat(1, num_classes)
        # print("highest classes all")
        # print(highest_classes_all)
        # scores all
        scores_all = class_weights[highest_classes_all]
        scores_all.to(device="cuda:0")
        return scores_all



class MixMatchImageList(ImageList):


    """
    Custom ImageList with filter function
    """
    def filter_train(self, num_items, seed = 23488):
        """
        Takes a number of observations as labeled, assumes that the evaluation observations are in the test folder
        :param num_items:
        :param seed: The seed is fixed for reproducibility
        :return: return the filtering function by itself
        """
        global args
        path_unlabeled = args.path_unlabeled
        if (args.path_unlabeled == ""):
          path_unlabeled = args.path_labeled
        #this means that a customized unlabeled dataset is not to be used, just pick the rest of the labelled data as unlabelled
        if(path_unlabeled == args.path_labeled):
          train_idxs = np.array([i for i, observation in enumerate(self.items) if Path(observation).parts[-3] != "test"])
        else:
          # IGNORE THE DATA ALREADY IN THE UNLABELED DATASET
          dataset_unlabeled =  torchvision.datasets.ImageFolder(path_unlabeled + "/train/")
          list_file_names_unlabeled = dataset_unlabeled.imgs
          for i in range(0, len(list_file_names_unlabeled)):
            #delete root of path
            list_file_names_unlabeled[i] = list_file_names_unlabeled[i][0].replace(path_unlabeled, "")
          list_train = []
          #add  to train if is not in the unlabeled dataset
          for i, observation in enumerate(self.items):
            path_1 = str(Path(observation))
            sub_str = args.path_labeled
            path_2 = path_1.replace(sub_str, "")
            path_2 = path_2.replace("train/", "")
            is_path_in_unlabeled = path_2 in list_file_names_unlabeled
            #add the observation to the train list, if is not in the unlabeled dataset
            if( not "test" in path_2 and not is_path_in_unlabeled):
              list_train += [i]
          #store the train idxs c
          train_idxs = np.array(list_train)            
          logger.info("Customized number of unlabeled observations " + str(len(list_file_names_unlabeled)))
          
        valid_idxs = np.array([i for i, observation in enumerate(self.items) if Path(observation).parts[-3] == "test"])
        # for reproducibility
        np.random.seed(seed)
        # keep the number of items desired, 500 by default
        keep_idxs = np.random.choice(train_idxs, num_items, replace=False)
        
        logger.info("Number of labeled observations: " + str(len(keep_idxs)))
        logger.info("First labeled id: " + str(keep_idxs[0]))
        logger.info("Number of validation observations: " + str(len(valid_idxs)))
        logger.info("Number of training observations " + str(len(train_idxs)))
        self.items = np.array([o for i, o in enumerate(self.items) if i in np.concatenate([keep_idxs, valid_idxs])])
        return self

class PartialTrainer(LearnerCallback):
    def on_epoch_end(self, epoch, last_metrics, smooth_loss, last_loss, **kwargs):
        train_loss = float(smooth_loss)
        val_loss = float(last_metrics[0])
        val_accuracy = float(last_metrics[1])
        mlflow.log_metric(key= 'train_loss', value=train_loss, step=epoch)#last_loss
        mlflow.log_metric(key= 'val_loss', value=val_loss, step=epoch)#last_metric #1
        mlflow.log_metric(key= 'val_accuracy', value=val_accuracy, step=epoch)

class MixMatchTrainer(LearnerCallback):
    """
    Mix match trainer functions
    """

    def on_train_begin(self, **kwargs):
        """
        Callback used when the trainer is beginning, inits variables
        :param kwargs:
        :return:
        """
        global data_labeled
        self.l_dl = iter(data_labeled.train_dl)
        #metrics recorder
        self.smoothL, self.smoothUL = SmoothenValue(0.98), SmoothenValue(0.98)
        #metrics to be displayed in the table
        self.it = 0

    def mixup(self, a_x, a_y, b_x, b_y):
        """
        Mixup augments data by mixing labels and pseudo labels and its observations
        :param a_x:
        :param a_y:
        :param b_x:
        :param b_y:
        :param alpha:
        :return:
        """
        global args
        alpha = args.alpha_mix
        l = np.random.beta(alpha, alpha)
        l = max(l, 1 - l)
        x = l * a_x + (1 - l) * b_x
        y = l * a_y + (1 - l) * b_y
        return x, y

    def sharpen(self, p):
        global args
        """
        Sharpens the distribution output, to encourage confidence
        :param p:
        :param T:
        :return:
        """
        T = args.T_sharpening
        u = p ** (1 / T)
        return u / u.sum(dim=1, keepdim=True)

    def on_batch_begin(self, train, last_input, last_target, **kwargs):
        """
        Called on batch training at the begining
        :param train:
        :param last_input:
        :param last_target:
        :param kwargs:
        :return:
        """
        global data_labeled, args
        if not train: return
        try:
            x_l, y_l = next(self.l_dl)
        except:
            self.l_dl = iter(data_labeled.train_dl)
            x_l, y_l = next(self.l_dl)
        x_ul = last_input
        with torch.no_grad():
            #calculates the pseudo sharpened labels
            ul_labels = self.sharpen(
                torch.softmax(torch.stack([self.learn.model(x_ul[:, i]) for i in range(x_ul.shape[1])], dim=1),
                              dim=2).mean(dim=1))
        #create torch array of unlabeled data
        x_ul = torch.cat([x for x in x_ul])

        #WE CAN CALCULATE HERE THE CONFIDENCE COEFFICIENT

        ul_labels = torch.cat([y.unsqueeze(0).expand(args.K_transforms, -1) for y in ul_labels])

        l_labels = torch.eye(data_labeled.c).cuda()[y_l]

        w_x = torch.cat([x_l, x_ul])
        w_y = torch.cat([l_labels, ul_labels])
        idxs = torch.randperm(w_x.shape[0])
        #create mixed input and targets
        mixed_input, mixed_target = self.mixup(w_x, w_y, w_x[idxs], w_y[idxs])
        bn_idxs = torch.randperm(mixed_input.shape[0])
        unsort = [0] * len(bn_idxs)
        for i, j in enumerate(bn_idxs): unsort[j] = i
        mixed_input = mixed_input[bn_idxs]

        ramp = self.it / args.rampup_coefficient if self.it < args.rampup_coefficient else 1.0
        return {"last_input": mixed_input, "last_target": (mixed_target, unsort, ramp, x_l.shape[0])}

    def on_batch_end(self, train, **kwargs):
        """
        Add the metrics at the end of the batch training
        :param train:
        :param kwargs:
        :return:
        """
        if not train: return
        self.smoothL.add_value(self.learn.loss_func.loss_x)
        self.smoothUL.add_value(self.learn.loss_func.loss_u)
        self.it += 1

        """def on_epoch_end(self, last_metrics, **kwargs):
        Avoid adding weird stuff on metrics table
        When the epoch ends, add the accmulated metric values
        :param last_metrics:
        :param kwargs:
        :return:
        
        return add_metrics(last_metrics, [self.smoothL.smooth, self.smoothUL.smooth])
        """
    
    def on_epoch_end(self, epoch, last_metrics, smooth_loss, last_loss, **kwargs):
        train_loss = float(smooth_loss)
        val_loss = float(last_metrics[0])
        val_accuracy = float(last_metrics[1])
        mlflow.log_metric(key= 'train_loss', value=train_loss, step=epoch)#last_loss
        mlflow.log_metric(key= 'val_loss', value=val_loss, step=epoch)#last_metric #1
        mlflow.log_metric(key= 'val_accuracy', value=val_accuracy, step=epoch)
        

def get_dataset_stats(args):
    #note: these are just used as a placeholder, the actual standardization stats are calculated on per batch basis when the data is read
    if(args.norm_stats.strip() == "MNIST"):
        # stats for MNIST
        meanDatasetComplete = [0.1307, 0.1307, 0.1307]
        stdDatasetComplete = [0.3081, 0.3081, 0.3081]

    return (meanDatasetComplete, stdDatasetComplete)

def calculate_weights(list_labels):
    """
    Calculate the class weights according to the number of observations
    :param list_labels:
    :return:
    """
    global logger, args
    array_labels = np.array(list_labels)
    logger.info("Using balanced loss: " + str(args.balanced))
    list_classes = np.unique(array_labels)
    weight_classes = np.zeros(len(list_classes))
    for curr_class in list_classes:

        number_observations_class = len(array_labels[array_labels == curr_class])
        logger.info("Number observations " + str(number_observations_class) + " for class " + str(curr_class))
        weight_classes[curr_class] = 1 / number_observations_class

    weight_classes = weight_classes / weight_classes.sum()
    logger.info("Weights to use: " + str(weight_classes))
    weight_classes_tensor = torch.tensor(weight_classes, device ="cuda:0" )
    return weight_classes_tensor

def get_datasets():
    """
    Get datasets  (FAST AI data bunches ) for labeled, unlabeled and validation
    :return: data_labeled (limited labeled data), data_unlabeled , data_full (complete labeled dataset)
    """
    global args, data_labeled, logger, class_weights
    path_labeled = args.path_labeled
    path_unlabeled = args.path_unlabeled
    if (args.path_unlabeled == ""):
        path_unlabeled = path_labeled
    #get dataset mean and std
    norm_stats = get_dataset_stats(args)
    logger.info("Loading labeled data from: " + path_labeled)
    logger.info("Loading unlabeled data from: " + path_unlabeled)
    # Create two databunch objects for the labeled and unlabled images. A fastai databunch is a container for train, validation, and
    # test dataloaders which automatically processes transforms and puts the data on the gpu.
    # https://docs.fast.ai/vision.transform.html
    
    #COMPUTE BATCH NORMALIZATION STATS FOR LABELED DATA
    data_labeled = (MixMatchImageList.from_folder(path_labeled)
                    .filter_train(args.number_labeled)  # Use 500 labeled images for traning
                    .split_by_folder(valid="test")  # test on all 10000 images in test set
                    .label_from_folder()
                    .transform(get_transforms(do_flip = True, flip_vert = True, max_zoom=1, max_warp=None, p_affine=0, p_lighting = 0),
                               size=args.size_image)
                    # On windows, must set num_workers=0. Otherwise, remove the argument for a potential performance improvement
                    .databunch(bs=args.batch_size, num_workers=args.workers)
                    .normalize(norm_stats))
    train_set = set(data_labeled.train_ds.x.items)
    #logging the labeled inputs to artifacts/inputs/labelled
    labeled_array_list = []
    for labeled in train_set:
        mlflow.log_artifact(labeled, artifact_path='inputs/labelled')
        image = imageio.imread(labeled)
        labeled_array_list.append(image)
    labeled_shape = image.shape
    labeled_array = np.array(labeled_array_list)/255.
    if len(labeled_array.shape) < 4: #for grayscale data we copy the last chanel three times
        norm_stats_labeled = (list(np.mean(labeled_array[:,:,:,np.newaxis], axis=(0,1,2)))*3, list(np.std(labeled_array[:,:,:,np.newaxis], axis=(0,1,2)))*3)
    else:
        norm_stats_labeled = (list(np.mean(labeled_array, axis=(0,1,2))), list(np.std(labeled_array, axis=(0,1,2))))
    
    #CREATE DATA BUNCH WITH BATCH STATS FOR LABELED DATA
    data_labeled = (MixMatchImageList.from_folder(path_labeled)
                    .filter_train(args.number_labeled)  # Use 500 labeled images for traning
                    .split_by_folder(valid="test")  # test on all 10000 images in test set
                    .label_from_folder()
                    .transform(get_transforms(do_flip = True, flip_vert = True, max_zoom=1, max_warp=None, p_affine=0, p_lighting = 0),
                               size=args.size_image)
                    # On windows, must set num_workers=0. Otherwise, remove the argument for a potential performance improvement
                    .databunch(bs=args.batch_size, num_workers=args.workers)
                    .normalize(norm_stats_labeled))
    # normalize_funcs(mean:FloatTensor, std:FloatTensor, do_x:bool=True, do_y:bool=False)
    train_set = set(data_labeled.train_ds.x.items)
    
    #get the list of labels for the dataset
    list_labels = data_labeled.train_ds.y.items
    #calculate the class weights
    class_weights = calculate_weights(list_labels)
    # load the unlabeled data
    #filter picks the labeled images not contained in the unlabeled dataset, in the case of SSDL
    #the test set is in the unlabeled folder

    src = (ImageList.from_folder(path_unlabeled)
           .filter_by_func(lambda x: x not in train_set)
           .split_by_folder(valid="test")
           )
    unlabeled_array_list = []
    #logging iod and ood unlabelled data
    for class_id in os.listdir(path_unlabeled+'/train'):
        for unlabelled in os.listdir(path_unlabeled+'/train'+'/'+class_id):
            if 'ood' in unlabelled:
                mlflow.log_artifact(path_unlabeled+'/train'+'/'+class_id+'/'+unlabelled, artifact_path='inputs/unlabelled/ood/'+class_id)
                image = imageio.imread(path_unlabeled+'/train'+'/'+class_id+'/'+unlabelled)
                image = transform.resize(image, (args.size_image, args.size_image, 3),preserve_range=True)
            else:
                #mlflow.log_artifact(path_unlabeled+'/train'+'/'+class_id+'/'+unlabelled, artifact_path='inputs/unlabelled/iod/'+class_id)
                mlflow.log_artifact(path_unlabeled+'/train'+'/'+class_id+'/'+unlabelled, artifact_path='inputs/unlabelled/iod/'+class_id)
                image = imageio.imread(path_unlabeled+'/train'+'/'+class_id+'/'+unlabelled)
                image = transform.resize(image, (args.size_image, args.size_image, 3), preserve_range=True)
            unlabeled_array_list.append(image)
    
    unlabeled_array = np.array(unlabeled_array_list)/255.
    print('##################################')
    print(unlabeled_array.shape)
    print('##################################')
    if len(unlabeled_array.shape) < 4: #for grayscale data we copy the last chanel three times
        norm_stats_unlabeled = (list(np.mean(unlabeled_array[:,:,:,np.newaxis], axis=(0,1,2)))*3, list(np.std(unlabeled_array[:,:,:,np.newaxis], axis=(0,1,2)))*3)
    else:
        norm_stats_unlabeled = (list(np.mean(unlabeled_array, axis=(0,1,2))), list(np.std(unlabeled_array, axis=(0,1,2))))
    mlflow.log_param(key="norm_stats_labeled", value=str(norm_stats_labeled))
    mlflow.log_param(key="norm_stats_unlabeled", value=str(norm_stats_unlabeled))
    
    #AUGMENT THE DATA
    src.train._label_list = MultiTransformLabelList
    # https://docs.fast.ai/vision.transform.html
    # data not in the train_set and splitted by test folder is used as unlabeled
    data_unlabeled = (src.label_from_folder()
                      .transform(get_transforms(do_flip = True, flip_vert = True, max_zoom=1, max_warp=None, p_affine=0, p_lighting = 0), size=args.size_image)
                      .databunch(bs=args.batch_size, collate_fn=MixmatchCollate, num_workers=10)
                      .normalize(norm_stats_unlabeled))


    # Databunch with all 50k images labeled, for baseline
    data_full = (ImageList.from_folder(path_labeled)
                 .split_by_folder(valid="test")
                 .label_from_folder()
                 .transform(get_transforms(do_flip = True, flip_vert = True, max_zoom=1, max_warp=None, p_affine=0, p_lighting = 0),
                            size=args.size_image)
                 .databunch(bs=args.batch_size, num_workers=args.workers)
                 .normalize(norm_stats))
    return (data_labeled, data_unlabeled, data_full)




def train_mix_match():
    """
    Train the mix match model
    :param path_labeled:
    :param path_unlabeled:
    :param number_epochs:
    :param learning_rate:
    :param mode:
    :return:
    """
    global data_labeled, is_colab, logger, args
    learning_rate = args.lr
    number_epochs = args.epochs
    logger = logging.getLogger('main')
    (data_labeled, data_unlabeled, data_full)= get_datasets()

    #start_nf the initial number of features
    """
    Wide ResNet with num_groups and a width of k.
    Each group contains N blocks. start_nf the initial number of features. Dropout of drop_p is applied in between the two convolutions in each block. The expected input channel size is fixed at 3.
    Structure: initial convolution -> num_groups x N blocks -> final layers of regularization and pooli
    """
    if(args.model == "wide_resnet"):
        model = models.WideResNet(num_groups=3,N=4,num_classes=args.num_classes,k = 2,start_nf=args.size_image)
    elif(args.model == "densenet"):
        model = models.densenet121(num_classes=args.num_classes)
    elif(args.model == "squeezenet"):
        model = models.squeezenet1_1(num_classes=args.num_classes)
    elif(args.model.strip() == "alexnet"):
        logger.info("Using alexnet")
        model = models.alexnet(num_classes=args.num_classes)

    if (args.mode.strip() == "fully_supervised"):
        logger.info("Training fully supervised model")
        # Edit: We can find the answer ‘Note that metrics are always calculated on the validation set.’ on this page: https://docs.fast.ai/training.html 42.
        if (is_colab):
            learn = Learner(data_full, model, metrics=[accuracy])
        else: #, callback_fns = [CSVLogger]
            learn = Learner(data_full, model, metrics=[accuracy], callback_fns = [CSVLogger])



    if (args.mode.strip() == "partial_supervised"):
        logger.info("Training supervised model with a limited set of labeled data")
        if(is_colab):
            #uses loss_func=FlattenedLoss of CrossEntropyLoss()
            learn = Learner(data_labeled, model, metrics=[accuracy])
        else:
            if(args.balanced == 5):
                logger.info("Using balanced cross entropy")
                calculate_cross_entropy = nn.CrossEntropyLoss(weight=class_weights.float())
                learn = Learner(data_labeled, model, metrics=[accuracy], callback_fns = [PartialTrainer, CSVLogger], loss_func = calculate_cross_entropy)
            else:
                learn = Learner(data_labeled, model, metrics=[accuracy], callback_fns=[PartialTrainer, CSVLogger])


        #learn.fit_one_cycle(number_epochs, learning_rate, wd=args.weight_decay)

    """
    fit[source][test]
    fit(epochs:int, lr:Union[float, Collection[float], slice]=slice(None, 0.003, None), wd:Floats=None, callbacks:Collection[Callback]=None)
    Fit the model on this learner with lr learning rate, wd weight decay for epochs with callbacks.
    """
    
    if (args.mode.strip() == "ssdl"):
        logger.info("Training semi supervised model with limited set of labeled data")
        # https://datascience.stackexchange.com/questions/15989/micro-average-vs-macro-average-performance-in-a-multiclass-classification-settin
        if(is_colab):
            learn = Learner(data_unlabeled, model, loss_func=MixupLoss(), callback_fns=[MixMatchTrainer], metrics=[accuracy])
        else:
            learn = Learner(data_unlabeled, model, loss_func=MixupLoss(), callback_fns=[MixMatchTrainer, CSVLogger],
                            metrics=[accuracy])

    #train the model
    learn.fit_one_cycle(number_epochs, learning_rate, wd=args.weight_decay)
    #if it is not colab, write the csv to harddrive
    if(not is_colab):
        logged_frame = learn.csv_logger.read_logged_file()


		
def main_colab():
    global args, logger, is_colab
    is_colab = True
    dateInfo = "{date:%Y-%m-%d_%H_%M_%S}".format(date=datetime.now())
    logging.basicConfig(filename="log_" + dateInfo + ".txt", level=logging.INFO, format='%(message)s')
    logger = logging.getLogger('main')    
    #Get the default arguments
    args = create_parser().parse_args(args=[])
    #args.balanced = False
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.info("Arguments: " + str(args))    
    train_mix_match()

if __name__ == '__main__':
    global args, counter, context, logger, is_colab
    is_colab = False
    args = cli.parse_commandline_args()
    print("Balanced loss: ")
    #args.balanced = False
    print(args.balanced)

    print("Rampup coefficient: ", args.rampup_coefficient)

    logger = logging.getLogger('main')
    logger.info("Learning rate " + str(args.lr))

    
    #mlflow logging
    _, batch_info = args.path_unlabeled.rsplit('/',1)
    batch, batch_num, batch_stats = batch_info.split('_', 2)
    num_labeled = str(args.number_labeled)
    _, _, num_unlabeled, _, _, ood_perc_pp = batch_stats.split('_')
    
    experiment_name = args.dataset+'-'+num_labeled+'-'+ood_perc_pp
    run_name = batch+'_'+batch_num
    mlflow.set_experiment(experiment_name=experiment_name) #create the experiment
    if args.exp_creator == "Yes":
        quit()
    mlflow.start_run(run_name=run_name) #start the mlflow run for logging

    mlflow.log_params(params=vars(args)) #log all parameters in one go using log_batch
    mlflow.log_param(key='batch number', value=batch+' '+batch_num)
    mlflow.log_param(key='batch stats', value = batch_stats)
    
    train_mix_match()
    mlflow.end_run()
