import json
import time,os,datetime
from collections import defaultdict
import numpy as np
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold
from sklearn.metrics import auc,roc_auc_score,cohen_kappa_score,roc_curve,confusion_matrix
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
from pyro.infer.autoguide import AutoDelta,AutoNormal,AutoDiagonalNormal
from  pyro.infer import SVI
import pyro.poutine as poutine
import pyro
import vegvisir
import vegvisir.utils as VegvisirUtils
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots
import vegvisir.models as VegvisirModels
import vegvisir.guides as VegvisirGuides
ModelLoad = namedtuple("ModelLoad",["args","max_len","seq_max_len","n_data","input_dim","aa_types","blosum"])


def train_loop(svi,Vegvisir,guide,data_loader, args):
    """Regular batch training
    :param pyro.infer svi
    :param nn.Module,PyroModule Vegvisir: Neural net architecture
    :param guide: EasyGuide or pyro.infer.autoguides
    :param DataLoader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    Vegvisir.train() #Highlight: look at https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    train_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    latent_spaces = []
    for batch_number, batch_dataset in enumerate(data_loader):
        batch_data_blosum = batch_dataset["batch_data_blosum"]
        batch_data_int = batch_dataset["batch_data_int"]
        batch_data_onehot = batch_dataset["batch_data_onehot"]
        batch_data_blosum_norm = batch_dataset["batch_data_blosum_norm"]
        batch_mask = batch_dataset["batch_mask"]
        if args.use_cuda:
            batch_data_blosum = batch_data_blosum.cuda()
            batch_data_int = batch_data_int.cuda()
            batch_data_onehot = batch_data_onehot.cuda()
            batch_data_blosum_norm = batch_data_blosum_norm.cuda()
            batch_mask = batch_mask.cuda()
        true_labels = batch_data_blosum[:,0,0,0]
        batch_data = {"blosum":batch_data_blosum,"int":batch_data_int,"onehot":batch_data_onehot,"norm":batch_data_blosum_norm}
        #Forward & Backward pass
        loss = svi.step(batch_data,batch_mask)
        guide_estimates = guide(batch_data,batch_mask)
        sampling_output = Vegvisir.sample(batch_data,batch_mask,guide_estimates)
        predicted_labels = sampling_output.predicted_labels
        latent_space = sampling_output.latent_space
        latent_spaces.append(latent_space.detach().cpu().numpy())
        total += true_labels.size(0)
        correct += (predicted_labels == true_labels).sum().item()
        predictions.append(predicted_labels.detach().cpu().numpy())
        train_loss += loss
    #Normalize train loss
    train_loss /= len(data_loader)
    predictions_arr = np.concatenate(predictions,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    accuracy= 100 * correct // total
    return train_loss,accuracy,predictions_arr,latent_arr
def valid_loop(svi,Vegvisir,guide, data_loader, args):
    """Regular batch training
    :param svi: pyro infer engine
    :param dataloader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    Vegvisir.eval()
    valid_loss = 0.0
    total = 0.
    correct = 0.
    predictions = []
    latent_spaces = []
    with torch.no_grad(): #do not update parameters with the evaluation data
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_data_int = batch_dataset["batch_data_int"]
            batch_data_onehot = batch_dataset["batch_data_onehot"]
            batch_data_blosum_norm = batch_dataset["batch_data_blosum_norm"]
            batch_mask = batch_dataset["batch_mask"]
            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda() #TODO: Automatize for any kind of input (blosum encoding, integers, one-hot)
                batch_data_int = batch_data_int.cuda()
                batch_data_onehot = batch_data_onehot.cuda()
                batch_data_blosum_norm = batch_data_blosum_norm.cuda()
                batch_mask = batch_mask.cuda()
            true_labels = batch_data_blosum[:,0,0,0]
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,"norm":batch_data_blosum_norm}
            loss = svi.step(batch_data,batch_mask)
            guide_estimates = guide(batch_data,batch_mask)
            sampling_out = Vegvisir.sample(batch_data,batch_mask,guide_estimates)
            predicted_labels = sampling_out.predicted_labels
            latent_space = sampling_out.latent_space
            latent_spaces.append(latent_space.detach().cpu().numpy())
            total += true_labels.size(0)
            correct += (predicted_labels == true_labels).sum().item()
            predictions.append(predicted_labels.cpu().numpy())
            valid_loss += loss #TODO: Multiply by the data size?
    valid_loss /= len(data_loader)
    predictions_arr = np.concatenate(predictions,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    accuracy= 100 * correct // total
    return valid_loss,accuracy,predictions_arr,latent_arr
def test_loop(Vegvisir,guide,data_loader,args):
    Vegvisir.train(False)
    correct = 0
    total = 0
    latent_spaces = []
    predictions = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_data_int = batch_dataset["batch_data_int"]
            batch_data_onehot = batch_dataset["batch_data_onehot"]
            batch_data_blosum_norm = batch_dataset["batch_data_blosum_norm"]
            batch_mask = batch_dataset["batch_mask"]
            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda()
                batch_data_int = batch_data_int.cuda()
                batch_data_onehot = batch_data_onehot.cuda()
                batch_data_blosum_norm = batch_data_blosum_norm.cuda()
                batch_mask = batch_mask.cuda()
            true_labels = batch_data_blosum[:,0,0,0]
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,"norm":batch_data_blosum_norm}
            guide_estimates = guide(batch_data,batch_mask)
            sampling_out = Vegvisir.sample(batch_data,batch_mask,guide_estimates)
            predicted_labels = sampling_out.predicted_labels
            latent_space = sampling_out.latent_space
            latent_spaces.append(latent_space.detach().cpu().numpy())
            total += true_labels.size(0)
            correct += (predicted_labels == true_labels).sum().item()
            predictions.append(predicted_labels.cpu().numpy())

    predictions_arr = np.concatenate(predictions,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    accuracy = 100 * correct // total
    print(f'Accuracy of the TCR-pMHC: {100 * correct // total} %')

    return predictions_arr,accuracy,latent_arr
def save_script(results_dir,output_name,script_name):
    """Saves the python script and its contents"""
    out_file = open("{}/{}.py".format(results_dir,output_name), "a+")
    script_file = open("{}/{}.py".format(os.path.dirname(vegvisir.__file__),script_name), "r+")
    text = script_file.readlines()
    out_file.write("".join(text))
    out_file.close()
def select_quide(Vegvisir,model_load,n_data,choice="autodelta"):
    """Select the guide type
    :param nn.module Vegvisir
    :param namedtuple model_load
    :param str choice: guide name"""

    print("Using {} as guide".format(choice))
    #guide = GleipnirGuides.GLEIPNIRGUIDES(Gleipnir.model,model_load,Gleipnir)
    guide = {"autodelta":AutoDelta(Vegvisir.model),
             "autonormal":AutoNormal(Vegvisir.model,init_scale=0.1),
             "custom":VegvisirGuides.VEGVISIRGUIDES(Vegvisir.model,model_load,Vegvisir)}
    return guide[choice]
    #return poutine.scale(guide[choice],scale=1.0/n_data) #Scale the ELBo to the data size
def select_model(model_load,results_dir,fold):
    """Select among the available models at models.py"""
    if model_load.seq_max_len == model_load.max_len:
        vegvisir_model = VegvisirModels.VegvisirModel5a(model_load)
    else:
        vegvisir_model = VegvisirModels.VegvisirModel5b(model_load)
    if fold == 0:
        text_file = open("{}/Hyperparameters.txt".format(results_dir), "a")
        text_file.write("Model Class:  {} \n".format(vegvisir_model.get_class()))
        text_file.close()
        save_script(results_dir, "ModelFunction", "models")
        save_script(results_dir, "ModelUtilsFunction", "model_utils")
    #Initialize the weights
    with torch.no_grad():
        vegvisir_model.apply(init_weights)
    return vegvisir_model
def config_build(args,results_dir):
    """Select a default configuration dictionary. It can load a string dictionary from the command line (using json) or use the default parameters
    :param namedtuple args"""
    # if args.parameter_search:
    #     config = json.loads(args.config_dict)
    # else:
    "Default hyperparameters (Clipped Adam optimizer), z dim and GRU"
    config = {
        "lr": 1e-3, #default is 1e-3
        "beta1": 0.95, #coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        "beta2": 0.999,
        "eps": 1e-8,#term added to the denominator to improve numerical stability (default: 1e-8)
        "weight_decay": 0,#weight_decay: weight decay (L2 penalty) (default: 0)
        "clip_norm": 10,#clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
        "lrd": 1,#0.1 ** (1 / args.num_epochs), #rate at which learning rate decays (default: 1.0) #https://pyro.ai/examples/svi_part_iv.html
        "z_dim": 30,
        "gru_hidden_dim": 60, #60
        "momentum":0.9
    }
    json.dump(config, open('{}/params_dict.txt'.format(results_dir), 'w'), indent=2)

    return config
def init_weights(m):
    """Xavier or Glorot parameter initialization is meant to be used with Tahn activation
    kaiming or He parameter initialization is for ReLU activation
    nn.Linear is initialized with kaiming_uniform by default
    Notes:
        -https://shiyan.medium.com/xavier-initialization-and-batch-normalization-my-understanding-b5b91268c25c
        -https://medium.com/ml-cheat-sheet/how-to-avoid-the-vanishing-exploding-gradients-problem-f9ccb4446c5a
    """
    if isinstance(m, nn.Module) and hasattr(m, 'weight') and not isinstance(m,nn.BatchNorm1d):
        nn.init.kaiming_normal_(m.weight,nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def clip_backprop(model, clip_value):
    "Norm Clip the gradients of the model parameters to orient them towards the minimum"
    handles = []
    for p in model.parameters():
        if p.requires_grad:
            func = lambda grad: torch.clamp(grad,
                                            -clip_value,
                                            clip_value)
            handle = p.register_hook(func)
            handles.append(handle)
    return handles
def fold_auc(predictions_fold,labels,accuracy,fold,results_dir,mode="Train"):
    if isinstance(labels,torch.Tensor):
        labels = labels.numpy()
    # total_predictions = np.column_stack(predictions_fold)
    # model_predictions = stats.mode(total_predictions, axis=1) #mode_predictions.mode
    auc_score = roc_auc_score(y_true=labels, y_score=predictions_fold)
    auk_score = VegvisirUtils.AUK(predictions_fold, labels).calculate_auk()
    fpr, tpr, threshold = roc_curve(y_true=labels, y_score=predictions_fold)
    VegvisirPlots.plot_ROC_curve(fpr,tpr,auc_score,auk_score,"{}/{}".format(results_dir,mode),fold)
    print("Fold : {}, {} AUC score : {}, AUK score {}".format(fold,mode, auc_score,auk_score))
    print("Fold : {}, {} AUC score : {}, AUK score {}".format(fold,mode, auc_score,auk_score),file=open("{}/AUC_out.txt".format(results_dir),"a"))
    tn, fp, fn, tp = confusion_matrix(y_true=labels, y_pred=predictions_fold).ravel()
    confusion_matrix_df = pd.DataFrame([[tn, fp], [fn, tp]],
                                    columns=["Negative", "Positive"],
                                    index=["Negative", "Positive"])
    VegvisirPlots.plot_confusion_matrix(confusion_matrix_df,accuracy,"{}/{}".format(results_dir,mode))
    return auc_score,auk_score
def dataset_proportions(data,results_dir,type="TrainEval"):
    """Calculates distribution of data points based on their labeling"""
    if isinstance(data,np.ndarray):
        data = torch.from_numpy(data)
    if data.ndim == 4:
        positives = torch.sum(data[:,0,0,0])
    else:
        positives = torch.sum(data[:,0,0])
    positives_proportion = (positives*100)/torch.tensor([data.shape[0]])
    negatives = data.shape[0] - positives
    negatives_proportion = 100-positives_proportion
    print("{} dataset: \n \t Total number of data points: {} \n \t Number positives : {}; \n \t Proportion positives : {} ; \n \t Number negatives : {} ; \n \t Proportion negatives : {}".format(type,data.shape[0],positives,positives_proportion.item(),negatives,negatives_proportion.item()))
    return (positives,positives_proportion),(negatives,negatives_proportion)
def trainevaltest_split_kfolds(data,args,results_dir,method="predefined_partitions"):
    """Perform kfolds and test split"""
    if method == "predefined_partitions":
        traineval_data, test_data = data[data[:, 0,0, 3] == 1.], data[data[:, 0,0, 3] == 0.]
        dataset_proportions(traineval_data, results_dir)
        dataset_proportions(test_data, results_dir, type="Test")
        partitions = traineval_data[:, 0,0, 2]
        unique_partitions = np.unique(partitions)
        assert args.kfolds <= len(unique_partitions), "kfold number is too high, please select a number lower than {}".format(len(unique_partitions))
        i = 1
        kfolds = []
        for part_num in unique_partitions:
            # train_idx = traineval_data[traineval_data[:,0,2] != part_num]
            train_idx = (traineval_data[:, 0,0, 2][..., None] != part_num).any(-1)
            valid_idx = (traineval_data[:, 0,0, 2][..., None] == part_num).any(-1)
            kfolds.append((train_idx, valid_idx))
            if args.k_folds <= i :
                break
            else:
                i+=1
        return traineval_data, test_data, kfolds
    elif method == "stratified_group_partitions":
        traineval_data,test_data = data[data[:,0,0,3] == 1.], data[data[:,0,0,3] == 0.]
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir,type="Test")
        kfolds = StratifiedGroupKFold(n_splits=args.k_folds).split(traineval_data, traineval_data[:,0,0,0], traineval_data[:,0,0,2])
        return traineval_data,test_data,kfolds
    elif method == "random_stratified":
        data_labels = data[:,0,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir, type="Test")
        kfolds = StratifiedShuffleSplit(n_splits=args.k_folds, random_state=13, test_size=0.2).split(traineval_data,traineval_data[:,0,0,0])
        return traineval_data,test_data,kfolds
    elif method == "discard_test":
        """Discard the test dataset (hard case) and use one of the partitions as the test instead. The rest of the dataset is used for the kfold partitions"""
        partition_idx = np.random.randint(0, 4)  # random selection of a partition as the test
        train_data = data[data[:, 0, 0, 2] != partition_idx]
        test_data = data[data[:, 0, 0, 2] == partition_idx]  # data[data[:, 0, 0, 3] == 1.],
        dataset_proportions(train_data, results_dir)
        dataset_proportions(test_data, results_dir, type="Test")
        partitions = train_data[:, 0, 0, 2]
        unique_partitions = np.unique(partitions)
        assert args.kfolds <= len(unique_partitions), "kfold number is too high, please select a number lower than {}".format(len(unique_partitions))
        i = 1
        kfolds = []
        for part_num in unique_partitions:
            # train_idx = traineval_data[traineval_data[:,0,2] != part_num]
            train_idx = (train_data[:, 0, 0, 2][..., None] != part_num).any(-1)
            valid_idx = (train_data[:, 0, 0, 2][..., None] == part_num).any(-1)
            kfolds.append((train_idx, valid_idx))
            if args.k_folds <= i:
                break
            else:
                i += 1
        return train_data, test_data, kfolds

    else:
        raise ValueError("train test split method not available")
def trainevaltest_split(data,args,results_dir,method="predefined_partitions"):
    """Perform train-valid-test split"""
    info_file = open("{}/dataset_info.txt".format(results_dir),"a+")
    if method == "random_stratified":
        data_labels = data[:,0,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(data, test_size=0.1, random_state=13, stratify=traineval_labels,shuffle=True)
        dataset_proportions(train_data,results_dir, type="Train")
        dataset_proportions(valid_data,results_dir, type="Valid")
        dataset_proportions(test_data,results_dir, type="Test")
        return train_data, valid_data, test_data
    elif method == "random_stratified_discard_test":
        """Discard the predefined test dataset"""
        data = data[data[:,0,0,3] == 1] #pick only the pre assigned training data
        data_labels = data[:,0,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(traineval_data, test_size=0.1, random_state=13, stratify=traineval_labels,shuffle=True)
        dataset_proportions(train_data,results_dir, type="Train")
        dataset_proportions(valid_data,results_dir, type="Valid")
        dataset_proportions(test_data,results_dir, type="Test")
        return train_data,valid_data,test_data

    elif method == "predefined_partitions_discard_test":
        """Discard the test dataset (because it seems a hard case) and use one of the partitions as the test instead. 
        The rest of the dataset is used for the training, and a portion for validation"""
        partition_idx = np.random.randint(0,4) #random selection of a partition as the test
        traineval_data = data[data[:, 0, 0, 2] != partition_idx]
        test_data = data[data[:, 0, 0, 2] == partition_idx] #data[data[:, 0, 0, 3] == 1.]
        traineval_labels = traineval_data[:,0,0,0]
        train_data, valid_data = train_test_split(traineval_data, test_size=0.1, random_state=13,stratify=traineval_labels, shuffle=True)
        dataset_proportions(train_data, results_dir, type="Train")
        dataset_proportions(valid_data, results_dir, type="Valid")
        dataset_proportions(test_data, results_dir, type="Test")
        info_file.write("-------------------------------------------------")
        info_file.write("Using partition {} as test:".format(partition_idx))

        return train_data, valid_data, test_data

    else:
        raise ValueError("train test split method not available")
def kfold_crossvalidation(dataset_info,additional_info,args):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_blosum_norm = dataset_info.data_array_blosum_norm

    seq_max_len = dataset_info.seq_max_len

    n_data = data_blosum.shape[0]
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
    results_dir = additional_info.results_dir
    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    #TODO: Detect and correct batch_size automatically?
    #Highlight: Train- Test split and kfold generator
    #TODO: Develop method to partition sequences, sequences in train and test must differ. Partitions must have similar distributions (Tree based on distance matrix?
    # In the loop computer another cosine similarity among the vectors of cos sim of each sequence?)
    traineval_data_blosum,test_data_blosum,kfolds = trainevaltest_split_kfolds(data_blosum,args,results_dir,method="predefined_partitions")

    #Highlight:Also split the rest of arrays
    traineval_idx = (data_blosum[:,0,0,1][..., None] == traineval_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    traineval_mask = data_array_blosum_encoding_mask[traineval_idx]
    test_mask = data_array_blosum_encoding_mask[~traineval_idx]
    traineval_data_int = data_int[traineval_idx]
    test_data_int = data_int[~traineval_idx]
    traineval_data_onehot = data_onehot[traineval_idx]
    test_data_onehot = data_onehot[~traineval_idx]
    traineval_data_norm = data_blosum_norm[traineval_idx]
    test_data_norm = data_blosum_norm[~traineval_idx]
    #Split the rest of the data (train_data) for train and validation
    batch_size = args.batch_size
    check_point_epoch = [5 if args.num_epochs < 100 else int(args.num_epochs / 50)][0]
    model_load = ModelLoad(args=args,
                           max_len =dataset_info.max_len,
                           seq_max_len=seq_max_len,
                           n_data = dataset_info.n_data,
                           input_dim = dataset_info.input_dim,
                           aa_types = dataset_info.corrected_aa_types,
                           blosum = dataset_info.blosum)

    valid_predictions_fold = None
    train_predictions_fold = None
    valid_accuracy = None
    train_accuracy=None
    for fold, (train_idx, valid_idx) in enumerate(kfolds): #returns k-splits for train and validation

        # #Highlight: Minmax scale the confidence scores #TODO: function or for loop?
        fold_train_data_blosum = traineval_data_blosum[train_idx]
        fold_train_data_int = traineval_data_int[train_idx]
        fold_train_data_onehot = traineval_data_onehot[train_idx]
        # #Highlight: valid
        fold_valid_data_blosum = traineval_data_blosum[valid_idx]
        fold_valid_data_int = traineval_data_int[valid_idx]
        fold_valid_data_onehot = traineval_data_onehot[valid_idx]
        print("---------------------------------------------------------------------")
        print('Fold number : {}'.format(fold))
        print('\t Number train data points: {}; Proportion: {}'.format(fold_train_data_blosum.shape[0],(fold_train_data_blosum.shape[0]*100)/traineval_data_blosum.shape[0]))
        print('\t Number valid data points: {}; Proportion: {}'.format(fold_valid_data_blosum.shape[0],(fold_valid_data_blosum.shape[0]*100)/traineval_data_blosum.shape[0]))

        custom_dataset_train = VegvisirLoadUtils.CustomDataset(fold_train_data_blosum,
                                                               fold_train_data_int,
                                                               fold_train_data_onehot,
                                                               traineval_data_norm[train_idx],
                                                                traineval_mask[train_idx])
        custom_dataset_valid = VegvisirLoadUtils.CustomDataset(fold_valid_data_blosum,
                                                               fold_valid_data_int,
                                                               fold_valid_data_onehot,
                                                               traineval_data_norm[valid_idx],
                                                               traineval_mask[valid_idx])
        train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
        valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device),**kwargs)


        #Restart the model each fold
        Vegvisir = select_model(model_load, additional_info.results_dir,fold)
        params_config = config_build(args,results_dir)
        if args.optimizer_name == "Adam":
            adam_args = {"lr":params_config["lr"],
                        "betas": (params_config["beta1"], params_config["beta2"]),
                        "eps": params_config["eps"],
                        "weight_decay": params_config["weight_decay"]}
            optimizer = pyro.optim.Adam(adam_args)
        elif args.optimizer_name == "ClippedAdam":
            clippedadam_args = {"lr": params_config["lr"],
                            "betas": (params_config["beta1"], params_config["beta2"]),
                            "eps": params_config["eps"],
                            "weight_decay": params_config["weight_decay"],
                            "clip_norm": params_config["clip_norm"],
                            "lrd": params_config["lrd"]}
            optimizer = pyro.optim.ClippedAdam(clippedadam_args)
        else:
            raise ValueError("selected optimizer {} not implemented".format(args.optimizer_name))
        loss_func = Vegvisir.loss()
        guide = select_quide(Vegvisir,model_load,n_data,args.guide)
        #svi = SVI(poutine.scale(Vegvisir.model,scale=1.0/n_data), guide, optimizer, loss_func)
        svi = SVI(Vegvisir.model, guide, optimizer, loss_func)

        #TODO: Dictionary that gathers the results from each fold
        train_loss = []
        valid_loss = []
        epochs_list = []
        train_accuracies = []
        valid_accuracies= []
        train_auc = []
        valid_auc = []
        train_auk = []
        valid_auk = []
        epoch = 0.
        gradient_norms = defaultdict(list)
        while epoch <= args.num_epochs:
            start = time.time()
            #svi,Vegvisir,guide,data_loader, args
            train_epoch_loss,train_accuracy,train_predictions = train_loop(svi,Vegvisir,guide, train_loader, args)
            stop = time.time()
            memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
            print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
            train_loss.append(train_epoch_loss)
            train_accuracies.append(train_accuracy)
            if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
                for name_i, value in pyro.get_param_store().named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                    value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
                valid_epoch_loss,valid_accuracy,valid_predictions = valid_loop(svi,Vegvisir,guide, valid_loader, args)
                valid_loss.append(valid_epoch_loss)
                epochs_list.append(epoch)
                valid_accuracies.append(valid_accuracy)
                train_auc_score = roc_auc_score(y_true=fold_train_data_blosum[:,0,0,0], y_score=train_predictions)
                train_auk_score = VegvisirUtils.AUK(probabilities= train_predictions,labels=fold_train_data_blosum[:,0,0,0].numpy()).calculate_auk()
                train_auk.append(train_auk_score)
                train_auc.append(train_auc_score)
                valid_auc_score = roc_auc_score(y_true=fold_valid_data_blosum[:,0,0,0], y_score=valid_predictions)
                valid_auk_score = VegvisirUtils.AUK(probabilities= valid_predictions,labels = fold_valid_data_blosum[:,0,0,0].numpy()).calculate_auk()
                valid_auk.append(valid_auk_score)
                valid_auc.append(valid_auc_score)
                VegvisirPlots.plot_ELBO(train_loss,valid_loss,epochs_list,fold,additional_info.results_dir)
                VegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,fold,additional_info.results_dir)
                VegvisirPlots.plot_classification_score(train_auc,valid_auc,epochs_list,fold,additional_info.results_dir,method="AUC")
                VegvisirPlots.plot_classification_score(train_auk,valid_auk,epochs_list,fold,additional_info.results_dir,method="AUK")
                Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)
                if epoch == args.num_epochs:
                    print("Saving final results")
                    train_predictions_fold = train_predictions
                    valid_predictions_fold = valid_predictions
                    VegvisirPlots.plot_gradients(gradient_norms, results_dir, fold)
                    Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)
                    # params = vegvisir_model.capture_parameters([name for name,val in vegvisir_model.named_parameters()])
                    # gradients = vegvisir_model.capture_gradients([name for name,val in vegvisir_model.named_parameters()])
                    # activations = vegvisir_model.attach_hooks([name for name,val in vegvisir_model.named_parameters() if name.starstwith("a")])

            torch.cuda.empty_cache()
            epoch += 1 #TODO: early stop?
        #predictions_fold,labels,accuracy,fold,results_dir
        fold_auc(valid_predictions_fold,fold_valid_data_blosum[:,0,0,0],valid_accuracy,fold,results_dir,mode="Valid")
        fold_auc(train_predictions_fold,fold_train_data_blosum[:,0,0,0],train_accuracy,fold,results_dir,mode="Train")


    if args.test: #TODO: Function for training
        print("Final training & testing")
        custom_dataset_train = VegvisirLoadUtils.CustomDataset(traineval_data_blosum,
                                                               traineval_data_int,
                                                               traineval_data_onehot,
                                                               traineval_data_norm,
                                                               traineval_mask)
        train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))

        Vegvisir = select_model(model_load, additional_info.results_dir,fold=0)
        params_config = config_build(args, results_dir)
        if args.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(Vegvisir.parameters(), lr=params_config["lr"],
                                         betas=(params_config["beta1"], params_config["beta2"]),
                                         eps=params_config["eps"], weight_decay=params_config["weight_decay"])
        elif args.optimizer_name == "ClippedAdam":
            clippedadam_args = {"lr": params_config["lr"],
                                "betas": (params_config["beta1"], params_config["beta2"]),
                                "eps": params_config["eps"],
                                "weight_decay": params_config["weight_decay"],
                                "clip_norm": params_config["clip_norm"],
                                "lrd": params_config["lrd"]}
            optimizer = pyro.optim.ClippedAdam(clippedadam_args)
        else:
            raise ValueError("selected optimizer {} not implemented".format(args.optimizer_name))
        loss_func = Vegvisir.loss
        guide = select_quide(Vegvisir, model_load, args.guide)
        svi = SVI(Vegvisir.model, guide, optimizer, loss_func)
        train_loss = []
        epochs_list = []
        train_accuracies = []
        train_auc = []
        train_auk = []
        epoch = 0.
        gradient_norms = defaultdict(list)
        while epoch <= args.num_epochs:
            start = time.time()
            # svi,Vegvisir,guide,data_loader, args
            train_epoch_loss, train_accuracy, train_predictions = train_loop(svi, Vegvisir, guide, train_loader, args)
            stop = time.time()
            memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
            print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (
            epoch, train_epoch_loss, stop - start, memory_usage_mib))
            train_loss.append(train_epoch_loss)
            train_accuracies.append(train_accuracy)
            if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs:
                for name_i, value in pyro.get_param_store().named_parameters():  # TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                    value.register_hook(
                        lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
                epochs_list.append(epoch)
                train_auc_score = roc_auc_score(y_true=traineval_data_blosum[:, 0, 0, 0], y_score=train_predictions)
                train_auk_score = VegvisirUtils.AUK(probabilities=train_predictions,
                                                    labels=traineval_data_blosum[:, 0, 0, 0].numpy()).calculate_auk()
                train_auk.append(train_auk_score)
                train_auc.append(train_auc_score)
                VegvisirPlots.plot_ELBO(train_loss, None, epochs_list, "final", additional_info.results_dir)
                VegvisirPlots.plot_accuracy(train_accuracies, None, epochs_list, "final",additional_info.results_dir)
                VegvisirPlots.plot_classification_score(train_auc, None, epochs_list, "final",additional_info.results_dir, method="AUC")
                VegvisirPlots.plot_classification_score(train_auk, None, epochs_list, "final",additional_info.results_dir, method="AUK")
                if epoch == args.num_epochs:
                    print("Saving final results")
                    train_predictions_fold = train_predictions
                    VegvisirPlots.plot_gradients(gradient_norms, results_dir, "final")
                    Vegvisir.save_checkpoint("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer)
            epoch += 1
            torch.cuda.empty_cache()
        fold_auc(train_predictions_fold,traineval_data_blosum[:,0,0,0],train_accuracy,"final",results_dir,mode="Train")
        #Highlight: Testing
        test_data_blosum[:,0,0,4] = VegvisirUtils.minmax_scale(test_data_blosum[:,0,0,4])
        test_data_int[:,0,4] = test_data_blosum[:,0,0,4]
        test_data_onehot[:,0,0,4] = test_data_blosum[:,0,0,4]
        custom_dataset_test = VegvisirLoadUtils.CustomDataset(test_data_blosum,
                                                              test_data_int,
                                                              test_data_onehot,
                                                              test_data_norm,
                                                              test_mask)
        test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True,
                                 generator=torch.Generator(device=args.device), **kwargs)
        predictions = test_loop(Vegvisir,guide,test_loader,args)
        score = roc_auc_score(y_true=test_data_blosum[:, 0, 0, 0].numpy(), y_score=predictions)
        print("Final AUC score : {}".format( score))
def train_model(dataset_info,additional_info,args):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_blosum_norm = dataset_info.data_array_blosum_norm
    seq_max_len = dataset_info.seq_max_len
    n_data = data_blosum.shape[0]
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
    results_dir = additional_info.results_dir
    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    #TODO: Detect and correct batch_size automatically?
    #Highlight: Train- Test split and kfold generator
    #TODO: Develop method to partition sequences, sequences in train and test must differ. Partitions must have similar distributions (Tree based on distance matrix?
    # In the loop computer another cosine similarity among the vectors of cos sim of each sequence?)
    train_data_blosum,valid_data_blosum,test_data_blosum = trainevaltest_split(data_blosum,args,results_dir,method="predefined_partitions_discard_test")

    #Highlight:Also split the rest of arrays
    train_idx = (data_blosum[:,0,0,1][..., None] == train_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    valid_idx = (data_blosum[:,0,0,1][..., None] == valid_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    test_idx = (data_blosum[:,0,0,1][..., None] == test_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not

    #Split the rest of the data (train_data) for train and validation
    batch_size = args.batch_size
    check_point_epoch = [5 if args.num_epochs < 100 else int(args.num_epochs / 50)][0]
    model_load = ModelLoad(args=args,
                           max_len =dataset_info.max_len,
                           seq_max_len= seq_max_len,
                           n_data = dataset_info.n_data,
                           input_dim = dataset_info.input_dim,
                           aa_types = dataset_info.corrected_aa_types,
                           blosum = dataset_info.blosum)

    print('\t Number train data points: {}; Proportion: {}'.format(train_data_blosum.shape[0],(train_data_blosum.shape[0]*100)/train_data_blosum.shape[0]))
    print('\t Number eval data points: {}; Proportion: {}'.format(valid_data_blosum.shape[0],(valid_data_blosum.shape[0]*100)/valid_data_blosum.shape[0]))

    custom_dataset_train = VegvisirLoadUtils.CustomDataset(train_data_blosum,
                                                           data_int[train_idx],
                                                           data_onehot[train_idx],
                                                           data_blosum_norm[train_idx],
                                                           data_array_blosum_encoding_mask[train_idx])
    custom_dataset_valid = VegvisirLoadUtils.CustomDataset(data_blosum[valid_idx],
                                                           data_int[valid_idx],
                                                           data_onehot[valid_idx],
                                                           data_blosum_norm[valid_idx],
                                                           data_array_blosum_encoding_mask[valid_idx])

    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))

    #Restart the model each fold
    Vegvisir = select_model(model_load, additional_info.results_dir,"all")
    params_config = config_build(args,results_dir)
    if args.optimizer_name == "Adam":
        adam_args = {"lr":params_config["lr"],
                    "betas": (params_config["beta1"], params_config["beta2"]),
                    "eps": params_config["eps"],
                    "weight_decay": params_config["weight_decay"]}
        optimizer = pyro.optim.Adam(adam_args)
    elif args.optimizer_name == "ClippedAdam":
        clippedadam_args = {"lr": params_config["lr"],
                        "betas": (params_config["beta1"], params_config["beta2"]),
                        "eps": params_config["eps"],
                        "weight_decay": params_config["weight_decay"],
                        "clip_norm": params_config["clip_norm"],
                        "lrd": params_config["lrd"]}
        optimizer = pyro.optim.ClippedAdam(clippedadam_args)
    else:
        raise ValueError("selected optimizer {} not implemented".format(args.optimizer_name))
    loss_func = Vegvisir.loss()
    guide = select_quide(Vegvisir,model_load,n_data,args.guide)
    svi = SVI(poutine.scale(Vegvisir.model,scale=1.0/n_data), guide, optimizer, loss_func)

    #TODO: Dictionary that gathers the results from each fold
    start = time.time()
    epochs_list = []
    train_loss = []
    train_accuracies = []
    train_auc = []
    train_auk = []
    valid_loss = []
    valid_accuracies = []
    valid_auc = []
    valid_auk = []
    epoch = 0.
    train_predictions = None
    valid_predictions = None
    train_accuracy = None
    valid_accuracy = None
    gradient_norms = defaultdict(list)
    while epoch <= args.num_epochs:
        start = time.time()
        #svi,Vegvisir,guide,data_loader, args
        train_epoch_loss,train_accuracy,train_predictions, train_latent_space = train_loop(svi,Vegvisir,guide, train_loader, args)
        stop = time.time()
        memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
        train_loss.append(train_epoch_loss)
        train_accuracies.append(train_accuracy)
        if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
            for name_i, value in pyro.get_param_store().named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
            epochs_list.append(epoch)
            valid_epoch_loss, valid_accuracy, valid_predictions, valid_latent_space = valid_loop(svi, Vegvisir, guide, valid_loader, args)
            valid_loss.append(valid_epoch_loss)
            valid_accuracies.append(valid_accuracy)
            train_auc_score = roc_auc_score(y_true=train_data_blosum[:,0,0,0], y_score=train_predictions)
            train_auk_score = VegvisirUtils.AUK(probabilities= train_predictions,labels=train_data_blosum[:,0,0,0].numpy()).calculate_auk()
            train_auk.append(train_auk_score)
            train_auc.append(train_auc_score)

            valid_auc_score = roc_auc_score(y_true=valid_data_blosum[:, 0, 0, 0], y_score=valid_predictions)
            valid_auk_score = VegvisirUtils.AUK(probabilities=valid_predictions,
                                                labels=valid_data_blosum[:, 0, 0, 0].numpy()).calculate_auk()
            valid_auk.append(valid_auk_score)
            valid_auc.append(valid_auc_score)

            VegvisirPlots.plot_ELBO(train_loss,valid_loss,epochs_list,"all",additional_info.results_dir)
            VegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,"all",additional_info.results_dir)
            VegvisirPlots.plot_classification_score(train_auc,valid_auc,epochs_list,"all",additional_info.results_dir,method="AUC")
            VegvisirPlots.plot_classification_score(train_auk,valid_auk,epochs_list,"all",additional_info.results_dir,method="AUK")
            Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer)
            if epoch == args.num_epochs:
                print("Saving final results")
                train_predictions = train_predictions
                valid_predictions = valid_predictions
                VegvisirPlots.plot_gradients(gradient_norms, results_dir, "all")
                VegvisirPlots.plot_latent_space(train_latent_space, train_predictions, "all",results_dir, method="Train")
                VegvisirPlots.plot_latent_space(valid_latent_space,valid_predictions, "all",results_dir, method="Valid")
                Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)
                # params = vegvisir_model.capture_parameters([name for name,val in vegvisir_model.named_parameters()])
                # gradients = vegvisir_model.capture_gradients([name for name,val in vegvisir_model.named_parameters()])
                # activations = vegvisir_model.attach_hooks([name for name,val in vegvisir_model.named_parameters() if name.starstwith("a")])

        torch.cuda.empty_cache()
        epoch += 1 #TODO: early stop?
    fold_auc(train_predictions,train_data_blosum[:,0,0,0],train_accuracy,"all",results_dir,mode="Train")
    fold_auc(valid_predictions,valid_data_blosum[:,0,0,0],valid_accuracy,"all",results_dir,mode="Valid")
    stop = time.time()
    print('Final timing: {}'.format(str(datetime.timedelta(seconds=stop-start))))


    if args.test: #TODO: Function for training
        print("Final testing")

        custom_dataset_test = VegvisirLoadUtils.CustomDataset(data_blosum[test_idx],
                                                              data_int[test_idx],
                                                              data_onehot[test_idx],
                                                              data_blosum_norm[test_idx],
                                                              data_array_blosum_encoding_mask[test_idx])
        test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True,
                                 generator=torch.Generator(device=args.device), **kwargs)
        test_predictions,test_accuracy,test_latent_space = test_loop(Vegvisir,guide,test_loader,args)
        fold_auc(test_predictions, test_data_blosum[:, 0, 0, 0], test_accuracy,"all", results_dir, mode="Test")


