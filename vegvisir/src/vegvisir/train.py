import time

import numpy as np
import pyro.optim
from scipy import stats
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold
from sklearn.metrics import auc,roc_auc_score,cohen_kappa_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import vegvisir.utils as vegvisirUtils
import vegvisir.load_utils as vegvisirLoadUtils
import vegvisir.plots as vegvisirPlots
import vegvisir.models as vegvisirModels
import vegvisir.model_utils as vegvisirModelUtils
ModelLoad = namedtuple("ModelLoad",["args","max_len","n_sequences","input_dim"])


def train_loop(model,loss_func,optimizer, data_loader, args):
    """Regular batch training
    :param nn.Module model:
    :param loss_func: loss function
    :param optimizer: gradient descent method
    :param DataLoader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    model.train() #I think this is useless?
    train_loss = 0.0
    correct = 0
    total = 0
    predictions = []
    for batch_number, batch_dataset in enumerate(data_loader):
        batch_data_blosum = batch_dataset["batch_data_blosum"]
        batch_mask = batch_dataset["batch_mask"]
        if args.use_cuda:
            batch_data_blosum = batch_data_blosum.cuda()
            batch_mask = batch_mask.cuda()
        true_labels = batch_data_blosum[:,0,0,0]
        optimizer.zero_grad() #set gradients to zero
        #Forward pass
        model_outputs = model(batch_data_blosum,batch_mask)
        _, predicted_labels = torch.max(model_outputs,dim= 1) #find the class with highest energy
        total += true_labels.size(0)
        correct += (predicted_labels == true_labels).sum().item()
        predictions.append(predicted_labels.detach().cpu().numpy())
        loss = loss_func(model_outputs,true_labels,batch_mask) #TODO: Change to logits?
        #loss.requires_grad = True #not sure why it does not do this automatically for the nnl loss
        #Backward pass: backpropagate the error loss
        loss.backward()#retain_graph=True
        optimizer.step()
        train_loss += loss.item()
    #TODO: normalize loss?
    predictions_arr = np.concatenate(predictions,axis=0)
    accuracy= 100 * correct // total
    return train_loss,accuracy,predictions_arr
def valid_loop(model,loss_func, data_loader, args):
    """Regular batch training
    :param svi: pyro infer engine
    :param dataloader train_loader: Pytorch dataloader
    :param namedtuple args
    """
    model.eval() #TODO: necessary?
    valid_loss = 0.0
    total = 0.
    correct = 0.
    predictions = []
    for batch_number, batch_dataset in enumerate(data_loader):
        batch_data_blosum = batch_dataset["batch_data_blosum"]
        batch_mask = batch_dataset["batch_mask"]
        if args.use_cuda:
            batch_data_blosum = batch_data_blosum.cuda()
            batch_mask = batch_mask.cuda()
        true_labels = batch_data_blosum[:,0,0,0] #TODO: Automatize for any kind of input (blosum encoding, integers, one-hot)
        model_outputs = model(batch_data_blosum,batch_mask)
        _, predicted_labels = torch.max(model_outputs,dim= 1) #find the class with highest energy
        total += true_labels.size(0)
        correct += (predicted_labels == true_labels).sum().item()
        predictions.append(predicted_labels.cpu().numpy())
        loss = loss_func(model_outputs,true_labels,batch_mask)
        valid_loss += loss.item() #TODO: Multiply by the data size?
    #TODO: normalize loss?
    predictions_arr = np.concatenate(predictions,axis=0)
    accuracy= 100 * correct // total
    return valid_loss,accuracy,predictions_arr
def test_loop(data_loader,model,args):
    correct = 0
    total = 0
    predictions = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data = batch_dataset["batch_data_blosum"]
            batch_mask = batch_dataset["batch_mask"]
            if args.use_cuda:
                batch_data = batch_data.cuda()
                batch_mask = batch_mask.cuda()
            true_labels = batch_data[:,0,0,0]
            # calculate outputs by running images through the network
            outputs = model(batch_data,batch_mask).detach()
            # the class with the highest energy is what we choose as prediction
            _, predicted_labels = torch.max(outputs, dim=1)
            total += true_labels.size(0)
            correct += (predicted_labels == true_labels).sum().item()
            predictions.append(predicted_labels.cpu().numpy())

    predictions_arr = np.concatenate(predictions,axis=0)
    print(f'Accuracy of the TCR-pMHC: {100 * correct // total} %')

    return predictions_arr

def select_model(model_load,results_dir,fold):
    """Select among the available models at models.py"""
    vegvisir_model = vegvisirModels.vegvisirDiffPool(model_load)
    if fold == 0:
        text_file = open("{}/Hyperparameters.txt".format(results_dir), "a")
        text_file.write("Model Class:  {} \n".format(vegvisir_model.get_class()))
        text_file.close()
    return vegvisir_model

def config_build(args):
    """Select a default configuration dictionary. It can load a string dictionary from the command line (using json) or use the default parameters
    :param namedtuple args"""
    # if args.parameter_search:
    #     config = json.loads(args.config_dict)
    # else:
    "Default hyperparameters (Clipped Adam optimizer), z dim and GRU"
    config = {
        "lr": 1e-3,
        "beta1": 0.9, #coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        "beta2": 0.999,
        "eps": 1e-8,#term added to the denominator to improve numerical stability (default: 1e-8)
        "weight_decay": 0,#weight_decay: weight decay (L2 penalty) (default: 0)
        "clip_norm": 10,#clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
        "lrd": 1, #rate at which learning rate decays (default: 1.0)
        "z_dim": 30,
        "gru_hidden_dim": 60, #60
    }
    return config


def reset_weights(m):
    if isinstance(m, nn.RNN) or isinstance(m, nn.Linear) or isinstance(m, nn.GRU):
        m.reset_parameters()
def fold_auc(predictions_fold,labels,fold,mode="Training"):
    #TODO: Implement per peptide

    # total_predictions = np.column_stack(predictions_fold)
    # model_predictions = stats.mode(total_predictions, axis=1) #mode_predictions.mode
    auc_score = roc_auc_score(y_true=labels.numpy(), y_score=predictions_fold)
    auk_score = vegvisirUtils.AUK(predictions_fold, labels.numpy()).calculate_auk()
    print("Fold : {}, {} AUC score : {}, AUK score {}".format(fold,mode, auc_score,auk_score))
def dataset_proportions(data,type="Train",encoding="blosum_encoding"):
    """Calculates distribution of data points based on their labeling"""
    #TODO: make it work (the indexing) both for blosum encoding and else
    positives = torch.sum(data[:,0,0,0])
    positives_proportion = (positives*100)/torch.tensor([data.shape[0]])
    negatives = data.shape[0] - positives
    negatives_proportion = 100-positives_proportion
    print("{} dataset: \n \t Number positives : {}; \n \t Proportion positives : {} ; \n \t Number negatives : {} ; \n \t Proportion negatives : {}".format(type,positives,positives_proportion.item(),negatives,negatives_proportion.item()))
    return (positives,positives_proportion),(negatives,negatives_proportion)

def trainevaltest_split(data,args,method="predefined_partitions"):
    """Perform train-test split"""
    if method == "predefined_partitions":
        #Train - Test split
        traineval_data,test_data = data[data[:,0,0,3] == 1.], data[data[:,0,0,3] == 0.]
        dataset_proportions(traineval_data)
        dataset_proportions(test_data, type="Test")
        #Train - Eval split
        kfolds = StratifiedGroupKFold(n_splits=args.k_folds).split(traineval_data, traineval_data[:,0,0,0], traineval_data[:,0,0,2])
        return traineval_data,test_data,kfolds
    elif method == "random_stratified": #TODO: not reviewed
        data_labels = data[:,0,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        dataset_proportions(traineval_data)
        dataset_proportions(test_data, type="Test")
        # Train - Eval split
        kfolds = StratifiedShuffleSplit(n_splits=args.k_folds, random_state=13, test_size=0.2).split(traineval_data,traineval_data[:,0,0,0])
        return traineval_data,test_data,kfolds
    else:
        raise ValueError("train test split method not available")


def kfold_crossvalidation(dataset_info,additional_info,args):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into model...")
    data = dataset_info.data_array_blosum_encoding
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask

    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    #TODO: Detect and correct batch_size automatically?
    #Highlight: Train- Test split and kfold generator

    traineval_data,test_data,kfolds = trainevaltest_split(data,args,method="predefined_partitions")

    exit()
    #Also split the adjacency and edge matrices, according to the identifiers # Highlight. Double check again
    traineval_idx = (data[:,0,0,1][..., None] == traineval_data[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    train_mask = data_array_blosum_encoding_mask[traineval_idx]
    test_mask = data_array_blosum_encoding_mask[~traineval_idx]
    #Split the rest of the data (train_data) for train and validation
    batch_size = args.batch_size


    check_point_epoch = [5 if args.num_epochs < 100 else int(args.num_epochs / 50)][0]
    model_load = ModelLoad(args=args,
                           max_len =dataset_info.max_len,
                           n_sequences = dataset_info.n_sequences,
                           input_dim = dataset_info.input_dim)

    # vegvisir_model = select_model(model_load,additional_info.results_dir)
    # params_config = config_build(args)
    # if args.optimizer_name == "Adam":
    #     optimizer = torch.optim.Adam(vegvisir_model.parameters(), lr=params_config["lr"],betas=(params_config["beta1"], params_config["beta2"]),eps=params_config["eps"],weight_decay=params_config["weight_decay"])
    # elif args.optimizer_name == "ClippedAdam":
    #     optimizer = pyro.optim.ClippedAdam(vegvisir_model.parameters())   #TODO: Easy introduction of Clipped Adam or just use pyro?
    #
    # loss_func_dict = {"nll": nn.NLLLoss(),"softloss":vegvisir_model.softloss}
    # loss_func = loss_func_dict[args.loss_func]
    valid_predictions_fold = None
    train_predictions_fold = None
    for fold, (train_idx, valid_idx) in enumerate(kfolds): #returns k-splits for train and validation
        print("---------------------------------------------------------------------")
        print('Fold number : {}'.format(fold))
        print('\t Number train data points: {}; Proportion: {}'.format(len(train_idx),(len(train_idx)*100)/train_data.shape[0]))
        print('\t Number valid data points: {}; Proportion: {}'.format(len(valid_idx),(len(valid_idx)*100)/train_data.shape[0]))

        custom_dataset_train = vegvisirLoadUtils.CustomDataset(train_data[train_idx],
                                             train_mask[train_idx])
        custom_dataset_valid = vegvisirLoadUtils.CustomDataset(train_data[valid_idx],
                                             train_mask[valid_idx])
        fold_train_data = train_data[train_idx]
        valid_data = train_data[valid_idx]
        train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
        valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device),**kwargs)

        #Restart the model each fold
        vegvisir_model = select_model(model_load, additional_info.results_dir,fold)
        params_config = config_build(args)
        if args.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(vegvisir_model.parameters(), lr=params_config["lr"],
                                         betas=(params_config["beta1"], params_config["beta2"]),
                                         eps=params_config["eps"], weight_decay=params_config["weight_decay"])
        elif args.optimizer_name == "ClippedAdam":
            optimizer = pyro.optim.ClippedAdam(
                vegvisir_model.parameters())  # TODO: Easy introduction of Clipped Adam or just use pyro?

        loss_func_dict = {"nll": nn.NLLLoss(), "softloss": vegvisir_model.softloss}
        loss_func = loss_func_dict[args.loss_func]


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
        while epoch <= args.num_epochs: #TODO: Plot gradients? perhaps expensive
            start = time.time()
            train_epoch_loss,train_accuracy,train_predictions = train_loop(vegvisir_model, loss_func, optimizer, train_loader, args)
            stop = time.time()
            memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
            print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
            train_loss.append(train_epoch_loss)
            train_accuracies.append(train_accuracy)
            if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
                valid_epoch_loss,valid_accuracy,valid_predictions = valid_loop(vegvisir_model, loss_func, valid_loader, args)
                valid_loss.append(valid_epoch_loss)
                epochs_list.append(epoch)
                valid_accuracies.append(valid_accuracy)
                train_auc_score = roc_auc_score(y_true=fold_train_data[:,0,0,0], y_score=train_predictions)
                train_auk_score = vegvisirUtils.AUK(probabilities= train_predictions,labels=fold_train_data[:,0,0,0].numpy()).calculate_auk()
                train_auk.append(train_auk_score)
                train_auc.append(train_auc_score)
                valid_auc_score = roc_auc_score(y_true=valid_data[:,0,0,0], y_score=valid_predictions)
                valid_auk_score = vegvisirUtils.AUK(probabilities= valid_predictions,labels = valid_data[:,0,0,0].numpy()).calculate_auk()
                valid_auk.append(valid_auk_score)
                valid_auc.append(valid_auc_score)
                vegvisirPlots.plot_ELBO(train_loss,valid_loss,epochs_list,fold,additional_info.results_dir)
                vegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,fold,additional_info.results_dir)
                vegvisirPlots.plot_classification_score(train_auc,valid_auc,fold,additional_info.results_dir,method="AUC")
                vegvisirPlots.plot_classification_score(train_auk,valid_auk,fold,additional_info.results_dir,method="AUK")
                if epoch == args.num_epochs:
                    print("Saving final results")
                    train_predictions_fold = train_predictions
                    valid_predictions_fold = valid_predictions
            torch.cuda.empty_cache()
            epoch += 1 #TODO: early stop?
        fold_auc(valid_predictions_fold,valid_data[:,0,0,0],fold,mode="Validation")
        fold_auc(train_predictions_fold,fold_train_data[:,0,0,0],fold,mode="Training")
        #Highlight: Reset the parameters for each fold!!!!!!!!!---> watch out because each nn function needs to be specified manually, so if anything new is added
        #vegvisir_model.apply(fn=vegvisirModelUtils.reset_all_weights)


    if args.test:
        print("Final testing")
        custom_dataset_test = vegvisirLoadUtils.CustomDataset(test_data,
                                                              test_mask)
        test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True,
                                 generator=torch.Generator(device=args.device), **kwargs)
        predictions = test_loop(test_loader, vegvisir_model, args)
        score = roc_auc_score(y_true=test_data[:, 0, 0, 0].numpy(), y_score=predictions)
        print("Final AUC score : {}".format( score))



