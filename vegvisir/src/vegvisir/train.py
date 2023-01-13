import json
import time,os
from collections import defaultdict
import numpy as np
import pyro.optim
from scipy import stats
from sklearn.model_selection import KFold,train_test_split,StratifiedShuffleSplit,StratifiedGroupKFold
from sklearn.metrics import auc,roc_auc_score,cohen_kappa_score,roc_curve
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import vegvisir
import vegvisir.utils as VegvisirUtils
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots
import vegvisir.models as VegvisirModels
import vegvisir.model_utils as VegvisirModelUtils
ModelLoad = namedtuple("ModelLoad",["args","max_len","n_data","input_dim","aa_types","blosum"])


def train_loop(model,loss_func,optimizer, data_loader, args):
    """Regular batch training
    :param nn.Module model:
    :param loss_func: loss function
    :param optimizer: gradient descent method
    :param DataLoader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    model.train() #Highlight: look at https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
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
        confidence_scores = batch_data_blosum[:,0,0,5]
        optimizer.zero_grad() #set gradients to zero
        #Forward pass
        model_outputs = model(batch_data_blosum,batch_mask)
        _, predicted_labels = torch.max(model_outputs,dim= 1) #find the class with highest energy
        total += true_labels.size(0)
        correct += (predicted_labels == true_labels).sum().item()
        predictions.append(predicted_labels.detach().cpu().numpy())
        loss = loss_func(confidence_scores,true_labels,model_outputs)
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
    :param dataloader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    model.eval()
    valid_loss = 0.0
    total = 0.
    correct = 0.
    predictions = []
    with torch.no_grad(): #do not update parameters with the evaluation data
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_mask = batch_dataset["batch_mask"]
            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda() #TODO: Automatize for any kind of input (blosum encoding, integers, one-hot)
                batch_mask = batch_mask.cuda()
            true_labels = batch_data_blosum[:,0,0,0] #
            confidence_scores = batch_data_blosum[:, 0, 0, 5]
            model_outputs = model(batch_data_blosum,batch_mask)
            _, predicted_labels = torch.max(model_outputs,dim= 1) #find the class with highest energy
            total += true_labels.size(0)
            correct += (predicted_labels == true_labels).sum().item()
            predictions.append(predicted_labels.cpu().numpy())
            loss = loss_func(confidence_scores, true_labels, model_outputs) #NOTE: not backprogagate error since evaluating
            valid_loss += loss.item() #TODO: Multiply by the data size?
    #TODO: normalize loss?
    predictions_arr = np.concatenate(predictions,axis=0)
    accuracy= 100 * correct // total
    return valid_loss,accuracy,predictions_arr
def test_loop(data_loader,model,args):
    model.train(False)
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
def save_script(results_dir,output_name,script_name):
    """Saves the python script and its contents"""
    out_file = open("{}/{}.py".format(results_dir,output_name), "a+")
    script_file = open("{}/{}.py".format(os.path.dirname(vegvisir.__file__),script_name), "r+")
    text = script_file.readlines()
    out_file.write("".join(text))
    out_file.close()
def select_model(model_load,results_dir,fold):
    """Select among the available models at models.py"""
    vegvisir_model = VegvisirModels.VegvisirModel1(model_load)
    if fold == 0:
        text_file = open("{}/Hyperparameters.txt".format(results_dir), "a")
        text_file.write("Model Class:  {} \n".format(vegvisir_model.get_class()))
        text_file.close()
        save_script(results_dir, "ModelFunction", "models")
        save_script(results_dir, "ModelUtilsFunction", "model_utils")
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
        "beta1": 0.9, #coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        "beta2": 0.999,
        "eps": 1e-8,#term added to the denominator to improve numerical stability (default: 1e-8)
        "weight_decay": 0,#weight_decay: weight decay (L2 penalty) (default: 0)
        "clip_norm": 10,#clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
        "lrd": 1, #rate at which learning rate decays (default: 1.0)
        "z_dim": 30,
        "gru_hidden_dim": 60, #60
        "momentum":0.9
    }
    json.dump(config, open('{}/params_dict.txt'.format(results_dir), 'w'), indent=2)

    return config


def reset_weights(m):
    if isinstance(m, nn.RNN) or isinstance(m, nn.Linear) or isinstance(m, nn.GRU):
        m.reset_parameters()
def fold_auc(predictions_fold,labels,fold,results_dir,mode="Train"):
    #TODO: Implement per peptide

    # total_predictions = np.column_stack(predictions_fold)
    # model_predictions = stats.mode(total_predictions, axis=1) #mode_predictions.mode
    auc_score = roc_auc_score(y_true=labels.numpy(), y_score=predictions_fold)
    auk_score = VegvisirUtils.AUK(predictions_fold, labels.numpy()).calculate_auk()
    fpr, tpr, threshold = roc_curve(y_true=labels.numpy(), y_score=predictions_fold)
    VegvisirPlots.plot_ROC_curve(fpr,tpr,auc_score,auk_score,"{}/{}".format(results_dir,mode),fold)
    print("Fold : {}, {} AUC score : {}, AUK score {}".format(fold,mode, auc_score,auk_score))
def dataset_proportions(data,results_dir,type="TrainEval"):
    """Calculates distribution of data points based on their labeling"""
    #TODO: make it work (the indexing) both for blosum encoding and else
    positives = torch.sum(data[:,0,0,0])
    positives_proportion = (positives*100)/torch.tensor([data.shape[0]])
    negatives = data.shape[0] - positives
    negatives_proportion = 100-positives_proportion
    print("{} dataset: \n \t Total number of data points: {} \n \t Number positives : {}; \n \t Proportion positives : {} ; \n \t Number negatives : {} ; \n \t Proportion negatives : {}".format(type,data.shape[0],positives,positives_proportion.item(),negatives,negatives_proportion.item()))
    return (positives,positives_proportion),(negatives,negatives_proportion)

def trainevaltest_split(data,args,results_dir,method="predefined_partitions"):
    """Perform train-test split"""
    if method == "predefined_partitions":
        #Train - Test split
        traineval_data,test_data = data[data[:,0,0,3] == 1.], data[data[:,0,0,3] == 0.]
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir,type="Test")
        #Train - Eval split
        kfolds = StratifiedGroupKFold(n_splits=args.k_folds).split(traineval_data, traineval_data[:,0,0,0], traineval_data[:,0,0,2])
        return traineval_data,test_data,kfolds
    elif method == "random_stratified":
        data_labels = data[:,0,0,0]
        traineval_data, test_data = train_test_split(data, test_size=0.1, random_state=13, stratify=data_labels,shuffle=True)
        dataset_proportions(traineval_data,results_dir)
        dataset_proportions(test_data,results_dir, type="Test")
        # Train - Eval split
        kfolds = StratifiedShuffleSplit(n_splits=args.k_folds, random_state=13, test_size=0.2).split(traineval_data,traineval_data[:,0,0,0])
        return traineval_data,test_data,kfolds
    else:
        raise ValueError("train test split method not available")


def kfold_crossvalidation(dataset_info,additional_info,args):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
    results_dir = additional_info.results_dir
    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    #TODO: Detect and correct batch_size automatically?
    #Highlight: Train- Test split and kfold generator
    #TODO: Develop method to partition sequences, sequences in train and test must differ. Partitions must have similar distributions (Tree based on distance matrix?
    # In the loop computer another cosine similarity among the vectors of cos sim of each sequence?)
    traineval_data_blosum,test_data_blosum,kfolds = trainevaltest_split(data_blosum,args,results_dir,method="predefined_partitions")

    #Highlight:Also split the rest of arrays
    traineval_idx = (data_blosum[:,0,0,1][..., None] == traineval_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    traineval_mask = data_array_blosum_encoding_mask[traineval_idx]
    test_mask = data_array_blosum_encoding_mask[~traineval_idx]
    traineval_data_int = data_int[traineval_idx]
    test_data_int = data_int[~traineval_idx]
    traineval_data_onehot = data_onehot[traineval_idx]
    test_data_onehot = data_onehot[~traineval_idx]
    #Split the rest of the data (train_data) for train and validation
    batch_size = args.batch_size
    check_point_epoch = [5 if args.num_epochs < 100 else int(args.num_epochs / 50)][0]
    model_load = ModelLoad(args=args,
                           max_len =dataset_info.max_len,
                           n_data = dataset_info.n_data,
                           input_dim = dataset_info.input_dim,
                           aa_types = dataset_info.corrected_aa_types,
                           blosum = dataset_info.blosum)

    valid_predictions_fold = None
    train_predictions_fold = None
    for fold, (train_idx, valid_idx) in enumerate(kfolds): #returns k-splits for train and validation
        print("---------------------------------------------------------------------")
        print('Fold number : {}'.format(fold))
        print('\t Number train data points: {}; Proportion: {}'.format(len(train_idx),(len(train_idx)*100)/traineval_data_blosum.shape[0]))
        print('\t Number valid data points: {}; Proportion: {}'.format(len(valid_idx),(len(valid_idx)*100)/traineval_data_blosum.shape[0]))

        #Highlight: Minmax scale the confidence scores #TODO: function or for loop?
        fold_train_data_blosum = traineval_data_blosum[train_idx]
        fold_train_data_int = traineval_data_int[train_idx]
        fold_train_data_onehot = traineval_data_onehot[train_idx]
        fold_train_data_blosum[:,0,0,4] = VegvisirUtils.minmax_scale(fold_train_data_blosum[:,0,0,4])
        idx_train = (fold_train_data_blosum[:,0,0,4][..., None] == 0.).any(-1)*(fold_train_data_blosum[:,0,0,4][..., None] == 1.).any(-1)
        #Assign weight <1 to the classification scores with int(1) or int(0), else 1 + (1-classification score)
        fold_train_data_blosum[idx_train, 0, 0, 5] = 1
        fold_train_data_blosum[~idx_train, 0, 0, 5] = 1 + (1 - fold_train_data_blosum[:,0,0,4])
        fold_train_data_int[:,0,5] = fold_train_data_blosum[:,0,0,5]
        fold_train_data_onehot[:,0,0,5] = fold_train_data_blosum[:,0,0,5]
        #Highlight: valid
        fold_valid_data_blosum = traineval_data_blosum[valid_idx]
        fold_valid_data_int = traineval_data_int[valid_idx]
        fold_valid_data_onehot = traineval_data_onehot[valid_idx]
        fold_valid_data_blosum[:,0,0,4] = VegvisirUtils.minmax_scale(fold_valid_data_blosum[:,0,0,4])
        idx_valid = (fold_valid_data_blosum[:,0,0,4][..., None] == 0.).any(-1)*(fold_valid_data_blosum[:,0,0,4][..., None] == 1.).any(-1)
        fold_valid_data_blosum[idx_valid, 0, 0, 5] = 1
        fold_valid_data_blosum[~idx_valid, 0, 0, 5] = 1+ (1 - fold_valid_data_blosum[:, 0, 0, 4])
        fold_valid_data_int[:,0,5] = fold_valid_data_blosum[:,0,0,5]
        fold_valid_data_onehot[:,0,0,5] = fold_valid_data_blosum[:,0,0,5]



        custom_dataset_train = VegvisirLoadUtils.CustomDataset(fold_train_data_blosum,
                                                               fold_train_data_int,
                                                               fold_train_data_onehot,
                                                                traineval_mask[train_idx])
        custom_dataset_valid = VegvisirLoadUtils.CustomDataset(fold_valid_data_blosum,
                                                               fold_valid_data_int,
                                                               fold_valid_data_onehot,
                                                               traineval_mask[valid_idx])
        train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
        valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device),**kwargs)


        #Restart the model each fold
        vegvisir_model = select_model(model_load, additional_info.results_dir,fold)


        params_config = config_build(args,results_dir)
        if args.optimizer_name == "Adam":
            optimizer = torch.optim.Adam(vegvisir_model.parameters(), lr=params_config["lr"],
                                         betas=(params_config["beta1"], params_config["beta2"]),
                                         eps=params_config["eps"], weight_decay=params_config["weight_decay"])
        elif args.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(vegvisir_model.parameters(),lr=params_config["lr"],momentum=params_config["momentum"])
        else:
            raise ValueError("selected optimizer {} not implemented".format(args.optimizer_name))
        loss_func = vegvisir_model.loss

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
            train_epoch_loss,train_accuracy,train_predictions = train_loop(vegvisir_model, loss_func, optimizer, train_loader, args)
            stop = time.time()
            memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
            print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
            train_loss.append(train_epoch_loss)
            train_accuracies.append(train_accuracy)
            if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
                for name_i, value in vegvisir_model.named_parameters():
                    value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().item()))
                valid_epoch_loss,valid_accuracy,valid_predictions = valid_loop(vegvisir_model, loss_func, valid_loader, args)
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
                if epoch == args.num_epochs:
                    print("Saving final results")
                    train_predictions_fold = train_predictions
                    valid_predictions_fold = valid_predictions
                    VegvisirPlots.plot_gradients(gradient_norms, results_dir, fold)
            torch.cuda.empty_cache()
            epoch += 1 #TODO: early stop?
        fold_auc(valid_predictions_fold,fold_valid_data_blosum[:,0,0,0],fold,results_dir,mode="Valid")
        fold_auc(train_predictions_fold,fold_train_data_blosum[:,0,0,0],fold,results_dir,mode="Train")


    if args.test:
        print("Final testing")
        test_data_blosum[:,0,0,4] = VegvisirUtils.minmax_scale(test_data_blosum[:,0,0,4])
        test_data_int[:,0,4] = test_data_blosum[:,0,0,4]
        test_data_onehot[:,0,0,4] = test_data_blosum[:,0,0,4]
        custom_dataset_test = VegvisirLoadUtils.CustomDataset(test_data_blosum,
                                                              test_data_int,
                                                              test_data_onehot,
                                                              test_mask)
        test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True,
                                 generator=torch.Generator(device=args.device), **kwargs)
        predictions = test_loop(test_loader, vegvisir_model, args)
        score = roc_auc_score(y_true=test_data_blosum[:, 0, 0, 0].numpy(), y_score=predictions)
        print("Final AUC score : {}".format( score))



