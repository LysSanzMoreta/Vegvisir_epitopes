import json
import time,os,math
import pandas as pd
from collections import defaultdict
import numpy as np
from sklearn.metrics import auc,roc_auc_score,cohen_kappa_score,roc_curve,confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import namedtuple
import vegvisir
import vegvisir.utils as VegvisirUtils
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots
import vegvisir.models as VegvisirModels
ModelLoad = namedtuple("ModelLoad",["args","seq_max_len","max_len","n_data","input_dim","aa_types","blosum","class_weights"])


def train_loop(model,loss_func,optimizer, data_loader, args):
    """Regular batch training
    :param nn.Module model: Neural net architecture
    :param loss_func: loss function
    :param optimizer: gradient descent method
    :param DataLoader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    model.train() #Highlight: look at https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    train_loss = 0.0
    predictions_binary = []
    predictions_logits = []
    predictions_probs = []
    labels = []
    for batch_number, batch_dataset in enumerate(data_loader):
        batch_data_blosum = batch_dataset["batch_data_blosum"]
        batch_data_onehot = batch_dataset["batch_data_onehot"]
        batch_data_int = batch_dataset["batch_data_int"]
        batch_mask = batch_dataset["batch_mask"]
        if args.use_cuda:
            batch_data_blosum = batch_data_blosum.cuda()
            batch_data_onehot = batch_data_onehot.cuda()
            batch_mask = batch_mask.cuda()
        true_labels = batch_data_blosum[:,0,0,0]
        confidence_scores = batch_data_blosum[:,0,0,5]
        optimizer.zero_grad() #set gradients to zero
        #Forward pass
        batch_data = {"blosum":batch_data_blosum,"int":batch_data_int,"onehot":batch_data_onehot}
        model_outputs = model(batch_data,batch_mask)
        logits = model_outputs.class_logits.detach().cpu().numpy()
        predicted_labels = np.argmax(logits, axis=1)  # find the class with highest energy
        predictions_binary.append(predicted_labels)
        predictions_logits.append(logits)
        # predictions_probs.append(torch.exp(logits) / (1 + torch.exp(logits)))
        predictions_probs.append(1 / (1 + np.exp(-logits)))
        labels.append(true_labels.detach().cpu().numpy())
        loss = loss_func(confidence_scores,true_labels,model_outputs,batch_data_onehot)
        #loss.requires_grad = True #not sure why it does not do this automatically for the nnl loss
        #Backward pass: backpropagate the error loss
        loss.backward()#retain_graph=True
        optimizer.step()
        if args.clip_gradients:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5, norm_type=2)
        train_loss += loss.item()
    #Normalize train loss
    train_loss /= len(data_loader)
    predictions_binary_arr = np.concatenate(predictions_binary,axis=0)
    predictions_logits_arr = np.concatenate(predictions_logits,axis=0)
    predictions_probs_arr = np.concatenate(predictions_probs,axis=0)
    labels_arr = np.concatenate(labels,axis=0)
    accuracy = (predictions_binary_arr == labels_arr).mean()
    predictions_dict = {"binary": predictions_binary_arr,
                        "logits":predictions_logits_arr,
                        "probs":predictions_probs_arr}

    return train_loss,accuracy,predictions_dict
def valid_loop(model,loss_func, data_loader, args):
    """Regular batch training
    :param svi: pyro infer engine
    :param dataloader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    model.eval()
    valid_loss = 0.0
    predictions_binary = []
    predictions_logits = []
    predictions_probs = []

    labels = []
    with torch.no_grad(): #do not update parameters with the evaluation data
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_data_onehot = batch_dataset["batch_data_onehot"]
            batch_data_int = batch_dataset["batch_data_int"]

            batch_mask = batch_dataset["batch_mask"]
            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda() #TODO: Automatize for any kind of input (blosum encoding, integers, one-hot)
                batch_data_onehot = batch_data_onehot.cuda()
                batch_mask = batch_mask.cuda()
            true_labels = batch_data_blosum[:,0,0,0] #
            confidence_scores = batch_data_blosum[:, 0, 0, 5]
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot}
            model_outputs = model(batch_data,batch_mask)
            logits = model_outputs.class_logits.detach().cpu().numpy()
            predicted_labels = np.argmax(logits, axis=1)  # find the class with highest energy
            predictions_binary.append(predicted_labels)
            predictions_logits.append(logits)
            # predictions_probs.append(torch.exp(logits) / (1 + torch.exp(logits)))
            predictions_probs.append(1 / (1 + np.exp(-logits)))
            labels.append(true_labels.detach().cpu().numpy())
            loss = loss_func(confidence_scores, true_labels, model_outputs,batch_data_onehot) #NOTE: not backprogagate error since evaluating
            valid_loss += loss.item()
    valid_loss /= len(data_loader)
    predictions_binary_arr = np.concatenate(predictions_binary,axis=0)
    predictions_logits_arr = np.concatenate(predictions_logits,axis=0)
    predictions_probs_arr = np.concatenate(predictions_probs,axis=0)
    labels_arr = np.concatenate(labels,axis=0)
    accuracy = (predictions_binary_arr == labels_arr).mean()
    predictions_dict = {"binary": predictions_binary_arr,
                        "logits":predictions_logits_arr,
                        "probs":predictions_probs_arr}

    return valid_loss,accuracy,predictions_dict
def test_loop(model,loss_func, data_loader, args):
    model.train(False)
    correct = 0
    total = 0
    predictions_binary = []
    predictions_probs = []
    predictions_logits = []

    labels = []
    test_loss = 0.
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_data_onehot = batch_dataset["batch_data_onehot"]
            batch_data_int = batch_dataset["batch_data_int"]
            batch_mask = batch_dataset["batch_mask"]
            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda()
                batch_data_onehot = batch_data_onehot.cuda()
                batch_mask = batch_mask.cuda()
            true_labels = batch_data_blosum[:,0,0,0]
            confidence_scores = batch_data_blosum[:, 0, 0, 5]
            # calculate outputs by running images through the network
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot}
            model_outputs = model(batch_data,batch_mask).detach()
            logits = model_outputs.class_logits.detach().cpu().numpy()
            predicted_labels = np.argmax(logits, axis=1)  # find the class with highest energy
            predictions_binary.append(predicted_labels)
            predictions_logits.append(logits)
            # predictions_probs.append(torch.exp(logits) / (1 + torch.exp(logits)))
            predictions_probs.append(1 / (1 + np.exp(-logits)))
            total += true_labels.size(0)
            correct += (predicted_labels == true_labels).sum().item()
            loss = loss_func(confidence_scores, true_labels, model_outputs,batch_data_onehot) #NOTE: not backprogagate error since evaluating
            test_loss += loss

    predictions_binary_arr = np.concatenate(predictions_binary,axis=0)
    predictions_logits_arr = np.concatenate(predictions_logits,axis=0)
    predictions_probs_arr = np.concatenate(predictions_probs,axis=0)

    labels_arr = np.concatenate(labels,axis=0)
    accuracy = (predictions_binary_arr == labels_arr).mean()
    predictions_dict = {"binary": predictions_binary_arr,
                        "logits":predictions_logits_arr,
                        "probs":predictions_probs_arr}

    return test_loss,accuracy,predictions_dict
def save_script(results_dir,output_name,script_name):
    """Saves the python script and its contents"""
    out_file = open("{}/{}.py".format(results_dir,output_name), "a+")
    script_file = open("{}/{}.py".format(os.path.dirname(vegvisir.__file__),script_name), "r+")
    text = script_file.readlines()
    out_file.write("".join(text))
    out_file.close()
def select_model(model_load,results_dir,fold):
    """Select among the available models at models.py"""
    if model_load.seq_max_len == model_load.max_len:
        Vegvisir_model = VegvisirModels.VegvisirModel1(model_load)
    else:
        raise ValueError("Not Implemented")
        Vegvisir_model = VegvisirModels.VegvisirModel1(model_load)
    if fold == 0 or fold == "all":
        text_file = open("{}/Hyperparameters.txt".format(results_dir), "a")
        text_file.write("Model Class:  {} \n".format(Vegvisir_model.get_class()))
        text_file.close()
        save_script("{}/Scripts".format(results_dir), "ModelFunction", "models")
        save_script("{}/Scripts".format(results_dir), "ModelUtilsFunction", "model_utils")
        save_script("{}/Scripts".format(results_dir), "TrainFunction", "train")
    #Initialize the weights
    with torch.no_grad():
        Vegvisir_model.apply(init_weights)
    return Vegvisir_model
def config_build(args,results_dir):
    """Select a default configuration dictionary. It can load a string dictionary from the command line (using json) or use the default parameters
    :param namedtuple args"""
    # if args.parameter_search:
    #     config = json.loads(args.config_dict)
    # else:
    "Default hyperparameters (Clipped Adam optimizer), z dim and GRU"
    config = {
        "lr": 0.05, #default is 1e-3
        "beta1": 0.9, #coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        "beta2": 0.999,
        "eps": 1e-8,#term added to the denominator to improve numerical stability (default: 1e-8)
        "weight_decay": 0,#weight_decay: weight decay (L2 penalty) (default: 0)
        "clip_norm": 10,#clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
        "lrd": 1, #rate at which learning rate decays (default: 1.0)
        "z_dim": 30,
        "gru_hidden_dim": 60, #60
        "momentum":0 #Default is 0.9
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
def train_model(dataset_info,additional_info,args):
    """Set up ordinary train,val,test split without k-fold cross validation.The validation set changes every time (random pick form pre defined partitions)"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
    data_blosum_norm = dataset_info.data_array_blosum_norm
    seq_max_len = dataset_info.seq_max_len

    results_dir = additional_info.results_dir
    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    #TODO: Detect and correct batch_size automatically?
    #Highlight: Train- Test split and kfold generator
    #TODO: Develop method to partition sequences, sequences in train and test must differ. Partitions must have similar distributions (Tree based on distance matrix?
    # In the loop computer another cosine similarity among the vectors of cos sim of each sequence?)
    train_data_blosum,valid_data_blosum,test_data_blosum = VegvisirLoadUtils.trainevaltest_split(data_blosum,
                                                                                                  args,results_dir,
                                                                                                  seq_max_len,dataset_info.max_len,
                                                                                                  dataset_info.features_names,method="predefined_partitions_discard_test")

    #Highlight:Also split the rest of arrays
    train_idx = (data_blosum[:,0,0,1][..., None] == train_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    valid_idx = (data_blosum[:,0,0,1][..., None] == valid_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    test_idx = (data_blosum[:,0,0,1][..., None] == test_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not


    #Split the rest of the data (train_data) for train and validation
    batch_size = args.batch_size
    check_point_epoch = [5 if args.num_epochs < 100 else int(args.num_epochs / 50)][0]
    model_load = ModelLoad(args=args,
                           seq_max_len = dataset_info.seq_max_len,
                           max_len =dataset_info.max_len,
                           n_data = dataset_info.n_data,
                           input_dim = dataset_info.input_dim,
                           aa_types = dataset_info.corrected_aa_types,
                           blosum = dataset_info.blosum,
                           class_weights= VegvisirLoadUtils.calculate_class_weights(train_data_blosum,args))


    fold = "all"
    print("---------------------TRAINING / VALIDATING ----------------------------------")
    print('\t Number train data points: {}; Proportion: {}'.format(train_data_blosum.shape[0],(train_data_blosum.shape[0]*100)/(train_data_blosum.shape[0] + valid_data_blosum.shape[0])))
    print('\t Number valid data points: {}; Proportion: {}'.format(valid_data_blosum.shape[0],(valid_data_blosum.shape[0]*100)/(train_data_blosum.shape[0] + valid_data_blosum.shape[0])))

    custom_dataset_train = VegvisirLoadUtils.CustomDataset(train_data_blosum,
                                                           data_int[train_idx],
                                                           data_onehot[train_idx],
                                                           data_blosum_norm[train_idx],
                                                           data_array_blosum_encoding_mask[train_idx])
    custom_dataset_valid = VegvisirLoadUtils.CustomDataset(valid_data_blosum,
                                                           data_int[valid_idx],
                                                           data_onehot[valid_idx],
                                                           data_blosum_norm[valid_idx],
                                                           data_array_blosum_encoding_mask[valid_idx])

    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffled_Ibel? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device),**kwargs)


    #Restart the model each fold
    Vegvisir_model = select_model(model_load, additional_info.results_dir,fold)
    params_config = config_build(args,results_dir)
    if args.optimizer_name == "Adam":
        optimizer = torch.optim.Adam(Vegvisir_model.parameters(), lr=params_config["lr"],
                                     betas=(params_config["beta1"], params_config["beta2"]),
                                     eps=params_config["eps"], weight_decay=params_config["weight_decay"])
    elif args.optimizer_name == "SGD":
        optimizer = torch.optim.SGD(Vegvisir_model.parameters(),lr=params_config["lr"],momentum=params_config["momentum"])
    else:
        raise ValueError("selected optimizer {} not implemented".format(args.optimizer_name))
    loss_func = Vegvisir_model.loss

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
    valid_predictions_dict = None
    train_predictions_dict = None
    gradient_norms = defaultdict(list)
    while epoch <= args.num_epochs:
        start = time.time()
        train_epoch_loss,train_accuracy,train_predictions_dict = train_loop(Vegvisir_model, loss_func, optimizer, train_loader, args)
        stop = time.time()
        memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
        train_loss.append(train_epoch_loss)
        train_accuracies.append(train_accuracy)
        if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
            for name_i, value in Vegvisir_model.named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
            valid_epoch_loss,valid_accuracy,valid_predictions_dict = valid_loop(Vegvisir_model, loss_func, valid_loader, args)
            valid_loss.append(valid_epoch_loss)
            epochs_list.append(epoch)
            valid_accuracies.append(valid_accuracy)
            train_auc_score = roc_auc_score(y_true=train_data_blosum[:,0,0,0], y_score=train_predictions_dict["binary"])
            train_auk_score = VegvisirUtils.AUK(probabilities= train_predictions_dict["binary"],labels=train_data_blosum[:,0,0,0].numpy()).calculate_auk()
            train_auk.append(train_auk_score)
            train_auc.append(train_auc_score)
            valid_auc_score = roc_auc_score(y_true=valid_data_blosum[:,0,0,0], y_score=valid_predictions_dict["binary"])
            valid_auk_score = VegvisirUtils.AUK(probabilities= valid_predictions_dict["binary"],labels = valid_data_blosum[:,0,0,0].numpy()).calculate_auk()
            valid_auk.append(valid_auk_score)
            valid_auc.append(valid_auc_score)
            VegvisirPlots.plot_loss(train_loss,valid_loss,epochs_list,fold,additional_info.results_dir)
            VegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,fold,additional_info.results_dir)
            VegvisirPlots.plot_classification_score(train_auc,valid_auc,epochs_list,fold,additional_info.results_dir,method="AUC")
            VegvisirPlots.plot_classification_score(train_auk,valid_auk,epochs_list,fold,additional_info.results_dir,method="AUK")
            Vegvisir_model.save_checkpoint("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer)
            if epoch == args.num_epochs:
                print("Saving final results")
                train_predictions_dict = train_predictions_dict
                valid_predictions_dict = valid_predictions_dict
                VegvisirPlots.plot_gradients(gradient_norms, results_dir, fold)
                Vegvisir_model.save_checkpoint("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)

        torch.cuda.empty_cache()
        epoch += 1 #TODO: early stop?

    VegvisirUtils.fold_auc(valid_predictions_dict,valid_data_blosum[:,0,0,0],fold,results_dir,mode="Valid")
    VegvisirUtils.fold_auc(train_predictions_dict,train_data_blosum[:,0,0,0],fold,results_dir,mode="Train")

    if args.test:
        print("Final testing")
        custom_dataset_test = VegvisirLoadUtils.CustomDataset(test_data_blosum,
                                                              data_int[test_idx],
                                                              data_onehot[test_idx],
                                                              data_blosum_norm[test_idx],
                                                              data_array_blosum_encoding_mask[test_idx])
        test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True,
                                 generator=torch.Generator(device=args.device), **kwargs)
        test_loss, test_accuracy,test_predictions_dict = test_loop(Vegvisir_model, loss_func, test_loader, args)

        VegvisirUtils.fold_auc(test_predictions_dict, test_data_blosum[:, 0, 0, 0], fold, results_dir,mode="Test")
def kfold_crossvalidation(dataset_info,additional_info,args):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
    data_blosum_norm = dataset_info.data_array_blosum_norm
    seq_max_len = dataset_info.seq_max_len

    results_dir = additional_info.results_dir
    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU
    #TODO: Detect and correct batch_size automatically?
    #Highlight: Train- Test split and kfold generator
    #TODO: Develop method to partition sequences, sequences in train and test must differ. Partitions must have similar distributions (Tree based on distance matrix?
    # In the loop computer another cosine similarity among the vectors of cos sim of each sequence?)
    traineval_data_blosum,test_data_blosum,kfolds = VegvisirLoadUtils.trainevaltest_split_kfolds(data_blosum,args,results_dir,method="predefined_partitions")

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
                           seq_max_len = dataset_info.seq_max_len,
                           max_len =dataset_info.max_len,
                           n_data = dataset_info.n_data,
                           input_dim = dataset_info.input_dim,
                           aa_types = dataset_info.corrected_aa_types,
                           blosum = dataset_info.blosum,
                           class_weights= VegvisirLoadUtils.calculate_class_weights(traineval_data_blosum,args))

    valid_predictions_fold = None
    train_predictions_fold = None
    for fold, (train_idx, valid_idx) in enumerate(kfolds): #returns k-splits for train and validation

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
        train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffled_Ibel? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
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
        epochs_list = []
        train_accuracies = []
        train_auc = []
        train_auk = []
        train_accuracy_fold = None

        valid_loss = []
        valid_accuracies= []
        valid_auc = []
        valid_auk = []
        valid_accuracy_fold = None

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
                for name_i, value in vegvisir_model.named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                    value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
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
                VegvisirPlots.plot_loss(train_loss,valid_loss,epochs_list,fold,additional_info.results_dir)
                VegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,fold,additional_info.results_dir)
                VegvisirPlots.plot_classification_score(train_auc,valid_auc,epochs_list,fold,additional_info.results_dir,method="AUC")
                VegvisirPlots.plot_classification_score(train_auk,valid_auk,epochs_list,fold,additional_info.results_dir,method="AUK")
                if epoch == args.num_epochs:
                    print("Saving final results")
                    train_predictions_fold = train_predictions
                    valid_predictions_fold = valid_predictions
                    train_accuracy_fold = train_accuracy
                    valid_accuracy_fold = valid_accuracy
                    VegvisirPlots.plot_gradients(gradient_norms, results_dir, fold)
                    vegvisir_model.save_checkpoint("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)
                    # params = vegvisir_model.capture_parameters([name for name,val in vegvisir_model.named_parameters()])
                    # gradients = vegvisir_model.capture_gradients([name for name,val in vegvisir_model.named_parameters()])
                    # activations = vegvisir_model.attach_hooks([name for name,val in vegvisir_model.named_parameters() if name.starstwith("a")])

            torch.cuda.empty_cache()
            epoch += 1 #TODO: early stop?
        VegvisirUtils.fold_auc(valid_predictions_fold,fold_valid_data_blosum[:,0,0,0],valid_accuracy_fold,fold,results_dir,mode="Valid")
        VegvisirUtils.fold_auc(train_predictions_fold,fold_train_data_blosum[:,0,0,0],train_accuracy_fold,fold,results_dir,mode="Train")


    if args.test:
        print("Final testing")
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
        predictions = test_loop(test_loader, vegvisir_model, args)
        score = roc_auc_score(y_true=test_data_blosum[:, 0, 0, 0].numpy(), y_score=predictions)
        print("Final AUC score : {}".format( score))


