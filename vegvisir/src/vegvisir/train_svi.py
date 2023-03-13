import json
from scipy import stats
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
from  pyro.infer import SVI,config_enumerate, Predictive
import pyro.poutine as poutine
import pyro
import vegvisir
import vegvisir.utils as VegvisirUtils
import vegvisir.load_utils as VegvisirLoadUtils
import vegvisir.plots as VegvisirPlots
import vegvisir.models as VegvisirModels
import vegvisir.guides as VegvisirGuides
ModelLoad = namedtuple("ModelLoad",["args","max_len","seq_max_len","n_data","input_dim","aa_types","blosum","class_weights"])


def train_loop(svi,Vegvisir,guide,data_loader, args,model_load):
    """Regular batch training
    :param pyro.infer svi
    :param nn.Module,PyroModule Vegvisir: Neural net architecture
    :param guide: EasyGuide or pyro.infer.autoguides
    :param DataLoader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    Vegvisir.train() #Highlight: look at https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    train_loss = 0.0
    reconstruction_accuracies = []
    predictions = []
    latent_spaces = []
    true_labels = []
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
        batch_data = {"blosum":batch_data_blosum,"int":batch_data_int,"onehot":batch_data_onehot,"norm":batch_data_blosum_norm}
        #Forward & Backward pass
        loss = svi.step(batch_data,batch_mask,sample=False)
        #guide_estimates = guide(batch_data,batch_mask)
        # sampling_output = Vegvisir.sample(batch_data,batch_mask,guide_estimates,argmax=True)
        # predicted_labels = sampling_output.predicted_labels.detach().cpu()
        # latent_space = sampling_output.latent_space
        # reconstructed_sequences = sampling_output.reconstructed_sequences.detach()
        sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=False)(batch_data,batch_mask,sample=True)
        if args.learning_type in ["semisupervised","unsupervised"]:
            predicted_labels = torch.concatenate([sampling_output[f"predictions_{t}"].squeeze(0).squeeze(0).detach() for t in range(batch_data_blosum.shape[0])],dim=0)
        else:
            predicted_labels = sampling_output["predictions"].squeeze(0).squeeze(0).squeeze(0).detach()
        reconstructed_sequences = sampling_output["sequences"].squeeze(0).squeeze(0).squeeze(0).detach()
        print(batch_data_int[0:2,1,:model_load.seq_max_len])
        print(reconstructed_sequences[0:2])

        latent_space = sampling_output["latent_z"].squeeze(0).squeeze(0).detach()

        identifiers = batch_data["blosum"][:,0,0,1]
        true_labels_batch = batch_data["blosum"][:,0,0,0]
        confidence_score = batch_data["blosum"][:,0,0,5]
        immunodominace_score = batch_data["blosum"][:, 0, 0, 4]

        latent_space = torch.column_stack([identifiers, true_labels_batch, confidence_score, immunodominace_score, latent_space])
        mask_seq = batch_mask[:, 1:,:,0].squeeze(1)
        equal_aa = torch.Tensor((batch_data_int[:,1,:model_load.seq_max_len] == reconstructed_sequences)*mask_seq)
        reconstruction_accuracy = (equal_aa.sum(dim=1))/mask_seq.sum(dim=1)
        reconstruction_accuracies.append(reconstruction_accuracy.cpu().numpy())
        latent_spaces.append(latent_space.detach().cpu().numpy())
        true_labels.append(true_labels_batch.detach().cpu().numpy())
        predictions.append(predicted_labels.detach().cpu().numpy())
        train_loss += loss
    #Normalize train loss
    train_loss /= len(data_loader)
    true_labels_arr = np.concatenate(true_labels,axis=0)
    predictions_arr = np.concatenate(predictions,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    target_accuracy= 100 * ((true_labels_arr == predictions_arr).sum()/true_labels_arr.shape[0])
    reconstruction_accuracies = np.concatenate(reconstruction_accuracies)
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies.mean(),"std":reconstruction_accuracies.std()}
    return train_loss,target_accuracy,predictions_arr,latent_arr, reconstruction_accuracies_dict
def valid_loop(svi,Vegvisir,guide, data_loader, args,model_load):
    """Regular batch training
    :param svi: pyro infer engine
    :param dataloader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    Vegvisir.eval()
    valid_loss = 0.0
    predictions = []
    latent_spaces = []
    reconstruction_accuracies = []
    true_labels = []
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
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,"norm":batch_data_blosum_norm}
            loss = svi.step(batch_data,batch_mask,sample=False)
            # guide_estimates = guide(batch_data,batch_mask)
            # sampling_output = Vegvisir.sample(batch_data,batch_mask,guide_estimates,argmax=True)
            # predicted_labels = sampling_output.predicted_labels.detach()
            # reconstructed_sequences = sampling_output.reconstructed_sequences.detach()
            #latent_space = sampling_output.latent_space.detach()

            sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=False)(
                batch_data, batch_mask,sample=True)
            if args.learning_type in ["semisupervised", "unsupervised"]:
                predicted_labels = torch.concatenate(
                    [sampling_output[f"predictions_{t}"].squeeze(0).squeeze(0).detach() for t in
                     range(batch_data_blosum.shape[0])], dim=0)
            else:
                predicted_labels = sampling_output["predictions"].squeeze(0).squeeze(0).squeeze(0).detach()
            reconstructed_sequences = sampling_output["sequences"].squeeze(0).squeeze(0).squeeze(0).detach()
            latent_space = sampling_output["latent_z"].squeeze(0).squeeze(0).detach()
            identifiers = batch_data["blosum"][:, 0, 0, 1]
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0]
            confidence_score = batch_data["blosum"][:, 0, 0, 5]
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
            latent_space = torch.column_stack([identifiers, true_labels_batch, confidence_score, immunodominace_score, latent_space])

            mask_seq = batch_mask[:, 1:, :, 0].squeeze(1)
            equal_aa = torch.Tensor((batch_data_int[:, 1, :model_load.seq_max_len] == reconstructed_sequences) * mask_seq)
            reconstruction_accuracy = (equal_aa.sum(dim=1)) / mask_seq.sum(dim=1)
            reconstruction_accuracies.append(reconstruction_accuracy.detach().cpu().numpy())
            latent_spaces.append(latent_space.cpu().detach().numpy())
            true_labels.append(true_labels_batch.cpu().detach().numpy())
            predictions.append(predicted_labels.cpu().detach().numpy())
            valid_loss += loss #TODO: Multiply by the data size?
    valid_loss /= len(data_loader)
    predictions_arr = np.concatenate(predictions,axis=0)
    true_labels_arr = np.concatenate(true_labels,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    target_accuracy= 100 * ((true_labels_arr == predictions_arr).sum()/true_labels_arr.shape[0])
    reconstruction_accuracies = np.concatenate(reconstruction_accuracies)
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies.mean(),"std":reconstruction_accuracies.std()}
    return valid_loss,target_accuracy,predictions_arr,latent_arr, reconstruction_accuracies_dict
def test_loop(Vegvisir,guide,data_loader,args,model_load):
    Vegvisir.train(False)
    correct = 0
    total = 0
    latent_spaces = []
    predictions = []
    reconstruction_accuracies = []
    true_labels = []
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
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,"norm":batch_data_blosum_norm}
            # guide_estimates = guide(batch_data,batch_mask)
            # sampling_output = Vegvisir.sample(batch_data,batch_mask,guide_estimates,argmax=True)
            # predicted_labels = sampling_output.predicted_labels.detach()
            # reconstructed_sequences = sampling_output.reconstructed_sequences.detach()
            #latent_space = sampling_output.latent_space.detach()
            sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=True)(batch_data, batch_mask,sample=True)
            #predicted_labels = sampling_output["predictions"].squeeze(0).squeeze(0).squeeze(0).detach()
            if args.learning_type in ["semisupervised", "unsupervised"]:
                predicted_labels = torch.concatenate(
                    [sampling_output[f"predictions_{t}"].squeeze(0).squeeze(0).detach() for t in
                     range(batch_data_blosum.shape[0])], dim=0)
            else:
                predicted_labels = sampling_output["predictions"].squeeze(0).squeeze(0).squeeze(0).detach()
            reconstructed_sequences = sampling_output["sequences"].squeeze(0).detach()
            latent_space = sampling_output["latent_space"].squeeze(0).squeeze(0).detach()
            identifiers = batch_data["blosum"][:, 0, 0, 1]
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0]
            confidence_score = batch_data["blosum"][:, 0, 0, 5]
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
            latent_space = torch.column_stack(
                [identifiers, true_labels_batch, confidence_score, immunodominace_score, latent_space])

            mask_seq = batch_mask[:, 1:, :, 0].squeeze(1)
            equal_aa = torch.Tensor((batch_data_int[:, 1, :model_load.seq_max_len] == reconstructed_sequences) * mask_seq)
            reconstruction_accuracy = (equal_aa.sum(dim=1)) / mask_seq.sum(dim=1)
            reconstruction_accuracies.append(reconstruction_accuracy.cpu().numpy())
            latent_spaces.append(latent_space.cpu().numpy())
            predictions.append(predicted_labels.detach().cpu().numpy())
            true_labels.append(true_labels_batch.detach().cpu().numpy())

    predictions_arr = np.concatenate(predictions,axis=0)
    true_labels_arr = np.concatenate(true_labels,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    target_accuracy = 100 * ((true_labels_arr == predictions_arr).sum() / true_labels_arr.shape[0])
    print(f'Accuracy of the TCR-pMHC: {100 * correct // total} %')
    reconstruction_accuracies = np.concatenate(reconstruction_accuracies)
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies.mean(),"std":reconstruction_accuracies.std()}
    return predictions_arr,target_accuracy,latent_arr, reconstruction_accuracies_dict
def sample_loop1(Vegvisir,guide,data_loader,args,custom=False):
    Vegvisir.train(False)
    print("Collecting {} samples".format(args.num_samples))
    with torch.no_grad():
        total_samples = []
        for sample in range(args.num_samples):
            batch_sample = []
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
                batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,"norm":batch_data_blosum_norm}
                if custom:
                    guide_estimates = guide(batch_data, batch_mask)
                    sampling_out = Vegvisir.sample(batch_data,batch_mask,guide_estimates,argmax=True)
                    predicted_labels = sampling_out.predicted_labels.detach()
                else: #TODO: more samples?
                    sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1,return_sites=(), parallel=False)(batch_data, batch_mask,sample=True)
                    predicted_labels = sampling_output["predictions"].squeeze(0).squeeze(0).detach()

                batch_sample.append(predicted_labels)
            data_sample = torch.cat(batch_sample,dim=0)
            total_samples.append(data_sample)
    total_samples_arr = torch.column_stack(total_samples)


    return total_samples_arr
def sample_loop(Vegvisir,guide,data_loader,args,custom=False):
    Vegvisir.train(False)
    print("Collecting {} samples".format(args.num_samples))
    with torch.no_grad():
        batch_samples = []
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
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,"norm":batch_data_blosum_norm}
            if custom: #TODO: NOT REVIEWED
                batch_sample=[]
                for sample in range(args.num_samples):
                    guide_estimates = guide(batch_data, batch_mask)
                    sampling_out = Vegvisir.sample(batch_data,batch_mask,guide_estimates,argmax=True)
                    batch_sample.append(sampling_out.prediction.detach())
                batch_sample = torch.column_stack(batch_sample)
                batch_samples.append(batch_sample)

            else: #TODO: more samples?
                sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=args.num_samples,return_sites=(), parallel=False)(batch_data, batch_mask,sample=True)
                if args.learning_type in ["semisupervised", "unsupervised"]:
                    predicted_labels = torch.concatenate([sampling_output[f"predictions_{t}"].squeeze(1).squeeze(1).detach()[None,:] for t in
                         range(batch_data_blosum.shape[0])], dim=0)
                else:
                    if sampling_output["predictions"].ndim == 4:
                        predicted_labels = sampling_output["predictions"].squeeze(1).squeeze(1).detach().T
                    else:
                        predicted_labels = sampling_output["predictions"].squeeze(1).detach().T
                batch_samples.append(predicted_labels)
        total_samples = torch.cat(batch_samples,dim=0)

    return total_samples
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
    guide = {"autodelta":AutoDelta(Vegvisir.model),
             "autonormal":AutoNormal(Vegvisir.model,init_scale=0.1),
             "autodiagonalnormal": AutoDiagonalNormal(Vegvisir.model, init_scale=0.1), #Mean Field approximation, only diagonal variance
             "custom":VegvisirGuides.VEGVISIRGUIDES(Vegvisir.model,model_load,Vegvisir)}
    return guide[choice]
    #return poutine.scale(guide[choice],scale=1.0/n_data) #Scale the ELBo to the data size
def select_model(model_load,results_dir,fold):
    """Select among the available models at models.py"""
    if model_load.seq_max_len == model_load.max_len:
        vegvisir_model = VegvisirModels.VegvisirModel5a(model_load)
    else:
        vegvisir_model = VegvisirModels.VegvisirModel5c(model_load)
    if fold == 0 or fold == "all":
        text_file = open("{}/Hyperparameters.txt".format(results_dir), "a")
        text_file.write("Model Class:  {} \n".format(vegvisir_model.get_class()))
        text_file.close()
        save_script("{}/Scripts".format(results_dir), "ModelFunction", "models")
        save_script("{}/Scripts".format(results_dir), "ModelUtilsFunction", "model_utils")
        save_script("{}/Scripts".format(results_dir), "GuidesFunction", "guides")
        save_script("{}/Scripts".format(results_dir), "TrainFunction", "train_svi")
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
    traineval_data_blosum,test_data_blosum,kfolds = VegvisirLoadUtils.trainevaltest_split_kfolds(data_blosum,args,results_dir,seq_max_len,dataset_info.max_len,dataset_info.features_names,method="predefined_partitions_discard_test")

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
                           blosum = dataset_info.blosum,
                           class_weights=VegvisirLoadUtils.calculate_class_weights(traineval_data_blosum, args))

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
            train_epoch_loss,train_accuracy,train_predictions,train_reconstruction_accuracies_dict = train_loop(svi,Vegvisir,guide, train_loader, args,model_load)
            stop = time.time()
            memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
            print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
            train_loss.append(train_epoch_loss)
            train_accuracies.append(train_accuracy)
            if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
                for name_i, value in pyro.get_param_store().named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                    value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
                valid_epoch_loss,valid_accuracy,valid_predictions,valid_reconstruction_accuracies_dict = valid_loop(svi,Vegvisir,guide, valid_loader, args,model_load)
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
                Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)
                if epoch == args.num_epochs:
                    print("Calculating Monte Carlo estimate of the posterior predictive")
                    train_predictions_fold = sample_loop(Vegvisir, guide, train_loader, args)
                    valid_predictions_fold = sample_loop(Vegvisir, guide, valid_loader, args)
                    train_predictions_fold_mode = stats.mode(train_predictions_fold.cpu().numpy(), axis=1,
                                                        keepdims=True).mode.squeeze(-1)
                    valid_predictions_fold_mode = stats.mode(valid_predictions_fold.cpu().numpy(), axis=1,
                                                        keepdims=True).mode.squeeze(-1)
                    VegvisirPlots.plot_gradients(gradient_norms, results_dir, fold)
                    Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)
                    # params = vegvisir_model.capture_parameters([name for name,val in vegvisir_model.named_parameters()])
                    # gradients = vegvisir_model.capture_gradients([name for name,val in vegvisir_model.named_parameters()])
                    # activations = vegvisir_model.attach_hooks([name for name,val in vegvisir_model.named_parameters() if name.starstwith("a")])

            torch.cuda.empty_cache()
            epoch += 1 #TODO: early stop?
        #predictions_fold,labels,accuracy,fold,results_dir
        VegvisirUtils.fold_auc(valid_predictions_fold_mode,fold_valid_data_blosum[:,0,0,0],valid_accuracy,fold,results_dir,mode="Valid")
        VegvisirUtils.fold_auc(train_predictions_fold_mode,fold_train_data_blosum[:,0,0,0],train_accuracy,fold,results_dir,mode="Train")
        pyro.clear_param_store()

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
            train_epoch_loss, train_accuracy, train_predictions,train_reconstruction_accuracies_dict = train_loop(svi, Vegvisir, guide, train_loader, args,model_load)
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
                VegvisirPlots.plot_loss(train_loss, None, epochs_list, "final", additional_info.results_dir)
                VegvisirPlots.plot_accuracy(train_accuracies, None, epochs_list, "final",additional_info.results_dir)
                VegvisirPlots.plot_classification_score(train_auc, None, epochs_list, "final",additional_info.results_dir, method="AUC")
                VegvisirPlots.plot_classification_score(train_auk, None, epochs_list, "final",additional_info.results_dir, method="AUK")
                if epoch == args.num_epochs:
                    print("Calculating Monte Carlo estimate of the posterior predictive")
                    train_predictions_fold = sample_loop(Vegvisir, guide, train_loader, args)
                    train_predictions_fold_mode = stats.mode(train_predictions_fold.cpu().numpy(), axis=1,keepdims=True).mode.squeeze(-1)
                    VegvisirPlots.plot_gradients(gradient_norms, results_dir, "final")
                    Vegvisir.save_checkpoint("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer)
            epoch += 1
            torch.cuda.empty_cache()
        VegvisirUtils.fold_auc(train_predictions_fold_mode,traineval_data_blosum[:,0,0,0],train_accuracy,"final",results_dir,mode="Train")
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
        predictions = test_loop(Vegvisir,guide,test_loader,args,model_load)
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
    train_data_blosum,valid_data_blosum,test_data_blosum = VegvisirLoadUtils.trainevaltest_split(data_blosum,args,results_dir,seq_max_len,dataset_info.max_len,dataset_info.features_names,method="predefined_partitions_discard_test")

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
                           blosum = dataset_info.blosum,
                           class_weights=VegvisirLoadUtils.calculate_class_weights(train_data_blosum, args)
                           )

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
    if args.optimizer_name == "Adam" and not args.clip_gradients:
        adam_args = {"lr":params_config["lr"],
                    "betas": (params_config["beta1"], params_config["beta2"]),
                    "eps": params_config["eps"],
                    "weight_decay": params_config["weight_decay"]}
        optimizer = pyro.optim.Adam(adam_args)
    elif args.optimizer_name == "ClippedAdam" or (args.optimizer_name == "Adam" and args.clip_gradients):
        clippedadam_args = {"lr": params_config["lr"],
                        "betas": (params_config["beta1"], params_config["beta2"]),
                        "eps": params_config["eps"],
                        "weight_decay": params_config["weight_decay"],
                        "clip_norm": params_config["clip_norm"],
                        "lrd": params_config["lrd"]}
        optimizer = pyro.optim.ClippedAdam(clippedadam_args)
    else:
        raise ValueError("selected optimizer <{}> not implemented with <{}> clip gradients".format(args.optimizer_name,args.clip_gradients))
    loss_func = Vegvisir.loss()
    if args.learning_type in ["semisupervised","unsupervised"]:
        guide = config_enumerate(select_quide(Vegvisir,model_load,n_data,args.guide))
    else:
        guide = select_quide(Vegvisir,model_load,n_data,args.guide)
    #svi = SVI(poutine.scale(Vegvisir.model,scale=1.0/n_data), poutine.scale(guide,scale=1.0/n_data), optimizer, loss_func)
    n = 50
    data_args_0 = {"blosum":train_data_blosum.to(args.device)[:n],"norm":data_blosum_norm[train_idx].to(args.device)[:n],"int":data_int[train_idx].to(args.device)[:n]}
    data_args_1 = data_array_blosum_encoding_mask[train_idx].to(args.device)[:n]
    trace = pyro.poutine.trace(Vegvisir.model).get_trace(data_args_0,data_args_1)
    #print(trace.nodes["predictions"])
    #print(trace.nodes["sequences"])
    guide_tr = poutine.trace(guide).get_trace(data_args_0,data_args_1,sample=False)
    model_tr = poutine.trace(poutine.replay(Vegvisir.model, trace=guide_tr)).get_trace(data_args_0,data_args_1,sample=False)
    monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()
    #print(monte_carlo_elbo)
    #obs_mask = trace.nodes["predictions"]
    #Highlight: Draw the graph model
    pyro.render_model(Vegvisir.model, model_args=(data_args_0,data_args_1,False), filename="{}/model_graph.png".format(results_dir),render_distributions=True,render_params=True)
    pyro.render_model(guide, model_args=(data_args_0,data_args_1,False), filename="{}/guide_graph.png".format(results_dir),render_distributions=True,render_params=True)
    svi = SVI(Vegvisir.model, guide, optimizer, loss_func)

    #TODO: Dictionary that gathers the results from each fold
    start = time.time()
    epochs_list = []
    train_loss = []
    train_accuracies = []
    train_reconstruction_accuracies_dict = {"mean":[],"std":[]}
    valid_reconstruction_accuracies_dict = {"mean":[],"std":[]}
    train_auc = []
    train_auk = []
    valid_loss = []
    valid_accuracies = []
    valid_auc = []
    valid_auk = []
    epoch = 0.
    train_predictions_dict = None
    valid_predictions_dict = None
    gradient_norms = defaultdict(list)
    while epoch <= args.num_epochs:
        start = time.time()
        #svi,Vegvisir,guide,data_loader, args
        train_epoch_loss,train_accuracy,train_predictions, train_latent_space,train_reconstruction_accuracy_dict = train_loop(svi,Vegvisir,guide, train_loader, args,model_load)
        stop = time.time()
        memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
        train_loss.append(train_epoch_loss)
        train_accuracies.append(train_accuracy)
        train_reconstruction_accuracies_dict["mean"].append(train_reconstruction_accuracy_dict["mean"])
        train_reconstruction_accuracies_dict["std"].append(train_reconstruction_accuracy_dict["std"])

        if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
            for name_i, value in pyro.get_param_store().named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
            epochs_list.append(epoch)
            valid_epoch_loss, valid_accuracy, valid_predictions, valid_latent_space,valid_reconstruction_accuracy_dict = valid_loop(svi, Vegvisir, guide, valid_loader, args,model_load)
            valid_loss.append(valid_epoch_loss)
            valid_accuracies.append(valid_accuracy)
            valid_reconstruction_accuracies_dict["mean"].append(valid_reconstruction_accuracy_dict["mean"])
            valid_reconstruction_accuracies_dict["std"].append(valid_reconstruction_accuracy_dict["std"])
            train_auc_score = roc_auc_score(y_true=train_data_blosum[:,0,0,0], y_score=train_predictions)
            train_auk_score = VegvisirUtils.AUK(probabilities= train_predictions,labels=train_data_blosum[:,0,0,0].numpy()).calculate_auk()
            train_auk.append(train_auk_score)
            train_auc.append(train_auc_score)

            valid_auc_score = roc_auc_score(y_true=valid_data_blosum[:, 0, 0, 0], y_score=valid_predictions)
            valid_auk_score = VegvisirUtils.AUK(probabilities=valid_predictions,labels=valid_data_blosum[:, 0, 0, 0].numpy()).calculate_auk()
            valid_auk.append(valid_auk_score)
            valid_auc.append(valid_auc_score)

            VegvisirPlots.plot_loss(train_loss,valid_loss,epochs_list,"all",additional_info.results_dir)
            VegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,"all",additional_info.results_dir)
            VegvisirPlots.plot_accuracy(train_reconstruction_accuracies_dict,valid_reconstruction_accuracies_dict,epochs_list,"all",additional_info.results_dir)
            VegvisirPlots.plot_classification_score(train_auc,valid_auc,epochs_list,"all",additional_info.results_dir,method="AUC")
            VegvisirPlots.plot_classification_score(train_auk,valid_auk,epochs_list,"all",additional_info.results_dir,method="AUK")
            Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer)
            if epoch == args.num_epochs:
                print("Calculating Monte Carlo estimate of the posterior predictive")
                train_predictions_samples = sample_loop(Vegvisir,guide,train_loader,args)
                valid_predictions_samples = sample_loop(Vegvisir,guide,valid_loader,args)
                train_predictions_mode = stats.mode(train_predictions_samples.cpu().numpy(), axis=1,keepdims=True).mode.squeeze(-1)
                valid_predictions_mode = stats.mode(valid_predictions_samples.cpu().numpy(), axis=1,keepdims=True).mode.squeeze(-1)
                train_frequencies = torch.stack([torch.bincount(x_i, minlength=args.num_classes) for i, x_i in enumerate(torch.unbind(train_predictions_samples.type(torch.int64), dim=0), 0)], dim=0)
                train_frequencies = train_frequencies/args.num_samples
                valid_frequencies = torch.stack([torch.bincount(x_i, minlength=args.num_classes) for i, x_i in enumerate(torch.unbind(valid_predictions_samples.type(torch.int64), dim=0), 0)], dim=0)
                valid_frequencies = valid_frequencies/args.num_samples

                train_predictions_dict = {"samples_mode":train_predictions_mode,
                                          "frequencies": train_frequencies.detach().cpu().numpy(),
                                          "predictions":train_predictions
                                          # "y_perc_5": svi_gdp.kthvalue(int(len(svi_gdp) * 0.05), dim=0)[
                                          #     0].detach().cpu().numpy(),
                                          # "y_perc_95": svi_gdp.kthvalue(int(len(svi_gdp) * 0.95), dim=0)[
                                          #     0].detach().cpu().numpy(),
                                          }

                valid_predictions_dict = {"samples_mode": valid_predictions_mode,
                                          "frequencies": valid_frequencies.detach().cpu().numpy(),
                                          "predictions":valid_predictions
                                          }
                VegvisirPlots.plot_gradients(gradient_norms, results_dir, "all")
                VegvisirPlots.plot_latent_space(train_latent_space, train_predictions_dict, "all",results_dir, method="Train")
                VegvisirPlots.plot_latent_space(valid_latent_space,valid_predictions_dict, "all",results_dir, method="Valid")
                VegvisirPlots.plot_latent_vector(train_latent_space, train_predictions_dict, "all",results_dir, method="Train")
                VegvisirPlots.plot_latent_vector(valid_latent_space,valid_predictions_dict, "all",results_dir, method="Valid")
                Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer)

        torch.cuda.empty_cache()
        epoch += 1 #TODO: early stop?
    VegvisirUtils.fold_auc(train_predictions_dict,train_data_blosum[:,0,0,0],"_mode_samples",results_dir,mode="Train")
    VegvisirUtils.fold_auc(valid_predictions_dict,valid_data_blosum[:,0,0,0],"_mode_samples",results_dir,mode="Valid")
    # VegvisirUtils.fold_auc(train_predictions,train_data_blosum[:,0,0,0],train_accuracy,"argmax_sample",results_dir,mode="Train")
    # VegvisirUtils.fold_auc(valid_predictions,valid_data_blosum[:,0,0,0],valid_accuracy,"argmax_sample",results_dir,mode="Valid")
    stop = time.time()
    print('Final timing: {}'.format(str(datetime.timedelta(seconds=stop-start))))


    if args.test: #TODO: Fix , it is  a mess
        print("Final testing")
        print("NEEDS TO BE FIXED!!!!!!!!!1")
        exit()
        custom_dataset_test = VegvisirLoadUtils.CustomDataset(data_blosum[test_idx],
                                                              data_int[test_idx],
                                                              data_onehot[test_idx],
                                                              data_blosum_norm[test_idx],
                                                              data_array_blosum_encoding_mask[test_idx])
        test_loader = DataLoader(custom_dataset_test, batch_size=batch_size, shuffle=True,
                                 generator=torch.Generator(device=args.device), **kwargs)
        test_predictions,test_accuracy,test_latent_space, test_reconstruction_accuracies_dict = test_loop(Vegvisir,guide,test_loader,args,model_load)

        VegvisirPlots.plot_latent_space(test_latent_space, train_predictions_mode, "all", results_dir, method="Train")
        print("Calculating Monte Carlo estimate of the posterior predictive")
        test_predictions = sample_loop(Vegvisir, guide, test_loader, args)
        test_predictions_mode = stats.mode(test_predictions.cpu().numpy(), axis=1, keepdims=True).mode.squeeze(-1)
        VegvisirUtils.fold_auc(test_predictions_mode, test_data_blosum[:, 0, 0, 0], test_accuracy,"all", results_dir, mode="Test")


