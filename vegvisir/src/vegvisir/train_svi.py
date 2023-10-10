import gc
import json
import signal
import warnings
from argparse import Namespace

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
import vegvisir.similarities as VegvisirSimilarities

from ray.air import Checkpoint, session

ModelLoad = namedtuple("ModelLoad",["args","max_len","seq_max_len","n_data","input_dim","aa_types","blosum","class_weights"])
minidatasetinfo = namedtuple("minidatasetinfo", ["seq_max_len", "corrected_aa_types","num_classes","num_obs_classes","storage_folder"])



def train_loop(svi,Vegvisir,guide,data_loader, args,model_load,epoch):
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
    binary_predictions = []
    logits_predictions = []
    probs_predictions = []
    latent_spaces = []
    z_locs = []
    z_scales = []
    true_labels = []
    confidence_scores = []
    training_assignation = []
    attention_weights = []
    encoder_hidden_states = []
    encoder_final_hidden_states = []
    decoder_hidden_states = []
    decoder_final_hidden_states = []
    reconstruction_logits = []
    data_int = []
    data_masks = []

    for batch_number, batch_dataset in enumerate(data_loader):
        batch_data_blosum = batch_dataset["batch_data_blosum"]
        batch_data_int = batch_dataset["batch_data_int"]
        batch_data_onehot = batch_dataset["batch_data_onehot"]
        batch_data_blosum_norm = batch_dataset["batch_data_blosum_norm"]
        batch_mask = batch_dataset["batch_mask"]
        batch_positional_mask = batch_dataset["batch_positional_mask"]

        if args.use_cuda:
            batch_data_blosum = batch_data_blosum.cuda()
            batch_data_int = batch_data_int.cuda()
            batch_data_onehot = batch_data_onehot.cuda()
            batch_data_blosum_norm = batch_data_blosum_norm.cuda()
            batch_mask = batch_mask.cuda()
            batch_positional_mask = batch_positional_mask.cuda()
        batch_data = {"blosum":batch_data_blosum,
                      "int":batch_data_int,
                      "onehot":batch_data_onehot,
                      "norm":batch_data_blosum_norm,
                      "positional_mask":batch_positional_mask}
        curr_bsize = batch_data["blosum"].shape[0]
        data_int.append(batch_data_int.detach().cpu().numpy())
        data_masks.append(batch_mask.detach().cpu().numpy())
        #Forward & Backward pass
        guide_estimates = guide(batch_data,batch_mask,epoch,None,sample=False)
        loss = svi.step(batch_data,batch_mask,epoch,guide_estimates,sample=False)
        sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=False)(batch_data,batch_mask,epoch = 0,guide_estimates=guide_estimates,sample=True)

        binary_class_prediction = VegvisirUtils.squeeze_tensor(1,sampling_output["predictions"]).detach()
        logits_class_prediction = VegvisirUtils.squeeze_tensor(2,sampling_output["class_logits"]).detach()
        probs_class_prediction = torch.nn.Sigmoid()(logits_class_prediction)
        attn_weights = VegvisirUtils.squeeze_tensor(3,sampling_output["attn_weights"]).detach().cpu().numpy()
        attention_weights.append(attn_weights)
        encoder_hidden = VegvisirUtils.squeeze_tensor(4,sampling_output["encoder_hidden_states"]).detach().cpu().numpy()
        encoder_hidden_states.append(encoder_hidden)
        decoder_hidden = VegvisirUtils.squeeze_tensor(4,sampling_output["decoder_hidden_states"]).detach().cpu().numpy()
        decoder_hidden_states.append(decoder_hidden)
        encoder_final_hidden = VegvisirUtils.squeeze_tensor(2,sampling_output["encoder_final_hidden"]).detach().cpu()
        decoder_final_hidden = VegvisirUtils.squeeze_tensor(2,sampling_output["decoder_final_hidden"]).detach().cpu()
        reconstructed_sequences = VegvisirUtils.squeeze_tensor(2,sampling_output["sequences"]).detach().cpu()
        reconstruction_logits_batch = VegvisirUtils.squeeze_tensor(3,sampling_output["sequences_logits"]).detach().cpu()
        reconstruction_logits.append(reconstruction_logits_batch)


        z_loc = VegvisirUtils.squeeze_tensor(2,guide_estimates["z_loc"]).detach().cpu().numpy()
        z_locs.append(z_loc)
        z_scale = VegvisirUtils.squeeze_tensor(2,guide_estimates["z_scale"]).detach().cpu().numpy()
        z_scales.append(z_scale)
        latent_space = VegvisirUtils.squeeze_tensor(2,sampling_output["latent_z"]).detach().cpu()
        true_labels_batch = batch_data["blosum"][:, 0, 0, 0].detach().cpu()
        identifiers = batch_data["blosum"][:, 0, 0, 1].detach().cpu()
        partitions = batch_data["blosum"][:, 0, 0, 2].detach().cpu()
        training = batch_data["blosum"][:, 0, 0, 3].detach().cpu()
        immunodominace_score = batch_data["blosum"][:, 0, 0, 4].detach().cpu()
        confidence_score = batch_data["blosum"][:, 0, 0, 5].detach().cpu()
        latent_space = torch.column_stack([true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, latent_space])
        encoder_final_hidden= torch.column_stack(
            [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, encoder_final_hidden])
        decoder_final_hidden= torch.column_stack(
            [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, decoder_final_hidden])
        encoder_final_hidden_states.append(encoder_final_hidden.numpy())
        decoder_final_hidden_states.append(decoder_final_hidden.numpy())

        mask_seq = batch_mask[:, 1:,:,0].squeeze(1).detach().cpu()
        equal_aa = torch.Tensor((batch_data_int[:,1,:model_load.seq_max_len].detach().cpu() == reconstructed_sequences.long())*mask_seq)
        reconstruction_accuracy = (equal_aa.sum(dim=1))/mask_seq.sum(dim=1)

        reconstruction_accuracies.append(reconstruction_accuracy.cpu().numpy())
        latent_spaces.append(latent_space.detach().cpu().numpy())
        true_labels.append(true_labels_batch.detach().cpu().numpy())
        confidence_scores.append(confidence_score.detach().cpu().numpy())
        training_assignation.append(training.detach().cpu().numpy())

        binary_predictions.append(binary_class_prediction.detach().cpu().numpy())
        logits_predictions.append(logits_class_prediction.detach().cpu().numpy())
        probs_predictions.append(probs_class_prediction.detach().cpu().numpy())

        train_loss += loss
    #Normalize train loss
    train_loss /= len(data_loader)
    data_int_arr = np.concatenate(data_int,axis=0)
    data_mask_arr = np.concatenate(data_masks,axis=0)

    true_labels_arr = np.concatenate(true_labels,axis=0)
    binary_predictions_arr = np.concatenate(binary_predictions,axis=0)
    logits_predictions_arr = np.concatenate(logits_predictions,axis=0)
    probs_predictions_arr = np.concatenate(probs_predictions,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    z_locs_arr = np.concatenate(z_locs,axis=0)
    z_scales_arr = np.concatenate(z_scales,axis=0)
    confidence_scores_arr = np.concatenate(confidence_scores, axis=0)
    training_assignation_arr = np.concatenate(training_assignation, axis=0)

    attention_weights_arr = np.concatenate(attention_weights,axis=0)
    encoder_hidden_arr = np.concatenate(encoder_hidden_states,axis=0)
    decoder_hidden_arr = np.concatenate(decoder_hidden_states,axis=0)
    encoder_final_hidden_arr = np.concatenate(encoder_final_hidden_states,axis=0)
    decoder_final_hidden_arr = np.concatenate(decoder_final_hidden_states,axis=0)

    reconstruction_accuracies_arr = np.concatenate(reconstruction_accuracies)
    reconstruction_logits_arr = np.concatenate(reconstruction_logits,axis=0)
    reconstruction_entropy = VegvisirUtils.compute_sites_entropies(reconstruction_logits_arr, true_labels_arr) #Highlight: first term is the label, then the entropy per position
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies_arr.mean(),
                                      "std":reconstruction_accuracies_arr.std(),
                                      "entropies": np.mean(reconstruction_entropy[:,1:],axis=0)}

    if args.num_classes == args.num_obs_classes:
        observed_labels = true_labels_arr
        target_accuracy = 100 * ((true_labels_arr == binary_predictions_arr).sum() / true_labels_arr.shape[0])

    else:
        confidence_mask = (true_labels_arr[..., None] != 2).any(-1) #Highlight: unlabelled data has been assigned labelled 2
        observed_labels = true_labels_arr.copy()
        observed_labels[~confidence_mask] = 0 #fake class 0 for unobserved data,will be ignored
        target_accuracy = 100 * ((true_labels_arr[confidence_mask] == binary_predictions_arr[confidence_mask]).sum() / true_labels_arr[confidence_mask].shape[0])

    true_onehot = np.zeros((true_labels_arr.shape[0],args.num_obs_classes))
    true_onehot[np.arange(0,true_labels_arr.shape[0]),observed_labels.astype(int)] = 1
    predictions_dict = {"data_int":data_int_arr,
                        "data_mask": data_mask_arr,
                        "binary":binary_predictions_arr,
                        "logits":logits_predictions_arr,
                        "probs":probs_predictions_arr,
                        "true":true_labels_arr,
                        "true_onehot":true_onehot,
                        "observed":observed_labels,
                        "confidence_scores":confidence_scores_arr,
                        "training_assignation":training_assignation_arr,
                        "attention_weights":attention_weights_arr,
                        "encoder_hidden_states":encoder_hidden_arr,
                        "decoder_hidden_states":decoder_hidden_arr,
                        "encoder_final_hidden_state": encoder_final_hidden_arr,
                        "decoder_final_hidden_state": decoder_final_hidden_arr,
                        "z_locs": z_locs_arr,
                        "z_scales":z_scales_arr,
                        "latent_z": latent_arr
                        }
    return train_loss,target_accuracy,predictions_dict,latent_arr, reconstruction_accuracies_dict
def valid_loop(svi,Vegvisir,guide, data_loader, args,model_load,epoch):
    """
    :param svi: pyro infer engine
    :param dataloader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    Vegvisir.train(False)
    Vegvisir.eval()
    valid_loss = 0.0
    binary_predictions = []
    logits_predictions = []
    probs_predictions = []
    z_locs = []
    z_scales= []
    latent_spaces = []
    reconstruction_accuracies = []
    true_labels = []
    confidence_scores = []
    training_assignation = []
    attention_weights=[]
    encoder_hidden_states = []
    encoder_final_hidden_states = []
    decoder_hidden_states = []
    decoder_final_hidden_states = []
    reconstruction_logits = []
    data_int=[]
    data_masks = []
    with torch.no_grad(): #do not update parameters with the evaluation data
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_data_int = batch_dataset["batch_data_int"]
            batch_data_onehot = batch_dataset["batch_data_onehot"]
            batch_data_blosum_norm = batch_dataset["batch_data_blosum_norm"]
            batch_mask = batch_dataset["batch_mask"]
            batch_positional_mask = batch_dataset["batch_positional_mask"]

            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda() #TODO: Automatize for any kind of input (blosum encoding, integers, one-hot)
                batch_data_int = batch_data_int.cuda()
                batch_data_onehot = batch_data_onehot.cuda()
                batch_data_blosum_norm = batch_data_blosum_norm.cuda()
                batch_mask = batch_mask.cuda()
                batch_positional_mask = batch_positional_mask.cuda()
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int,
                          "onehot": batch_data_onehot,"norm":batch_data_blosum_norm,
                          "positional_mask":batch_positional_mask}
            curr_bsize = batch_data["blosum"].shape[0]
            data_int.append(batch_data_int.detach().cpu().numpy())
            data_masks.append(batch_mask.detach().cpu().numpy())

            guide_estimates = guide(batch_data, batch_mask,epoch, None, sample=False)

            loss = svi.step(batch_data, batch_mask, epoch, guide_estimates, sample=False)

            sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=False)(batch_data, batch_mask,epoch = 0,guide_estimates=guide_estimates,sample=True)


            binary_class_prediction = VegvisirUtils.squeeze_tensor(1, sampling_output["predictions"]).detach()
            logits_class_prediction = VegvisirUtils.squeeze_tensor(2, sampling_output["class_logits"]).detach()
            probs_class_prediction = torch.nn.Sigmoid()(logits_class_prediction)
            attn_weights = VegvisirUtils.squeeze_tensor(3, sampling_output["attn_weights"]).detach().cpu().numpy()
            attention_weights.append(attn_weights)
            encoder_hidden = VegvisirUtils.squeeze_tensor(4, sampling_output["encoder_hidden_states"]).detach().cpu().numpy()
            encoder_hidden_states.append(encoder_hidden)
            decoder_hidden = VegvisirUtils.squeeze_tensor(4, sampling_output["decoder_hidden_states"]).detach().cpu().numpy()
            decoder_hidden_states.append(decoder_hidden)
            encoder_final_hidden = VegvisirUtils.squeeze_tensor(2,sampling_output["encoder_final_hidden"]).detach().cpu()
            decoder_final_hidden = VegvisirUtils.squeeze_tensor(2,sampling_output["decoder_final_hidden"]).detach().cpu()
            reconstructed_sequences = VegvisirUtils.squeeze_tensor(2, sampling_output["sequences"]).detach().cpu()
            reconstruction_logits_batch = VegvisirUtils.squeeze_tensor(3, sampling_output["sequences_logits"]).detach().cpu()
            reconstruction_logits.append(reconstruction_logits_batch)

            z_loc = VegvisirUtils.squeeze_tensor(2, guide_estimates["z_loc"]).detach().cpu().numpy()
            z_locs.append(z_loc)
            z_scale = VegvisirUtils.squeeze_tensor(2, guide_estimates["z_scale"]).detach().cpu().numpy()
            z_scales.append(z_scale)

            latent_space = VegvisirUtils.squeeze_tensor(2,sampling_output["latent_z"]).detach().cpu()
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0].detach().cpu()
            identifiers = batch_data["blosum"][:, 0, 0, 1].detach().cpu()
            partitions = batch_data["blosum"][:,0,0,2].detach().cpu()
            training = batch_data["blosum"][:,0,0,3].detach().cpu()
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4].detach().cpu()
            confidence_score = batch_data["blosum"][:, 0, 0, 5].detach().cpu()

            latent_space = torch.column_stack([true_labels_batch,identifiers, partitions, immunodominace_score,confidence_score, latent_space])
            encoder_final_hidden = torch.column_stack([true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,encoder_final_hidden])
            decoder_final_hidden = torch.column_stack([true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,decoder_final_hidden])

            encoder_final_hidden_states.append(encoder_final_hidden.numpy())
            decoder_final_hidden_states.append(decoder_final_hidden.numpy())
            mask_seq = batch_mask[:, 1:, :, 0].squeeze(1).detach().cpu()
            equal_aa = torch.Tensor((batch_data_int[:, 1, :model_load.seq_max_len].detach().cpu() == reconstructed_sequences) * mask_seq)
            reconstruction_accuracy = (equal_aa.sum(dim=1)) / mask_seq.sum(dim=1)
            reconstruction_accuracies.append(reconstruction_accuracy.detach().cpu().numpy())
            latent_spaces.append(latent_space.cpu().detach().numpy())
            true_labels.append(true_labels_batch.cpu().detach().numpy())
            confidence_scores.append(confidence_score.detach().cpu().numpy())
            training_assignation.append(training.detach().cpu().numpy())

            binary_predictions.append(binary_class_prediction.detach().cpu().numpy())
            logits_predictions.append(logits_class_prediction.detach().cpu().numpy())
            probs_predictions.append(probs_class_prediction.detach().cpu().numpy())
            valid_loss += loss #TODO: Multiply by the data size?
    valid_loss /= len(data_loader)
    data_int_arr = np.concatenate(data_int,axis=0)
    data_mask_arr = np.concatenate(data_masks,axis=0)

    binary_predictions_arr = np.concatenate(binary_predictions, axis=0)
    logits_predictions_arr = np.concatenate(logits_predictions, axis=0)
    probs_predictions_arr = np.concatenate(probs_predictions, axis=0)
    true_labels_arr = np.concatenate(true_labels,axis=0)
    confidence_scores_arr = np.concatenate(confidence_scores, axis=0)
    training_asignation_arr = np.concatenate(training_assignation, axis=0)

    z_locs_arr = np.concatenate(z_locs,axis=0)
    z_scales_arr = np.concatenate(z_scales,axis=0)
    latent_arr = np.concatenate(latent_spaces,axis=0)
    attention_weights_arr = np.concatenate(attention_weights,axis=0)
    encoder_hidden_arr = np.concatenate(encoder_hidden_states,axis=0)
    decoder_hidden_arr = np.concatenate(decoder_hidden_states,axis=0)
    encoder_final_hidden_arr = np.concatenate(encoder_final_hidden_states,axis=0)
    decoder_final_hidden_arr = np.concatenate(decoder_final_hidden_states,axis=0)
    reconstruction_accuracies_arr = np.concatenate(reconstruction_accuracies)
    reconstruction_logits_arr= np.concatenate(reconstruction_logits,axis=0)
    reconstruction_entropy = VegvisirUtils.compute_sites_entropies(reconstruction_logits_arr, true_labels_arr) #Highlight: first term is the label, then the entropy per position
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies_arr.mean(),
                                      "std":reconstruction_accuracies_arr.std(),
                                      "entropies": np.mean(reconstruction_entropy[:,1:],axis=0)}
    if args.num_classes == args.num_obs_classes:
        observed_labels = true_labels_arr
        target_accuracy = 100 * ((true_labels_arr == binary_predictions_arr).sum() / true_labels_arr.shape[0])

    else:
        confidence_mask = (true_labels_arr[..., None] != 2).any(-1)  # Highlight: unlabelled data has been assigned labelled 2, we give high confidence to the labelled data (for now)
        observed_labels = true_labels_arr.copy()
        observed_labels[~confidence_mask] = 0
        target_accuracy = 100 * ((true_labels_arr[confidence_mask] == binary_predictions_arr[confidence_mask]).sum() / true_labels_arr[confidence_mask].shape[0])

    true_onehot = np.zeros((true_labels_arr.shape[0],args.num_obs_classes))
    true_onehot[np.arange(0,true_labels_arr.shape[0]),observed_labels.astype(int)] = 1
    predictions_dict = {"data_int": data_int_arr ,
                        "data_mask": data_mask_arr,
                        "binary":binary_predictions_arr,
                        "logits":logits_predictions_arr,
                        "probs":probs_predictions_arr,
                        "true": true_labels_arr,
                        "true_onehot": true_onehot,
                        "observed": observed_labels,
                        "confidence_scores":confidence_scores_arr,
                        "training_assignation":training_asignation_arr,
                        "attention_weights":attention_weights_arr,
                        "encoder_hidden_states": encoder_hidden_arr,
                        "decoder_hidden_states": decoder_hidden_arr,
                        "encoder_final_hidden_state": encoder_final_hidden_arr,
                        "decoder_final_hidden_state": decoder_final_hidden_arr,
                        "z_locs":z_locs_arr,
                        "z_scales":z_scales_arr,
                        "latent_z":latent_arr
                        }

    return valid_loss,target_accuracy,predictions_dict,latent_arr, reconstruction_accuracies_dict
def sample_loop(svi, Vegvisir, guide, data_loader, args, model_load):
    """
    :param svi: pyro infer engine
    :param dataloader data_loader: Pytorch dataloader
    :param namedtuple args
    """
    Vegvisir.train(False)
    Vegvisir.eval()
    sample_loss = 0.0
    binary_predictions = []
    logits_predictions = []
    probs_predictions = []
    z_locs = []
    z_scales = []
    latent_spaces = []
    reconstruction_accuracies = []
    attention_weights = []
    encoder_hidden_states = []
    encoder_final_hidden_states = []
    decoder_hidden_states = []
    decoder_final_hidden_states = []
    true_labels = []
    confidence_scores = []
    training_asignation = []
    data_int=[]
    data_masks= []
    with torch.no_grad():  # do not update parameters with the evaluation data
        for batch_number, batch_dataset in enumerate(data_loader):
            batch_data_blosum = batch_dataset["batch_data_blosum"]
            batch_data_int = batch_dataset["batch_data_int"]
            batch_data_onehot = batch_dataset["batch_data_onehot"]
            batch_data_blosum_norm = batch_dataset["batch_data_blosum_norm"]
            batch_mask = batch_dataset["batch_mask"]
            batch_positional_mask = batch_dataset["batch_positional_mask"]
            if args.use_cuda:
                batch_data_blosum = batch_data_blosum.cuda()  # TODO: Automatize for any kind of input (blosum encoding, integers, one-hot)
                batch_data_int = batch_data_int.cuda()
                batch_data_onehot = batch_data_onehot.cuda()
                batch_data_blosum_norm = batch_data_blosum_norm.cuda()
                batch_mask = batch_mask.cuda()
                batch_positional_mask = batch_positional_mask.cuda()
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,
                          "norm": batch_data_blosum_norm,
                          "positional_mask":batch_positional_mask}
            data_int.append(batch_data_int.detach().cpu().numpy())
            data_masks.append(batch_mask.detach().cpu().numpy())

            guide_estimates = guide(batch_data, batch_mask, epoch=0,guide_estimates=None,sample=False)
            sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=args.num_samples, return_sites=(), parallel=False)(batch_data, batch_mask,epoch=0,guide_estimates=guide_estimates, sample=True)

            z_loc = VegvisirUtils.squeeze_tensor(2, guide_estimates["z_loc"]).detach().cpu().numpy()
            z_locs.append(z_loc)
            z_scale = VegvisirUtils.squeeze_tensor(2, guide_estimates["z_scale"]).detach().cpu().numpy()
            z_scales.append(z_scale)

            if sampling_output["predictions"].shape == (args.num_samples,batch_data["blosum"].shape[0]):
                binary_class_prediction = sampling_output["predictions"].detach().T
            else:
                binary_class_prediction = sampling_output["predictions"].squeeze(1).detach().T
                binary_class_prediction = binary_class_prediction.squeeze(1) #necessary sometimes when adding .to_event(1)

            logits_class_prediction = sampling_output["class_logits"].detach().permute(1,0,2)
            probs_class_prediction = torch.nn.Sigmoid()(logits_class_prediction)
            reconstructed_sequences = sampling_output["sequences"].detach().permute(1,0,2)
            attn_weights = sampling_output["attn_weights"].squeeze(0).detach().cpu().permute(1,0,2,3).numpy()
            attention_weights.append(attn_weights)
            encoder_hidden = sampling_output["encoder_hidden_states"].squeeze(0).detach().cpu().permute(1,0,2,3,4).numpy()
            #encoder_hidden = encoder_hidden.mean(axis=1)
            encoder_hidden_states.append(encoder_hidden)
            decoder_hidden = sampling_output["decoder_hidden_states"].squeeze(0).detach().cpu().permute(1,0,2,3,4).numpy()
            #decoder_hidden = decoder_hidden.mean(axis=1)
            decoder_hidden_states.append(decoder_hidden)
            encoder_final_hidden = sampling_output["encoder_final_hidden"].squeeze(0).detach().cpu().permute(1,0,2)
            encoder_final_hidden = encoder_final_hidden.mean(dim=1) #TODO: this might not be correct, think about it
            decoder_final_hidden = sampling_output["decoder_final_hidden"].squeeze(0).detach().cpu().permute(1,0,2)
            decoder_final_hidden = decoder_final_hidden.mean(dim=1)

            if sampling_output["latent_z"].ndim == 4:
                latent_space = sampling_output["latent_z"].squeeze(1).detach().permute(1,0,2)[:,0,:].detach().cpu()
            else:
                latent_space = sampling_output["latent_z"].detach().permute(1,0,2)[:,0,:].detach().cpu()
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0].detach().cpu()
            identifiers = batch_data["blosum"][:, 0, 0, 1].detach().cpu()
            partitions = batch_data["blosum"][:, 0, 0, 2].detach().cpu()
            training = batch_data["blosum"][:, 0, 0, 3].detach().cpu()
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4].detach().cpu()
            confidence_score = batch_data["blosum"][:, 0, 0, 5].detach().cpu()


            latent_space = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, latent_space])
            encoder_final_hidden = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,
                 encoder_final_hidden])
            decoder_final_hidden = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,
                 decoder_final_hidden])
            encoder_final_hidden_states.append(encoder_final_hidden.detach().cpu().numpy())
            decoder_final_hidden_states.append(decoder_final_hidden.detach().cpu().numpy())

            mask_seq = batch_mask[:, 1:, :, 0].squeeze(1)
            true_seqs = batch_data_int[:, 1, :model_load.seq_max_len]
            equal_aa = torch.Tensor((true_seqs[:,None] == reconstructed_sequences) * mask_seq[:,None])
            reconstruction_accuracy = (equal_aa.sum(dim=2)) / mask_seq.sum(dim=1)[:,None] #Reconstruction accuracy of each sample against the true sequence
            reconstruction_accuracies.append(reconstruction_accuracy.detach().cpu().numpy())
            latent_spaces.append(latent_space.detach().cpu().numpy())
            true_labels.append(true_labels_batch.detach().cpu().numpy())
            confidence_scores.append(confidence_score.detach().cpu().numpy())
            training_asignation.append(training.detach().cpu().numpy())
            binary_predictions.append(binary_class_prediction.detach().cpu().numpy())
            logits_predictions.append(logits_class_prediction.detach().cpu().numpy())
            probs_predictions.append(probs_class_prediction.detach().cpu().numpy())
            #sample_loss += loss
    sample_loss /= len(data_loader)
    data_int_arr = np.concatenate(data_int,axis=0)
    data_mask_arr = np.concatenate(data_masks,axis=0)

    binary_predictions_arr = np.concatenate(binary_predictions, axis=0)
    logits_predictions_arr = np.concatenate(logits_predictions, axis=0)
    probs_predictions_arr = np.concatenate(probs_predictions, axis=0)
    true_labels_arr = np.concatenate(true_labels, axis=0)
    confidence_scores_arr = np.concatenate(confidence_scores, axis=0)
    training_assignation_arr = np.concatenate(training_asignation, axis=0)

    attention_weights_arr = np.concatenate(attention_weights,axis=0)
    encoder_hidden_arr = np.concatenate(encoder_hidden_states,axis=0)
    decoder_hidden_arr = np.concatenate(decoder_hidden_states,axis=0)
    encoder_final_hidden_arr = np.concatenate(encoder_final_hidden_states,axis=0)
    decoder_final_hidden_arr = np.concatenate(decoder_final_hidden_states,axis=0)

    latent_arr = np.concatenate(latent_spaces, axis=0)
    z_locs_arr = np.concatenate(z_locs, axis=0)
    z_scales_arr = np.concatenate(z_scales, axis=0)

    target_accuracy = 100 * ((true_labels_arr[:,None] == binary_predictions_arr).astype(float).mean(axis=1).mean(axis=0))
    reconstruction_accuracies_arr = np.concatenate(reconstruction_accuracies).mean(axis=1) #[N,num_samples,1]
    reconstruction_accuracies_dict = {"mean": reconstruction_accuracies_arr.mean(), "std": reconstruction_accuracies_arr.std()}
    if args.num_classes == args.num_obs_classes:
        observed_labels = true_labels_arr
    else:
        confidence_mask = (true_labels_arr[..., None] != 2).any(-1)  # Highlight: unlabelled data has been assigned labelled 2, we give high confidence to the labelled data (for now)
        observed_labels = true_labels_arr.copy()
        observed_labels[~confidence_mask] = 0
    true_onehot = np.zeros((true_labels_arr.shape[0],args.num_obs_classes))
    true_onehot[np.arange(0,true_labels_arr.shape[0]),observed_labels.astype(int)] = 1
    predictions_dict = {"data_int":data_int_arr,
                        "data_mask":data_mask_arr,
                        "binary": binary_predictions_arr,
                        "logits": logits_predictions_arr,
                        "probs": probs_predictions_arr,
                        "true":true_labels_arr,
                        "true_onehot": true_onehot,
                        "observed": observed_labels,
                        "accuracy":target_accuracy,
                        "training_assignation":training_assignation_arr,
                        "confidence_scores":confidence_scores_arr,
                        "attention_weights":attention_weights_arr,
                        "encoder_hidden_states": encoder_hidden_arr,
                        "decoder_hidden_states": decoder_hidden_arr,
                        "encoder_final_hidden_state":encoder_final_hidden_arr,
                        "decoder_final_hidden_state": decoder_final_hidden_arr,
                        "z_locs":z_locs_arr,
                        "z_scales":z_scales_arr,
                        "latent_z":latent_arr
                        }
    return sample_loss, target_accuracy, predictions_dict, latent_arr, reconstruction_accuracies_dict
def generate_loop(svi, Vegvisir, guide, data_loader, args, model_load,dataset_info,additional_info,train_predictive_samples_dict):
    Vegvisir.train(False)
    Vegvisir.eval()
    num_synthetic_peptides = args.num_synthetic_peptides
    assert num_synthetic_peptides < 10000, "Please generate less than 10000 peptides, otherwise the computations might not be posible"
    argmax = args.generate_argmax
    with torch.no_grad():  # do not update parameters with the evaluation data
        #Highlight: Initalize fake dummy data (not used)
        batch_data = {"blosum":torch.randint(low=-7,high=7,size=(num_synthetic_peptides,2,model_load.seq_max_len,model_load.aa_types)).double().to(device=args.device),
                      "onehot":torch.randint(low=0,high=1,size=(num_synthetic_peptides,2,model_load.seq_max_len,model_load.aa_types)).double().to(device=args.device),
                      "norm": torch.randn(size=(num_synthetic_peptides,2,model_load.seq_max_len,model_load.aa_types)).double().to(device=args.device),
                      "int": torch.randint(low=0,high=21,size=(num_synthetic_peptides,2,model_load.seq_max_len,model_load.aa_types)).double().to(device=args.device),
                      "positional_mask": torch.ones((num_synthetic_peptides,model_load.seq_max_len)).bool().to(device=args.device),
                      }
        if len(dataset_info.unique_lens) > 1:
            lenghts_sample = np.random.choice([8,9,10,11],(num_synthetic_peptides,),replace=True,p=[0.1,0.7,0.1,0.1]).tolist()
            blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(dataset_info.corrected_aa_types,
                                                                                       args.subs_matrix,
                                                                                       zero_characters=["#"],
                                                                                       include_zero_characters=True)
            zero_character = 0
        else:
            lenghts_sample = np.ones((num_synthetic_peptides,))*model_load.seq_max_len
            lenghts_sample = lenghts_sample.astype(int)
            blosum_array, blosum_dict, blosum_array_dict = VegvisirUtils.create_blosum(dataset_info.corrected_aa_types,
                                                                                       args.subs_matrix,
                                                                                       zero_characters=[],
                                                                                       include_zero_characters=False)
            zero_character = None
        batch_mask = list(map(lambda length: VegvisirUtils.generate_mask(model_load.seq_max_len,length),lenghts_sample))
        batch_mask = torch.from_numpy(np.concatenate(batch_mask,axis=0))

        batch_mask_blosum = np.broadcast_to(batch_mask[:, None, :, None], (num_synthetic_peptides, 2, model_load.seq_max_len,model_load.aa_types)).copy()
        batch_mask_blosum = torch.from_numpy(batch_mask_blosum).to(args.device)

        h_0_GUIDE = [param for key,param in guide.named_parameters() if key == "h_0_GUIDE"][0]


        guide_estimates = {
                           "rnn_hidden":h_0_GUIDE.expand(1 * 2, num_synthetic_peptides,args.hidden_dim*2).contiguous(),
                           #"rnn_hidden":None,
                           "rnn_final_hidden":torch.ones((num_synthetic_peptides, args.hidden_dim*2)).to(device=args.device),
                           "rnn_final_hidden_bidirectional": h_0_GUIDE.expand(1 * 2, num_synthetic_peptides,args.hidden_dim*2).contiguous(),#Highlight: Not used
                           "rnn_hidden_states_bidirectional" : torch.ones((num_synthetic_peptides, 2, dataset_info.seq_max_len, args.hidden_dim*2)).to(device=args.device),
                           "rnn_hidden_states" : torch.ones((num_synthetic_peptides, dataset_info.seq_max_len, args.hidden_dim*2)).to(device=args.device),
                           "latent_z": train_predictive_samples_dict["latent_z"],
                           "z_scales": train_predictive_samples_dict["z_scales"],
                           "generate":True
                           }
        #guide_estimates = None

        #sampling_output = Vegvisir.sample(batch_data, batch_mask_blosum, epoch=0, guide_estimates=guide_estimates,sample=True,argmax=argmax)
        sampling_output = Predictive(Vegvisir.model, guide=None, num_samples=args.generate_num_samples, return_sites=(),parallel=False)(batch_data, batch_mask_blosum, epoch=0, guide_estimates=guide_estimates,sample=True)

        custom_features_dicts = VegvisirUtils.build_features_dicts(dataset_info)
        aminoacids_dict_reversed = custom_features_dicts["aminoacids_dict_reversed"]

        #Highlight: majority vote? most likely?
        if argmax:
            sequences_logits = sampling_output["sequences_logits"].detach().cpu().permute(1,0,2,3)
            generated_sequences_int = torch.argmax(sequences_logits,dim=-1)
            generated_sequences_int = torch.mode(generated_sequences_int,dim=1).values.numpy()
        else:
            generated_sequences_int = sampling_output["sequences"].detach().cpu().permute(1, 0, 2)
            generated_sequences_int = torch.mode(generated_sequences_int,dim=1).values.numpy()


        #Highlight: Plot before removing duplicates
        generated_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(generated_sequences_int)

        if sampling_output["predictions"].shape == (args.num_samples, num_synthetic_peptides):
            binary_predictions = sampling_output["predictions"].detach().cpu().permute(1, 0).numpy()
        else:
            binary_predictions = sampling_output["predictions"].squeeze(1).detach().cpu().permute(1, 0).numpy()
        binary_mode = stats.mode(binary_predictions, axis=1, keepdims=True).mode.squeeze(-1)
        binary_frequencies = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.num_classes), axis=1,arr=binary_predictions.astype("int64"))
        binary_frequencies = binary_frequencies / args.generate_num_samples

        VegvisirUtils.numpy_to_fasta(generated_sequences_raw, binary_mode, binary_frequencies,"{}/Generated".format(additional_info.results_dir),"_NOT_FILTERED")

        # Highlight: Remove inner duplicates (before discarding the ones that have # in strange places)
        unique_sequences, unique_idx = np.unique(generated_sequences_int, axis=0, return_index=True)
        # Highlight: Remove identical sequences to the training dataset
        train_dataset = torch.concatenate([batch_data["batch_data_int"][:, 1].squeeze(1) for batch_data in data_loader],dim=0).detach().cpu().numpy()
        # train_raw = np.vectorize(aminoacids_dict_reversed.get)(train_dataset)
        # VegvisirPlots.plot_logos(list(map(lambda seq: "{}".format("".join(seq).replace("#","-")), train_raw.tolist())), additional_info.results_dir, "_TRAIN_raw")

        identical_to_train_bool = np.any(np.array(generated_sequences_int[:, None] == train_dataset[None, :]).all((-1)) == True, axis=0)
        identical_to_train_idx, = np.where(identical_to_train_bool == True)

        unique_idx = np.array(np.arange(num_synthetic_peptides)[..., None] == unique_idx).any(-1)
        identical_to_train_idx = np.invert(np.array(np.arange(num_synthetic_peptides)[..., None] == identical_to_train_idx).any(-1))

        # Highlight: Merge the indicators of the NON duplicates (discard the duplicates, keep the unique ones)
        unique_idx = unique_idx * identical_to_train_idx

        #Highlight: Remove the sequences that have a gap in positions < 8
        if zero_character is not None:
            clean_results = list(map(lambda seq_int,seq_mask: VegvisirUtils.clean_generated_sequences(seq_int,seq_mask,zero_character,min_len=8), generated_sequences_int.tolist(),batch_mask.numpy().tolist()))
            discarded_sequences =  list(map(lambda v,i: i if v is None else None, clean_results,list(range(len(clean_results)))))
            discarded_sequences =  np.array(list(filter(lambda i: i is not None, discarded_sequences)))
            clean_results = list(filter(lambda v: v is not None, clean_results))

            clean_results = list(zip(*clean_results))
            #clean_generated_sequences = clean_results[0]
            clean_generated_masks = clean_results[1]
            #generated_sequences_int = np.concatenate(clean_generated_sequences,axis=0)
            generated_sequences_mask = np.concatenate(clean_generated_masks,axis=0) #contains the truncated masks

            keep_idx = np.invert((np.arange(num_synthetic_peptides)[..., None] == discarded_sequences).any(-1))
            batch_mask = batch_mask.numpy()
            batch_mask[keep_idx] = generated_sequences_mask

        else:
            generated_sequences_mask = np.ones_like(generated_sequences_int).astype(bool)
            #discarded_sequences = None
            keep_idx = np.ones(num_synthetic_peptides).astype(bool)


    keep_idx = keep_idx * unique_idx

    generated_sequences_int = generated_sequences_int[keep_idx]
    batch_mask = generated_sequences_mask = batch_mask[keep_idx]


    generated_sequences_raw = np.vectorize(aminoacids_dict_reversed.get)(generated_sequences_int)
    generated_sequences_blosum = np.vectorize(blosum_array_dict.get, signature='()->(n)')(generated_sequences_int)

    if generated_sequences_blosum.shape[0] < 10000:
        print(generated_sequences_blosum.shape[0])

        def handle_timeout(signum, frame):
            raise TimeoutError

        signal.signal(signal.SIGALRM, handle_timeout)
        signal.alarm(600)  # 10 minutes
        try:
            generated_sequences_cosine_similarity = VegvisirSimilarities.cosine_similarity(generated_sequences_blosum,
                                                                                           generated_sequences_blosum,
                                                                                           correlation_matrix=False,
                                                                                           parallel=False)
            batch_size = 100 if generated_sequences_blosum.shape[0] > 100 else generated_sequences_blosum.shape[0]
            positional_weights = VegvisirSimilarities.importance_weight(generated_sequences_cosine_similarity, model_load.seq_max_len, generated_sequences_mask,batch_size=batch_size, neighbours=1)
            VegvisirPlots.plot_heatmap(positional_weights, "Cosine similarity \n positional weights","{}/Generated/Generated_epitopes_positional_weights.png".format(additional_info.results_dir))
        except:
            print("Could not calculate conservational positional weights. Time exceeded or some other error")
        finally:
            signal.alarm(0)

    n_seqs_clean = generated_sequences_int.shape[0]
    class_logits = sampling_output["class_logits"].detach().cpu().permute(1,0,2)
    class_probabilities = torch.nn.Sigmoid()(class_logits)

    if sampling_output["predictions"].shape == (args.num_samples,num_synthetic_peptides):
        binary_predictions = sampling_output["predictions"].detach().cpu().permute(1,0).numpy()[keep_idx]
    else:
        binary_predictions = sampling_output["predictions"].squeeze(1).detach().cpu().permute(1,0).numpy()[keep_idx]

    confidence_score = torch.zeros(n_seqs_clean).detach().cpu()
    #Highlight: Label the generated sequences as unobserved
    true_labels = np.ones(n_seqs_clean)*2
    true_onehot = np.zeros((n_seqs_clean, 3))
    true_onehot[np.arange(0, n_seqs_clean), true_labels.astype(int)] = 1

    binary_mode = stats.mode(binary_predictions, axis=1,keepdims=True).mode.squeeze(-1)
    binary_frequencies = np.apply_along_axis(lambda x: np.bincount(x, minlength=args.num_classes), axis=1, arr=binary_predictions.astype("int64"))
    binary_frequencies = binary_frequencies / args.generate_num_samples

    VegvisirUtils.numpy_to_fasta(generated_sequences_raw,binary_mode,binary_frequencies,"{}/Generated".format(additional_info.results_dir))

    #Highlight: Save sequences
    #TODO: Finish dict and look into transferrin the h_0_encoder to h_0_decoder somehow
    generated_out_dict =  {"data_int":generated_sequences_int,
                        "data_raw": generated_sequences_raw,
                        "data_mask":generated_sequences_mask,
                        "binary": binary_predictions,
                        "logits": class_logits.numpy()[keep_idx],
                        "probs": class_probabilities[keep_idx],
                        "true":true_labels,
                        "true_onehot": true_onehot,
                        "confidence_scores":confidence_score.detach().cpu().numpy(),
                        "attention_weights":sampling_output["attn_weights"].permute(1,0,2,3).detach().cpu().numpy()[keep_idx],
                        "encoder_hidden_states": sampling_output["encoder_hidden_states"].permute(1,0,2,3,4).detach().cpu().numpy()[keep_idx],
                        "decoder_hidden_states": sampling_output["decoder_hidden_states"].permute(1,0,2,3,4).detach().cpu().numpy()[keep_idx],
                        "encoder_final_hidden_state":sampling_output["encoder_final_hidden"].permute(1,0,2).detach().cpu().numpy()[keep_idx],
                        "decoder_final_hidden_state": sampling_output["decoder_final_hidden"].permute(1,0,2).detach().cpu().numpy()[keep_idx]
                        }


    latent_space = sampling_output["latent_z"].permute(1,0,2).detach().cpu()[keep_idx,0]
    #Highlight: fake infomation to maintain same functions
    identifiers = torch.ones(n_seqs_clean).detach().cpu()
    partitions = torch.ones(n_seqs_clean).detach().cpu()
    immunodominace_score = torch.zeros(n_seqs_clean).detach().cpu()
    latent_space = torch.column_stack([torch.from_numpy(true_labels).detach().cpu(), identifiers, partitions, immunodominace_score, confidence_score, latent_space])

    return generated_out_dict,latent_space.numpy()
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
def select_model(model_load,results_dir,fold,args):
    """Select among the available models at models.py"""
    print(args.learning_type )
    if model_load.seq_max_len == model_load.max_len:
        if args.learning_type == "supervised":
            vegvisir_model = VegvisirModels.VegvisirModel5a_supervised(model_load)
        elif args.learning_type == "unsupervised":
            vegvisir_model = VegvisirModels.VegvisirModel5a_unsupervised(model_load)
        elif args.learning_type == "semisupervised":
            vegvisir_model = VegvisirModels.VegvisirModel5a_semisupervised(model_load)
        elif args.learning_type == "supervised_no_decoder":
            vegvisir_model = VegvisirModels.VegvisirModel5a_supervised_no_decoder(model_load)
    else:
        print("Setting to default supervised mode for blosum encoded sequences")

        vegvisir_model = VegvisirModels.VegvisirModel5a_supervised(model_load)
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
    if args.hpo or args.config_dict:
        #config = json.loads(args.config_dict)
        config = args.config_dict
    else:
        "Default hyperparameters (Clipped Adam optimizer), z dim and GRU"
        config = {
            "lr": 1e-3, #default is 1e-3
            "beta1": 0.95, #coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
            "beta2": 0.999,
            "eps": 1e-8,#term added to the denominator to improve numerical stability (default: 1e-8)
            "weight_decay": 0,#weight_decay: weight decay (L2 penalty) (default: 0)
            "clip_norm": 10,#clip_norm: magnitude of norm to which gradients are clipped (default: 10.0)
            "lrd": 1,#0.1 ** (1 / args.num_epochs), #rate at which learning rate decays (default: 1.0) #https://pyro.ai/examples/svi_part_iv.html
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
    if isinstance(m, nn.Module) and hasattr(m, 'weight') and not (isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.Embedding)):
        nn.init.kaiming_normal_(m.weight,nonlinearity="leaky_relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def reset_weights(m):
    if isinstance(m, nn.Module) and hasattr(m, 'weight') and not (isinstance(m,nn.BatchNorm1d)):
        m.reset_parameters()
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
def set_configuration(args,config,results_dir):
    """Sets the hyperparameter values, either to
    i) values sampled to perform Hyperparameter Optimization with Ray Tune
    ii) The best configuration defined via HPO
    iii) the values set via args NamedTuple in Vegvisir_example.py
    :param Namedtuple args: Current hyperparameter configuration defined in Vegvisir_example (general configuration) and config_build (sets optimizer hyperparameters)
    :param dict or None. If None, not HPO is running

    """
    keys_list = ["batch_size", "likelihood_scale", "encoding", "num_epochs", "hidden_dim", "num_samples","z_dim"]
    if config is not None: #i
        print("Using hyperparameter search from Ray-tune")
        args_dict = vars(args)
        config_dict1 = config.copy()
        for key in keys_list:
            args_dict[key] = config[key] #add the sampled hyperparam
            config_dict1.pop(key,None)
        args_dict["config_dict"] = config_dict1
        #print(args_dict)
        #print("---------------")
        args = Namespace(**args_dict)
        pyro.enable_validation(False)
        if args.use_cuda:
            torch.set_default_tensor_type(torch.cuda.DoubleTensor)
        else:
            torch.set_default_tensor_type(torch.DoubleTensor)
        #json.dump(args_dict, open('{}/commandline_args.txt'.format(results_dir), 'w'), indent=2)
        return args
    elif args.config_dict is not None: #ii
        print("Using best hyperparameter set defined by Ray Tune (or some dict provided by args.config_dict, set to None if you do not want to change them)")
        if isinstance(args.config_dict,str):#if both validate and test == True, the first time we load it from a str path, the second time, we already have load it
            best_hyperparameters_dict = json.load(open(args.config_dict,"r+")) #contains all the hyperparameters
            general_config = best_hyperparameters_dict["general_config"] #batch_size, num_samples ...
            optimizer_config = best_hyperparameters_dict["optimizer_config"] #learning rate, momemtum ...
            args_dict = vars(args)
            for key in keys_list:
                args_dict[key] = general_config[key] #change values to the optimized one
            args_dict["config_dict"] = optimizer_config #
            args = Namespace(**args_dict)
            json.dump(args_dict, open('{}/commandline_args.txt'.format(results_dir), 'w'), indent=2)
            return args
        else:
            print("Second time loading args")
            return args
    else:  #iii
        return args
def kfold_loop(kfolds,dataset_info, args, additional_info,config=None):
    """K-fold cross validation training loop"""

    keys_list = ["train_loss","ROC_AUC_train","average_precision_train","valid_loss","ROC_AUC_valid","average_precision_valid"]
    metrics_summary_dict = dict.fromkeys(keys_list,0)
    for fold, (train_idx, valid_idx) in enumerate(kfolds): #returns k-splits for train and validation
        fold_metrics_summary_dict = epoch_loop(train_idx, valid_idx, dataset_info, args, additional_info, mode="Valid_fold_{}".format(fold),fold="_fold_{}".format(fold),config=config)
        if args.hpo:
            for key,val in metrics_summary_dict.items():
                metrics_summary_dict[key] += fold_metrics_summary_dict[key]
        torch.cuda.empty_cache()
    if args.hpo:
        for key,val in metrics_summary_dict.items():
            metrics_summary_dict[key] = val/args.k_folds #report the average
        session.report(metrics=metrics_summary_dict)
def kfold_crossvalidation(config=None,dataset_info=None,additional_info=None,args=None):
    """Set up k-fold cross validation for the training loop"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    seq_max_len = dataset_info.seq_max_len
    results_dir = additional_info.results_dir
    #Highlight: Train- Test split and kfold generator
    if args.predefined_partitions:
        partitioning_method = ["predefined_partitions" if args.test else"predefined_partitions_discard_test"][0]
        if args.dataset_name in ["viral_dataset6","viral_dataset7"]:
            partitioning_method = "predefined_partitions_diffused_test_create_new_test" if args.test else "predefined_partitions_diffused_test"
    else:
        partitioning_method = "random_stratified"


    traineval_data_blosum,test_data_blosum,kfolds = VegvisirLoadUtils.trainevaltest_split_kfolds(data_blosum,
                                                                                                 args,results_dir,
                                                                                                 seq_max_len,dataset_info.max_len,
                                                                                                 dataset_info.features_names,
                                                                                                 None,method=partitioning_method)



    #Highlight:Also split the rest of arrays
    traineval_idx = (data_blosum[:,0,0,1][..., None] == traineval_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    if args.test:
        test_idx = (data_blosum[:,0,0,1][..., None] == test_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
        print('\t Number test data points: {}; Proportion: {}'.format(test_data_blosum.shape[0],
                                                                      (test_data_blosum.shape[0] * 100) /
                                                                      test_data_blosum.shape[0]))

    print('\t Number Train-valid data points: {}; Proportion: {}'.format(traineval_data_blosum.shape[0],(traineval_data_blosum.shape[0]*100)/traineval_data_blosum.shape[0]))
    if not args.test and args.validate:
        print("K-fold cross validation (Training & Validation)")
        kfold_loop(kfolds, dataset_info, args, additional_info,config)
    elif args.test and args.validate:
        print("Training, Validation and testing")
        print("FIRST: Training & Validation")
        kfold_loop(kfolds, dataset_info, args, additional_info,config)
        print("SECOND: Training + Validation & Testing")
        if args.dataset_name == "viral_dataset7" and not args.test:
            warnings.warn("Test == Valid for dataset7, since the test is diffused onto the train and validation")
        else:
            print("Joining Training & validation datasets to perform testing...")
        for fold in range(args.k_folds):
            epoch_loop(traineval_idx, test_idx, dataset_info, args, additional_info,mode="Test_fold_{}".format(fold),fold="_fold_{}".format(fold),config=config)
    else:
        if args.dataset_name == "viral_dataset7":
            warnings.warn("Test == Valid for dataset7, since the test is diffused onto the train and validation")
        else:
            print("Training & testing  (not validation) ...")
        for fold in range(args.k_folds):
            epoch_loop(traineval_idx, test_idx, dataset_info, args, additional_info,mode="Test_fold_{}".format(fold),fold="_fold_{}".format(fold),config=config)
def epoch_loop(train_idx,valid_idx,dataset_info,args,additional_info,mode="Valid",fold="",config=None):
    print("Remaining objects: {}".format(len(gc.get_objects())))

    args = set_configuration(args,config,additional_info.results_dir)
    #Split the rest of the data (train_data) for train and validation
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_blosum_norm = dataset_info.data_array_blosum_norm
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
    data_array_onehot_encoding_mask = dataset_info.data_array_onehot_encoding_mask

    data_positional_weights_mask = dataset_info.positional_weights_mask

    train_data_blosum = data_blosum[train_idx]
    valid_data_blosum = data_blosum[valid_idx]
    assert (train_data_blosum[:,0,0,1] == data_int[train_idx,0,1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (train_data_blosum[:,0,0,1] == data_onehot[train_idx,0,0,1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (train_data_blosum[:,0,0,1] == data_blosum_norm[train_idx,0,1]).all(), "The data is shuffled and the data frames are comparing the wrong things"

    assert (valid_data_blosum[:,0,0,1] == data_int[valid_idx,0,1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (valid_data_blosum[:,0,0,1] == data_onehot[valid_idx,0,0,1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (valid_data_blosum[:,0,0,1] == data_blosum_norm[valid_idx,0,1]).all(), "The data is shuffled and the data frames are comparing the wrong things"


    n_data = data_blosum.shape[0]
    batch_size = int(args.batch_size)
    results_dir = additional_info.results_dir
    check_point_epoch = [5 if args.num_epochs < 100 else int(args.num_epochs / 50)][0]
    model_load = ModelLoad(args=args,
                           max_len =dataset_info.max_len,
                           seq_max_len= dataset_info.seq_max_len,
                           n_data = dataset_info.n_data,
                           input_dim = dataset_info.input_dim,
                           aa_types = dataset_info.corrected_aa_types,
                           blosum = torch.from_numpy(dataset_info.blosum),
                           class_weights=VegvisirLoadUtils.calculate_class_weights(train_data_blosum, args)
                           )
    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU

    print("-----------------Train_{}--------------------------".format(mode))
    print('\t Number train data points: {}; Proportion: {}'.format(train_data_blosum.shape[0],(train_data_blosum.shape[0]*100)/train_data_blosum.shape[0]))
    print('\t Number eval data points: {}; Proportion: {}'.format(valid_data_blosum.shape[0],(valid_data_blosum.shape[0]*100)/valid_data_blosum.shape[0]))

    custom_dataset_train = VegvisirLoadUtils.CustomDataset(train_data_blosum,
                                                           data_int[train_idx],
                                                           data_onehot[train_idx],
                                                           data_blosum_norm[train_idx],
                                                           data_array_blosum_encoding_mask[train_idx],
                                                           data_positional_weights_mask[train_idx],
                                                           )
    custom_dataset_valid = VegvisirLoadUtils.CustomDataset(data_blosum[valid_idx],
                                                           data_int[valid_idx],
                                                           data_onehot[valid_idx],
                                                           data_blosum_norm[valid_idx],
                                                           data_array_blosum_encoding_mask[valid_idx],
                                                           data_positional_weights_mask[valid_idx],
                                                           )
    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)
    valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)


    #Restart the model each fold
    Vegvisir = select_model(model_load, additional_info.results_dir,"all",args)

    params_config = config_build(args,additional_info.results_dir)

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
    # if args.learning_type in ["semisupervised","unsupervised"]:
    #     guide = config_enumerate(select_quide(Vegvisir,model_load,n_data,args.guide))
    # else:
    guide = select_quide(Vegvisir,model_load,n_data,args.guide)
    #svi = SVI(poutine.scale(Vegvisir.model,scale=1.0/n_data), poutine.scale(guide,scale=1.0/n_data), optimizer, loss_func)
    n = 50
    data_args_0 = {"blosum":train_data_blosum.to(args.device)[:n],
                   "norm":data_blosum_norm[train_idx].to(args.device)[:n],
                   "int":data_int[train_idx].to(args.device)[:n],
                   "onehot":data_onehot[train_idx].to(args.device)[:n],
                   "positional_mask":data_positional_weights_mask[train_idx].to(args.device)[:n]}


    data_args_1 = data_array_onehot_encoding_mask[train_idx].to(args.device)[:n] if args.encoding == "onehot" else data_array_blosum_encoding_mask[train_idx].to(args.device)[:n]
    model_trace = pyro.poutine.trace(Vegvisir.model).get_trace(data_args_0,data_args_1,0,None,False)
    info_file = open("{}/dataset_info.txt".format(results_dir),"a+")
    info_file.write("\n ---------TRACE SHAPES------------\n {}".format(str(model_trace.format_shapes())))
    #print(model_trace.format_shapes())
    # print(model_trace.nodes["sequences"])
    # print(model_trace.nodes["predictions"])

    #Highlight: Draw the graph model

    pyro.render_model(Vegvisir.model, model_args=(data_args_0,data_args_1,0,None,False), filename="{}/model_graph.png".format(results_dir),render_distributions=True,render_params=False)
    pyro.render_model(guide, model_args=(data_args_0,data_args_1,0,None,False), filename="{}/guide_graph.png".format(results_dir),render_distributions=True,render_params=False)
    svi = SVI(Vegvisir.model, guide, optimizer, loss_func)

    #Highlight: Hyperparameter optimization:

    #TODO: Dictionary that gathers the results from each fold
    start = time.time()
    epochs_list = []
    train_loss = []
    train_accuracies = []
    train_reconstruction_dict = {"mean":[],"std":[],"entropies":[]}
    valid_reconstruction_dict = {"mean":[],"std":[],"entropies":[]}
    train_auc = []
    train_auk = []
    valid_loss = []
    valid_accuracies = []
    valid_auc = []
    valid_auk = []

    epoch = 0.
    train_summary_dict = None
    valid_summary_dict = None
    gradient_norms = defaultdict(list)

    while epoch <= args.num_epochs:
        start = time.time()
        train_epoch_loss,train_accuracy,train_predictions_dict, train_latent_space,train_reconstruction_accuracy_dict = train_loop(svi,Vegvisir,guide, train_loader, args,model_load,epoch)
        stop = time.time()
        memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
        print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
        train_loss.append(train_epoch_loss)
        train_accuracies.append(train_accuracy)
        train_reconstruction_dict["mean"].append(train_reconstruction_accuracy_dict["mean"])
        train_reconstruction_dict["std"].append(train_reconstruction_accuracy_dict["std"])
        train_reconstruction_dict["entropies"].append(train_reconstruction_accuracy_dict["entropies"])

        if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
            for name_i, value in pyro.get_param_store().named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
            epochs_list.append(epoch)
            valid_epoch_loss, valid_accuracy, valid_predictions_dict, valid_latent_space,valid_reconstruction_accuracy_dict = valid_loop(svi, Vegvisir, guide, valid_loader, args,model_load,epoch)
            valid_loss.append(valid_epoch_loss)
            valid_accuracies.append(valid_accuracy)
            valid_reconstruction_dict["mean"].append(valid_reconstruction_accuracy_dict["mean"])
            valid_reconstruction_dict["std"].append(valid_reconstruction_accuracy_dict["std"])
            valid_reconstruction_dict["entropies"].append(valid_reconstruction_accuracy_dict["entropies"])
            train_observed_idx = (train_predictions_dict["true"][..., None] != 2).any(-1)
            #train_true_prob = train_predictions_dict["probs"][np.arange(0, train_true.shape[0]), train_true.long()] #pick the probability of the true target
            #train_pred_prob = np.argmax(train_predictions_dict["probs"],axis=-1) #return probability of the most likely class predicted by the model

            train_micro_roc_auc_ovr = roc_auc_score(
                train_predictions_dict["true_onehot"][train_observed_idx],
                train_predictions_dict["probs"][train_observed_idx],
                multi_class="ovr",
                average="micro",
            )
            train_auk_score = VegvisirUtils.AUK(probabilities= train_predictions_dict["binary"][train_observed_idx],labels=train_predictions_dict["true"][train_observed_idx]).calculate_auk()
            train_auk.append(train_auk_score)
            train_auc.append(train_micro_roc_auc_ovr)

            #valid_true_prob = valid_predictions_dict["probs"][np.arange(0, valid_true.shape[0]), valid_true.long()]  # pick the probability of the true target
            #valid_pred_prob = np.argmax(valid_predictions_dict["probs"],axis=-1)  # return probability of the most likely class predicted by the model
            valid_observed_idx = (valid_predictions_dict["true"][..., None] != 2).any(-1)

            valid_micro_roc_auc_ovr = roc_auc_score(
                valid_predictions_dict["true_onehot"][valid_observed_idx],
                valid_predictions_dict["probs"][valid_observed_idx],
                multi_class="ovr",
                average="micro",
            )
            valid_auk_score = VegvisirUtils.AUK(probabilities=valid_predictions_dict["binary"][valid_observed_idx], labels=valid_predictions_dict["true"][valid_observed_idx]).calculate_auk()
            valid_auk.append(valid_auk_score)
            valid_auc.append(valid_micro_roc_auc_ovr)

            VegvisirPlots.plot_loss(train_loss,valid_loss,epochs_list,"Train_{}".format(mode),results_dir)
            VegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,"Train_{}_single_sample".format(mode),results_dir)
            VegvisirPlots.plot_accuracy(train_reconstruction_dict,valid_reconstruction_dict,epochs_list,"Train_{}_single_sample".format(mode),results_dir)
            VegvisirPlots.plot_logits_entropies(train_reconstruction_dict,valid_reconstruction_dict,epochs_list,"Train_{}_single_sample".format(mode),results_dir)
            VegvisirPlots.plot_classification_score(train_auc,valid_auc,epochs_list,"Train_{}".format(mode),additional_info.results_dir,method="AUC")
            VegvisirPlots.plot_classification_score(train_auk,valid_auk,epochs_list,"Train_{}".format(mode),additional_info.results_dir,method="AUK")
            Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer,guide)
            # Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_train{}.p".format(results_dir,fold),
            #                            {"latent_space": train_latent_space,
            #                             "predictions_dict":train_predictions_dict})
            # Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_{}.p".format(results_dir,mode.lower()),
            #                            {"latent_space": valid_latent_space,
            #                             "predictions_dict":valid_predictions_dict})
            if epoch == args.num_epochs:
                print("Calculating Monte Carlo estimate of the posterior predictive")
                train_predictive_samples_loss, train_predictive_samples_accuracy, train_predictive_samples_dict, train_predictive_samples_latent_space,train_predictive_samples_reconstruction_accuracy_dict = sample_loop(svi, Vegvisir, guide, train_loader, args, model_load)
                valid_predictive_samples_loss, valid_predictive_samples_accuracy, valid_predictive_samples_dict, valid_predictive_samples_latent_space, valid_predictive_samples_reconstruction_accuracy_dict = sample_loop(svi, Vegvisir, guide, valid_loader, args, model_load)
                if args.generate:
                    print("Generating neo-epitopes ...")
                    generative_dict,generated_latent_space = generate_loop(svi, Vegvisir, guide, train_loader, args, model_load,dataset_info,additional_info,train_predictive_samples_dict)
                    generative_summary_dict = VegvisirUtils.manage_predictions_generative(args,generative_dict)
                else:
                    generative_summary_dict = None

                train_summary_dict = VegvisirUtils.manage_predictions(train_predictive_samples_dict,args,train_predictions_dict)
                valid_summary_dict = VegvisirUtils.manage_predictions(valid_predictive_samples_dict,args,valid_predictions_dict)

                if args.plot_all:
                    #VegvisirPlots.plot_gradients(gradient_norms, results_dir, "Train_{}".format(mode))
                    ##VegvisirPlots.plot_latent_space(args,dataset_info,train_latent_space, train_summary_dict, "single_sample",results_dir, method="Train{}".format(fold))
                    ##VegvisirPlots.plot_latent_space(args,dataset_info,valid_latent_space,valid_summary_dict, "single_sample",results_dir, method=mode)
                    VegvisirPlots.plot_latent_space(args,dataset_info,train_predictive_samples_latent_space, train_summary_dict, "samples",results_dir, method="Train{}".format(fold))
                    VegvisirPlots.plot_latent_space(args,dataset_info,valid_predictive_samples_latent_space,valid_summary_dict, "samples",results_dir, method=mode)
                    if args.generate:
                        VegvisirPlots.plot_latent_space(args,dataset_info,generated_latent_space,generative_summary_dict, "samples" if args.generate_num_samples > 1 else "single_sample",results_dir, method="Generated")
                    #VegvisirPlots.plot_latent_vector(train_latent_space, train_summary_dict, "single_sample",results_dir, method="Train{}".format(fold))
                    #VegvisirPlots.plot_latent_vector(valid_latent_space,valid_summary_dict, "single_sample",results_dir, method=mode)
                    VegvisirPlots.plot_attention_weights(train_summary_dict,dataset_info,results_dir,method="Train{}".format(fold))
                    VegvisirPlots.plot_attention_weights(valid_summary_dict,dataset_info,results_dir,method=mode)

                    VegvisirPlots.plot_hidden_dimensions(train_summary_dict, dataset_info, results_dir,args, method="Train{}".format(fold))
                    VegvisirPlots.plot_hidden_dimensions(valid_summary_dict, dataset_info, results_dir,args, method=mode)

                Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer,guide)

                if args.save_all:
                    Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_train{}{}.p".format(results_dir,mode.lower().split("_")[0],fold),
                                               {"latent_space": train_predictive_samples_latent_space,
                                                #"predictions_dict":train_predictions_dict,
                                                "summary_dict": train_summary_dict,
                                                "dataset_info":dataset_info})
                    Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_{}.p".format(results_dir,mode.lower()),
                                               {"latent_space": valid_predictive_samples_latent_space,
                                                #"predictions_dict": valid_predictions_dict,
                                                "summary_dict": valid_summary_dict,
                                                "dataset_info":dataset_info})

                else:
                    selection_keys = ["true_samples","true_onehot_samples","confidence_scores_samples","class_probs_predictions_samples_average","data_int_samples"]
                    miniinfo = minidatasetinfo(seq_max_len=dataset_info.seq_max_len,
                                               corrected_aa_types=dataset_info.corrected_aa_types,
                                               storage_folder=dataset_info.storage_folder,
                                               num_classes=args.num_classes,
                                               num_obs_classes=args.num_obs_classes)
                    Vegvisir.save_model_output(
                        "{}/Vegvisir_checkpoints/model_outputs_train_{}{}.p".format(results_dir,mode.lower().split("_")[0], fold),
                        {"latent_space": train_predictive_samples_latent_space,
                         "summary_dict": {key: train_summary_dict[key] for key in selection_keys},
                         "dataset_info": miniinfo,
                         "args":args})

                    Vegvisir.save_model_output(
                        "{}/Vegvisir_checkpoints/model_outputs_{}.p".format(results_dir, mode.lower()),
                        {"latent_space": valid_predictive_samples_latent_space,
                         "summary_dict": {key: valid_summary_dict[key] for key in selection_keys},
                         "dataset_info": miniinfo,
                         "args":args})



        torch.cuda.empty_cache()
        epoch += 1 #TODO: early stop?

    total_number_parameters = 0
    for param in Vegvisir.parameters(): #for name_i, value in pyro.get_param_store().named_parameters()
        total_number_parameters += param.numel()
    info_file.write("\n ------------------------------------- \n ")
    info_file.write("\n Total number parameters: {} \n ".format(total_number_parameters))

    train_metrics_summary_dict = VegvisirPlots.plot_classification_metrics(args,train_summary_dict,"{}".format(fold),results_dir,mode="Train{}".format(fold))
    valid_metrics_summary_dict = VegvisirPlots.plot_classification_metrics(args,valid_summary_dict,"{}".format(fold),results_dir,mode=mode)
    #VegvisirPlots.plot_classification_metrics_per_species(dataset_info,args,train_summary_dict,"all",results_dir,mode="Train{}".format(fold),per_sample=False)
    #VegvisirPlots.plot_classification_metrics_per_species(dataset_info,args,valid_summary_dict,"all",results_dir,mode=mode,per_sample=False)

    if args.dataset_name == "viral_dataset7": #Highlight: Sectioning out the old test data points to calculate the AUC isolated
        #Highlight: Extract the predictions of the test dataset from train and validation and calculate ROC
        print("Calculating classification metrics for old test dataset....")
        test_train_summary_dict,test_valid_summary_dict,test_all_summary_dict =VegvisirUtils.extract_group_old_test(train_summary_dict,valid_summary_dict,args)
        VegvisirPlots.plot_classification_metrics(args, test_train_summary_dict, "viral_dataset3_test_in_train", results_dir,mode="Test")
        VegvisirPlots.plot_classification_metrics(args, test_valid_summary_dict, "viral_dataset3_test_in_valid", results_dir,mode="Test")
        VegvisirPlots.plot_classification_metrics(args, test_all_summary_dict, "viral_dataset3_test_in_train_and_valid", results_dir,mode="Test")

    if args.hpo:
        metrics_summary_dict = {"train_loss":train_epoch_loss,
             "ROC_AUC_train": (train_metrics_summary_dict["samples"]["ALL"]["roc_auc"][0] + train_metrics_summary_dict["samples"]["ALL"]["roc_auc"][1] )/2,
             "average_precision_train": (train_metrics_summary_dict["samples"]["ALL"]["average_precision"][0] + train_metrics_summary_dict["samples"]["ALL"]["average_precision"][1])/2,
             "valid_loss": valid_epoch_loss,
             "ROC_AUC_valid": (valid_metrics_summary_dict["samples"]["ALL"]["roc_auc"][0] + valid_metrics_summary_dict["samples"]["ALL"]["roc_auc"][1])/2,
             "average_precision_valid": (valid_metrics_summary_dict["samples"]["ALL"]["average_precision"][0] + valid_metrics_summary_dict["samples"]["ALL"]["average_precision"][1])/2,
             }



    stop = time.time()
    print('Final timing: {}'.format(str(datetime.timedelta(seconds=stop-start))))
    print("Clearing CPU , GPU memory allocations. Clearing Pyro parameter store")
    print("Remaining objects: {}".format(len(gc.get_objects())))
    del train_predictive_samples_dict,valid_predictive_samples_dict,train_predictions_dict,train_summary_dict,valid_predictions_dict,valid_summary_dict,train_latent_space,valid_latent_space
    del train_predictive_samples_loss, train_predictive_samples_accuracy, train_predictive_samples_latent_space,train_predictive_samples_reconstruction_accuracy_dict
    del valid_predictive_samples_loss, valid_predictive_samples_accuracy,valid_predictive_samples_latent_space, valid_predictive_samples_reconstruction_accuracy_dict
    del gradient_norms
    del epochs_list ,train_loss ,train_accuracies ,train_reconstruction_dict,valid_reconstruction_dict ,train_auc ,train_auk ,valid_loss,valid_accuracies ,valid_auc ,valid_auk
    pyro.clear_param_store()
    Vegvisir.cpu()
    guide.cpu()
    Vegvisir.apply(reset_weights)
    del Vegvisir
    del guide
    del optimizer
    del loss_func
    del data_args_0,data_args_1,train_data_blosum,valid_data_blosum
    del train_loader,valid_loader,custom_dataset_train,custom_dataset_valid
    del data_blosum,data_int,data_onehot,data_blosum_norm,data_array_blosum_encoding_mask,data_positional_weights_mask
    torch.cuda.empty_cache()
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    gc.collect()
    print("Remaining objects AFTER clearing: {}".format(len(gc.get_objects())))

    if args.hpo:
        if args.k_folds <= 1:
            session.report(metrics=metrics_summary_dict)
        else:
            return metrics_summary_dict
def train_model(config=None,dataset_info=None,additional_info=None,args=None):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    seq_max_len = dataset_info.seq_max_len
    results_dir = additional_info.results_dir
    #Highlight: Train- Test split and kfold generator
    if args.predefined_partitions:
        partitioning_method = "predefined_partitions" if args.test else"predefined_partitions_discard_test"
        if args.dataset_name in ["viral_dataset6","viral_dataset7"]:
            partitioning_method = "predefined_partitions_diffused_test_create_new_test" if args.test else "predefined_partitions_diffused_test"
    else:
        partitioning_method = "random_stratified"

    train_data_blosum,valid_data_blosum,test_data_blosum = VegvisirLoadUtils.trainevaltest_split(data_blosum,
                                                                                                 args,results_dir,
                                                                                                 seq_max_len,dataset_info.max_len,
                                                                                                 dataset_info.features_names,
                                                                                                 None,method=partitioning_method)

    #Highlight:Also split the rest of arrays
    train_idx = (data_blosum[:,0,0,1][..., None] == train_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    valid_idx = (data_blosum[:,0,0,1][..., None] == valid_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    test_idx = (data_blosum[:,0,0,1][..., None] == test_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not

    print('\t Number train data points: {}; Proportion: {}'.format(train_data_blosum.shape[0],(train_data_blosum.shape[0]*100)/train_data_blosum.shape[0]))
    print('\t Number eval data points: {}; Proportion: {}'.format(valid_data_blosum.shape[0],(valid_data_blosum.shape[0]*100)/valid_data_blosum.shape[0]))
    print('\t Number test data points: {}; Proportion: {}'.format(test_data_blosum.shape[0],(test_data_blosum.shape[0]*100)/test_data_blosum.shape[0]))
    if args.pretrained_model is not None:
        if not args.test and args.validate:
            print("Only Training & Validation")
            load_model(train_idx, valid_idx, dataset_info, args, additional_info)
        elif args.test and args.validate:
            print("Training, Validation and testing")
            print("FIRST: Training & Validation")
            load_model( train_idx, valid_idx, dataset_info, args, additional_info,mode="Valid")
            print("SECOND: Training + Validation & Testing")
            if args.dataset_name == "viral_dataset7" and not args.test:
                warnings.warn("Test == Valid for dataset7, since the test is diffused onto the train and validation")
            print("Joining Training & validation datasets to perform testing...")
            train_idx = (train_idx.int() + valid_idx.int()).bool()
            load_model(train_idx, test_idx, dataset_info, args, additional_info)
        else:
            print("Joining Training & validation datasets to perform testing...")
            train_idx = (train_idx.int() + valid_idx.int()).bool()
            load_model(train_idx, test_idx, dataset_info, args, additional_info, mode="Test")

    else:
        if not args.test and args.validate:
            print("Only Training & Validation")
            epoch_loop( train_idx, valid_idx, dataset_info, args, additional_info,config=config)
        elif args.test and args.validate:
            print("Training, Validation and testing")
            print("FIRST: Training & Validation")
            epoch_loop( train_idx, valid_idx, dataset_info, args, additional_info,mode="Valid",config=config)
            print("SECOND: Training + Validation & Testing")
            if args.dataset_name == "viral_dataset7" and not args.test:
                warnings.warn("Test == Valid for dataset7, since the test is diffused onto the train and validation")
            else:
                print("Joining Training & validation datasets to perform testing...")
            train_idx = (train_idx.int() + valid_idx.int()).bool()
            epoch_loop(train_idx, test_idx, dataset_info, args, additional_info, mode="Test",config=config)
        else:
            if args.dataset_name == "viral_dataset7" and not args.test:
                warnings.warn("Test == Valid for dataset7, since the test is diffused onto the train and validation")
            else:
                print("Joining Training & validation datasets to perform testing...")
            train_idx = (train_idx.int() + valid_idx.int()).bool()
            epoch_loop(train_idx, test_idx, dataset_info, args, additional_info,mode="Test",config=config)
def load_model(train_idx,valid_idx,dataset_info,args,additional_info,mode="Valid",fold=""):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into pre-trained model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_blosum_norm = dataset_info.data_array_blosum_norm
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
    data_array_onehot_encoding_mask = dataset_info.data_array_onehot_encoding_mask

    data_positional_weights_mask = dataset_info.positional_weights_mask

    train_data_blosum = data_blosum[train_idx]
    valid_data_blosum = data_blosum[valid_idx]
    assert (train_data_blosum[:, 0, 0, 1] == data_int[train_idx, 0, 1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (train_data_blosum[:, 0, 0, 1] == data_onehot[train_idx, 0, 0, 1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (train_data_blosum[:, 0, 0, 1] == data_blosum_norm[train_idx, 0, 1]).all(), "The data is shuffled and the data frames are comparing the wrong things"

    assert (valid_data_blosum[:, 0, 0, 1] == data_int[valid_idx, 0, 1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (valid_data_blosum[:, 0, 0, 1] == data_onehot[valid_idx, 0, 0, 1]).all(), "The data is shuffled and the data frames are comparing the wrong things"
    assert (valid_data_blosum[:, 0, 0, 1] == data_blosum_norm[valid_idx, 0, 1]).all(), "The data is shuffled and the data frames are comparing the wrong things"

    n_data = data_blosum.shape[0]
    batch_size = args.batch_size
    results_dir = additional_info.results_dir
    check_point_epoch = 5 if args.num_epochs < 100 else int(args.num_epochs / 50)
    model_load = ModelLoad(args=args,
                           max_len=dataset_info.max_len,
                           seq_max_len=dataset_info.seq_max_len,
                           n_data=dataset_info.n_data,
                           input_dim=dataset_info.input_dim,
                           aa_types=dataset_info.corrected_aa_types,
                           blosum=torch.from_numpy(dataset_info.blosum),
                           class_weights=VegvisirLoadUtils.calculate_class_weights(train_data_blosum, args)
                           )
    pretrained_model_path = args.pretrained_model


    print('\t Number train data points: {}; Proportion: {}'.format(train_data_blosum.shape[0],(train_data_blosum.shape[0]*100)/train_data_blosum.shape[0]))
    print('\t Number eval data points: {}; Proportion: {}'.format(valid_data_blosum.shape[0],(valid_data_blosum.shape[0]*100)/valid_data_blosum.shape[0]))

    kwargs = {'num_workers': 0, 'pin_memory': args.use_cuda}  # pin-memory has to do with transferring CPU tensors to GPU

    print("-----------------Train_{}--------------------------".format(mode))
    print('\t Number train data points: {}; Proportion: {}'.format(train_data_blosum.shape[0],(train_data_blosum.shape[0]*100)/train_data_blosum.shape[0]))
    print('\t Number eval data points: {}; Proportion: {}'.format(valid_data_blosum.shape[0],(valid_data_blosum.shape[0]*100)/valid_data_blosum.shape[0]))

    custom_dataset_train = VegvisirLoadUtils.CustomDataset(train_data_blosum,
                                                           data_int[train_idx],
                                                           data_onehot[train_idx],
                                                           data_blosum_norm[train_idx],
                                                           data_array_blosum_encoding_mask[train_idx],
                                                           data_positional_weights_mask[train_idx],
                                                           )
    custom_dataset_valid = VegvisirLoadUtils.CustomDataset(data_blosum[valid_idx],
                                                           data_int[valid_idx],
                                                           data_onehot[valid_idx],
                                                           data_blosum_norm[valid_idx],
                                                           data_array_blosum_encoding_mask[valid_idx],
                                                           data_positional_weights_mask[valid_idx],
                                                           )

    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffled_Ibel? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffled_Ibel? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))


    Vegvisir = select_model(model_load, additional_info.results_dir,"all",args)
    params_config = config_build(args,additional_info.results_dir)
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
    # if args.learning_type in ["semisupervised","unsupervised"]:
    #     guide = config_enumerate(select_quide(Vegvisir,model_load,n_data,args.guide))
    # else:
    guide = select_quide(Vegvisir,model_load,n_data,args.guide)
    #svi = SVI(poutine.scale(Vegvisir.model,scale=1.0/n_data), poutine.scale(guide,scale=1.0/n_data), optimizer, loss_func)
    n = 50
    data_args_0 = {"blosum":train_data_blosum.to(args.device)[:n],
                   "norm":data_blosum_norm[train_idx].to(args.device)[:n],
                   "int":data_int[train_idx].to(args.device)[:n],
                   "onehot":data_onehot[train_idx].to(args.device)[:n],
                   "positional_mask":data_positional_weights_mask[train_idx].to(args.device)[:n]}
    data_args_1 = data_array_onehot_encoding_mask[train_idx].to(args.device)[:n] if args.encoding == "onehot" else data_array_blosum_encoding_mask[train_idx].to(args.device)[:n]
    model_trace = pyro.poutine.trace(Vegvisir.model).get_trace(data_args_0,data_args_1,0,None,False)
    info_file = open("{}/dataset_info.txt".format(results_dir),"a+")
    info_file.write("\n ---------TRACE SHAPES------------\n {}".format(str(model_trace.format_shapes())))

    #Highlight: Draw the graph model
    pyro.render_model(Vegvisir.model, model_args=(data_args_0,data_args_1,0,None,False), filename="{}/model_graph.png".format(results_dir),render_distributions=True,render_params=False)
    pyro.render_model(guide, model_args=(data_args_0,data_args_1,0,None,False), filename="{}/guide_graph.png".format(results_dir),render_distributions=True,render_params=False)
    svi = SVI(Vegvisir.model, guide, optimizer, loss_func)
    #Highlight: Load pretrained model
    pretrained_params_dict_guide = torch.load("{}/Vegvisir_checkpoints/checkpoints.pt".format(pretrained_model_path))["guide_state_dict"]
    pretrained_params_dict_model = torch.load("{}/Vegvisir_checkpoints/checkpoints.pt".format(pretrained_model_path))["model_state_dict"]
    print("Loading parameters from pretrained model at {}".format("{}/Vegvisir_checkpoints/checkpoints.pt".format(pretrained_model_path)))
    with torch.no_grad():
        for name, parameter in guide.named_parameters():
            parameter.copy_(pretrained_params_dict_guide[name])
        for name, parameter in Vegvisir.named_parameters():
            parameter.copy_(pretrained_params_dict_model[name])
    #TODO: Dictionary that gathers the results from each fold

    print("Calculating Monte Carlo estimate of the posterior predictive")
    train_predictive_samples_loss, train_predictive_samples_accuracy, train_predictive_samples_dict, train_predictive_samples_latent_space, \
        train_predictive_samples_reconstruction_accuracy_dict = sample_loop(
        svi, Vegvisir, guide, train_loader, args, model_load)
    valid_predictive_samples_loss, valid_predictive_samples_accuracy, valid_predictive_samples_dict, valid_predictive_samples_latent_space, \
        valid_predictive_samples_reconstruction_accuracy_dict = sample_loop(
        svi, Vegvisir, guide, valid_loader, args, model_load)
    train_summary_dict = VegvisirUtils.manage_predictions(train_predictive_samples_dict, args, None)
    valid_summary_dict = VegvisirUtils.manage_predictions(valid_predictive_samples_dict, args, None)

    if args.generate:
        print("Generating neo-epitopes ...")
        generative_dict, generated_latent_space = generate_loop(svi, Vegvisir, guide, train_loader, args, model_load,dataset_info, additional_info)
        generative_summary_dict = VegvisirUtils.manage_predictions_generative(args, generative_dict)
    else:
        generative_summary_dict = None

    if args.plot_all:
        VegvisirPlots.plot_latent_space(args, dataset_info, train_predictive_samples_latent_space, train_summary_dict,"samples", results_dir, method="Train{}".format(fold))
        VegvisirPlots.plot_latent_space(args, dataset_info, valid_predictive_samples_latent_space, valid_summary_dict,"samples", results_dir, method=mode)

        if args.generate:
            VegvisirPlots.plot_latent_space(args, dataset_info, generated_latent_space, generative_summary_dict,
                                            "samples" if args.generate_num_samples > 1 else "single_sample",
                                            results_dir, method="Generated")
        VegvisirPlots.plot_attention_weights(train_summary_dict, dataset_info, results_dir, method="Train{}".format(fold))
        VegvisirPlots.plot_attention_weights(valid_summary_dict, dataset_info, results_dir, method=mode)

        VegvisirPlots.plot_hidden_dimensions(train_summary_dict, dataset_info, results_dir, args,method="Train{}".format(fold))
        VegvisirPlots.plot_hidden_dimensions(valid_summary_dict, dataset_info, results_dir, args, method=mode)


    VegvisirPlots.plot_classification_metrics(args,train_summary_dict,"all",results_dir,mode="Train{}".format(fold))
    VegvisirPlots.plot_classification_metrics(args,valid_summary_dict,"all",results_dir,mode=mode)

    Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer, guide)

    if args.save_all:
        Vegvisir.save_model_output(
            "{}/Vegvisir_checkpoints/model_outputs_train{}{}.p".format(results_dir, mode.lower().split("_")[0], fold),
            {"latent_space": train_predictive_samples_latent_space,
             "summary_dict": train_summary_dict,
             "dataset_info": dataset_info})
        Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_{}.p".format(results_dir, mode.lower()),
                                   {"latent_space": valid_predictive_samples_latent_space,
                                    # "predictions_dict": valid_predictions_dict,
                                    "summary_dict": valid_summary_dict,
                                    "dataset_info": dataset_info})

    else:
        selection_keys = ["true_samples", "true_onehot_samples", "confidence_scores_samples",
                          "class_probs_predictions_samples_average", "data_int_samples"]
        miniinfo = minidatasetinfo(seq_max_len=dataset_info.seq_max_len,
                                   corrected_aa_types=dataset_info.corrected_aa_types,
                                   storage_folder=dataset_info.storage_folder,
                                   num_classes=args.num_classes,
                                   num_obs_classes=args.num_obs_classes)
        Vegvisir.save_model_output(
            "{}/Vegvisir_checkpoints/model_outputs_train_{}{}.p".format(results_dir, mode.lower().split("_")[0], fold),
            {"latent_space": train_predictive_samples_latent_space,
             "summary_dict": {key: train_summary_dict[key] for key in selection_keys},
             "dataset_info": miniinfo,
             "args": args})

        Vegvisir.save_model_output(
            "{}/Vegvisir_checkpoints/model_outputs_{}.p".format(results_dir, mode.lower()),
            {"latent_space": valid_predictive_samples_latent_space,
             "summary_dict": {key: valid_summary_dict[key] for key in selection_keys},
             "dataset_info": miniinfo,
             "args": args})


