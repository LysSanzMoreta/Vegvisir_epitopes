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

def batch_loop():
    """"""

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
        data_int.append(batch_data_int.detach().cpu().numpy())
        data_masks.append(batch_mask.detach().cpu().numpy())
        #Forward & Backward pass
        guide_estimates = guide(batch_data,batch_mask,epoch,None,sample=False)
        loss = svi.step(batch_data,batch_mask,epoch,guide_estimates,sample=False)
        sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=False)(batch_data,batch_mask,epoch = 0,guide_estimates=guide_estimates,sample=True)

        binary_class_prediction = sampling_output["predictions"].squeeze(0).squeeze(0).detach()
        logits_class_prediction = sampling_output["class_logits"].squeeze(0).squeeze(0).squeeze(0).detach()
        probs_class_prediction = torch.nn.Sigmoid()(logits_class_prediction)
        attn_weights = sampling_output["attn_weights"].squeeze(0).detach().cpu().numpy()
        attention_weights.append(attn_weights)
        encoder_hidden = sampling_output["encoder_hidden_states"].squeeze(0).detach().cpu().numpy()
        encoder_hidden_states.append(encoder_hidden)
        decoder_hidden = sampling_output["decoder_hidden_states"].squeeze(0).detach().cpu().numpy()
        decoder_hidden_states.append(decoder_hidden)
        encoder_final_hidden = sampling_output["encoder_final_hidden"].squeeze(0).detach().cpu()
        decoder_final_hidden = sampling_output["decoder_final_hidden"].squeeze(0).detach().cpu()
        reconstructed_sequences = sampling_output["sequences"].squeeze(0).squeeze(0).squeeze(0).detach().cpu()
        reconstruction_logits_batch = sampling_output["sequences_logits"].squeeze(0).detach().cpu()
        reconstruction_logits.append(reconstruction_logits_batch)

        latent_space = sampling_output["latent_z"].squeeze(0).squeeze(0).detach()
        true_labels_batch = batch_data["blosum"][:, 0, 0, 0]

        identifiers = batch_data["blosum"][:, 0, 0, 1]
        partitions = batch_data["blosum"][:, 0, 0, 2]
        training = batch_data["blosum"][:, 0, 0, 3]
        immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
        confidence_score = batch_data["blosum"][:, 0, 0, 5]
        latent_space = torch.column_stack(
            [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, latent_space])
        encoder_final_hidden= torch.column_stack(
            [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, encoder_final_hidden])
        decoder_final_hidden= torch.column_stack(
            [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score, decoder_final_hidden])
        encoder_final_hidden_states.append(encoder_final_hidden.numpy())
        decoder_final_hidden_states.append(decoder_final_hidden.numpy())

        mask_seq = batch_mask[:, 1:,:,0].squeeze(1)
        equal_aa = torch.Tensor((batch_data_int[:,1,:model_load.seq_max_len] == reconstructed_sequences.long())*mask_seq)
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
    confidence_scores_arr = np.concatenate(confidence_scores, axis=0)
    training_assignation_arr = np.concatenate(training_assignation, axis=0)

    attention_weights_arr = np.concatenate(attention_weights,axis=0)
    encoder_hidden_arr = np.concatenate(encoder_hidden_states,axis=0)
    decoder_hidden_arr = np.concatenate(decoder_hidden_states,axis=0)
    encoder_final_hidden_arr = np.concatenate(encoder_final_hidden_states,axis=0)
    decoder_final_hidden_arr = np.concatenate(decoder_final_hidden_states,axis=0)
    target_accuracy = 100 * ((true_labels_arr == binary_predictions_arr).sum() / true_labels_arr.shape[0])

    reconstruction_accuracies_arr = np.concatenate(reconstruction_accuracies)
    reconstruction_logits_arr = np.concatenate(reconstruction_logits,axis=0)
    reconstruction_entropy = VegvisirUtils.compute_sites_entropies(reconstruction_logits_arr, true_labels_arr) #Highlight: first term is the label, then the entropy per position
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies_arr.mean(),
                                      "std":reconstruction_accuracies_arr.std(),
                                      "entropies": np.mean(reconstruction_entropy[:,1:],axis=0)}
    true_onehot = np.zeros((true_labels_arr.shape[0],args.num_classes))
    true_onehot[np.arange(0,true_labels_arr.shape[0]),true_labels_arr.astype(int)] = 1
    predictions_dict = {"data_int":data_int_arr,
                        "data_mask": data_mask_arr,
                        "binary":binary_predictions_arr,
                        "logits":logits_predictions_arr,
                        "probs":probs_predictions_arr,
                        "true":true_labels_arr,
                        "true_onehot":true_onehot,
                        "confidence_scores":confidence_scores_arr,
                        "training_assignation":training_assignation_arr,
                        "attention_weights":attention_weights_arr,
                        "encoder_hidden_states":encoder_hidden_arr,
                        "decoder_hidden_states":decoder_hidden_arr,
                        "encoder_final_hidden_state": encoder_final_hidden_arr,
                        "decoder_final_hidden_state": decoder_final_hidden_arr,
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
            data_int.append(batch_data_int.detach().cpu().numpy())
            data_masks.append(batch_mask.detach().cpu().numpy())

            guide_estimates = guide(batch_data, batch_mask,epoch, None, sample=False)
            loss = svi.step(batch_data, batch_mask, epoch, guide_estimates, sample=False)

            sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=False)(batch_data, batch_mask,epoch = 0,guide_estimates=guide_estimates,sample=True)

            binary_class_prediction = sampling_output["predictions"].squeeze(0).squeeze(0).detach()
            logits_class_prediction = sampling_output["class_logits"].squeeze(0).squeeze(0).squeeze(0).detach()
            attn_weights = sampling_output["attn_weights"].squeeze(0).detach().cpu().numpy()
            attention_weights.append(attn_weights)
            encoder_hidden = sampling_output["encoder_hidden_states"].squeeze(0).detach().cpu().numpy()
            encoder_hidden_states.append(encoder_hidden)
            decoder_hidden = sampling_output["decoder_hidden_states"].squeeze(0).detach().cpu().numpy()
            decoder_hidden_states.append(decoder_hidden)
            encoder_final_hidden = sampling_output["encoder_final_hidden"].squeeze(0).detach().cpu()
            decoder_final_hidden = sampling_output["decoder_final_hidden"].squeeze(0).detach().cpu()
            probs_class_prediction = torch.nn.Sigmoid()(logits_class_prediction)
            reconstructed_sequences = sampling_output["sequences"].squeeze(0).squeeze(0).squeeze(0).detach().cpu()
            reconstruction_logits_batch = sampling_output["sequences_logits"].squeeze(0).detach().cpu()
            reconstruction_logits.append(reconstruction_logits_batch)
            latent_space = sampling_output["latent_z"].squeeze(0).squeeze(0).detach()
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0]
            identifiers = batch_data["blosum"][:, 0, 0, 1]
            partitions = batch_data["blosum"][:,0,0,2]
            training = batch_data["blosum"][:,0,0,3]
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
            confidence_score = batch_data["blosum"][:, 0, 0, 5]
            # latent_space = torch.column_stack(
            #     [identifiers, true_labels_batch, confidence_score, immunodominace_score, latent_space])
            latent_space = torch.column_stack(
                [true_labels_batch,identifiers, partitions, immunodominace_score,confidence_score, latent_space])
            encoder_final_hidden = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,
                 encoder_final_hidden])
            decoder_final_hidden = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,
                 decoder_final_hidden])
            encoder_final_hidden_states.append(encoder_final_hidden.numpy())
            decoder_final_hidden_states.append(decoder_final_hidden.numpy())
            mask_seq = batch_mask[:, 1:, :, 0].squeeze(1)
            equal_aa = torch.Tensor((batch_data_int[:, 1, :model_load.seq_max_len] == reconstructed_sequences) * mask_seq)
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

    latent_arr = np.concatenate(latent_spaces,axis=0)
    attention_weights_arr = np.concatenate(attention_weights,axis=0)
    encoder_hidden_arr = np.concatenate(encoder_hidden_states,axis=0)
    decoder_hidden_arr = np.concatenate(decoder_hidden_states,axis=0)
    encoder_final_hidden_arr = np.concatenate(encoder_final_hidden_states,axis=0)
    decoder_final_hidden_arr = np.concatenate(decoder_final_hidden_states,axis=0)
    target_accuracy= 100 * ((true_labels_arr == binary_predictions_arr).sum()/true_labels_arr.shape[0])
    reconstruction_accuracies_arr = np.concatenate(reconstruction_accuracies)
    reconstruction_logits_arr= np.concatenate(reconstruction_logits,axis=0)
    reconstruction_entropy = VegvisirUtils.compute_sites_entropies(reconstruction_logits_arr, true_labels_arr) #Highlight: first term is the label, then the entropy per position
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies_arr.mean(),
                                      "std":reconstruction_accuracies_arr.std(),
                                      "entropies": np.mean(reconstruction_entropy[:,1:],axis=0)}
    true_onehot = np.zeros((true_labels_arr.shape[0],args.num_classes))
    true_onehot[np.arange(0,true_labels_arr.shape[0]),true_labels_arr.astype(int)] = 1
    predictions_dict = {"data_int": data_int_arr ,
                        "data_mask": data_mask_arr,
                        "binary":binary_predictions_arr,
                        "logits":logits_predictions_arr,
                        "probs":probs_predictions_arr,
                        "true": true_labels_arr,
                        "true_onehot": true_onehot,
                        "confidence_scores":confidence_scores_arr,
                        "training_assignation":training_asignation_arr,
                        "attention_weights":attention_weights_arr,
                        "encoder_hidden_states": encoder_hidden_arr,
                        "decoder_hidden_states": decoder_hidden_arr,
                        "encoder_final_hidden_state": encoder_final_hidden_arr,
                        "decoder_final_hidden_state": decoder_final_hidden_arr,
                        }

    return valid_loss,target_accuracy,predictions_dict,latent_arr, reconstruction_accuracies_dict
def test_loop(svi,Vegvisir,guide,data_loader,args,model_load,epoch): #TODO: remove?
    Vegvisir.train(False)
    Vegvisir.eval()
    test_loss = 0.
    latent_spaces = []
    binary_predictions = []
    logits_predictions = []
    probs_predictions = []
    reconstruction_accuracies = []
    true_labels = []
    confidence_scores = []
    training_assignation = []
    attention_weights = []
    encoder_hidden_states = []
    encoder_final_hidden_states = []
    decoder_hidden_states = []
    decoder_final_hidden_states = []
    data_int=[]
    data_masks = []
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
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
            batch_data = {"blosum": batch_data_blosum, "int": batch_data_int, "onehot": batch_data_onehot,
                          "norm":batch_data_blosum_norm,
                          "positional_mask":batch_positional_mask}
            data_int.append(batch_data_int.detach().cpu().numpy())
            data_masks.append(batch_mask.detach().cpu().numpy())

            guide_estimates = guide(batch_data, batch_mask,epoch, None, sample=False)
            loss = svi.step(batch_data, batch_mask, epoch,guide_estimates, sample=False)
            # guide_estimates = guide(batch_data,batch_mask)
            # sampling_output = Vegvisir.sample(batch_data,batch_mask,guide_estimates,argmax=True)
            # predicted_labels = sampling_output.predicted_labels.detach()
            # reconstructed_sequences = sampling_output.reconstructed_sequences.detach()
            #latent_space = sampling_output.latent_space.detach()
            sampling_output = Predictive(Vegvisir.model, guide=guide, num_samples=1, return_sites=(), parallel=True)(batch_data, batch_mask,guide_estimates=guide_estimates,sample=True)
            #predicted_labels = sampling_output["predictions"].squeeze(0).squeeze(0).squeeze(0).detach()
            binary_class_prediction = sampling_output["predictions"].squeeze(0).detach()
            logits_class_prediction = sampling_output["class_logits"].squeeze(0).detach()
            attn_weights = sampling_output["attn_weights"].squeeze(0).detach().cpu().numpy()
            attention_weights.append(attn_weights)
            encoder_hidden = sampling_output["encoder_hidden_states"].squeeze(0).detach().cpu().numpy()
            encoder_hidden_states.append(encoder_hidden)
            decoder_hidden = sampling_output["decoder_hidden_states"].squeeze(0).detach().cpu().numpy()
            decoder_hidden_states.append(decoder_hidden)
            encoder_final_hidden = sampling_output["encoder_final_hidden"].squeeze(0).detach().cpu()
            decoder_final_hidden = sampling_output["decoder_final_hidden"].squeeze(0).detach().cpu()
            probs_class_prediction = torch.nn.Sigmoid()(logits_class_prediction)

            reconstructed_sequences = sampling_output["sequences"].squeeze(0).detach()
            latent_space = sampling_output["latent_space"].squeeze(0).squeeze(0).detach()
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0]
            identifiers = batch_data["blosum"][:, 0, 0, 1]
            partitions = batch_data["blosum"][:,0,0,2]
            training = batch_data["blosum"][:,0,0,3]
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
            confidence_score = batch_data["blosum"][:, 0, 0, 5]
            latent_space = torch.column_stack(
                [true_labels_batch,identifiers, partitions, immunodominace_score,confidence_score, latent_space])
            encoder_final_hidden = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,
                 encoder_final_hidden])
            decoder_final_hidden = torch.column_stack(
                [true_labels_batch, identifiers, partitions, immunodominace_score, confidence_score,
                 decoder_final_hidden])
            encoder_final_hidden_states.append(encoder_final_hidden.numpy())
            decoder_final_hidden_states.append(decoder_final_hidden.numpy())

            mask_seq = batch_mask[:, 1:, :, 0].squeeze(1)
            equal_aa = torch.Tensor((batch_data_int[:, 1, :model_load.seq_max_len] == reconstructed_sequences) * mask_seq)
            reconstruction_accuracy = (equal_aa.sum(dim=1)) / mask_seq.sum(dim=1)
            reconstruction_accuracies.append(reconstruction_accuracy.cpu().numpy())
            latent_spaces.append(latent_space.cpu().numpy())
            binary_predictions.append(binary_class_prediction.detach().cpu().numpy())
            logits_predictions.append(logits_class_prediction.detach().cpu().numpy())
            probs_predictions.append(probs_class_prediction.detach().cpu().numpy())
            true_labels.append(true_labels_batch.detach().cpu().numpy())
            confidence_scores.append(confidence_score.detach().cpu().numpy())
            training_assignation.append(training.detach().cpu().numpy())

            test_loss += loss

    test_loss /= len(data_loader)
    data_int_arr = np.concatenate(data_int,axis=0)
    data_mask_arr = np.concatenate(data_masks,axis=0)

    binary_predictions_arr = np.concatenate(binary_predictions, axis=0)
    logits_predictions_arr = np.concatenate(logits_predictions, axis=0)
    probs_predictions_arr = np.concatenate(probs_predictions, axis=0)
    true_labels_arr = np.concatenate(true_labels,axis=0)
    confidence_scores_arr = np.concatenate(confidence_scores, axis=0)
    training_assignation_arr = np.concatenate(training_assignation, axis=0)

    latent_arr = np.concatenate(latent_spaces,axis=0)
    attention_weights_arr = np.concatenate(attention_weights,axis=0)
    encoder_hidden_arr = np.concatenate(encoder_hidden_states,axis=0)
    decoder_hidden_arr = np.concatenate(decoder_hidden_states,axis=0)
    encoder_final_hidden_arr = np.concatenate(encoder_final_hidden_states,axis=0)
    decoder_final_hidden_arr = np.concatenate(decoder_final_hidden_states,axis=0)
    target_accuracy = 100 * ((true_labels_arr == binary_predictions_arr).sum() / true_labels_arr.shape[0])
    reconstruction_accuracies = np.concatenate(reconstruction_accuracies)
    reconstruction_accuracies_dict = {"mean":reconstruction_accuracies.mean(),"std":reconstruction_accuracies.std()}
    true_onehot = np.zeros((true_labels_arr.shape[0],args.num_classes))
    true_onehot[np.arange(0,true_labels_arr.shape[0]),true_labels_arr.astype(int)] = 1
    predictions_dict = {"data_int":data_int_arr,
                        "data_mask": data_mask_arr,
                        "binary":binary_predictions_arr,
                        "logits":logits_predictions_arr,
                        "probs":probs_predictions_arr,
                        "true": true_labels_arr,
                        "true_onehot": true_onehot,
                        "confidence_scores":confidence_scores_arr,
                        "training_assignation":training_assignation_arr,
                        "attention_weights":attention_weights_arr,
                        "encoder_hidden_states": encoder_hidden_arr,
                        "decoder_hidden_states": decoder_hidden_arr,
                        "encoder_final_hidden_state": encoder_final_hidden_arr,
                        "decoder_final_hidden_state": decoder_final_hidden_arr,
                        }
    return test_loss,target_accuracy,predictions_dict,latent_arr, reconstruction_accuracies_dict #TODO: remove?
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

            if sampling_output["predictions"].shape == (args.num_samples,batch_data["blosum"].shape[0]):
                binary_class_prediction = sampling_output["predictions"].detach().T
            else:
                binary_class_prediction = sampling_output["predictions"].squeeze(1).detach().T
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
            encoder_final_hidden = encoder_final_hidden.mean(dim=1) #TODO: this is not correct, think about it
            decoder_final_hidden = sampling_output["decoder_final_hidden"].squeeze(0).detach().cpu().permute(1,0,2)
            decoder_final_hidden = decoder_final_hidden.mean(dim=1)

            if sampling_output["latent_z"].ndim == 4:
                latent_space = sampling_output["latent_z"].squeeze(1).detach().permute(1,0,2)[:,0,:]
            else:
                latent_space = sampling_output["latent_z"].detach().permute(1,0,2)[:,0,:]
            true_labels_batch = batch_data["blosum"][:, 0, 0, 0]
            identifiers = batch_data["blosum"][:, 0, 0, 1]
            partitions = batch_data["blosum"][:, 0, 0, 2]
            training = batch_data["blosum"][:, 0, 0, 3]
            immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
            confidence_score = batch_data["blosum"][:, 0, 0, 5]
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
    target_accuracy = 100 * ((true_labels_arr[:,None] == binary_predictions_arr).astype(float).mean(axis=1).mean(axis=0))
    reconstruction_accuracies_arr = np.concatenate(reconstruction_accuracies).mean(axis=1) #[N,num_samples,1]
    reconstruction_accuracies_dict = {"mean": reconstruction_accuracies_arr.mean(), "std": reconstruction_accuracies_arr.std()}
    true_onehot = np.zeros((true_labels_arr.shape[0],args.num_classes))
    true_onehot[np.arange(0,true_labels_arr.shape[0]),true_labels_arr.astype(int)] = 1
    predictions_dict = {"data_int":data_int_arr,
                        "data_mask":data_mask_arr,
                        "binary": binary_predictions_arr,
                        "logits": logits_predictions_arr,
                        "probs": probs_predictions_arr,
                        "true":true_labels_arr,
                        "true_onehot": true_onehot,
                        "accuracy":target_accuracy,
                        "training_assignation":training_assignation_arr,
                        "confidence_scores":confidence_scores_arr,
                        "attention_weights":attention_weights_arr,
                        "encoder_hidden_states": encoder_hidden_arr,
                        "decoder_hidden_states": decoder_hidden_arr,
                        "encoder_final_hidden_state":encoder_final_hidden_arr,
                        "decoder_final_hidden_state": decoder_final_hidden_arr,
                        }
    return sample_loss, target_accuracy, predictions_dict, latent_arr, reconstruction_accuracies_dict
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
    if model_load.seq_max_len == model_load.max_len:
        if args.learning_type == "supervised":
            vegvisir_model = VegvisirModels.VegvisirModel5a_supervised(model_load)
        elif args.learning_type == "unsupervised":
            vegvisir_model = VegvisirModels.VegvisirModel5a_unsupervised(model_load)
        elif args.learning_type == "semisupervised":
            vegvisir_model = VegvisirModels.VegvisirModel5a_semisupervised(model_load)
    else:
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
    if isinstance(m, nn.Module) and hasattr(m, 'weight') and not (isinstance(m,nn.BatchNorm1d) or isinstance(m,nn.Embedding)):
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
            train_epoch_loss,train_accuracy,train_predictions,train_reconstruction_accuracies_dict = train_loop(svi,Vegvisir,guide, train_loader, args,model_load,epoch)
            stop = time.time()
            memory_usage_mib = torch.cuda.max_memory_allocated() * 9.5367 * 1e-7  # convert byte to MiB
            print("[epoch %03d]  average training loss: %.4f %.5g time/epoch %.2f MiB/epoch" % (epoch, train_epoch_loss, stop - start, memory_usage_mib))
            train_loss.append(train_epoch_loss)
            train_accuracies.append(train_accuracy)
            if (check_point_epoch > 0 and epoch > 0 and epoch % check_point_epoch == 0) or epoch == args.num_epochs :
                for name_i, value in pyro.get_param_store().named_parameters(): #TODO: https://stackoverflow.com/questions/68634707/best-way-to-detect-vanishing-exploding-gradient-in-pytorch-via-tensorboard
                    value.register_hook(lambda g, name_i=name_i: gradient_norms[name_i].append(g.norm().detach().item()))
                valid_epoch_loss,valid_accuracy,valid_predictions,valid_reconstruction_accuracies_dict = valid_loop(svi,Vegvisir,guide, valid_loader, args,model_load,epoch)
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

        Vegvisir = select_model(model_load, additional_info.results_dir,0,args)
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
        predictions = test_loop(Vegvisir,guide,test_loader,args,model_load,epoch=0)
        score = roc_auc_score(y_true=test_data_blosum[:, 0, 0, 0].numpy(), y_score=predictions)
        print("Final AUC score : {}".format( score))
def epoch_loop(train_idx,valid_idx,dataset_info,args,additional_info,mode="Valid"):

    #Split the rest of the data (train_data) for train and validation
    data_blosum = dataset_info.data_array_blosum_encoding
    data_int = dataset_info.data_array_int
    data_onehot = dataset_info.data_array_onehot_encoding
    data_blosum_norm = dataset_info.data_array_blosum_norm
    data_array_blosum_encoding_mask = dataset_info.data_array_blosum_encoding_mask
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
    batch_size = args.batch_size
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
                                                           data_positional_weights_mask[train_idx])
    custom_dataset_valid = VegvisirLoadUtils.CustomDataset(data_blosum[valid_idx],
                                                           data_int[valid_idx],
                                                           data_onehot[valid_idx],
                                                           data_blosum_norm[valid_idx],
                                                           data_array_blosum_encoding_mask[valid_idx],
                                                           data_positional_weights_mask[valid_idx])

    train_loader = DataLoader(custom_dataset_train, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))
    valid_loader = DataLoader(custom_dataset_valid, batch_size=batch_size,shuffle=True,generator=torch.Generator(device=args.device), **kwargs)  # also shuffle? collate_fn=lambda x: tuple(x_.to(device) for x_ in default_collate(x))

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
    if args.learning_type in ["semisupervised","unsupervised"]:
        guide = config_enumerate(select_quide(Vegvisir,model_load,n_data,args.guide))
    else:
        guide = select_quide(Vegvisir,model_load,n_data,args.guide)
    #svi = SVI(poutine.scale(Vegvisir.model,scale=1.0/n_data), poutine.scale(guide,scale=1.0/n_data), optimizer, loss_func)
    n = 50
    data_args_0 = {"blosum":train_data_blosum.to(args.device)[:n],
                   "norm":data_blosum_norm[train_idx].to(args.device)[:n],
                   "int":data_int[train_idx].to(args.device)[:n],
                   "onehot":data_onehot[train_idx].to(args.device)[:n],
                   "positional_mask":data_positional_weights_mask[train_idx].to(args.device)[:n]}
    data_args_1 = data_array_blosum_encoding_mask[train_idx].to(args.device)[:n]
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


            #train_true_prob = train_predictions_dict["probs"][np.arange(0, train_true.shape[0]), train_true.long()] #pick the probability of the true target
            #train_pred_prob = np.argmax(train_predictions_dict["probs"],axis=-1) #return probability of the most likely class predicted by the model
            train_micro_roc_auc_ovr = roc_auc_score(
                train_predictions_dict["true_onehot"],
                train_predictions_dict["probs"],
                multi_class="ovr",
                average="micro",
            )
            train_auk_score = VegvisirUtils.AUK(probabilities= train_predictions_dict["binary"],labels=train_predictions_dict["true"]).calculate_auk()
            train_auk.append(train_auk_score)
            train_auc.append(train_micro_roc_auc_ovr)

            #valid_true_prob = valid_predictions_dict["probs"][np.arange(0, valid_true.shape[0]), valid_true.long()]  # pick the probability of the true target
            #valid_pred_prob = np.argmax(valid_predictions_dict["probs"],axis=-1)  # return probability of the most likely class predicted by the model
            valid_micro_roc_auc_ovr = roc_auc_score(
                valid_predictions_dict["true_onehot"],
                valid_predictions_dict["probs"],
                multi_class="ovr",
                average="micro",
            )
            valid_auk_score = VegvisirUtils.AUK(probabilities=valid_predictions_dict["binary"], labels=valid_predictions_dict["true"]).calculate_auk()
            valid_auk.append(valid_auk_score)
            valid_auc.append(valid_micro_roc_auc_ovr)

            VegvisirPlots.plot_loss(train_loss,valid_loss,epochs_list,"Train_{}".format(mode),results_dir)
            VegvisirPlots.plot_accuracy(train_accuracies,valid_accuracies,epochs_list,"Train_{}_single_sample".format(mode),results_dir)
            VegvisirPlots.plot_accuracy(train_reconstruction_dict,valid_reconstruction_dict,epochs_list,"Train_{}_single_sample".format(mode),results_dir)
            VegvisirPlots.plot_logits_entropies(train_reconstruction_dict,valid_reconstruction_dict,epochs_list,"Train_{}_single_sample".format(mode),results_dir)
            VegvisirPlots.plot_classification_score(train_auc,valid_auc,epochs_list,"Train_{}".format(mode),additional_info.results_dir,method="AUC")
            VegvisirPlots.plot_classification_score(train_auk,valid_auk,epochs_list,"Train_{}".format(mode),additional_info.results_dir,method="AUK")
            Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir), optimizer,guide)
            Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_train.p".format(results_dir),
                                       {"latent_space": train_latent_space,
                                        "predictions_dict":train_predictions_dict})
            Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_valid.p".format(results_dir),
                                       {"latent_space": valid_latent_space,
                                        "predictions_dict":valid_predictions_dict})
            if epoch == args.num_epochs:
                print("Calculating Monte Carlo estimate of the posterior predictive")
                train_predictive_samples_loss, train_predictive_samples_accuracy, train_predictive_samples_dict, train_predictive_samples_latent_space,\
                    train_predictive_samples_reconstruction_accuracy_dict = sample_loop(
                    svi, Vegvisir, guide, train_loader, args, model_load)
                valid_predictive_samples_loss, valid_predictive_samples_accuracy, valid_predictive_samples_dict, valid_predictive_samples_latent_space, \
                    valid_predictive_samples_reconstruction_accuracy_dict = sample_loop(
                    svi, Vegvisir, guide, valid_loader, args, model_load)
                train_summary_dict = VegvisirUtils.manage_predictions(train_predictive_samples_dict,args,train_predictions_dict)
                valid_summary_dict = VegvisirUtils.manage_predictions(valid_predictive_samples_dict,args,valid_predictions_dict)
                VegvisirPlots.plot_gradients(gradient_norms, results_dir, "Train_{}".format(mode))
                VegvisirPlots.plot_latent_space(train_latent_space, train_summary_dict, "single_sample",results_dir, method="Train")
                VegvisirPlots.plot_latent_space(valid_latent_space,valid_summary_dict, "single_sample",results_dir, method=mode)
                VegvisirPlots.plot_latent_space(train_predictive_samples_latent_space, train_summary_dict, "samples",results_dir, method="Train")
                VegvisirPlots.plot_latent_space(valid_predictive_samples_latent_space,valid_summary_dict, "samples",results_dir, method=mode)

                VegvisirPlots.plot_latent_vector(train_latent_space, train_summary_dict, "single_sample",results_dir, method="Train")
                VegvisirPlots.plot_latent_vector(valid_latent_space,valid_summary_dict, "single_sample",results_dir, method=mode)

                VegvisirPlots.plot_attention_weights(train_summary_dict,dataset_info,results_dir,method="Train")
                VegvisirPlots.plot_attention_weights(valid_summary_dict,dataset_info,results_dir,method=mode)

                VegvisirPlots.plot_hidden_dimensions(train_summary_dict, dataset_info, results_dir,args, method="Train")
                VegvisirPlots.plot_hidden_dimensions(valid_summary_dict, dataset_info, results_dir,args, method=mode)

                Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer,guide)
                Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_train.p".format(results_dir),
                                           {"latent_space": train_latent_space,
                                            "predictions_dict":train_predictions_dict,
                                            "summary_dict": train_summary_dict})
                Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_valid.p".format(results_dir),
                                           {"latent_space": valid_latent_space,
                                            "predictions_dict": valid_predictions_dict,
                                            "summary_dict": valid_summary_dict})

        torch.cuda.empty_cache()
        epoch += 1 #TODO: early stop?
    VegvisirPlots.plot_classification_metrics(args,train_summary_dict,"all",results_dir,mode="Train")
    VegvisirPlots.plot_classification_metrics(args,valid_summary_dict,"all",results_dir,mode=mode)
    if args.dataset_name == "viral_dataset7": #Highlight: Sectioning out the old test data points to calculate the AUC isolated
        #Highlight: Extract the predictions of the test dataset from train and validation and calculate ROC
        print("Calculating classification metrics for old test dataset....")
        test_train_summary_dict,test_valid_summary_dict,test_all_summary_dict =VegvisirUtils.extract_group_old_test(train_summary_dict,valid_summary_dict,args)
        VegvisirPlots.plot_classification_metrics(args, test_train_summary_dict, "viral_dataset3_test_in_train", results_dir,mode="Test")
        VegvisirPlots.plot_classification_metrics(args, test_valid_summary_dict, "viral_dataset3_test_in_valid", results_dir,mode="Test")
        VegvisirPlots.plot_classification_metrics(args, test_all_summary_dict, "viral_dataset3_test_in_train_and_valid", results_dir,mode="Test")


    stop = time.time()
    print('Final timing: {}'.format(str(datetime.timedelta(seconds=stop-start))))
def train_model(dataset_info,additional_info,args):
    """Set up k-fold cross validation and the training loop"""
    print("Loading dataset into model...")
    data_blosum = dataset_info.data_array_blosum_encoding
    seq_max_len = dataset_info.seq_max_len
    results_dir = additional_info.results_dir
    #Highlight: Train- Test split and kfold generator
    partitioning_method = ["predefined_partitions" if args.test else"predefined_partitions_discard_test"][0]
    if args.dataset_name == "viral_dataset7":
        partitioning_method = "predefined_partitions_no_test"

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
    if not args.test:
        print("Only Training & Validation")
        epoch_loop( train_idx, valid_idx, dataset_info, args, additional_info)
    else:
        if args.dataset_name == "viral_dataset7":
            print("Test == Valid for dataset7")
        else:
            print("Training & testing...")
        train_idx = (train_idx.int() + valid_idx.int()).bool()
        epoch_loop(train_idx, test_idx, dataset_info, args, additional_info,mode="Test")
def load_model(dataset_info,additional_info,args):
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
    pretrained_model_path = args.pretrained_model
    partition_used = None
    for line in open("{}/dataset_info.txt".format(pretrained_model_path), "r+").readlines():
        if line.startswith(" Using as test partition:"):
            partition_used = int(line.split(":")[1])
    print("Loading previous command line arguments")
    commandline_args = json.load(open("{}/commandline_args.txt".format(pretrained_model_path),"r+"))
    args = namedtuple('args', commandline_args.keys())(*commandline_args.values())

    train_data_blosum,valid_data_blosum,test_data_blosum = VegvisirLoadUtils.trainevaltest_split(data_blosum,args,results_dir,
                                                                                                 seq_max_len,dataset_info.max_len,dataset_info.features_names,
                                                                                                 partition_used,
                                                                                                 method="predefined_partitions_discard_test")

    #Highlight:Also split the rest of arrays
    train_idx = (data_blosum[:,0,0,1][..., None] == train_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    valid_idx = (data_blosum[:,0,0,1][..., None] == valid_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not
    test_idx = (data_blosum[:,0,0,1][..., None] == test_data_blosum[:,0,0,1]).any(-1) #the data and the adjacency matrix have not been shuffled,so we can use it for indexing. It does not matter that train-data has been shuffled or not

    #Split the rest of the data (train_data) for train and validation
    batch_size = args.batch_size
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

    Vegvisir = select_model(model_load, additional_info.results_dir,"all",args)
    Vegvisir.load_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(pretrained_model_path))
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

    guide = select_quide(Vegvisir,model_load,n_data,args.guide)
    n = 50
    data_args_0 = {"blosum":train_data_blosum.to(args.device)[:n],"norm":data_blosum_norm[train_idx].to(args.device)[:n],"int":data_int[train_idx].to(args.device)[:n]}
    data_args_1 = data_array_blosum_encoding_mask[train_idx].to(args.device)[:n]
    trace = pyro.poutine.trace(Vegvisir.model).get_trace(data_args_0,data_args_1)

    info_file = open("{}/dataset_info.txt".format(results_dir),"a+")
    info_file.write("\n ---------TRACE SHAPES------------\n {}".format(str(trace.format_shapes())))

    #Highlight: Draw the graph model
    pyro.render_model(Vegvisir.model, model_args=(data_args_0,data_args_1,False), filename="{}/model_graph.png".format(results_dir),render_distributions=True,render_params=True)
    pyro.render_model(guide, model_args=(data_args_0,data_args_1,False), filename="{}/guide_graph.png".format(results_dir),render_distributions=True,render_params=True)
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
    train_predictive_samples_loss, train_predictive_samples_accuracy, train_predictive_samples_dict, train_predictive_samples_latent_space, train_predictive_samples_reconstruction_accuracy_dict = sample_loop(
        svi, Vegvisir, guide, train_loader, args, model_load)
    valid_predictive_samples_loss, valid_predictive_samples_accuracy, valid_predictive_samples_dict, valid_predictive_samples_latent_space, valid_predictive_samples_reconstruction_accuracy_dict = sample_loop(
        svi, Vegvisir, guide, valid_loader, args, model_load)
    train_true = train_data_blosum[:,0,0,0]
    valid_true = valid_data_blosum[:,0,0,0]
    train_summary_dict = VegvisirUtils.manage_predictions(train_predictive_samples_dict,args,None,train_true)
    valid_summary_dict = VegvisirUtils.manage_predictions(valid_predictive_samples_dict,args,None,valid_true)
    VegvisirPlots.plot_latent_space(train_predictive_samples_latent_space,train_summary_dict, "_samples",results_dir, method="Train")
    VegvisirPlots.plot_latent_space(valid_predictive_samples_latent_space,valid_summary_dict, "_samples",results_dir, method="Valid")
    VegvisirPlots.plot_latent_vector(train_predictive_samples_latent_space, train_summary_dict, "_samples",results_dir, method="Train")
    VegvisirPlots.plot_latent_vector(valid_predictive_samples_latent_space,valid_summary_dict, "-samples",results_dir, method="Valid")
    Vegvisir.save_checkpoint_pyro("{}/Vegvisir_checkpoints/checkpoints.pt".format(results_dir),optimizer,guide)
    Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_train.p".format(results_dir),
                               {"latent_space": train_predictive_samples_latent_space,
                                "predictions_dict":None,
                                "summary_dict": train_summary_dict})
    Vegvisir.save_model_output("{}/Vegvisir_checkpoints/model_outputs_valid.p".format(results_dir),
                               {"latent_space": valid_predictive_samples_latent_space,
                                "predictions_dict": None,
                                "summary_dict": valid_summary_dict})

    VegvisirUtils.fold_auc(train_summary_dict,train_true,"all",results_dir,mode="Train")
    VegvisirUtils.fold_auc(valid_summary_dict,valid_true,"all",results_dir,mode="Valid")

