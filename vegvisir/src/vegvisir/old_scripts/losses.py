#!/usr/bin/env python3
"""
=======================
2024: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
=======================
"""
import torch
import torch.nn as nn
from pyro.infer.trace_elbo import *
from pyro.infer.trace_elbo import _compute_log_r

class TaylorSoftmax(nn.Module):
    """
    Notes:
        - Implementation reference - https://github.com/CoinCheung/pytorch-loss/blob/master/pytorch_loss/taylor_softmax.py

        - Paper - https://www.ijcai.org/Proceedings/2020/0305.pdf"""
    def __init__(self, dim=1, t=2):
        """
        :param dim: Indicates the dimensions of the logits
        :param t: Indicates the number of approximations of the Taylor expansion
        """
        super(TaylorSoftmax, self).__init__()
        assert t % 2 == 0
        self.dim = dim
        self.t = t

    def forward(self, x):
        assert x.dim() >= 2, "Please provide at least one logit per class"
        fn = torch.ones_like(x)
        denor = 1.
        for i in range(1, self.t + 1):
            denor *= i
            fn = fn + x.pow(i) / denor
        out = fn / fn.sum(dim=self.dim, keepdims=True)
        return out


class VegvisirLosses(object):
    """"""
    def __init__(self,max_len,input_dim):
        self.sigmoid = nn.Sigmoid()
        self.max_len = max_len
        self.input_dim = input_dim

    def calculate_weights(self,true_labels,weights=None):
        """Weight class calculation"""
        if weights is not None:
            weights = weights
        else: #estimate weights per batch
            npositives = true_labels.sum() # we have more negatives in the raw data. Multiply by 10 to get 0.9 to 9 for example
            nnegatives = true_labels.shape[0] - npositives
            if npositives > nnegatives:
                weights = [torch.tensor([1,float(npositives/nnegatives)]) if nnegatives != 0 else torch.tensor([1,1])][0]
            elif npositives == 0:
                weights = torch.tensor([0.5,1.])
            elif nnegatives == 0:
                weights = torch.tensor([1.,0.5])
            else:
                weights = [torch.tensor([float(nnegatives/npositives),1]) if npositives != 0 else torch.tensor([1,1])][0]
        array_weights = true_labels.clone()
        array_weights[array_weights == 0] = weights[0]
        array_weights[array_weights == 1] = weights[1]
        return weights, array_weights

    def weighted_loss(self, y_true, y_pred, confidence_scores):
        """E(y_true,y_pred) = -y_true*log(y_pred) -(1 - y_true)*ln(1-y_pred)
        E(y_true,y_pred) = -weights[pos_weight*y_true*log(sigmoid(y_pred)) + (1-y_true)*log(1 -sigmoid(y_pred))]
        Notes:
            - Analyzing BCE: https://towardsdatascience.com/understanding-binary-cross-entropy-log-loss-a-visual-explanation-a3ac6025181a
            - https://www.mldawn.com/binary-classification-from-scratch-using-numpy/
            - Weighted BCE= https://github.com/KarimMibrahim/Sample-level-weighted-loss/blob/master/IM-WCE-MSCOCO.ipynb
            - Cross entropy: https://vitalflux.com/cross-entropy-loss-explained-with-python-examples/
            - Logistic regression with BCE: https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
            - https://hal.science/hal-02547012/document
            - https://www.researchgate.net/publication/336069340_A_Comparison_of_Loss_Weighting_Strategies_for_Multi_task_Learning_in_Deep_Neural_Networks
            - Multitask Learning Based on Improved Uncertainty Weighted Loss for Multi-Parameter Meteorological Data Prediction #TODO!!!
            - Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
           :param y_true: Probabilities values (min= 0, max= 1)
           NOTE: BCE seems to favour the prediction of class 1 correctly?
           y_true | y_pred  | BCE
           --------------------------------------
              1   |  0.8    | ((-1 * torch.log(0.8)) - ((1.0 - 1) * torch.log(1.0 - 0.8))) = 0.2231
              1   |  0.51   | ((-1 * torch.log(0.51)) - ((1.0 - 1) * torch.log(1.0 - 0.51))) = 0.6733
              0   |  0.8    | ((-0 * torch.log(0.8)) - ((1.0 - 0) * torch.log(1.0 - 0.8))) = 1.6094
              0   |  0.51   | ((-0 * torch.log(0.51)) - ((1.0 - 0) * torch.log(1.0 - 0.51))) = 0.7133


            """
        # beta = (y_true.shape[0] - y_true.sum())/y_true.shape[0] #posible class imbalance corrector
        # clip to prevent NaN's and Inf's
        y_pred = torch.clip(self.sigmoid(y_pred), 1e-7, 1 - 1e-7)
        # loss = ((-y_true * torch.log(y_pred)) - ((1.0 - y_true) * torch.log(1.0 - y_pred)))*confidence_scores
        # loss = beta*(-y_true * torch.log(y_pred)) - ((1 - beta)*(1.0 - y_true) * torch.log(1.0 - y_pred))
        loss = (-y_true * torch.log(y_pred)) - ((1.0 - y_true) * torch.log(1.0 - y_pred))

        return loss.mean()

    def focal_loss(self, y_true, y_pred, confidence_scores):
        """Notes:
        -https://pytorch.org/vision/0.12/_modules/torchvision/ops/focal_loss.html"""
        gamma = 0.5  # downweights the easy to classify examples
        y_pred = torch.clip(self.sigmoid(y_pred), 1e-7, 1 - 1e-7)
        ce_loss = (-y_true * torch.log(y_pred)) - ((1.0 - y_true) * torch.log(1.0 - y_pred))
        p_t = y_pred * y_true + (1 - y_pred) * (1 - y_true)
        loss = ce_loss * ((1 - p_t) ** gamma) * confidence_scores  # gamma = 0 ---> focal_loss = ce_loss

        return loss.mean()

    def dice_reconstruction_loss(self, reconstructed_sequences, onehot_sequences, smooth=1):
        """ Sørensen–Dice coefficient . Notes:
        - https://towardsdatascience.com/building-autoencoders-on-sparse-one-hot-encoded-data-53eefdfdbcc7"""
        # comment out if your model contains a sigmoid acitvation
        y_pred = self.sigmoid(reconstructed_sequences)

        # flatten label and prediction tensors
        y_pred = y_pred.view(-1)
        y_true = onehot_sequences.view(-1)

        intersection = (y_pred * y_true).sum()
        dice = (2. * intersection + smooth) / (y_pred.sum() + y_true.sum() + smooth)

        return 1 - dice

    def argmax_reconstruction_loss(self, reconstructed_sequences, onehot_sequences):
        assert reconstructed_sequences.shape == (onehot_sequences.shape[0], self.max_len, self.input_dim)
        idx_true = torch.argmax(onehot_sequences, dim=-1)
        idx_pred = torch.argmax(reconstructed_sequences, dim=-1)
        loss = 1 - (idx_true == idx_pred).float().mean()
        return loss

    def label_smoothing(self,y_pred,y_true,confidence_scores,num_classes):
        """https://stackoverflow.com/questions/55681502/label-smoothing-in-pytorch"""
        smoothing = 1
        smooth_factor = torch.zeros_like(y_pred)
        smooth_factor.fill_(smoothing / (num_classes -1 ))
        smooth_factor[torch.arange(0, y_true.shape[0]), y_true.long()] = confidence_scores
        return smooth_factor
        #return torch.mean(torch.sum(-true_dist * y_pred, dim=-1))

    def taylor_crossentropy_loss(self,y_true,y_pred,confidence_scores,num_classes,class_weights):
        """
        https://www.kaggle.com/code/yerramvarun/cassava-taylorce-loss-label-smoothing-combo
        https://github.com/CoinCheung/pytorch-loss/blob/master/taylor_softmax.py
        """
        log_probs = TaylorSoftmax()(y_pred).log()
        #log_probs  = log_probs[torch.arange(0, y_true.shape[0]), y_true.long()]
        #log_probs = torch.max(log_probs, dim=1).values.squeeze(-1)
        smooth_factor  = self.label_smoothing(log_probs, y_true,confidence_scores,num_classes)
        #print(confidence_scores)
        smoothed_probs = log_probs*smooth_factor
        #loss = nn.NLLLoss(reduction="mean")(smoothed_probs, y_true.long()) #ignore_index=-1
        #smoothed_probs = smoothed_probs[torch.arange(0, y_true.shape[0]), y_true.long()]
        #smoothed_probs,_ = torch.max(smoothed_probs,-1)
        #loss = nn.BCEWithLogitsLoss()(smoothed_probs,y_true)
        # print(log_probs)
        # print(confidence_scores)
        # print(smoothed_probs)

        loss = nn.NLLLoss(reduction="mean",weight=class_weights)(smoothed_probs, y_true.long())
        assert not torch.isnan(loss),"Nan loss detected. Check the confidence scores for nan"
        return loss

class Trace_ELBO_classification(ELBO):
    """
    A trace implementation of ELBO-based SVI. The estimator is constructed
    along the lines of references [1] and [2]. There are no restrictions on the
    dependency structure of the model or the guide. The gradient estimator includes
    partial Rao-Blackwellization for reducing the variance of the estimator when
    non-reparameterizable random variables are present. The Rao-Blackwellization is
    partial in that it only uses conditional independence information that is marked
    by :class:`~pyro.plate` contexts. For more fine-grained Rao-Blackwellization,
    see :class:`~pyro.infer.tracegraph_elbo.TraceGraph_ELBO`.

    References

    [1] Automated Variational Inference in Probabilistic Programming,
        David Wingate, Theo Weber

    [2] Black Box Variational Inference,
        Rajesh Ranganath, Sean Gerrish, David M. Blei
    """
    def __init__(self,max_len,input_dim,num_classes):
        super(Trace_ELBO_classification, self).__init__()
        self.losses = VegvisirLosses(max_len,input_dim)
        self.classification_loss = True
        self.num_classes = num_classes

    def _get_trace(self, model, guide, args, kwargs):
        """
        Returns a single trace from the guide, and the model that is run
        against it.
        """
        model_trace, guide_trace = get_importance_trace(
            "flat", self.max_plate_nesting, model, guide, args, kwargs
        )
        if is_validation_enabled():
            check_if_enumerated(guide_trace)
        return model_trace, guide_trace
    def loss(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Evaluates the ELBO with an estimator that uses num_particles many samples/particles.
        """
        elbo = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            elbo_particle = torch_item(model_trace.log_prob_sum()) - torch_item(
                guide_trace.log_prob_sum()
            )
            elbo += elbo_particle / self.num_particles

        loss = -elbo
        warn_if_nan(loss, "loss")
        return loss
    def _differentiable_loss_particle(self, model_trace, guide_trace):
        elbo_particle = 0
        surrogate_elbo_particle = 0
        log_r = None

        # compute elbo and surrogate elbo
        for name, site in model_trace.nodes.items():
            if site["type"] == "sample":
                elbo_particle = elbo_particle + torch_item(site["log_prob_sum"])
                surrogate_elbo_particle = surrogate_elbo_particle + site["log_prob_sum"]

        for name, site in guide_trace.nodes.items():
            if site["type"] == "sample":
                log_prob, score_function_term, entropy_term = site["score_parts"]

                elbo_particle = elbo_particle - torch_item(site["log_prob_sum"])

                if not is_identically_zero(entropy_term):
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle - entropy_term.sum()
                    )

                if not is_identically_zero(score_function_term):
                    if log_r is None:
                        log_r = _compute_log_r(model_trace, guide_trace)
                    site = log_r.sum_to(site["cond_indep_stack"])
                    surrogate_elbo_particle = (
                        surrogate_elbo_particle + (site * score_function_term).sum()
                    )

        return -elbo_particle, -surrogate_elbo_particle
    def differentiable_loss(self, model, guide, *args, **kwargs):
        """
        Computes the surrogate loss that can be differentiated with autograd
        to produce gradient estimates for the model and guide parameters
        """
        loss = 0.0
        surrogate_loss = 0.0
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            surrogate_loss += surrogate_loss_particle / self.num_particles
            loss += loss_particle / self.num_particles
        warn_if_nan(surrogate_loss, "loss")
        return loss + (surrogate_loss - torch_item(surrogate_loss))
    def euclidean_2d_norm(self,A,B,squared=True):
        """
        Computes euclidean distance among matrix/arrays according to https://medium.com/swlh/euclidean-distance-matrix-4c3e1378d87f
        Equivalent to scipy.spatial.distance_matrix(A,B)
        Note: To calculate vector euclidean distance or euclidean_1d_norm, use:
            euclidean_1d_norm = torch.sqrt(torch.sum((X1[:, None, :] - X2) ** 2,dim=2))  # equal to torch.cdist(X1,X2) or scipy.spatial.distance.cdist , which is for 1D space, for more dimensions we need the dot product
        """

        diag_AA_T = torch.sum(A**2,dim=1)[:,None]
        diag_BB_T = torch.sum(B**2,dim=1)
        third_component = -2*torch.mm(A,B.T)
        distance = diag_AA_T + third_component + diag_BB_T
        if squared:
            distance = torch.sqrt(distance)
            return distance.clamp(min=0) #to avoid nan/negative values, set them to 0
        else:
            return distance.clamp(min=0)
    def loss_and_grads(self, model, guide, *args, **kwargs):
        """
        :returns: returns an estimate of the ELBO
        :rtype: float

        Computes the ELBO as well as the surrogate ELBO that is used to form the gradient estimator.
        Performs backward on the latter. Num_particle many samples are used to form the estimators.
        """
        loss = 0.0
        # grab a trace from the generator
        for model_trace, guide_trace in self._get_traces(model, guide, args, kwargs):
            loss_particle, surrogate_loss_particle = self._differentiable_loss_particle(
                model_trace, guide_trace
            )
            loss += loss_particle / self.num_particles
            if self.classification_loss :
                #Reconstruction loss of the upper triangular adjacency matrix
                batch_data_blosum = model_trace.nodes["_INPUT"]["args"][0]["blosum"]
                y_true = batch_data_blosum[:,0,0,0]
                confidence_scores = batch_data_blosum[:,0,0,5]
                y_pred = model_trace.nodes["_RETURN"]["value"]["sequences_logits"]

                #classification_loss = self.losses.taylor_crossentropy_loss(y_true,y_pred,confidence_scores,self.num_classes)
                print(loss)
                #print(classification_loss)
                #TODO: Also make weighted loss for logits_classes, penalize over positive class prediction
                #loss += classification_loss/self.num_particles


            # collect parameters to train from model and guide
            trainable_params = any(
                site["type"] == "param"
                for trace in (model_trace, guide_trace)
                for site in trace.nodes.values()
            )

            if trainable_params and getattr(
                surrogate_loss_particle, "requires_grad", False
            ):
                surrogate_loss_particle = surrogate_loss_particle / self.num_particles
                surrogate_loss_particle.backward(retain_graph=self.retain_graph)
        warn_if_nan(loss, "loss")
        return loss

