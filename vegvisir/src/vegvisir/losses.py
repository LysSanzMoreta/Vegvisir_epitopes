import torch
import torch.nn as nn
class VegvisirLosses(object):
    """"""
    def __init__(self,max_len,input_dim):
        self.sigmoid = nn.Sigmoid()
        self.max_len = max_len
        self.input_dim = input_dim

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