"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import torch.nn as nn
import torch
from abc import abstractmethod
from vegvisir.model_utils import *
class VEGVISIRModelClass(nn.Module):
    def __init__(self, model_load):
        super(VEGVISIRModelClass, self).__init__()
        self.aa_types = model_load.aa_types
        self.embedding_dim = 50
        self.hidden_dim = 60
        self.max_len = model_load.max_len
        self.batch_size = model_load.args.batch_size
        self.input_dim = model_load.input_dim
        self.hidden_dim = model_load.args.hidden_dim
        self.device = model_load.args.device
        self.use_cuda = model_load.args.use_cuda
        #self.dropout = model_load.args.dropout
        self.num_classes = model_load.args.num_classes
        self.embedding_dim = model_load.args.embedding_dim
        self.blosum = model_load.blosum
        self.loss_type = model_load.args.loss_func
        if self.use_cuda:
            # calling cuda() here will put all the parameters of
            # the networks into gpu memory
            self.cuda()
    @abstractmethod
    def forward(self,batch_data,batch_mask):
        raise NotImplementedError
    @abstractmethod
    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name
class VegvisirModel1(VEGVISIRModelClass):
    """
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #self.mlp = MLP(self.aa_types*self.max_len,self.hidden_dim*2,self.num_classes,self.device)
        self.cnn = CNN_layers(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        self.gru_hidden_dim = self.hidden_dim*2
        #self.rnn = RNN_layers(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.device,self.loss_type)
        self.sigmoid = nn.Sigmoid()
        #self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.c_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)

    def forward(self,batch_data,batch_mask):
        """"""
        batch_sequences = batch_data[:,1].squeeze(1)
        batch_sequences = self.embedder(batch_sequences,None)
        #probs = self.mlp(batch_sequences.flatten(1),None)
        probs = self.cnn(batch_sequences.permute(0,2,1),None)
        #init_h_0 = self.h_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        # init_c_0 = self.c_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],
        #                                  self.gru_hidden_dim).contiguous()  # bidirectional
        #probs = self.rnn(batch_sequences,init_h_0,None)

        return probs

    def loss(self,confidence_scores,true_labels,predictions):
        """Weighted loss according to confidence score:
        Notes:
            - https://hal.science/hal-02547012/document
            - https://www.researchgate.net/publication/336069340_A_Comparison_of_Loss_Weighting_Strategies_for_Multi_task_Learning_in_Deep_Neural_Networks
            - Multitask Learning Based on Improved Uncertainty Weighted Loss for Multi-Parameter Meteorological Data Prediction #TODO!!!
            - Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics
            """
        predictions = torch.max(predictions, dim=1).values
        if self.loss_type == "weighted_bce":

            bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            output = bce_loss(predictions, true_labels)

            #output = self.weighted_loss(true_labels,predictions,confidence_scores)

            return output
        elif self.loss_type == "bce":
            #bce_loss = nn.BCELoss()
            bce_loss = nn.BCEWithLogitsLoss()
            output = bce_loss(predictions, true_labels)
            return output
    def weighted_loss(self,y_true, y_pred, confidence_scores):
        """E(y_true,y_pred) = -y_true*log(y_pred) -(1 - y_true)*ln(1-y_pred)
        E(y_true,y_pred) = -weights[pos_weight*y_true*log(sigmoid(y_pred)) + (1-y_true)*log(1 -sigmoid(y_pred))]
        Notes:
            - https://www.mldawn.com/binary-classification-from-scratch-using-numpy/
            - Weighted BCE= https://github.com/KarimMibrahim/Sample-level-weighted-loss/blob/master/IM-WCE-MSCOCO.ipynb
            - Cross entropy: https://vitalflux.com/cross-entropy-loss-explained-with-python-examples/
            - Logistic regression with BCE: https://developer.ibm.com/articles/implementing-logistic-regression-from-scratch-in-python/
           :param y_true: Probabilities values (min= 0, max= 1)
            """
        # Define the proposed weighted loss function
        # clip to prevent NaN's and Inf's
        y_pred = torch.clip(self.sigmoid(y_pred), 1e-7, 1 - 1e-7)
        loss = ((-y_true * torch.log(y_pred)) - ((1.0 - y_true) * torch.log(1.0 - y_pred)))*confidence_scores

        return loss.mean()