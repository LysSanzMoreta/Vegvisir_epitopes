"""
=======================
2023: Lys Sanz Moreta
Vegvisir :
=======================
"""
import numpy as np
import pyro
import torch.nn as nn
import torch
from collections import defaultdict,namedtuple
from abc import abstractmethod
from pyro.nn import PyroModule
import pyro.distributions as dist
from pyro.infer import Trace_ELBO,JitTrace_ELBO,TraceMeanField_ELBO, TraceEnum_ELBO
from vegvisir.model_utils import *
from vegvisir.losses import *
ModelOutput = namedtuple("ModelOutput",["reconstructed_sequences","class_out"])
SamplingOutput = namedtuple("SamplingOutput",["latent_space","predicted_labels","immunodominance_scores","reconstructed_sequences"])

class VEGVISIRModelClass(nn.Module):
    def __init__(self, model_load):
        super(VEGVISIRModelClass, self).__init__()
        self.args = model_load.args
        self.likelihood_scale = model_load.args.likelihood_scale #scaling the class log likelihood
        self.aa_types = model_load.aa_types
        self.seq_max_len = model_load.seq_max_len
        self.max_len = model_load.max_len
        self.batch_size = model_load.args.batch_size
        self.likelihood_scale = self.likelihood_scale if self.likelihood_scale < 100 else self.batch_size
        self.input_dim = model_load.input_dim
        self.hidden_dim = model_load.args.hidden_dim
        self.embedding_dim = model_load.args.embedding_dim
        self.z_dim = model_load.args.z_dim
        self.device = model_load.args.device
        self.use_cuda = model_load.args.use_cuda
        self.tensor_type = torch.cuda.DoubleTensor if self.use_cuda else torch.DoubleTensor
        #self.dropout = model_load.args.dropout
        self.generate_num_samples = model_load.args.generate_num_samples
        self.num_classes = model_load.args.num_obs_classes
        self.embedding_dim = model_load.args.embedding_dim
        self.blosum = model_load.blosum
        self.loss_type = model_load.args.loss_func
        self.learning_type = model_load.args.learning_type
        self.class_weights = model_load.class_weights
        self.num_iafs = 0
        if self.use_cuda:
            # calling cuda() here will put all the parameters of
            # the networks into gpu memory
            #self.cuda()
            self.to(self.device)

        self.gradients_dict = {}
        self.handles_dict = defaultdict(list)
        self.visualization_dict = {}
        self.parameters_dict = {}
        self._parameters = {}

    def build(self,parameter_list,suffix):
        """Function designed to store nn.Parameters"""
        self.wt_dict = nn.ParameterDict()
        for i,param in enumerate(parameter_list):
            self.wt_dict["weight_{}_{}".format(suffix,i)] = param
    @abstractmethod
    def forward(self,batch_data,batch_mask):
        raise NotImplementedError
    @abstractmethod
    def get_class(self):
        full_name = self.__class__
        name = str(full_name).split(".")[-1].replace("'>","")
        return name
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_checkpoint(self, filename,optimizer):
        """Stores the model weight parameters and optimizer status"""
        # Builds dictionary with all elements for resuming training
        checkpoint = {'model_state_dict': self.state_dict(),
                      'optimizer_state_dict': optimizer.state_dict()}
        torch.save(checkpoint, filename)
    def save_checkpoint_pyro(self, filename,optimizer,guide):
        """Stores the model weight parameters and optimizer status"""
        # Builds dictionary with all elements for resuming training
        try:
            checkpoint = {'model_state_dict': self.state_dict(),
                          'optimizer_state_dict': optimizer.get_state(),
                          'guide_state_dict':guide.state_dict()}
        except:
            checkpoint = {'model_state_dict': self.state_dict(),
                          'optimizer_state_dict': optimizer.get_state(),
                          'guide_state_dict':None}
        torch.save(checkpoint, filename)

    def load_checkpoint(self, filename,optimizer):
        # Loads dictionary
        checkpoint = torch.load(filename)
        # Restore state for model and optimizer
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.eval() # resume training
    def load_checkpoint_pyro(self, filename,optimizer=None):
        # Loads dictionary
        checkpoint = torch.load(filename)
        # Restore state for model and optimizer
        self.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.set_state(checkpoint['optimizer_state_dict'])
        self.eval() # resume training
    def save_model_output(self,filename,output_dict):
        """"""
        torch.save(output_dict, filename,pickle_protocol=4)

    def chebyshev_inverse_3d(self,a, iterations=10):
        """Computes an approximation of the inverse of a matrix for large matrices
        alpha = \frac{1}{(||A||_{1})*||A||_{\inf}}
        \begin{cases}
           N(0) = alpha.a^T
           N(t+1) = N(t)(3.In -a\cdotN(t)(3.In - a.N(t)))
        \end{cases}

        :param a : Matrix
        :param iterations: Number of chebyshev polynomial series to compute"""

        a_norm_1d = torch.linalg.matrix_norm(a, ord=1)
        a_norm_inf = torch.linalg.matrix_norm(a, ord=float('inf'))
        alpha = 1 / (a_norm_1d * a_norm_inf)
        N_0 = alpha[:, None, None] * a.permute(0, 2, 1)
        N_t = N_0
        In = torch.eye(a.shape[1]).expand(a.shape[0], a.shape[1], a.shape[2])
        for t in range(iterations):
            N_t_plus_1 = torch.matmul(N_t,(3 * In - torch.matmul(torch.matmul(a, N_t), (3 * In - torch.matmul(a, N_t)))))
            N_t = N_t_plus_1
        return N_t

    def conditional_sampling_fast(self,n_generated, n_train, guide_estimates):
        """Conditional sampling the internal nodes given the leaves from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
        :param guide_estimates: dictionary conatining the MAP estimates for the OU process parameters
        :param """

        # Highlight:  See Page 689 at Patter Recongnition and Ml (Bishop)
        # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves

        # Highlight: B.49 Λ−1aa
        inverse_generated = torch.eye(n_generated)  # [z_dim,n_test,n_test]
        # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
        # Highlight: µa
        OU_mean_generated = torch.zeros((n_generated,))  # [n_generated,]
        # Highlight: Λab
        inverse_generated_train = torch.eye(n_generated, n_train)  # [z_dim,n_generated,n_train]
        # Highlight: xb
        xb = torch.from_numpy(guide_estimates["latent_z"][:,5:]).T.to(OU_mean_generated.device)  # [z_dim,n_train]

        assert xb.shape == (self.z_dim,n_train), "Perhaps you forgot that the latent space has some columns stacked"
        # Highlight:µb
        OU_mean_train = torch.zeros((n_train,))
        # Highlight:µa|b---> Splitted Equation  B-50
        part1 = torch.matmul(torch.linalg.inv(inverse_generated), inverse_generated_train)  # [z_dim,n_test,n_train]
        part2 = xb - OU_mean_train[None, :]  # [z_dim,n_train]
        OU_mean = OU_mean_generated[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]

        latent_space = dist.MultivariateNormal(OU_mean.squeeze(-1),torch.linalg.inv(inverse_generated) + 1e-6).to_event(1).sample()
        latent_space = latent_space.T
        return latent_space

    def conditional_sampling(self,n_generated, n_train, guide_estimates):
        """Conditional sampling the synthetic epitopes given the learnt representations from the training dataset from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
        :param guide_estimates: dictionary conatining the MAP estimates for the OU process parameters
        :param """
        print("Sampling .........")
        # Highlight:  See Page 689 at Patter Recongnition and Ml (Bishop)
        # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves

        # Highlight: B.49 Λ−1aa
        inverse_generated = torch.eye(n_generated)  # [n_test,n_test]
        inverse_generated = inverse_generated[None,:].expand(self.z_dim,n_generated,n_generated) # [z_dim,n_test,n_test]
        z_scales = torch.from_numpy(guide_estimates["z_scales"]).mean(0).to(inverse_generated.device) #[z_dim]
        inverse_generated = z_scales[:,None,None]*inverse_generated

        # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
        # Highlight: µa
        OU_mean_generated = torch.zeros((n_generated,))  # [n_generated,]
        # Highlight: Λab
        inverse_generated_train = torch.eye(n_generated, n_train)  # [n_generated,n_train]
        inverse_generated_train = inverse_generated_train[None,:].expand(self.z_dim,n_generated,n_train) # [z_dim,n_generated,n_train]
        inverse_generated_train = z_scales[:,None,None]*inverse_generated_train

        # Highlight: xb
        xb = torch.from_numpy(guide_estimates["latent_z"][:,5:]).T.to(OU_mean_generated.device)  # [z_dim,n_train]

        assert xb.shape == (self.z_dim,n_train), "Perhaps you forgot that the latent space has some columns stacked"
        # Highlight:µb
        OU_mean_train = torch.zeros((n_train,))
        # Highlight:µa|b---> Splitted Equation  B-50
        part1 = torch.matmul(self.chebyshev_inverse_3d(inverse_generated), inverse_generated_train)  # [z_dim,n_test,n_train]
        part2 = xb - OU_mean_train[None, :]  # [z_dim,n_train]
        z_mean = OU_mean_generated[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]

        if n_generated > 3000: #this is very slow
            batch_size_init = 1000
            split_size = int(n_generated / batch_size_init) if not batch_size_init > n_generated else 1
            z_mean = z_mean.squeeze(-1)
            inverse_generated = self.chebyshev_inverse_3d(inverse_generated)
            idx = torch.zeros(n_generated).bool()
            idx0= 0
            batch_size = batch_size_init
            latent_space_list = []
            for i in range(split_size):
                idx_split = idx.clone()
                idx_split[idx0:batch_size] = True
                z_mean_split = z_mean[:,idx_split]
                inverse_generated_split = inverse_generated[:,idx_split]
                inverse_generated_split = inverse_generated_split[:,:,idx_split]

                idx0=batch_size
                batch_size += batch_size_init

                latent_space_split = dist.MultivariateNormal(z_mean_split.squeeze(-1),inverse_generated_split + 1e-6).to_event(1).sample()
                latent_space_split = latent_space_split.T
                latent_space_list.append(latent_space_split)

            latent_space = torch.concatenate(latent_space_list,dim=0)
        else:

            latent_space = dist.MultivariateNormal(z_mean.squeeze(-1),self.chebyshev_inverse_3d(inverse_generated) + 1e-6).to_event(1).sample()
            latent_space = latent_space.T
        return latent_space


class VegvisirModel1(VEGVISIRModelClass):
    """
    Multilayer Perceptron model
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: MLP
        self.mlp = MLP(self.aa_types*self.max_len,self.hidden_dim*2,self.num_classes,self.device)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        """
        batch_sequences = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: MLP
        class_out = self.mlp(batch_sequences.flatten(1),None)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        weights = self.class_weights
        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss()
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel2a(VEGVISIRModelClass):
    """
    Convolutional networks based models without information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: CNN
        self.cnn = CNN_layers(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        #self.letnet5 = LetNET5(self.aa_types, self.max_len, self.hidden_dim * 2, self.num_classes, self.device,self.loss_type)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: CNN
        class_out = self.cnn(batch_sequences.permute(0,2,1),None)

        #logits,class_out = self.letnet5(batch_sequences.permute(0, 2, 1), None)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        weights,array_weights = self.losses.calculate_weights(true_labels,self.class_weights)

        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels,predictions,confidence_scores,self.num_classes,weights)
            return loss
        elif self.loss_type == "bceprobs":

            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=array_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel2b(VEGVISIRModelClass):
    """
    Convolutional networks based models without information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        self.feats_dim = self.max_len - self.seq_max_len
        #Highlight: CNN
        self.cnn = CNN_layers(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        self.fcl3 = FCL3(self.feats_dim,self.hidden_dim*2,self.num_classes,self.device)
        #self.letnet5 = LetNET5(self.aa_types, self.max_len, self.hidden_dim * 2, self.num_classes, self.device,self.loss_type)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        :type batch_data: object
        """
        batch_sequences = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:,1,self.seq_max_len:,0]
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: CNN
        logits_seqs = self.cnn(batch_sequences.permute(0,2,1),None)
        logits_feats = self.fcl3(batch_features)
        # print("sequences")
        # print(logits_seqs)
        # print("features")
        #logits,class_out = self.letnet5(batch_sequences.permute(0, 2, 1), None)
        class_out = logits_seqs + logits_feats
        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        weights,array_weights = self.losses.calculate_weights(true_labels,self.class_weights)

        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels,predictions,confidence_scores,self.num_classes,weights)
            return loss
        elif self.loss_type == "bceprobs":

            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=array_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel3a(VEGVISIRModelClass):
    """
    Recurrent Neural Networks models with information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: RNN
        self.gru_hidden_dim = self.hidden_dim*2
        self.rnn = RNN_layers(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.device,self.loss_type)
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.h_0_MODEL_r = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.c_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device) #For LSTM
        self.losses = VegvisirLosses(self.max_len,self.input_dim)


    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: RNN
        init_h_0 = self.h_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        #init_h_0_r = self.h_0_MODEL_r.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        #init_c_0 = self.c_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        class_out = self.rnn(batch_sequences,None,init_h_0)
        #class_out = self.rnn(batch_sequences,None,init_h_0_r)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)


    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        weights,array_weights = self.losses.calculate_weights(true_labels,self.class_weights)

        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]


            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels, predictions, confidence_scores, self.num_classes,weights)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=array_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel3b(VEGVISIRModelClass):
    """
    Recurrent Neural Networks models with information bottleneck
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        self.feats_dim = self.max_len - self.seq_max_len
        #Highlight: RNN
        self.gru_hidden_dim = self.hidden_dim*2
        self.rnn = RNN_layers(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.device,self.loss_type)
        self.fcl3 = FCL3(self.feats_dim,self.hidden_dim*2,self.num_classes,self.device)
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.c_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device) #For LSTM
        self.losses = VegvisirLosses(self.max_len,self.input_dim)


    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:,1,self.seq_max_len:,0]

        #Highlight: RNN
        init_h_0 = self.h_0_MODEL.expand(self.rnn.num_layers * 2, batch_sequences.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        logits_seqs = self.rnn(batch_sequences,None,init_h_0)
        logits_feats = self.fcl3(batch_features)
        class_out = logits_seqs + logits_feats

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)


    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        weights,array_weights = self.losses.calculate_weights(true_labels,self.class_weights)

        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]


            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels, predictions, confidence_scores, self.num_classes,weights)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=array_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel4(VEGVISIRModelClass):
    """
    Simple Autoencoder
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: Autoencoder
        self.autoencoder = AutoEncoder(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device,self.loss_type)
        self.sigmoid = nn.Sigmoid()
        self.losses = VegvisirLosses(self.max_len,self.input_dim)


    def forward(self,batch_data,batch_mask):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        """
        batch_sequences = batch_data["blosum"][:,1].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #Highlight: Autoencoder
        reconstructed_seqs,class_out = self.autoencoder(batch_sequences.permute(0,2,1))

        return ModelOutput(reconstructed_sequences=reconstructed_seqs,
                           class_out=class_out)

    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return:
        :rtype: object
        """

        weights = self.class_weights

        if self.loss_type == "ae_loss":
            #reconstruction_loss = nn.CosineEmbeddingLoss(reduction='none')(onehot_sequences[:,1],model_outputs.reconstructed_sequences)
            reconstruction_loss = self.losses.argmax_reconstruction_loss(model_outputs.reconstructed_sequences,onehot_sequences[:,1])
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions,dim=1).values.squeeze(-1)
                #TODO: torch.round()
            else:
                predictions = predictions.squeeze(-1)
            classification_loss = nn.BCEWithLogitsLoss(pos_weight=weights)(predictions,true_labels)
            total_loss = reconstruction_loss + classification_loss.mean()

            return total_loss
        else:
            raise ValueError(
                "Error loss: {} not implemented for this model type: {}".format(self.loss_type, self.get_class()))

class VegvisirModel5a_supervised_no_decoder(VEGVISIRModelClass,PyroModule):
    """
    Variational Autoencoder with all dimensions dependent
    -Notes:
            https://pyro.ai/examples/cvae.html
            https://avandekleut.github.io/vae/
    -Notes: on nan values
            http://pyro.ai/examples/svi_part_iv.html
            https://forum.pyro.ai/t/my-guide-keeps-producing-nan-values-what-am-i-doing-wrong/2024/8
    -CSVAE:
            https://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        self.gru_hidden_dim = self.hidden_dim*2
        self.num_params = 2 #number of parameters of the beta distribution
        #self.decoder = RNN_model6(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.classifier_model = FCL4(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device)
        #self.classifier_model = CNN_layers(1,self.z_dim,self.hidden_dim,self.num_classes,self.device) #input_dim,max_len,hidden_dim,num_classes,device,loss_type
        #self.classifier_model = RNN_classifier(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.z_dim,self.device) #input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device
        #self.h_0_MODEL_encoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.h_0_MODEL_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.h_0_MODEL_classifier = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)
        #self.init_hidden = Init_Hidden(self.z_dim, self.max_len, self.gru_hidden_dim, self.device)

    # def model_non_glitch(self,batch_data,batch_mask,epoch,guide_estimates,sample=False):
    #     """
    #     :param batch_data:
    #     :param batch_mask:
    #     :return:
    #     - Notes:
    #         - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
    #         - https://maxhalford.github.io/blog/bayesian-linear-regression/
    #         - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
    #         - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
    #         - https://fehiepsi.github.io/rethinking-pyro/
    #     """
    #
    #     pyro.module("vae_model", self)
    #     batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
    #
    #     batch_sequences_int = batch_data["int"][:,1].squeeze(1)
    #     batch_sequences_norm = batch_data["norm"][:,1]
    #     batch_size = batch_sequences_blosum.shape[0]
    #     batch_mask_len = batch_mask[:,1:].squeeze(1)
    #     batch_mask_len = batch_mask_len[:,:,0]
    #     batch_sequences_lens = batch_mask_len.sum(dim=1)
    #     batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
    #     true_labels = batch_data["blosum"][:,0,0,0]
    #     #immunodominance_scores = batch_data["blosum"][:,0,0,4]
    #     confidence_scores = batch_data["blosum"][:,0,0,5]
    #     confidence_mask = (confidence_scores[..., None] > 0.7).any(-1) #now we try to predict those with a low confidence score
    #     confidence_mask_true = torch.ones_like(confidence_mask).bool()
    #     #init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
    #     #z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
    #     z_mean,z_scale = torch.zeros((batch_size,self.z_dim)), torch.ones((batch_size,self.z_dim))
    #     with pyro.plate("plate_batch",dim=-1,device=self.device):
    #         latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]
    #         #Highlight: Create fake variable tensors to avoid changing the metrics calculation pipeline
    #         attn_weights = torch.randn(batch_size,self.seq_max_len,self.seq_max_len)
    #         encoder_hidden_states = torch.rand(batch_size,2,self.seq_max_len,self.gru_hidden_dim)
    #         decoder_hidden_states = torch.rand(batch_size,2,self.seq_max_len,self.gru_hidden_dim)
    #         encoder_final_hidden = torch.rand(batch_size,self.gru_hidden_dim)
    #         decoder_final_hidden = torch.rand(batch_size,self.gru_hidden_dim)
    #         pyro.deterministic("attn_weights", attn_weights, event_dim=0)
    #         pyro.deterministic("encoder_hidden_states", encoder_hidden_states, event_dim=0)
    #         pyro.deterministic("decoder_hidden_states", decoder_hidden_states, event_dim=0)
    #         pyro.deterministic("encoder_final_hidden", encoder_final_hidden, event_dim=0)
    #         pyro.deterministic("decoder_final_hidden", decoder_final_hidden, event_dim=0)
    #         sequences_logits = self.logsoftmax(torch.rand(batch_size,self.seq_max_len,self.aa_types))
    #         pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)
    #         pyro.deterministic("sequences", batch_sequences_int, event_dim=0)
    #
    #         class_logits = self.classifier_model(latent_space,None)
    #         class_logits = self.logsoftmax(class_logits) #[N,num_classes]
    #         pyro.deterministic("class_logits", class_logits,event_dim=1)
    #         with pyro.poutine.mask(mask=confidence_mask_true):
    #             pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)  # [N,]
    #
    #
    #     return {"attn_weights":attn_weights}

    def model_glitched(self, batch_data, batch_mask, epoch, guide_estimates, sample=False):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
            - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
            - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
            - https://fehiepsi.github.io/rethinking-pyro/
        """
        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
        batch_sequences_int = batch_data["int"][:, 1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:, 1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]

        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] > 0.7).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        # init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        # z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
        z_mean, z_scale = torch.zeros((batch_size, self.z_dim)), torch.ones((batch_size, self.z_dim))
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]

            # Highlight: Create fake variable tensors to avoid changing the metrics calculation pipeline

            attn_weights = torch.randn(batch_size,self.seq_max_len,self.seq_max_len)
            encoder_hidden_states = torch.rand(batch_size,2,self.seq_max_len,self.gru_hidden_dim)
            decoder_hidden_states = torch.rand(batch_size,2,self.seq_max_len,self.gru_hidden_dim)
            encoder_final_hidden = torch.rand(batch_size,self.gru_hidden_dim)
            decoder_final_hidden = torch.rand(batch_size,self.gru_hidden_dim)
            pyro.deterministic("attn_weights", attn_weights, event_dim=0)
            pyro.deterministic("encoder_hidden_states", encoder_hidden_states, event_dim=0)
            pyro.deterministic("decoder_hidden_states", decoder_hidden_states, event_dim=0)
            pyro.deterministic("encoder_final_hidden", encoder_final_hidden, event_dim=0)
            pyro.deterministic("decoder_final_hidden", decoder_final_hidden, event_dim=0)
            sequences_logits = self.logsoftmax(torch.rand(batch_size,self.seq_max_len,self.aa_types))
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)
            pyro.deterministic("sequences", batch_sequences_int, event_dim=0)

            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=1)
            with pyro.poutine.mask(mask=confidence_mask_true):
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=[None if sample else true_labels][0])  # [N,]

        return {"attn_weights": attn_weights}

    def model(self, batch_data, batch_mask, epoch, guide_estimates, sample):
        if self.args.glitch:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)
        else:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)

    def sample(self,batch_data,batch_mask,guide_estimates,argmax=False):
        """"""
        raise ValueError("Not implemented")


    def loss(self):
        """
        """
        return Trace_ELBO(strict_enumeration_warning=False)

class VegvisirModel5a_supervised_og(VEGVISIRModelClass,PyroModule):
    """
    Variational Autoencoder with all dimensions dependent
    -Notes:
            https://pyro.ai/examples/cvae.html
            https://avandekleut.github.io/vae/
    -Notes: on nan values
            http://pyro.ai/examples/svi_part_iv.html
            https://forum.pyro.ai/t/my-guide-keeps-producing-nan-values-what-am-i-doing-wrong/2024/8
    -CSVAE:
            https://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        self.gru_hidden_dim = self.hidden_dim*2
        self.num_params = 2 #number of parameters of the beta distribution
        #self.decoder = RNN_model6(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.decoder = RNN_model7(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device).to(self.device) #Highlight: Reconstr accurac too high
        self.classifier_model = FCL4(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device).to(self.device)
        #self.classifier_model = CNN_layers(1,self.z_dim,self.hidden_dim,self.num_classes,self.device) #input_dim,max_len,hidden_dim,num_classes,device,loss_type
        #self.classifier_model = RNN_classifier(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.z_dim,self.device) #input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device
        #self.h_0_MODEL_encoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.bidirectional = [2 if self.decoder.bidirectional else 1][0]
        self.h_0_MODEL_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device) #this is used only for generative purposes, not training
        #self.h_0_MODEL_classifier = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)
        #self.init_hidden = Init_Hidden(self.z_dim, self.max_len, self.gru_hidden_dim, self.device)
        self.build([self.h_0_MODEL_decoder],suffix="_model")
        self.num_iafs = 0


    # def model_non_glitch(self,batch_data,batch_mask,epoch,guide_estimates,sample=False):
    #     """
    #     :param batch_data:
    #     :param batch_mask:
    #     :return:
    #     - Notes:
    #         - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
    #         - https://maxhalford.github.io/blog/bayesian-linear-regression/
    #         - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
    #         - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
    #         - https://fehiepsi.github.io/rethinking-pyro/
    #     """
    #
    #     pyro.module("vae_model", self)
    #     batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
    #
    #     batch_sequences_int = batch_data["int"][:,1].squeeze(1)
    #     batch_sequences_norm = batch_data["norm"][:,1]
    #     batch_size = batch_sequences_blosum.shape[0]
    #     batch_mask_len = batch_mask[:,1:].squeeze(1)
    #     batch_mask_len = batch_mask_len[:,:,0]
    #     batch_sequences_lens = batch_mask_len.sum(dim=1)
    #     batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
    #     true_labels = batch_data["blosum"][:,0,0,0]
    #     #immunodominance_scores = batch_data["blosum"][:,0,0,4]
    #     confidence_scores = batch_data["blosum"][:,0,0,5]
    #     confidence_mask = (confidence_scores[..., None] > 0.7).any(-1) #now we try to predict those with a low confidence score
    #     confidence_mask_true = torch.ones_like(confidence_mask).bool()
    #     #init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
    #     #z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
    #     z_mean,z_scale = torch.zeros((batch_size,self.z_dim)), torch.ones((batch_size,self.z_dim))
    #     with pyro.plate("plate_batch",dim=-1,device=self.device):
    #         latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]
    #
    #         latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len, self.z_dim) #[N,L,z_dim]
    #         #print(latent_z_seq)
    #         init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,self.gru_hidden_dim).contiguous()
    #         #init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
    #         #sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)
    #
    #         outputnn= self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder,z=latent_z_seq, mask=batch_mask_len, guide_estimates=guide_estimates)
    #
    #         pyro.deterministic("attn_weights",outputnn.attn_weights,event_dim=0)
    #         pyro.deterministic("encoder_hidden_states",outputnn.encoder_hidden_states,event_dim=0)
    #         pyro.deterministic("decoder_hidden_states",outputnn.decoder_hidden_states,event_dim=0)
    #         pyro.deterministic("encoder_final_hidden",outputnn.encoder_final_hidden,event_dim=0)
    #         pyro.deterministic("decoder_final_hidden",outputnn.decoder_final_hidden,event_dim=0)
    #         sequences_logits = self.logsoftmax(outputnn.output)
    #         pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)
    #         #with pyro.plate("plate_len", dim=-2, device=self.device):
    #         #with pyro.poutine.mask(mask=batch_mask_len_true):#highlight: removed .to_event(1)
    #         #with pyro.poutine.mask(mask=batch_mask_len):
    #         pyro.sample("sequences",dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=None if sample else batch_sequences_int)
    #         #init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
    #         class_logits = self.classifier_model(latent_space,None)
    #         class_logits = self.logsoftmax(class_logits) #[N,num_classes]
    #         pyro.deterministic("class_logits", class_logits,event_dim=1)
    #         with pyro.poutine.mask(mask=confidence_mask_true):
    #             pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)  # [N,]
    #
    #
    #     return {"attn_weights":outputnn.attn_weights}

    def model_glitched(self, batch_data, batch_mask, epoch, guide_estimates, sample=False):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
            - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
            - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
            - https://fehiepsi.github.io/rethinking-pyro/
        """

        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
        batch_sequences_int = batch_data["int"][:, 1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:, 1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]

        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] > 0.7).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        # init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        # z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
        z_mean, z_scale = torch.zeros((batch_size, self.z_dim)), torch.ones((batch_size, self.z_dim))
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            if self.num_iafs > 0:
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)) # [n,z_dim]
            else:
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1)) # [n,z_dim]

            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len,self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,self.gru_hidden_dim).contiguous() #[2,batch_size,gru_dim]
            #init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)

            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=0) #should be event_dim = 2, but for sampling convenience we leave it here
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=0) #should be event_dim = 3
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=0) #should be event_dim = 3
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=0) #should be event_dim = 2
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=0) #should be event_dim = 2
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=0) #should be event_dim = 2
            # with pyro.plate("plate_len", dim=-2, device=self.device):
            # with pyro.poutine.mask(mask=batch_mask_len_true):#highlight: removed .to_event(1)
            #with pyro.poutine.mask(mask=batch_mask_len):
            pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=None if sample else batch_sequences_int)
            # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=0) #should be event_dim = 1
            #with pyro.poutine.mask(mask=confidence_mask_true):
                #pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)  # [N,]
            with pyro.poutine.scale(None,self.likelihood_scale):
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)

        return {"attn_weights": outputnn.attn_weights,
                "encoder_hidden_states":outputnn.encoder_hidden_states,
                "decoder_hidden_states":outputnn.decoder_hidden_states,
                "encoder_final_hidden":outputnn.encoder_final_hidden,
                "decoder_final_hidden":outputnn.decoder_final_hidden,
                "sequences_logits":sequences_logits,
                "class_logits":class_logits}

    def model(self, batch_data, batch_mask, epoch, guide_estimates, sample):
        if self.args.glitch:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)
        else:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)

    def sample(self,batch_data, batch_mask, epoch, guide_estimates, sample,argmax):
        """"""
        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
        batch_sequences_int = batch_data["int"][:, 1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:, 1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]

        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] > 0.7).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        # init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        # z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
        z_mean, z_scale = torch.zeros((batch_size, self.z_dim)), torch.ones((batch_size, self.z_dim))
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            if self.num_iafs > 0:
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)).to(device=self.device)  # [n,z_dim]
            else:
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)).to_event(1).to(device=self.device)  # [n,z_dim]

            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len,
                                                                            self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,
                                                             self.gru_hidden_dim).contiguous()
            # init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)

            pyro.deterministic("attn_weights", outputnn.attn_weights,event_dim=0)  # should be event_dim = 2, but for sampling convenience we leave it here
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states,event_dim=0)  # should be event_dim = 3
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states,event_dim=0)  # should be event_dim = 3
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden,event_dim=0)  # should be event_dim = 2
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden,event_dim=0)  # should be event_dim = 2
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)  # should be event_dim = 2

            if argmax:
                generated_sequences = torch.argmax(sequences_logits,dim=-1)
            else:
                num_seq_samples = 1
                generated_sequences = dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1).sample([num_seq_samples])
            # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=0)  # should be event_dim = 1
            with pyro.poutine.scale(None, self.likelihood_scale): #with pyro.poutine.mask(mask=confidence_mask_true):
                #binary_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)
                binary_predictions = dist.Categorical(logits=class_logits).to_event(1).sample([self.generate_num_samples])

        return {"attn_weights": outputnn.attn_weights,
                "encoder_hidden_states": outputnn.encoder_hidden_states,
                "decoder_hidden_states": outputnn.decoder_hidden_states,
                "encoder_final_hidden": outputnn.encoder_final_hidden,
                "decoder_final_hidden": outputnn.decoder_final_hidden,
                "sequences_logits": sequences_logits,
                "class_logits": class_logits,
                "binary_predictions":binary_predictions,
                "generated_sequences":generated_sequences,
                "latent_z":latent_space}


    def loss(self):
        """
        """
        return Trace_ELBO(strict_enumeration_warning=False)

class VegvisirModel5a_supervised(VEGVISIRModelClass,PyroModule):
    """
    Variational Autoencoder with all dimensions dependent
    -Notes:
            https://pyro.ai/examples/cvae.html
            https://avandekleut.github.io/vae/
    -Notes: on nan values
            http://pyro.ai/examples/svi_part_iv.html
            https://forum.pyro.ai/t/my-guide-keeps-producing-nan-values-what-am-i-doing-wrong/2024/8
    -CSVAE:
            https://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        self.gru_hidden_dim = self.hidden_dim*2
        self.num_params = 2 #number of parameters of the beta distribution
        #self.decoder = RNN_model6(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.decoder = RNN_model7(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device).to(self.device) #Highlight: Reconstr accurac too high
        self.classifier_model = FCL4(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device).to(self.device)
        #self.classifier_model = CNN_layers(1,self.z_dim,self.hidden_dim,self.num_classes,self.device) #input_dim,max_len,hidden_dim,num_classes,device,loss_type
        #self.classifier_model = RNN_classifier(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.z_dim,self.device) #input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device
        #self.h_0_MODEL_encoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.bidirectional = [2 if self.decoder.bidirectional else 1][0]
        self.h_0_MODEL_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device) #this is used only for generative purposes, not training
        #self.h_0_MODEL_classifier = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)
        #self.init_hidden = Init_Hidden(self.z_dim, self.max_len, self.gru_hidden_dim, self.device)
        self.build([self.h_0_MODEL_decoder],suffix="_model")
        self.num_iafs = 0

    def model_glitched(self, batch_data, batch_mask, epoch, guide_estimates, sample=False):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
            - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
            - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
            - https://fehiepsi.github.io/rethinking-pyro/
        """

        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
        batch_sequences_int = batch_data["int"][:, 1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:, 1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]

        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
        batch_positional_mask = batch_data["positional_mask"]

        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] > 0.7).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()

        z_mean, z_scale = torch.zeros((batch_size, self.z_dim)), torch.ones((batch_size, self.z_dim))
        with pyro.plate("plate_batch", dim=-1, device=self.device):

            if guide_estimates is not None and "generate" in guide_estimates.keys():
                latent_space = self.conditional_sampling(batch_size,guide_estimates["latent_z"].shape[0],guide_estimates)
                pyro.deterministic("latent_z", latent_space,event_dim=0)  # should be event_dim = 2, but for sampling convenience we leave it as it is

            else:
                if self.num_iafs > 0:
                    latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)) # [n,z_dim]
                else:
                    latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1)) # [n,z_dim]


            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.seq_max_len,self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,self.gru_hidden_dim).contiguous() #[2,batch_size,gru_dim]
            #init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)

            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=0) #should be event_dim = 2, but for sampling convenience we leave it as it is
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=0) #should be event_dim = 3
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=0) #should be event_dim = 3
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=0) #should be event_dim = 2
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=0) #should be event_dim = 2
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=0) #should be event_dim = 2

            # with pyro.plate("plate_len", dim=-2, device=self.device):
            #     pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).mask(~batch_positional_mask),obs=None if sample else batch_sequences_int)
            # with pyro.poutine.scale(None, self.likelihood_scale):
            #pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).mask(~batch_positional_mask).to_event(1),obs=None if sample else batch_sequences_int)
            pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=None if sample else batch_sequences_int)

            # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=0) #should be event_dim = 1
            #with pyro.poutine.mask(mask=confidence_mask_true):
                #pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)  # [N,]
            with pyro.poutine.scale(None,self.likelihood_scale):
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels) #TODO: removed .to_event(1)

        return {"attn_weights": outputnn.attn_weights,
                "encoder_hidden_states":outputnn.encoder_hidden_states,
                "decoder_hidden_states":outputnn.decoder_hidden_states,
                "encoder_final_hidden":outputnn.encoder_final_hidden,
                "decoder_final_hidden":outputnn.decoder_final_hidden,
                "sequences_logits":sequences_logits,
                "class_logits":class_logits}

    def model(self, batch_data, batch_mask, epoch, guide_estimates, sample):
        if self.args.glitch:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)
        else:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)

    def sample(self,batch_data, batch_mask, epoch, guide_estimates, sample,argmax):
        """"""
        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
        batch_sequences_int = batch_data["int"][:, 1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:, 1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]

        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] > 0.7).any(
            -1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        # init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        # z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
        z_mean, z_scale = torch.zeros((batch_size, self.z_dim)), torch.ones((batch_size, self.z_dim))
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            if self.num_iafs > 0:
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)).to(device=self.device)  # [n,z_dim]
            else:
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)).to(device=self.device)  # [n,z_dim]

            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len,
                                                                            self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,
                                                             self.gru_hidden_dim).contiguous()
            # init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)

            pyro.deterministic("attn_weights", outputnn.attn_weights,event_dim=0)  # should be event_dim = 2, but for sampling convenience we leave it here
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states,event_dim=0)  # should be event_dim = 3
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states,event_dim=0)  # should be event_dim = 3
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden,event_dim=0)  # should be event_dim = 2
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden,event_dim=0)  # should be event_dim = 2
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)  # should be event_dim = 2

            #generated_sequences = pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=None if sample else batch_sequences_int)
            with pyro.poutine.scale(None,int(self.likelihood_scale)):
                with pyro.plate("plate_len", dim=-2, device=self.device):
                    if argmax:
                        generated_sequences = torch.argmax(sequences_logits,dim=-1)
                    else:
                        num_seq_samples = 1
                        #generated_sequences = dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1).sample([num_seq_samples])
                        generated_sequences = dist.Categorical(logits=sequences_logits).mask(batch_mask_len).sample([num_seq_samples])
            # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=0)  # should be event_dim = 1
            with pyro.poutine.scale(None, self.likelihood_scale): #with pyro.poutine.mask(mask=confidence_mask_true):
                #binary_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)
                binary_predictions = dist.Categorical(logits=class_logits).to_event(1).sample([self.generate_num_samples])

        return {"attn_weights": outputnn.attn_weights,
                "encoder_hidden_states": outputnn.encoder_hidden_states,
                "decoder_hidden_states": outputnn.decoder_hidden_states,
                "encoder_final_hidden": outputnn.encoder_final_hidden,
                "decoder_final_hidden": outputnn.decoder_final_hidden,
                "sequences_logits": sequences_logits,
                "class_logits": class_logits,
                "binary_predictions":binary_predictions,
                "predictions":binary_predictions,
                "generated_sequences":generated_sequences,
                "sequences":generated_sequences,
                "latent_z":latent_space}


    def loss(self):
        """
        """
        return Trace_ELBO(strict_enumeration_warning=False)

class VegvisirModel5a_unsupervised(VEGVISIRModelClass,PyroModule):
    """
    Variational Autoencoder with all dimensions dependent
    -Notes:
            https://pyro.ai/examples/cvae.html
            https://avandekleut.github.io/vae/
    -Notes: on nan values
            http://pyro.ai/examples/svi_part_iv.html
            https://forum.pyro.ai/t/my-guide-keeps-producing-nan-values-what-am-i-doing-wrong/2024/8
    -CSVAE:
            https://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        self.gru_hidden_dim = self.hidden_dim*2
        self.num_params = 2 #number of parameters of the beta distribution
        #self.decoder = RNN_model6(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.decoder = RNN_model7(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.bidirectional = [2 if self.decoder.bidirectional else 1][0]
        self.h_0_MODEL_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)
        #self.init_hidden = Init_Hidden(self.z_dim,self.max_len,self.gru_hidden_dim,self.device)
        self.build([self.h_0_MODEL_decoder],suffix="_model")

    # def model_non_glitched(self,batch_data,batch_mask,epoch,guide_estimates,sample=False):
    #     """
    #     :param batch_data:
    #     :param batch_mask:
    #     :return:
    #     - Notes:
    #         - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
    #         - https://maxhalford.github.io/blog/bayesian-linear-regression/
    #         - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
    #         - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
    #         - https://fehiepsi.github.io/rethinking-pyro/
    #     """
    #
    #     pyro.module("vae_model", self)
    #     batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
    #     batch_sequences_int = batch_data["int"][:,1].squeeze(1) #the squeeze is not necessary
    #     batch_sequences_norm = batch_data["norm"][:,1]
    #     batch_size = batch_sequences_blosum.shape[0]
    #     batch_mask_len = batch_mask[:, 1:].squeeze(1)
    #     batch_mask_len = batch_mask_len[:, :, 0]
    #     batch_sequences_lens = batch_mask_len.sum(dim=1)
    #     batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
    #     true_labels = batch_data["blosum"][:,0,0,0]
    #     #immunodominance_scores = batch_data["blosum"][:,0,0,4]
    #     confidence_scores = batch_data["blosum"][:,0,0,5]
    #     confidence_mask = (confidence_scores[..., None] > 0.7).any(-1) #now we try to predict those with a low confidence score
    #     confidence_mask_true = torch.ones_like(confidence_mask).bool()
    #     #init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
    #     #z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
    #
    #     z_mean,z_scale = torch.zeros((batch_size,self.z_dim)), torch.ones((batch_size,self.z_dim))
    #     with pyro.plate("plate_batch",dim=-1,device=self.device):
    #         latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]
    #         latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.seq_max_len, self.z_dim)
    #         #init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
    #         class_logits = torch.rand((batch_size,self.num_classes))
    #         class_logits = self.logsoftmax(class_logits)
    #         pyro.deterministic("class_logits",class_logits,event_dim=0)
    #         pyro.deterministic("predictions",true_labels,event_dim=0)
    #         init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,self.gru_hidden_dim).contiguous()
    #         assert torch.isnan(init_h_0_decoder).int().sum().item() == 0, "found nan in init_h_0_decoder"
    #         outputnn = self.decoder(batch_sequences_blosum,batch_sequences_lens,init_h_0_decoder,z=latent_z_seq,mask=batch_mask_len,guide_estimates=guide_estimates)
    #
    #         pyro.deterministic("attn_weights",outputnn.attn_weights,event_dim=0)
    #         pyro.deterministic("encoder_hidden_states",outputnn.encoder_hidden_states,event_dim=0)
    #         pyro.deterministic("decoder_hidden_states",outputnn.decoder_hidden_states,event_dim=0)
    #         pyro.deterministic("encoder_final_hidden",outputnn.encoder_final_hidden,event_dim=0)
    #         pyro.deterministic("decoder_final_hidden",outputnn.decoder_final_hidden,event_dim=0)
    #
    #         # sequences_logits = self.logsoftmax(sequences_logits)
    #         # #with pyro.plate("plate_len", dim=-2, device=self.device):
    #         # #with pyro.poutine.mask(mask=batch_mask_len):
    #         # #Highlight: Scaling up the log likelihood of the reconstruction loss of the non padded positions
    #         sequences_logits = self.logsoftmax(outputnn.output)
    #         pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)
    #         assert not torch.isnan(sequences_logits).any(), "found nan in sequences_logits"
    #
    #         pyro.sample("sequences",dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=[None if sample else batch_sequences_int][0])
    #
    #     return {"attn_weights":outputnn.attn_weights}

    def model_glitched(self, batch_data, batch_mask, epoch, guide_estimates, sample=False):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
            - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
            - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
            - https://fehiepsi.github.io/rethinking-pyro/
        """

        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
        batch_sequences_int = batch_data["int"][:, 1].squeeze(1)  # the squeeze is not necessary
        batch_sequences_norm = batch_data["norm"][:, 1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] > 0.7).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        # init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        z_mean, z_scale = torch.zeros((batch_size, self.z_dim)), torch.ones((batch_size, self.z_dim))
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]
            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.seq_max_len, self.z_dim)
            # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
            class_logits = torch.rand((batch_size, self.num_classes))
            class_logits = self.logsoftmax(class_logits)
            pyro.deterministic("class_logits", class_logits, event_dim=0)
            pyro.deterministic("predictions", true_labels, event_dim=0)
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,self.gru_hidden_dim).contiguous()
            #init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            assert torch.isnan(init_h_0_decoder).int().sum().item() == 0, "found nan in init_h_0_decoder"
            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)

            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=0)
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=0)
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=0)
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=0)
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=0)

            # sequences_logits = self.logsoftmax(sequences_logits)
            # #with pyro.plate("plate_len", dim=-2, device=self.device):
            # #with pyro.poutine.mask(mask=batch_mask_len):
            # #Highlight: Scaling up the log likelihood of the reconstruction loss of the non padded positions
            sequences_logits = self.logsoftmax(outputnn.output)

            pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)
            assert not torch.isnan(sequences_logits).any(), "found nan in sequences_logits"
            #with pyro.poutine.mask(mask=batch_mask_len):
            pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=[None if sample else batch_sequences_int][0])

        return {"attn_weights": outputnn.attn_weights}

    def model(self,batch_data,batch_mask,epoch,guide_estimates,sample):
        if self.args.glitch:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates,sample)
        else:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)

    def sample(self,batch_data,batch_mask,guide_estimates,argmax=False):
        """"""
        raise ValueError("Not implemented")


    def loss(self):
        """
        """
        return Trace_ELBO(strict_enumeration_warning=False)

class VegvisirModel5a_semisupervised(VEGVISIRModelClass,PyroModule):
    """
    Variational Autoencoder with all dimensions dependent
    -Notes:
            https://pyro.ai/examples/cvae.html
            https://avandekleut.github.io/vae/
    -Notes: on nan values
            http://pyro.ai/examples/svi_part_iv.html
            https://forum.pyro.ai/t/my-guide-keeps-producing-nan-values-what-am-i-doing-wrong/2024/8
    -CSVAE:
            https://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        self.gru_hidden_dim = self.hidden_dim * 2
        self.num_params = 2  # number of parameters of the beta distribution
        # self.decoder = RNN_model6(self.z_dim,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.decoder = RNN_model7(self.z_dim, self.seq_max_len, self.gru_hidden_dim, self.aa_types, self.z_dim,
                                  self.device)  # Highlight: Reconstr accurac too high
        self.classifier_model = FCL4(self.z_dim, self.max_len, self.hidden_dim, self.num_classes, self.device)
        # self.classifier_model = CNN_layers(1,self.z_dim,self.hidden_dim,self.num_classes,self.device) #input_dim,max_len,hidden_dim,num_classes,device,loss_type
        # self.classifier_model = RNN_classifier(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.z_dim,self.device) #input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device
        # self.h_0_MODEL_encoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.bidirectional = [2 if self.decoder.bidirectional else 1][0]
        self.h_0_MODEL_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.h_0_MODEL_classifier = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len, self.input_dim)
        #self.init_hidden = Init_Hidden(self.z_dim, self.max_len, self.gru_hidden_dim, self.device)
        self.build([self.h_0_MODEL_decoder],suffix="_model")


    def model_glitched(self,batch_data,batch_mask,epoch,guide_estimates,sample=False):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
            - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
            - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
            - https://fehiepsi.github.io/rethinking-pyro/
        """

        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
        batch_sequences_int = batch_data["int"][:, 1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:, 1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]

        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
        true_labels = batch_data["blosum"][:, 0, 0, 0]
        #immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        #confidence_mask = (confidence_scores[..., None] >= 0.35).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask = (true_labels[..., None] != 2).any(-1) #Highlight: unlabelled data has been assigned labelled 2, we give high confidence to the labelled data (for now)
        #confidence_mask = confidence_mask[:,None].tile(1,3)
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        # init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        z_mean, z_scale = torch.zeros((batch_size, self.z_dim)).to(device=self.device), torch.ones((batch_size, self.z_dim)).to(device=self.device)
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]

            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len,
                                                                            self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,
                                                             self.gru_hidden_dim).contiguous()
            # init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)
            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=0)
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=0)
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=0)
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=0)
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=0)
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)
            # with pyro.plate("plate_len", dim=-2, device=self.device):
            #with pyro.poutine.mask(mask=batch_mask_len_true):#highlight: removed .to_event(1)
            #with pyro.poutine.mask(mask=batch_mask_len):
            pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len_true).mask(batch_mask_len).to_event(1),obs=None if sample else batch_sequences_int)
            # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=1)
            with pyro.poutine.mask(mask=confidence_mask):
                #with pyro.poutine.mask(mask=confidence_mask_true):

                observed_labels = true_labels.clone()
                observed_labels[~confidence_mask] = 0. #TODO: random labels?
                #TODO: Try without to_event(1)
                pyro.sample("predictions", dist.Categorical(logits=class_logits).mask(confidence_mask).to_event(1),obs=None if sample else observed_labels*confidence_mask)#,obs_mask=confidence_mask)  # [N,]

        return {"attn_weights": outputnn.attn_weights,
                "encoder_hidden_states": outputnn.encoder_hidden_states,
                "decoder_hidden_states": outputnn.decoder_hidden_states,
                "encoder_final_hidden": outputnn.encoder_final_hidden,
                "decoder_final_hidden": outputnn.decoder_final_hidden,
                "sequences_logits": sequences_logits,
                "class_logits": class_logits}


    # def model_non_glitch(self,batch_data,batch_mask,epoch,guide_estimates,sample=False): #TODO: remove, some errors in the masking
    #     """
    #     :param batch_data:
    #     :param batch_mask:
    #     :return:
    #     - Notes:
    #         - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
    #         - https://maxhalford.github.io/blog/bayesian-linear-regression/
    #         - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
    #         - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
    #         - https://fehiepsi.github.io/rethinking-pyro/
    #     """
    #
    #     pyro.module("vae_model", self)
    #     batch_sequences_blosum = batch_data["blosum"][:, 1].squeeze(1)
    #
    #     batch_sequences_int = batch_data["int"][:, 1].squeeze(1)
    #     batch_sequences_norm = batch_data["norm"][:, 1]
    #     batch_size = batch_sequences_blosum.shape[0]
    #     batch_mask_len = batch_mask[:, 1:].squeeze(1)
    #     batch_mask_len = batch_mask_len[:, :, 0]
    #     batch_sequences_lens = batch_mask_len.sum(dim=1)
    #     batch_mask_len_true = torch.ones_like(batch_mask_len).bool()
    #     true_labels = batch_data["blosum"][:, 0, 0, 0]
    #     # immunodominance_scores = batch_data["blosum"][:,0,0,4]
    #     confidence_scores = batch_data["blosum"][:, 0, 0, 5]
    #     confidence_mask = (confidence_scores[..., None] > 0.4).any(-1)  # now we try to predict those with a low confidence score
    #     confidence_mask_true = torch.ones_like(confidence_mask).bool()
    #
    #     z_mean, z_scale = torch.zeros((batch_size, self.z_dim)), torch.ones((batch_size, self.z_dim))
    #     with pyro.plate("plate_batch", dim=-1, device=self.device):
    #         latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]
    #         latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len,self.z_dim)  # [N,L,z_dim]
    #         init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,
    #                                                          self.gru_hidden_dim).contiguous()
    #         # init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
    #         # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)
    #
    #         outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
    #                                 mask=batch_mask_len, guide_estimates=guide_estimates)
    #         pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=0)
    #         pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=0)
    #         pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=0)
    #         pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=0)
    #         pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=0)
    #         sequences_logits = self.logsoftmax(outputnn.output)
    #         pyro.deterministic("sequences_logits", sequences_logits, event_dim=0)
    #         # with pyro.plate("plate_len", dim=-2, device=self.device):
    #         # with pyro.poutine.mask(mask=batch_mask_len_true):#highlight: removed .to_event(1)
    #         #with pyro.poutine.mask(mask=batch_mask_len):
    #         pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=None if sample else batch_sequences_int)
    #         # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
    #         class_logits = self.classifier_model(latent_space, None)
    #         class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
    #         pyro.deterministic("class_logits", class_logits, event_dim=1)
    #         with pyro.poutine.mask(mask=confidence_mask_true):
    #             with pyro.poutine.mask(mask=confidence_mask):
    #                 pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)  # [N,]

        return {"attn_weights": outputnn.attn_weights}

    def model(self,batch_data,batch_mask,epoch,guide_estimates,sample):
        if self.args.glitch:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates,sample)
        else:
            return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)


    def sample(self,batch_data,batch_mask,guide_estimates,argmax=False):
        """"""
        raise ValueError("Not implemented. Using Predictive")
        return SamplingOutput(latent_space = None,
                              predicted_labels=None,
                              immunodominance_scores= None, #predicted_immunodominance_scores,
                              reconstructed_sequences = None)

    def loss(self):
        """
        """

        return Trace_ELBO(strict_enumeration_warning=False)

class VegvisirModel5b(VEGVISIRModelClass,PyroModule):
    """
    Variational Autoencoder
    -Notes:
            https://pyro.ai/examples/cvae.html
            https://avandekleut.github.io/vae/
    -Notes: on nan values
            http://pyro.ai/examples/svi_part_iv.html
            https://forum.pyro.ai/t/my-guide-keeps-producing-nan-values-what-am-i-doing-wrong/2024/8
    -CSVAE:
            https://bjlkeng.github.io/posts/semi-supervised-learning-with-variational-autoencoders/
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        self.gru_hidden_dim = self.hidden_dim*2
        self.num_params = 2 #number of parameters of the beta distribution
        self.encoder = RNN_guide(self.aa_types,self.max_len,self.gru_hidden_dim,self.z_dim,self.device)
        self.decoder = RNN_model(self.aa_types,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.classifier_model = FCL4(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device)
        #self.classifier_model = CNN_layers(1,self.z_dim,self.hidden_dim,self.num_classes,self.device) #input_dim,max_len,hidden_dim,num_classes,device,loss_type
        #self.classifier_model = RNN_classifier(1,self.max_len,self.gru_hidden_dim,self.num_classes,self.z_dim,self.device) #input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device
        self.h_0_MODEL_encoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.h_0_MODEL_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)

    def model(self,batch_data,batch_mask,epoch,guide_estimates,sample=False):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
            - https://link.springer.com/chapter/10.1007/978-3-031-06053-3_36
            - https://bookdown.org/robertness/causalml/docs/tutorial-on-deep-probabilitic-modeling-with-pyro.html
            - https://fehiepsi.github.io/rethinking-pyro/
        """

        pyro.module("vae_model", self)
        batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
        batch_sequences_int = batch_data["int"][:,1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:,1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask_len = batch_mask[:,1:].squeeze(1)
        batch_mask = batch_mask_len[:,:,0]
        true_labels = batch_data["blosum"][:,0,0,0]
        #immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] > 0.7).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool() #TODO: Check
        init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
        #z_mean,z_scale = torch.zeros((batch_size,self.z_dim)), torch.ones((batch_size,self.z_dim))
        #with pyro.poutine.scale(scale=self.beta):
        with pyro.plate("plate_batch", dim=-1,device=self.device):
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]
            #latent_z_seq = latent_space.repeat(1,1, self.max_len).reshape(latent_space.shape[0],batch_size, self.max_len, self.z_dim)
            latent_z_seq = latent_space.repeat(1, self.max_len).reshape(batch_size, self.max_len, self.z_dim)
            #batch_sequences_norm = (batch_sequences_norm*batch_mask)[:,:,None].expand(batch_sequences_norm.shape[0],batch_sequences_norm.shape[1],self.z_dim)
            #latent_z_seq += batch_sequences_norm
            #init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.poutine.mask(mask=[confidence_mask if self.learning_type in ["semisupervised"] else confidence_mask_true][0]):
                    class_logits = self.classifier_model(latent_space,None) #TODO: Is it better with latent_z_seq or latent_space?
                    #class_logits = self.classifier_model(latent_space[:,:,None],init_h_0_classifier)
                    class_logits = self.logsoftmax(class_logits)
                    #Highlight: Declaring first dimensions as conditionally independent is essential (.to_event(1))
                    if self.learning_type == "semisupervised":
                        pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1).mask(confidence_mask),obs=[None if sample else true_labels][0])
                    elif self.learning_type == "supervised":
                        pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=[None if sample else true_labels][0])
                    else:
                        pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1))

            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            with pyro.plate("plate_len",dim=-2, device=self.device):  #Highlight: not to_event(1) and with our without plate over the len dimension
                with pyro.poutine.mask(mask=batch_mask):
                    sequences_logits = self.decoder(latent_z_seq,init_h_0_decoder)
                    sequences_logits = self.logsoftmax(sequences_logits)
                    pyro.sample("sequences",dist.Categorical(logits=sequences_logits).mask(batch_mask),obs=[None if sample else batch_sequences_int][0])



        return {"sequences_logits":None}
                # "beta":beta,
                # "alpha":alpha}

    def sample(self,batch_data,batch_mask,guide_estimates,argmax=False):
        """"""
        batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
        batch_sequences_norm = batch_data["norm"][:,1]
        batch_size = batch_sequences_blosum.shape[0]
        batch_mask = batch_mask[:,1:].squeeze(1)
        batch_mask = batch_mask[:,:,0]

        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask_true = torch.ones_like(confidence_scores).bool() #include all data points
        init_h_0_encoder = self.h_0_MODEL_encoder.expand(self.encoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        z_mean,z_scale = self.encoder(batch_sequences_blosum,init_h_0_encoder)
        #z_mean,z_scale = torch.zeros((batch_size,self.z_dim)), torch.ones((batch_size,self.z_dim))

        #Highlight: Forward network
        with pyro.poutine.scale(scale=self.beta):
            with pyro.plate("plate_batch",dim=-1):
                latent_space = dist.Normal(z_mean, z_scale).sample()  # [n,z_dim]
                latent_z_seq = latent_space.repeat(1, self.max_len).reshape(latent_space.shape[0], self.max_len, self.z_dim)
                with pyro.poutine.mask(mask=confidence_mask_true):
                    class_logits = self.classifier_model(latent_space, None)
                    class_logits = self.logsoftmax(class_logits)
                    if argmax:
                        predicted_labels = torch.argmax(class_logits, dim=1) #TODO: Check sampling here
                    else:
                        predicted_labels = dist.Categorical(logits=class_logits).sample()
                init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
                #with pyro.plate("plate_len",dim=-2, device=self.device):  #Highlight: not to_event(1) and with our without plate over the len dimension
                with pyro.poutine.mask(mask=batch_mask):
                        # Highlight: Forward network
                        sequences_logits = self.decoder(latent_z_seq, init_h_0_decoder)
                        sequences_logits = self.logsoftmax(sequences_logits)
                        reconstructed_sequences = dist.Categorical(logits= sequences_logits).sample()
        identifiers = batch_data["blosum"][:,0,0,1]
        true_labels = batch_data["blosum"][:,0,0,0]
        confidence_score = batch_data["blosum"][:,0,0,5]
        immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
        latent_space = torch.column_stack([identifiers, true_labels, confidence_score, immunodominace_score, latent_space])

        return SamplingOutput(latent_space = latent_space,
                              predicted_labels=predicted_labels,
                              immunodominance_scores= None, #predicted_immunodominance_scores,
                              reconstructed_sequences = reconstructed_sequences)

    def loss(self):
        """
        """
        if self.learning_type in ["semisupervised", "unsupervised"]:
            return TraceEnum_ELBO(strict_enumeration_warning=True,max_plate_nesting=2)
        else:
            return Trace_ELBO(strict_enumeration_warning=True)

class VegvisirModel5c(VEGVISIRModelClass,PyroModule):
    """
    Variational Autoencoder with sequences and features
    -Notes:
         a) on nan values
            http://pyro.ai/examples/svi_part_iv.html
            https://forum.pyro.ai/t/my-guide-keeps-producing-nan-values-what-am-i-doing-wrong/2024/8
        b)
            https://www.jeremyjordan.me/variational-autoencoders/#:~:text=The%20main%20benefit%20of%20a,us%20to%20reproduce%20the%20input.
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        self.gru_hidden_dim = self.hidden_dim*2
        self.num_params = 2 #number of parameters of the beta distribution
        self.decoder = RNN_model(self.aa_types,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device,self.loss_type)
        self.feats_dim = self.max_len - self.seq_max_len
        self.classifier_model = FCL4(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device)
        self.h_0_MODEL = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)

    def model(self,batch_data,batch_mask,epoch,guide_estimates,sample=False):
        """
        :param batch_data:
        :param batch_mask:
        :return:
        - Notes:
            - https://medium.com/@amitnitdvaranasi/bayesian-classification-basics-svi-7cdceaf31230
            - https://maxhalford.github.io/blog/bayesian-linear-regression/
        """

        pyro.module("vae_model", self)


        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:, 1, self.seq_max_len:, 0]
        batch_sequences_int = batch_data["int"][:,1,:self.seq_max_len].squeeze(1)

        batch_sequences_norm = batch_data["norm"][:,1,:self.seq_max_len] #only sequences norm
        batch_sequences_feats = batch_data["norm"][:,1,self.seq_max_len:] #only features
        batch_sequences_norm_feats = batch_data["norm"][:,1] #both

        batch_mask = batch_mask[:, 1:].squeeze(1)
        batch_mask = batch_mask[:, :, 0]

        true_labels = batch_data["blosum"][:,0,0,0]
        #immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] >= 0.7).any(-1)

        mean = (batch_sequences_norm * batch_mask).mean(dim=1)
        mean = mean[:, None].expand(batch_sequences_norm.shape[0], self.z_dim)

        scale = (batch_sequences_norm * batch_mask).std(dim=1)
        scale = scale[:, None].expand(batch_sequences_norm.shape[0], self.z_dim)
        #print(batch_sequences_blosum.shape[0])
        with pyro.poutine.scale(scale=self.beta):
            with pyro.plate("plate_latent", batch_sequences_blosum.shape[0],device=self.device): #dim = -2
                latent_z = pyro.sample("latent_z", dist.Normal(mean, scale).to_event(1))  # [n,z_dim]
                # logits_class = self.fcl1(latent_z,None)
                # class_logits = self.logsoftmax(logits_class)
                # # smooth_factor = self.losses.label_smoothing(class_logits,true_labels,confidence_scores,self.num_classes)
                # # class_logits = class_logits*smooth_factor
                # if self.semi_supervised:
                #     pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs_mask=confidence_mask, obs=true_labels)
                # else:
                #     pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs=true_labels)

                #beta_params = self.fcl2(latent_z, None)
                #beta = pyro.sample("beta",dist.Uniform(0.4,0.6))
                #alpha = pyro.sample("alpha",dist.Uniform(0.5,0.7))
                # beta = beta_params[:, 0]
                # alpha = beta_params[:, 1]
                # #pyro.sample("immunodominance_prediction", dist.Beta(beta, alpha), obs_mask=confidence_mask,obs=immunodominance_scores)  # obs_mask  If provided, events with mask=True will be conditioned on obs and remaining events will be imputed by sampling.
                #pyro.sample("immunodominance_prediction", dist.Beta(beta, alpha),obs=immunodominance_scores)  # obs_mask  If provided, events with mask=True will be conditioned on obs and remaining events will be imputed by sampling.

        latent_z_seq = latent_z.repeat(1, self.seq_max_len).reshape(latent_z.shape[0], self.seq_max_len, self.z_dim)
        batch_sequences_norm = batch_sequences_norm[:,:,None].expand(batch_sequences_norm.shape[0],batch_sequences_norm.shape[1],self.z_dim)
        batch_sequences_feats = batch_sequences_feats[:,:,None].expand(batch_sequences_feats.shape[0],batch_sequences_feats.shape[1],self.z_dim)
        latent_z_seq += batch_sequences_norm
        with pyro.plate("plate_class",batch_sequences_blosum.shape[0],dim=-2,device=self.device):
            class_logits = self.classifier_model(torch.concatenate([latent_z_seq,batch_sequences_feats],dim=1),None)
            class_logits = self.logsoftmax(class_logits)
            #smooth_factor = self.losses.label_smoothing(class_logits,true_labels,confidence_scores,self.num_classes)
            #class_logits = class_logits*smooth_factor
            if self.semi_supervised:
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs_mask=confidence_mask,obs=true_labels)
            else:
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs=true_labels)
        init_h_0_decoder = self.h_0_MODEL.expand(self.model_rnn.num_layers * 2, batch_sequences_blosum.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.poutine.mask(mask=batch_mask):
            with pyro.plate("plate_len",self.seq_max_len,device=self.device): #dim=-1
                with pyro.plate("plate_seq", batch_sequences_blosum.shape[0],device=self.device): #dim=-2
                    #Highlight: Forward network
                    logits_seqs = self.decoder(latent_z_seq,init_h_0_decoder)
                    logits_seqs = self.logsoftmax(logits_seqs)
                    pyro.sample("sequences",dist.Categorical(logits=logits_seqs).to_event(2).mask(batch_mask),obs=batch_sequences_int)

        return {"sequences_logits":logits_seqs}
                # "beta":beta,
                # "alpha":alpha}

    def sample(self,batch_data,batch_mask,guide_estimates,argmax=False):
        """"""
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:, 1, self.seq_max_len:, 0]

        batch_sequences_norm = batch_data["norm"][:, 1, :self.seq_max_len]  # only sequences norm
        batch_sequences_feats = batch_data["norm"][:, 1, self.seq_max_len:]  # only features
        batch_sequences_norm_feats = batch_data["norm"][:, 1]  # both

        batch_mask = batch_mask[:, 1:].squeeze(1)
        batch_mask = batch_mask[:, :, 0]

        mean = (batch_sequences_norm * batch_mask).mean(dim=1)
        mean = mean[:, None].expand(batch_sequences_norm.shape[0], self.z_dim)

        scale = (batch_sequences_norm * batch_mask).std(dim=1)
        scale = scale[:, None].expand(batch_sequences_norm.shape[0], self.z_dim)

        #Highlight: Forward network
        with pyro.plate("plate_latent", batch_sequences_blosum.shape[0], device=self.device):
            latent_z = pyro.sample("latent_z", dist.Normal(mean, scale).to_event(1))  # [n,z_dim]
            #logits_class = self.fcl1(latent_z, None)
            #logits_feats = self.fcl3(batch_features)
            # class_logits = self.logsoftmax(logits_class)
            # if argmax:
            #     predicted_labels = torch.argmax(class_logits, dim=1)
            # else:
            #     predicted_labels = dist.Categorical(logits=class_logits).sample()
            # beta_params = self.fcl2(latent_z, None)
            # # beta = pyro.sample("beta",dist.Uniform(0.4,0.6))
            # # alpha = pyro.sample("alpha",dist.Uniform(0.5,0.7))
            # beta = beta_params[:, 0]
            # alpha = beta_params[:, 1]
            # predicted_immunodominance_scores= dist.Beta(beta, alpha)  # obs_mask  If provided, events with mask=True will be conditioned on obs and remaining events will be imputed by sampling.

        latent_z_seq = latent_z.repeat(1, self.seq_max_len).reshape(latent_z.shape[0], self.seq_max_len, self.z_dim)
        batch_sequences_norm = batch_sequences_norm[:, :, None].expand(batch_sequences_norm.shape[0],
                                                                       batch_sequences_norm.shape[1], self.z_dim)
        batch_sequences_feats = batch_sequences_feats[:, :, None].expand(batch_sequences_feats.shape[0],
                                                                         batch_sequences_feats.shape[1], self.z_dim)
        latent_z_seq += batch_sequences_norm
        with pyro.plate("plate_class", batch_sequences_blosum.shape[0], dim=-2, device=self.device):
            class_logits = self.fcl4(torch.concatenate([latent_z_seq, batch_sequences_feats], dim=1), None)
            class_logits = self.logsoftmax(class_logits)
            if argmax:
                predicted_labels = torch.argmax(class_logits, dim=1)
            else:
                predicted_labels = dist.Categorical(logits=class_logits).sample()
        init_h_0_decoder = self.h_0_MODEL.expand(self.model_rnn.num_layers * 2, batch_sequences_blosum.shape[0],
                                         self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.poutine.mask(mask=batch_mask):
            with pyro.plate("plate_len", self.seq_max_len,device=self.device):
                with pyro.plate("plate_seq", batch_sequences_blosum.shape[0], device=self.device):
                    # Highlight: Forward network
                    logits_seqs = self.model_rnn(latent_z_seq, init_h_0_decoder)
                    logits_seqs = self.logsoftmax(logits_seqs)
                    reconstructed_sequences = dist.Categorical(logits=logits_seqs).sample()

        identifiers = batch_data["blosum"][:,0,0,1]
        true_labels = batch_data["blosum"][:,0,0,0]
        confidence_score = batch_data["blosum"][:,0,0,5]
        immunodominace_score = batch_data["blosum"][:, 0, 0, 4]
        latent_space = torch.column_stack([identifiers, true_labels, confidence_score, immunodominace_score, latent_z])

        return SamplingOutput(latent_space = latent_space,
                              predicted_labels=predicted_labels,
                              immunodominance_scores= None, #predicted_immunodominance_scores,
                              reconstructed_sequences = reconstructed_sequences)

    def loss(self):
        """
        Notes:
            - Custom losses: https://pyro.ai/examples/custom_objectives.html
        """
        #return TraceMeanField_ELBO()
        return Trace_ELBO()
        #return Trace_ELBO_classification(self.max_len,self.input_dim,self.num_classes)

class VegvisirModel6a(VEGVISIRModelClass):
    """
    NNalign gimick
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: RNN
        self.nna = NNAlign(self.aa_types,self.max_len,self.hidden_dim*2,self.num_classes,self.device)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)

    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences_blosum = batch_data["blosum"][:,1].squeeze(1)
        batch_mask = batch_mask[:,1:].squeeze(1)
        #batch_sequences = self.embedder(batch_sequences,None)
        #seq_lens = batch_data["int"][:,1].bool().sum(1)
        #batch_sequences_int = batch_data["int"][:,1]
        #Highlight: NNAlign
        class_out = self.nna(batch_sequences_blosum,batch_mask)

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)


    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        weights,array_weights = self.losses.calculate_weights(true_labels,self.class_weights)

        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]


            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels, predictions, confidence_scores, self.num_classes,weights)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=array_weights)
            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))

class VegvisirModel6b(VEGVISIRModelClass):
    """
    NNalign gimick to work with sequences and features
    """
    def __init__(self, ModelLoad):
        VEGVISIRModelClass.__init__(self, ModelLoad)
        #self.embedder = Embedder(self.aa_types,self.hidden_dim,self.device)
        #Highlight: RNN
        self.nna = NNAlign2(self.aa_types,self.seq_max_len,self.hidden_dim*2,self.num_classes,self.device)
        self.losses = VegvisirLosses(self.max_len,self.input_dim)
        self.feats_dim = self.max_len - self.seq_max_len
        self.fcl3 = FCL3(self.feats_dim,self.hidden_dim*2,self.num_classes,self.device)


    def forward(self,batch_data,batch_mask):
        """
        """
        batch_sequences_blosum = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        batch_mask = batch_mask[:,1:].squeeze(1)
        batch_features = batch_data["blosum"][:,1,self.seq_max_len:,0]
        #batch_sequences = self.embedder(batch_sequences,None)
        #seq_lens = batch_data["int"][:,1].bool().sum(1)
        #batch_sequences_int = batch_data["int"][:,1]
        #Highlight: NNAlign
        logits_seqs = self.nna(batch_sequences_blosum,batch_mask)
        logits_feats = self.fcl3(batch_features)
        class_out = logits_seqs + logits_feats

        return ModelOutput(reconstructed_sequences=None,
                           class_out=class_out)


    def loss(self,confidence_scores,true_labels,model_outputs,onehot_sequences=None):
        """
        :param confidence_scores:
        :param true_labels:
        :param model_outputs:
        :param onehot_sequences:
        :return: tensor loss
        """
        weights,array_weights = self.losses.calculate_weights(true_labels,self.class_weights)


        if self.loss_type == "weighted_bce":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out) #TODO: Softmax?
            predictions = nn.Sigmoid()(model_outputs.class_out)
            predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]


            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=confidence_scores) #pos weights affects only the positive (1) labels
            #output = bce_loss(predictions, true_labels)

            #output = self.losses.weighted_loss(true_labels,predictions,confidence_scores)
            #output = self.losses.weighted_loss(confidence_scores,predictions,None)
            loss = self.losses.focal_loss(true_labels,predictions,confidence_scores)
            return loss
        elif self.loss_type == "softloss":
            predictions = model_outputs.class_out
            # if self.num_classes == 2:
            #     #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
            #     predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            # else:
            #     predictions = predictions.squeeze(-1)
            loss = self.losses.taylor_crossentropy_loss(true_labels, predictions, confidence_scores, self.num_classes,weights)
            return loss
        elif self.loss_type == "bceprobs":
            #predictions = nn.Softmax(dim=-1)(model_outputs.class_out)
            predictions = nn.Sigmoid()(model_outputs.class_out)
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values
            bce_loss = nn.BCELoss()
            loss = bce_loss(predictions, true_labels)
            return loss
        elif self.loss_type == "bcelogits": #combines a Sigmoid layer and the BCELoss in one single class, numerically more stable
            predictions = model_outputs.class_out
            if self.num_classes == 2:
                #predictions = predictions[torch.arange(0, true_labels.shape[0]), true_labels.long()]
                predictions = torch.max(predictions, dim=1).values.squeeze(-1)
            else:
                predictions = predictions.squeeze(-1)
            bce_loss = nn.BCEWithLogitsLoss(pos_weight=array_weights)
            #bce_loss = nn.BCEWithLogitsLoss(pos_weight=class_weights)
            loss = bce_loss(predictions, true_labels)
            return loss

        else:
            raise ValueError("Error loss: {} not implemented for this model type: {}".format(self.loss_type,self.get_class()))


