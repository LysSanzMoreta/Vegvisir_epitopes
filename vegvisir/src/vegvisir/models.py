#!/usr/bin/env python3
"""
=======================
2024: Lys Sanz Moreta
Vegvisir (VAE): T-cell epitope classifier
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
#from vegvisir.losses import *
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
        self.z_dim = model_load.args.z_dim
        self.device = model_load.args.device
        self.use_cuda = model_load.args.use_cuda
        self.tensor_type = torch.cuda.DoubleTensor if self.use_cuda else torch.DoubleTensor
        #self.dropout = model_load.args.dropout
        self.num_classes = model_load.args.num_obs_classes
        self.blosum = model_load.blosum
        self.blosum_weighted = model_load.blosum_weighted
        self.learning_type = model_load.args.learning_type
        self.class_weights = model_load.class_weights
        self.generate_sampling_type = model_load.args.generate_sampling_type
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
        REFERENCES:
            "Chebyshev-type methods and preconditioning techniques", Hou-Biao Li, Ting-Zhu Huang, Yong Zhang, Xing-Ping Liu, Tong-Xiang Gu
        :param a : Matrix
        :param iterations: Number of chebyshev polynomial series to compute"""

        a_norm_1d = torch.linalg.matrix_norm(a, ord=1)
        a_norm_inf = torch.linalg.matrix_norm(a, ord=float('inf'))
        alpha = 1 / (a_norm_1d * a_norm_inf)
        N_0 = alpha[:, None, None] * a.permute(0, 2, 1)
        N_t = N_0
        In = torch.eye(a.shape[1]).expand(a.shape[0], a.shape[1], a.shape[2]).detach()
        for t in range(iterations):
            N_t_plus_1 = torch.matmul(N_t,(3 * In - torch.matmul(torch.matmul(a, N_t), (3 * In - torch.matmul(a, N_t))))).detach()
            N_t = N_t_plus_1
            torch.cuda.empty_cache()
        return N_t

    def conditional_sampling_fast(self,n_generated, n_train, guide_estimates):
        """Conditional sampling generated sequences given the training dataset from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop).
        Uses a bradcastable identity matrix
        :param guide_estimates:
        """

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
        xb = torch.from_numpy(guide_estimates["latent_z"][:,6:]).T.to(OU_mean_generated.device)  # [z_dim,n_train]

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

    def conditional_sampling(self,n_generated, guide_estimates):
        """Conditional sampling the synthetic epitopes given the learnt representations from the training dataset from a Multivariate Normal according to page 698 at Pattern Recognition and ML (Bishop)
        :param guide_estimates: dictionary conatining the MAP estimates for the OU process parameters
        :param """
        #print("Sampling .........")
        n_train = guide_estimates["latent_z"].shape[0]
        if n_train > 10000:#select only a few data points
            idx_train = np.array(np.random.randn(n_train) > 0)
        else:
            idx_train = np.ones(n_train).astype(bool)
        n_train = guide_estimates["latent_z"][idx_train].shape[0]
        # Highlight:  See Page 689 at Patter Recongnition and Ml (Bishop)
        # Highlight: Formula is: p(xa|xb) = N (x|µa|b, Λ−1aa ) , a = test/internal; b= train/leaves

        # Highlight: B.49 Λ−1aa
        inverse_generated = torch.eye(n_generated)  # [n_test,n_test]
        inverse_generated = inverse_generated[None,:].expand(self.z_dim,n_generated,n_generated).detach() # [z_dim,n_test,n_test]
        assert not torch.isnan(torch.from_numpy(guide_estimates["z_scales"])).any(), "z scales contains nan"
        assert not torch.isnan(torch.from_numpy(guide_estimates["z_scales"][idx_train])).any(), "z scales idx contains nan"

        z_scales = torch.from_numpy(guide_estimates["z_scales"][idx_train]).mean(0).to(inverse_generated.device).detach() #[z_dim] #fill in the diagonal with an average of the inferred z_scales
        assert not torch.isnan(z_scales).any(), "z scales contains nan"

        inverse_generated = z_scales[:,None,None]*inverse_generated

        # Highlight: Conditional mean Mean ---->B-50:  µa|b = µa − Λ−1aa Λab(xb − µb)
        # Highlight: µa
        OU_mean_generated = torch.zeros((n_generated,))  # [n_generated,]
        # Highlight: Λab
        inverse_generated_train = torch.eye(n_generated, n_train)  # [n_generated,n_train]
        inverse_generated_train = inverse_generated_train[None,:].expand(self.z_dim,n_generated,n_train).detach() # [z_dim,n_generated,n_train]
        inverse_generated_train = z_scales[:,None,None]*inverse_generated_train

        # Highlight: xb
        xb = torch.from_numpy(guide_estimates["latent_z"][idx_train][:,6:]).T.to(OU_mean_generated.device)  # [z_dim,n_train]
        assert xb.shape == (self.z_dim,n_train), "Perhaps you forgot that the latent space has some columns stacked"
        # Highlight:µb
        OU_mean_train = torch.zeros((n_train,))
        # Highlight:µa|b---> Splitted Equation  B-50
        part1 = torch.matmul(self.chebyshev_inverse_3d(inverse_generated), inverse_generated_train)  # [z_dim,n_test,n_train]
        part2 = xb - OU_mean_train[None, :]  # [z_dim,n_train]
        z_mean = OU_mean_generated[None, :, None] - torch.matmul(part1, part2[:, :,None])  # [:,n_test,:] - [z_dim,n_test,None]

        if n_generated > 3000: #this is slower but works for larger amounts of sequences
            batch_size_init = 1000 #splits the process of conditional sampling
            split_size = int(n_generated / batch_size_init) if not batch_size_init > n_generated else 1
            z_mean = z_mean.squeeze(-1)
            inverse_generated = self.chebyshev_inverse_3d(inverse_generated)
            idx = torch.zeros(n_generated).bool().detach()
            idx0= 0
            batch_size = batch_size_init
            latent_space_list = []
            for i in range(split_size):
                idx_split = idx.clone()
                idx_split[idx0:batch_size] = True #select these sequences
                z_mean_split = z_mean[:,idx_split]
                inverse_generated_split = inverse_generated[:,idx_split]
                inverse_generated_split = inverse_generated_split[:,:,idx_split]

                idx0=batch_size  #new starting point
                batch_size += batch_size_init

                latent_space_split = dist.MultivariateNormal(z_mean_split.squeeze(-1),inverse_generated_split + 1e-6).to_event(1).sample().detach()
                latent_space_split = latent_space_split.T
                latent_space_list.append(latent_space_split)

            latent_space = torch.concatenate(latent_space_list,dim=0)
        else:
            assert not torch.isnan(inverse_generated).any(), "found nan in covariance cond sampling"
            assert not torch.isnan(torch.linalg.inv(inverse_generated)).any(), "found nan in inverse cond sampling"
            assert not torch.isnan(z_mean).any(),"found nan in zmean conditional sampling"
            try:
                latent_space = dist.MultivariateNormal(z_mean.squeeze(-1),torch.linalg.inv(inverse_generated) + 1e-6).to_event(1).sample().detach()
            except:
                print("Could not perform inversion or some other error. Falling back to simple Identity matrix for the covariance")
                inverse_generated = torch.eye(n_generated)
                latent_space = dist.MultivariateNormal(z_mean.squeeze(-1),torch.linalg.inv(inverse_generated) + 1e-6).to_event(1).sample().detach()

            latent_space = latent_space.T
            assert not torch.isnan(latent_space).any(), "found nan in latent space conditional sampling"
        torch.cuda.empty_cache()
        return latent_space.detach()

class VegvisirModel_supervised_no_decoder(VEGVISIRModelClass,PyroModule):
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
            pyro.deterministic("attn_weights", attn_weights, event_dim=2)
            pyro.deterministic("encoder_hidden_states", encoder_hidden_states, event_dim=3)
            pyro.deterministic("decoder_hidden_states", decoder_hidden_states, event_dim=3)
            pyro.deterministic("encoder_final_hidden", encoder_final_hidden, event_dim=2)
            pyro.deterministic("decoder_final_hidden", decoder_final_hidden, event_dim=2)
            sequences_logits = self.logsoftmax(torch.rand(batch_size,self.seq_max_len,self.aa_types))
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=2)
            pyro.deterministic("sequences", batch_sequences_int, event_dim=2) #not guaranteed


            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=1)
            with pyro.poutine.scale(None,self.likelihood_scale):
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=[None if sample else true_labels][0])  # [N,]

        return {"attn_weights": attn_weights}

    def model(self, batch_data, batch_mask, epoch, guide_estimates, sample):

        return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)

    def sample(self,batch_data,batch_mask,guide_estimates,argmax=False):
        """"""
        raise ValueError("Not implemented")


    def loss(self):
        """
        """
        return Trace_ELBO(strict_enumeration_warning=False)

class VegvisirModel_supervised(VEGVISIRModelClass,PyroModule):
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
        self.gru_hidden_dim = int(self.hidden_dim*2)
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
                if guide_estimates["sampling_type"] == "conditional":
                    latent_space = self.conditional_sampling(batch_size,guide_estimates)
                elif guide_estimates["sampling_type"] == "independent":
                    latent_space = dist.Normal(z_mean,z_scale).sample()
                pyro.deterministic("latent_z", latent_space,event_dim=2)  # should be event_dim = 2, but for sampling convenience we leave it as it is
            else:
                if self.num_iafs > 0:
                    latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)) # [n,z_dim]
                else:
                    latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1)) # [n,z_dim] #before

            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.seq_max_len,self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,self.gru_hidden_dim).contiguous() #[2,batch_size,gru_dim]
            #init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)

            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=2) #should be event_dim = 2, but for sampling convenience we leave it as it is
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=3) #should be event_dim = 3
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=3) #should be event_dim = 3
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=2) #should be event_dim = 2
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=2) #should be event_dim = 2
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=2) #should be event_dim = 2
            #
            # with pyro.plate("plate_len", dim=-2, device=self.device):
            #      pyro.sample("sequences", dist.Categorical(logits=sequences_logits),obs=None if sample else batch_sequences_int)
            # with pyro.poutine.scale(None, self.likelihood_scale):
            #pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).mask(~batch_positional_mask).to_event(1),obs=None if sample else batch_sequences_int)
            #pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=None if sample else batch_sequences_int)
            pyro.sample("sequences", dist.Categorical(logits=sequences_logits).to_event(1),obs=None if sample else batch_sequences_int)

            # init_h_0_classifier = self.h_0_MODEL_classifier.expand(self.classifier_model.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            class_logits = self.classifier_model(latent_space, None)
            class_logits = self.logsoftmax(class_logits)  # [N,num_classes]
            pyro.deterministic("class_logits", class_logits, event_dim=1) #should be event_dim = 1
            #with pyro.poutine.mask(mask=confidence_mask_true):
                #pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)  # [N,]
            with pyro.poutine.scale(None,self.likelihood_scale): #TODO: Implement https://pyro.ai/examples/custom_objectives.html
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels) #TODO: removed .to_event(1)

        return {"attn_weights": outputnn.attn_weights,
                "encoder_hidden_states":outputnn.encoder_hidden_states,
                "decoder_hidden_states":outputnn.decoder_hidden_states,
                "encoder_final_hidden":outputnn.encoder_final_hidden,
                "decoder_final_hidden":outputnn.decoder_final_hidden,
                "sequences_logits":sequences_logits,
                "class_logits":class_logits}

    def model(self, batch_data, batch_mask, epoch, guide_estimates, sample):
        # if self.args.glitch:
        #     return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)
        # else:
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
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale)).to(device=self.device)  # [n,z_dim]

            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len,
                                                                            self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,
                                                             self.gru_hidden_dim).contiguous()
            # init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)



            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=2) #should be event_dim = 2, but for sampling convenience we leave it as it is
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=3) #should be event_dim = 3
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=3) #should be event_dim = 3
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=2) #should be event_dim = 2
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=2) #should be event_dim = 2
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=2) #should be event_dim = 2

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
            pyro.deterministic("class_logits", class_logits, event_dim=1)  # should be event_dim = 1
            with pyro.poutine.scale(None, self.likelihood_scale): #with pyro.poutine.mask(mask=confidence_mask_true):
                #binary_predictions = pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),obs=None if sample else true_labels)
                binary_predictions = dist.Categorical(logits=class_logits).to_event(1).sample([self.num_samples])

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

class VegvisirModel_unsupervised(VEGVISIRModelClass,PyroModule):
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
        #self.init_hidden = Init_Hidden(self.z_dim,self.max_len,self.gru_hidden_dim,self.device)
        self.build([self.h_0_MODEL_decoder],suffix="_model")


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
            pyro.deterministic("class_logits", class_logits, event_dim=1)
            pyro.deterministic("predictions", true_labels, event_dim=1)
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,self.gru_hidden_dim).contiguous()
            #init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            assert torch.isnan(init_h_0_decoder).int().sum().item() == 0, "found nan in init_h_0_decoder"
            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)

            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=2)
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=3)
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=3)
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=2)
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=2)
            # #Highlight: Scaling up the log likelihood of the reconstruction loss of the non padded positions
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=2)
            assert not torch.isnan(sequences_logits).any(), "found nan in sequences_logits"
            #with pyro.poutine.mask(mask=batch_mask_len):
            pyro.sample("sequences", dist.Categorical(logits=sequences_logits).mask(batch_mask_len).to_event(1),obs=[None if sample else batch_sequences_int][0])

        return {"attn_weights": outputnn.attn_weights}

    def model(self,batch_data,batch_mask,epoch,guide_estimates,sample):

        return self.model_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)

    def sample(self,batch_data,batch_mask,guide_estimates,argmax=False):
        """"""
        raise ValueError("Not implemented")


    def loss(self):
        """
        """
        return Trace_ELBO(strict_enumeration_warning=False)

class VegvisirModel_semisupervised(VEGVISIRModelClass,PyroModule):
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

        #self.gru_hidden_dim = self.hidden_dim * 2 if not isinstance(self.hidden_dim,np.int64) else self.hidden_dim.item()*2
        self.gru_hidden_dim = self.hidden_dim*2


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
            if guide_estimates is not None and "generate" in guide_estimates.keys():
                if guide_estimates["sampling_type"] == "conditional":
                    latent_space = self.conditional_sampling(batch_size, guide_estimates)
                elif guide_estimates["sampling_type"] == "independent":
                    latent_space = dist.Normal(z_mean, z_scale).sample()
                pyro.deterministic("latent_z", latent_space,event_dim=0)  # should be event_dim = 2, but for sampling convenience we leave it as it is
            else:
                if self.num_iafs > 0:
                    latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale))  # [n,z_dim]
                else:
                    latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [n,z_dim]

            latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len,
                                                                            self.z_dim)  # [N,L,z_dim]
            init_h_0_decoder = self.h_0_MODEL_decoder.expand(self.decoder.num_layers * self.bidirectional, batch_size,
                                                             self.gru_hidden_dim).contiguous()
            # init_h_0_decoder = self.init_hidden(latent_space).expand(self.decoder.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # sequences_logits = self.decoder(batch_sequences_norm[:,:,None],batch_sequences_lens,init_h_0_decoder)

            outputnn = self.decoder(batch_sequences_blosum, batch_sequences_lens, init_h_0_decoder, z=latent_z_seq,
                                    mask=batch_mask_len, guide_estimates=guide_estimates)
            pyro.deterministic("attn_weights", outputnn.attn_weights, event_dim=2)
            pyro.deterministic("encoder_hidden_states", outputnn.encoder_hidden_states, event_dim=3)
            pyro.deterministic("decoder_hidden_states", outputnn.decoder_hidden_states, event_dim=3)
            pyro.deterministic("encoder_final_hidden", outputnn.encoder_final_hidden, event_dim=2)
            pyro.deterministic("decoder_final_hidden", outputnn.decoder_final_hidden, event_dim=2)
            sequences_logits = self.logsoftmax(outputnn.output)
            pyro.deterministic("sequences_logits", sequences_logits, event_dim=2)

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


        return {"attn_weights": outputnn.attn_weights}

    def model(self,batch_data,batch_mask,epoch,guide_estimates,sample):

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





