"""
=======================
2023: Lys Sanz Moreta
Vegvisir
=======================
"""
from pyro.contrib.easyguide import EasyGuide
from pyro.nn import PyroParam
from vegvisir.models import *
from vegvisir.model_utils import *
from vegvisir.losses import *
import torch.nn as nn
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule
from pyro.distributions import constraints
class VEGVISIRGUIDES(EasyGuide):
    def __init__(self,vegvisir_model,model_load, Vegvisir):
        """The guide provides a valid joint probability density over all the latent random variables in the model or variational distribution."""
        super(VEGVISIRGUIDES, self).__init__(vegvisir_model)
        #self.guide_type = ModelLoad.args.select_guide
        #self.Vegvisir = Vegvisir
        self.beta = model_load.args.beta_scale #scaling the KL divergence error
        self.learning_type = model_load.args.learning_type
        self.aa_types = model_load.aa_types
        self.max_len = model_load.max_len
        self.seq_max_len = model_load.seq_max_len
        self.hidden_dim = model_load.args.hidden_dim
        self.gru_hidden_dim = self.hidden_dim*2
        self.z_dim = model_load.args.z_dim
        self.device = model_load.args.device
        self.num_classes = model_load.args.num_classes
        self.feats_dim = self.max_len - self.seq_max_len
        self.input_dim = model_load.input_dim
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)
        self.h_0_GUIDE = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.h_0_GUIDE_classifier = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.h_0_GUIDE_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.encoder_guide = RNN_guide(self.aa_types,self.max_len,self.gru_hidden_dim,self.z_dim,self.device)
        self.decoder_guide = RNN_model(self.aa_types,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        #self.decoder_guide = RNN_model(self.aa_types,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        #self.h_0_MODEL_decoder = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        self.num_iafs = 0
        self.iaf_dim = self.hidden_dim
        self.iafs = [dist.transforms.affine_autoregressive(self.z_dim, hidden_dims=[self.iaf_dim]) for _ in range(self.num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        if self.learning_type in ["semisupervised","unsupervised"]:
            self.classifier_guide = FCL4(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device)
        #self.classifier_guide = FCL1(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device)
        #self.classifier_guide = RNN_classifier(1,self.max_len,self.gru_hidden_dim,self.num_classes,self.z_dim,self.device) #input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device


    def guide_a1(self, batch_data, batch_mask,sample=False):
        """
        Amortized inference with only sequences usian a mean field approximation
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        pyro.module("vae_guide", self)
        batch_mask = batch_mask[:, 1]
        batch_mask = batch_mask[:, :, 0]
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_blosum.shape[0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool() #TODO: Check

        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_batch",dim= -1,device=self.device): #dim = -2
            #z_mean, z_scale = self.encoder_guide(batch_sequences_norm[:,:,None], init_h_0)
            z_mean, z_scale = self.encoder_guide(batch_sequences_blosum, init_h_0)
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            if len(self.iafs) > 0:
                z_dist= dist.TransformedDistribution(dist.Normal(z_mean, z_scale), self.iafs)
                print(z_dist.event_shape)
                print(z_dist.batch_shape)
                latent_space = pyro.sample("latent_z",dist.TransformedDistribution(dist.Normal(z_mean, z_scale), self.iafs).to_event(1))
                # assert z_dist.event_shape == (self.z_q_0.size(0),)
                # assert z_dist.batch_shape[-1:] == (len(mini_batch),)
            else:
                latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))  # [z_dim,n]

                # assert z_dist.event_shape == ()
                # assert z_dist.batch_shape[-2:] == (len(mini_batch),self.z_q_0.size(0),)
                assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
                assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            #latent_z_seq = latent_space.repeat(1, self.max_len).reshape(batch_size, self.max_len, self.z_dim)
            # Highlight: We only need to specify a variational distribution over the class/class if class/label is unobserved
            if self.learning_type in ["semisupervised","unsupervised"]:
                with pyro.poutine.mask(mask=[confidence_mask if self.learning_type in ["semisupervised"] else confidence_mask_true][0]):
                    #with pyro.plate("plate_class_seq", batch_sequences_blosum.shape[0], dim=-1, device=self.device):
                            class_logits = self.classifier_guide(latent_space, None)
                            class_logits = self.logsoftmax(class_logits)
                            # smooth_factor = self.losses.label_smoothing(class_logits,true_labels,confidence_scores,self.num_classes)
                            # class_logits = class_logits*smooth_factor
                            if self.learning_type == "semisupervised":
                                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1).mask(confidence_mask),infer={'enumerate': 'parallel'})
                            else: #unsupervised
                                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1),infer={'enumerate': 'parallel'})



        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "class_predictions":None}


    def guide_a2(self, batch_data, batch_mask,sample=False):
        """
        Amortized inference with only sequences
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        pyro.module("vae_guide", self)
        batch_mask = batch_mask[:, 1]
        batch_mask = batch_mask[:, :, 0]
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_blosum.shape[0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm
        # mean = (batch_sequences_norm*batch_mask).mean(dim=1)
        # mean = mean[:,None].expand(batch_sequences_norm.shape[0],self.z_dim)
        #
        # scale = (batch_sequences_norm*batch_mask).std(dim = 1)
        # scale = scale[:,None].expand(batch_sequences_norm.shape[0],self.z_dim)
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        #confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_batch",dim= -1,device=self.device): #dim = -2
            z_mean, z_scale = self.encoder_guide(batch_sequences_blosum, init_h_0)
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))  # ,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            # Highlight: We only need to specify a variational distribution over the class/class if class/label is unobserved
            if self.learning_type in ["semisupervised","unsupervised"]:
                #with pyro.poutine.mask(mask=[confidence_mask if self.learning_type in ["semisupervised"] else confidence_mask_true][0]):
                        class_logits = self.classifier_guide(latent_space, None)
                        class_logits = self.logsoftmax(class_logits)
                        #smooth_factor = self.losses.label_smoothing(class_logits,true_labels,confidence_scores,self.num_classes)
                        #class_logits = class_logits*smooth_factor
                        if self.learning_type == "semisupervised":
                            #pyro.sample("predictions_unobserved", dist.Categorical(logits=class_logits).to_event(1).mask(confidence_mask),infer={'enumerate': 'parallel'})
                            for t, y in enumerate(class_logits):
                                pyro.sample(f"predictions_{t}_unobserved", dist.Categorical(class_logits[t]).mask(confidence_mask[t]),infer={"enumerate": "parallel"}) #TODO: mask or not?
                        else: #unsupervised
                            #pyro.sample("predictions", dist.Categorical(logits=class_logits),infer={'enumerate': 'parallel'})
                            for t, y in enumerate(class_logits):
                                pyro.sample(f"predictions_{t}_unobserved", dist.Categorical(class_logits[t]), infer={"enumerate": "parallel"})

            # latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len, self.z_dim)
            # init_h_0_decoder = self.h_0_GUIDE_decoder.expand(self.decoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # with pyro.plate("plate_len",dim=-2, device=self.device):  #Highlight: not to_event(1) and with our without plate over the len dimension
            #     sequences_logits = self.decoder_guide(latent_z_seq, init_h_0_decoder)
            #     sequences_logits = self.logsoftmax(sequences_logits)
            #     pyro.sample("sequences_unobserved", dist.Categorical(logits=sequences_logits).mask(~batch_mask),infer={"enumerate": "parallel"})

        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "class_predictions":None}



    def guide_b(self, batch_data,batch_mask,sample=False):
        """
        Amortized inference with features and sequences
        Notes:
            -https://pyro.ai/examples/easyguide.html
            -https://medium.com/analytics-vidhya/activity-detection-using-the-variational-autoencoder-d2b017da1a88
            -https://sites.google.com/illinois.edu/supervised-vae?pli=1
            -TODO: https://github.com/analytique-bourassa/VAE-Classifier
        :param batch_data:
        :param batch_mask:
        :return:
        """
        pyro.module("vae_guide", self)
        batch_mask = batch_mask[:, 1]
        batch_mask = batch_mask[:, :, 0]
        batch_sequences_blosum = batch_data["blosum"][:,1,:self.seq_max_len].squeeze(1)
        batch_features = batch_data["blosum"][:, 1, self.seq_max_len:, 0]

        batch_sequences_norm = batch_data["norm"][:, 1, :self.seq_max_len]  # only sequences norm
        batch_sequences_feats = batch_data["norm"][:, 1, self.seq_max_len:]  # only features
        batch_sequences_norm_feats = batch_data["norm"][:, 1]  # both

        true_labels = batch_data["blosum"][:, 0, 0, 0]
        # immunodominance_scores = batch_data["blosum"][:,0,0,4]
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        mean = (batch_sequences_norm * batch_mask).mean(dim=1)
        mean = mean[:, None].expand(batch_sequences_norm.shape[0], self.z_dim)

        scale = (batch_sequences_norm * batch_mask).std(dim=1)
        scale = scale[:, None].expand(batch_sequences_norm.shape[0], self.z_dim)

        init_h_0 = self.h_0_GUIDE.expand(self.guide_rnn.num_layers * 2, batch_sequences_norm_feats.shape[0],self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.poutine.scale(scale=self.beta):
            with pyro.plate("plate_latent", batch_sequences_norm_feats.shape[0],device=self.device):
                z_mean, z_scale = self.encoder(batch_sequences_norm_feats[:,:,None],init_h_0)
                assert z_mean.shape == (batch_sequences_norm_feats.shape[0],self.z_dim), "Wrong shape got {}".format(z_mean.shape)
                assert z_scale.shape == (batch_sequences_norm_feats.shape[0],self.z_dim), "Wrong shape got {}".format(z_scale.shape)
                z_mean += mean
                z_scale += scale
                latent_z = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))#,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]

        latent_z_seq = latent_z.repeat(1, self.seq_max_len).reshape(latent_z.shape[0], self.seq_max_len, self.z_dim)
        batch_sequences_norm = batch_sequences_norm[:,:,None].expand(batch_sequences_norm.shape[0],batch_sequences_norm.shape[1],self.z_dim)
        batch_sequences_feats = batch_sequences_feats[:,:,None].expand(batch_sequences_feats.shape[0],batch_sequences_feats.shape[1],self.z_dim)
        latent_z_seq += batch_sequences_norm
        with pyro.plate("plate_class",batch_sequences_blosum.shape[0],dim=-2,device=self.device):
            class_logits = self.classifier_guide(torch.concatenate([latent_z_seq,batch_sequences_feats],dim=1),None)
            class_logits = self.logsoftmax(class_logits)
            #smooth_factor = self.losses.label_smoothing(class_logits,true_labels,confidence_scores,self.num_classes)
            #class_logits = class_logits*smooth_factor
            if self.semi_supervised:
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs_mask=confidence_mask,obs=true_labels)
            else:
                pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1), obs=true_labels)

        return {"latent_z":latent_z,
                "z_mean":z_mean,
                "z_scale":z_scale,
                "class_predictions": None}



    def guide(self,batch_data,batch_mask,sample):
        if self.seq_max_len == self.max_len:
            return self.guide_a2(batch_data,batch_mask,sample)
        else:
            return self.guide_b(batch_data,batch_mask,sample)



