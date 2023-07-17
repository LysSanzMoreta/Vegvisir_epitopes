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
from collections import namedtuple
OutputNN = namedtuple("OutputNN",[])

class VEGVISIRGUIDES(EasyGuide):
    def __init__(self,vegvisir_model,model_load, Vegvisir):
        """The guide provides a valid joint probability density over all the latent random variables in the model or variational distribution."""
        super(VEGVISIRGUIDES, self).__init__(vegvisir_model)
        #self.guide_type = ModelLoad.args.select_guide
        #self.Vegvisir = Vegvisir
        self.beta = model_load.args.beta_scale #scaling the KL divergence error
        self.blosum = model_load.blosum
        self.learning_type = model_load.args.learning_type
        self.glitch = model_load.args.glitch
        self.aa_types = model_load.aa_types
        self.max_len = model_load.max_len
        self.seq_max_len = model_load.seq_max_len
        self.hidden_dim = model_load.args.hidden_dim
        self.embedding_dim = model_load.args.embedding_dim
        self.gru_hidden_dim = self.hidden_dim*2
        self.z_dim = model_load.args.z_dim
        self.device = model_load.args.device
        self.num_classes = model_load.args.num_classes
        self.encoding = model_load.args.encoding
        self.feats_dim = self.max_len - self.seq_max_len
        self.input_dim = model_load.input_dim
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.losses = VegvisirLosses(self.seq_max_len,self.input_dim)
        #self.embedding = Embed(self.blosum,self.embedding_dim,self.aa_types,self.device)
        self.h_0_GUIDE = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
        #self.decoder_guide = RNN_model(self.aa_types,self.seq_max_len,self.gru_hidden_dim,self.aa_types,self.z_dim ,self.device)
        self.num_iafs = 0
        self.iaf_dim = self.hidden_dim
        self.iafs = [dist.transforms.affine_autoregressive(self.z_dim, hidden_dims=[self.iaf_dim]) for _ in range(self.num_iafs)]
        self.iafs_modules = nn.ModuleList(self.iafs)
        if self.learning_type in ["supervised"]:
            self.encoder_guide = RNN_guide2(self.aa_types, self.max_len, self.gru_hidden_dim, self.z_dim, self.device)
            #self.classifier_guide = FCL4(self.z_dim,self.max_len,self.hidden_dim,self.num_classes,self.device)
            #self.h_0_GUIDE_classifier = nn.Parameter(torch.randn(self.gru_hidden_dim), requires_grad=True).to(self.device)
            #self.classifier_guide = RNN_classifier(self.aa_types,self.max_len,self.gru_hidden_dim,self.num_classes,self.z_dim,self.device) #input_dim,max_len,gru_hidden_dim,aa_types,z_dim,device
        else:
            self.encoder_guide = RNN_guide2(self.aa_types, self.max_len, self.gru_hidden_dim, self.z_dim, self.device)


    def guide_supervised_glitch(self, batch_data, batch_mask, epoch,guide_estimates, sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1]
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_blosum.shape[0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm
        batch_positional_mask = batch_data["positional_mask"]

        # Highlight: Rotational Glitch
        #rotate = torch.randn(1) > 0.5
        #rotate = torch.tensor([True])

        if  epoch%2 != 0:
            #print("Rotating ....epoch {}".format(epoch))
            #print("Translating ....epoch {}".format(epoch))
            batch_positional_mask = torch.ones_like(batch_positional_mask)
            batch_positional_mask[:,1] = False
            batch_positional_mask[:,3] = False
            batch_positional_mask[:,8] = False
            #print("-----------------------------------")
            #print(batch_sequences_blosum[0])
            #positional_mask = torch.tile(batch_positional_mask[:, :, None], (1, 1, batch_sequences_blosum.shape[-1]))
            #batch_sequences_blosum = self.rotate_blosum_batch(batch_sequences_blosum, batch_positional_mask)
            # print(batch_sequences_blosum[0])
            # print("----------------------------------")
            # exit()

        else:
            #print("not rotating....")
            pass
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,
                                         self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_batch", dim=-1, device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional = self.encoder_guide(
                batch_sequences_blosum, batch_sequences_lens, init_h_0)
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(
                z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(
                z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))
        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden": rnn_hidden,
                "rnn_final_hidden": rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states": rnn_hidden_states}

    def guide_supervised(self, batch_data, batch_mask,epoch,guide_estimates,sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1]
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_blosum.shape[0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm

        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_batch",dim= -1,device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional = self.encoder_guide(batch_sequences_blosum, batch_sequences_lens,init_h_0)
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))
        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden":rnn_hidden,
                "rnn_final_hidden":rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states":rnn_hidden_states}

    def guide_supervised_new(self, batch_data, batch_mask,epoch,guide_estimates,sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1]
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_sequences_encoded = batch_data[self.encoding][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_encoded.shape[0]

        confidence_scores = batch_data[self.encoding][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_batch",dim= -1,device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional = self.encoder_guide(batch_sequences_encoded, batch_sequences_lens,init_h_0)
            assert z_mean.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))
        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden":rnn_hidden,
                "rnn_final_hidden":rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states":rnn_hidden_states}

    def guide_unsupervised(self, batch_data, batch_mask,epoch,guide_estimates,sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_blosum.shape[0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_batch",dim= -1,device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional= self.encoder_guide(batch_sequences_blosum,batch_sequences_lens, init_h_0)
            assert torch.isnan(rnn_hidden_states).sum().item() == 0, "found nan in rnn hidden states"
            assert not torch.isnan(rnn_final_hidden_state).any(), "found nan in rnn final state"
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))  # ,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            assert not torch.isnan(latent_space).any(), "found nan in latent-space"
            # Highlight: We only need to specify a variational distribution over the class/class if class/label is unobserved
            #class_logits = self.classifier_guide(latent_space, None)
            # init_h_0_classifier = self.h_0_GUIDE_classifier.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
            # #latent_z_seq = latent_space.repeat(1, self.seq_max_len).reshape(batch_size, self.max_len, self.z_dim)
            # class_logits = self.classifier_guide(batch_sequences_blosum,init_h_0_classifier)
            # class_logits = self.logsoftmax(class_logits)
            # pyro.deterministic("class_logits",class_logits)
            # with pyro.poutine.mask(mask=confidence_mask_true):
            #     pyro.sample("predictions", dist.Categorical(logits=class_logits).to_event(1))

        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden":rnn_hidden,
                "rnn_final_hidden":rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states":rnn_hidden_states}

    def guide_unsupervised_new(self, batch_data, batch_mask, epoch, guide_estimates, sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        #batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_sequences_encoded = batch_data[self.encoding][:, 1, :self.seq_max_len].squeeze(1)

        batch_size = batch_sequences_encoded.shape[0]
        #batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(
            -1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,
                                         self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_batch", dim=-1, device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state, rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional = self.encoder_guide(
                batch_sequences_encoded, batch_sequences_lens, init_h_0)
            assert torch.isnan(rnn_hidden_states).sum().item() == 0, "found nan in rnn hidden states"
            assert not torch.isnan(rnn_final_hidden_state).any(), "found nan in rnn final state"
            assert z_mean.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(
                z_mean.shape)
            assert z_scale.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(
                z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(
                1))  # ,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            assert not torch.isnan(latent_space).any(), "found nan in latent-space"
            # Highlight: We only need to specify a variational distribution over the class/class if class/label is unobserved


        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden": rnn_hidden,
                "rnn_final_hidden": rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states": rnn_hidden_states}

    def guide_unsupervised_glitched(self, batch_data, batch_mask,epoch,guide_estimates,sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1:].squeeze(1)
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_blosum.shape[0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm
        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.7).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        batch_positional_mask = batch_data["positional_mask"]
        if epoch % 2 != 0:pass
            # print("Rotating ....epoch {}".format(epoch))
            # print("Translating ....epoch {}".format(epoch))
            # batch_positional_mask = torch.ones_like(batch_positional_mask)
            # batch_positional_mask[:, 1] = False
            # batch_positional_mask[:, 3] = False
            # batch_positional_mask[:, 8] = False
            # print("-----------------------------------")
            # print(batch_sequences_blosum[0])
            # positional_mask = torch.tile(batch_positional_mask[:, :, None], (1, 1, batch_sequences_blosum.shape[-1]))
            # batch_sequences_blosum = self.rotate_blosum_batch(batch_sequences_blosum, batch_positional_mask)
            # print(batch_sequences_blosum[0])
            # print("----------------------------------")
            # exit()

        else:
            # print("not rotating....")
            pass

        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional
        with pyro.plate("plate_batch",dim= -1,device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional= self.encoder_guide(batch_sequences_blosum,batch_sequences_lens, init_h_0)
            assert torch.isnan(rnn_hidden_states).sum().item() == 0, "found nan in rnn hidden states"
            assert not torch.isnan(rnn_final_hidden_state).any(), "found nan in rnn final state"
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))  # ,infer=dict(baseline={'nn_baseline': self.guide_rnn,'nn_baseline_input': batch_sequences_blosum}))  # [z_dim,n]
            assert not torch.isnan(latent_space).any(), "found nan in latent-space"
            # Highlight: We only need to specify a variational distribution over the class/class if class/label is unobserved


        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden":rnn_hidden,
                "rnn_final_hidden":rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional":rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states":rnn_hidden_states}

    def guide_semisupervised(self, batch_data, batch_mask,epoch,guide_estimates,sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1]
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_blosum.shape[0]
        batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm

        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.4).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_batch",dim= -1,device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional = self.encoder_guide(batch_sequences_blosum, batch_sequences_lens,init_h_0)
            assert z_mean.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_norm.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))
        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden":rnn_hidden,
                "rnn_final_hidden":rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states":rnn_hidden_states}

    def guide_semisupervised_new(self, batch_data, batch_mask,epoch,guide_estimates,sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1]
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        batch_sequences_encoded = batch_data[self.encoding][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_encoded.shape[0]
        #batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm

        confidence_scores = batch_data["blosum"][:,0,0,5]
        confidence_mask = (confidence_scores[..., None] < 0.4).any(-1) #now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_batch",dim= -1,device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional = self.encoder_guide(batch_sequences_encoded, batch_sequences_lens,init_h_0)
            assert z_mean.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(z_mean.shape)
            assert z_scale.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean,z_scale).to_event(1))
        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden":rnn_hidden,
                "rnn_final_hidden":rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states":rnn_hidden_states}

    def guide_semisupervised_glitched(self, batch_data, batch_mask, epoch, guide_estimates, sample=False):
        """
        Amortized inference with only sequences, all sites and sequences dependent
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
        batch_mask_len = batch_mask[:, 1]
        batch_mask_len = batch_mask_len[:, :, 0]
        batch_sequences_lens = batch_mask_len.sum(dim=1)
        #batch_sequences_blosum = batch_data["blosum"][:, 1, :self.seq_max_len].squeeze(1)
        batch_sequences_encoded = batch_data[self.encoding][:, 1, :self.seq_max_len].squeeze(1)
        batch_size = batch_sequences_encoded.shape[0]
        #batch_sequences_norm = batch_data["norm"][:, 1]  # only sequences norm
        batch_positional_mask = batch_data["positional_mask"]

        # Highlight: Rotational Glitch
        #rotate = torch.randn(1) > 0.5
        #rotate = torch.tensor([True])

        if  epoch%2 != 0: pass
            #print("Rotating ....epoch {}".format(epoch))
            #print("Translating ....epoch {}".format(epoch))
            # batch_positional_mask = torch.ones_like(batch_positional_mask)
            # batch_positional_mask[:,1] = False
            # batch_positional_mask[:,3] = False
            # batch_positional_mask[:,8] = False
            # #print("-----------------------------------")
            #print(batch_sequences_blosum[0])
            #positional_mask = torch.tile(batch_positional_mask[:, :, None], (1, 1, batch_sequences_blosum.shape[-1]))
            #batch_sequences_blosum = self.rotate_blosum_batch(batch_sequences_blosum, batch_positional_mask)
            # print(batch_sequences_blosum[0])
            # print("----------------------------------")
            # exit()

        else:
            #print("not rotating....")
            pass
        confidence_scores = batch_data["blosum"][:, 0, 0, 5]
        confidence_mask = (confidence_scores[..., None] < 0.4).any(-1)  # now we try to predict those with a low confidence score
        confidence_mask_true = torch.ones_like(confidence_mask).bool()
        init_h_0 = self.h_0_GUIDE.expand(self.encoder_guide.num_layers * 2, batch_size,
                                         self.gru_hidden_dim).contiguous()  # bidirectional

        with pyro.plate("plate_batch", dim=-1, device=self.device):
            z_mean, z_scale, rnn_hidden_states, rnn_hidden, rnn_final_hidden_state,rnn_final_hidden_state_bidirectional, rnn_hidden_states_bidirectional = self.encoder_guide(
                batch_sequences_encoded, batch_sequences_lens, init_h_0)
            assert z_mean.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(
                z_mean.shape)
            assert z_scale.shape == (batch_sequences_encoded.shape[0], self.z_dim), "Wrong shape got {}".format(
                z_scale.shape)
            latent_space = pyro.sample("latent_z", dist.Normal(z_mean, z_scale).to_event(1))
        return {"latent_z": latent_space,
                "z_mean": z_mean,
                "z_scale": z_scale,
                "rnn_hidden": rnn_hidden,
                "rnn_final_hidden": rnn_final_hidden_state,
                "rnn_final_hidden_bidirectional": rnn_final_hidden_state_bidirectional,
                "rnn_hidden_states_bidirectional": rnn_hidden_states_bidirectional,
                "rnn_hidden_states": rnn_hidden_states}


    def guide(self,batch_data,batch_mask,epoch,guide_estimates,sample):
        if self.seq_max_len == self.max_len:
            if self.learning_type == "supervised":
                if self.glitch:
                    return self.guide_supervised_glitch(batch_data,batch_mask,epoch,guide_estimates,sample)
                else:
                    return self.guide_supervised_new(batch_data,batch_mask,epoch, guide_estimates,sample)
            elif self.learning_type == "unsupervised":
                if self.glitch:
                    return self.guide_unsupervised_glitched(batch_data, batch_mask,epoch,guide_estimates, sample)
                else:
                    return self.guide_unsupervised_new(batch_data, batch_mask,epoch,guide_estimates, sample)
            elif self.learning_type == "semisupervised":
                if self.glitch:
                    return self.guide_semisupervised_glitched(batch_data, batch_mask, epoch, guide_estimates, sample)
                else:
                    return self.guide_semisupervised_new(batch_data, batch_mask, epoch, guide_estimates, sample)
        else:
            raise ValueError("guide not implemented for features, re-do")



