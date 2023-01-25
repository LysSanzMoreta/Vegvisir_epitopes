import torch.nn as nn
import torch
import math
def glorot_init(input_dim, output_dim):
    init_range = torch.sqrt(torch.tensor(6/(input_dim + output_dim)))
    initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
    return initial
class Embedder(nn.Module):
    def __init__(self,aa_types,embedding_dim,device):
        super(Embedder, self).__init__()
        self.aa_types = aa_types
        self.embedding_dim = embedding_dim
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.fc1 = nn.Linear(self.aa_types,self.embedding_dim)
        #self.weight1 = nn.Parameter(glorot_init(self.aa_types, self.embedding_dim), requires_grad=True).to(device)
        self.fc2 = nn.Linear(self.embedding_dim,self.aa_types)
        #self.weight2 = nn.Parameter(glorot_init(self._dim,self.aa_types), requires_grad=True).to(device)


    def forward(self,input,mask):
        #output = torch.matmul(input,self.weight1) #.type(torch.cuda.IntTensor)
        #output = torch.matmul(output,self.weight2)
        output = self.fc2(self.fc1(input))
        output = self.logsoftmax(output)
        #output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        if mask is not None:
            output = output.masked_fill(mask == 0, 1e10) #Highlight: This one does not seem crucial
        return output
class MLP(nn.Module):
    def __init__(self,input_dim,hidden_dim,num_classes,device):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.device = device
        self.fc1 = nn.Linear(self.input_dim,self.hidden_dim,bias=True)
        self.fc2 = nn.Linear(self.hidden_dim,self.num_classes,bias=True)
        self.relu = nn.ReLU()
    def forward(self,input,mask):

        output = self.relu(self.fc1(input))
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.relu(self.fc2(output))
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        return output

class CNN_layers(nn.Module):
    def __init__(self,input_dim,max_len,embedding_dim,num_classes,device,loss_type):
        super(CNN_layers, self).__init__()
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.max_len = max_len
        self.device = device
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=0)

        self.k_size = 3
        self.conv1 = nn.Conv1d(in_channels=self.input_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.embedding_dim,
                               kernel_size=self.k_size,
                               stride=1,
                               bias=True,
                               padding=int((self.k_size-1)/2)) # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]

        # # self.cnn_out_1 = (self.max_len + 2*int(self.conv1.padding[0])- self.conv1.dilation[0]*(self.k_size - 1) -1) / self.conv1.stride[0] + 1
        self.avgpool1 = nn.AvgPool1d(kernel_size=self.k_size, stride=1,padding=int((self.k_size-1)/2))
        self.conv2 = nn.Conv1d(in_channels=self.embedding_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=int(self.embedding_dim*2),
                               kernel_size=self.k_size,
                               stride=1,
                               bias=True,
                               padding=int((self.k_size-1)/2)) # Without padding the output has shape [N, out_channels, (max_len - kernel_size + 1)], with padding [N, out_channels, max_len]
        self.avgpool2 = nn.AvgPool1d(kernel_size=self.k_size, stride=1,padding=int((self.k_size-1)/2))

        #self.h = int(self.max_len*self.input_dim)
        self.h = int(self.embedding_dim*2)

        self.fc1 = nn.Linear(self.h,int(self.h/2))
        self.fc2 = nn.Linear(int(self.h/2),self.num_classes)
        #self.rbf = RBF(int(self.h/2),self.num_classes,"linear")
        self.dropout= nn.Dropout(p=0)


    def forward(self, input, mask):
        """https://github.com/bentrevett/pytorch-seq2seq/blob/master/5%20-%20Convolutional%20Sequence%20to%20Sequence%20Learning.ipynb
        TODO: CNN practices
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        https://cognitivemedium.com/assets/rmnist/Simard.pdf
        https://towardsdatascience.com/a-guide-for-building-convolutional-neural-networks-e4eefd17f4fd
        https://www.jeremyjordan.me/convnet-architectures/
        https://www.upgrad.com/blog/basic-cnn-architecture/#:~:text=LeNet%2D5%20CNN%20Architecture,-In%201998%2C%20the&text=It%20is%20one%20of%20the,resulting%20in%20dimension%20of%2028x28x6.
        TODO: LetNet5
        https://blog.paperspace.com/writing-lenet5-from-scratch-in-python/
        https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
        TODO: manifold inspection:
        https://arxiv.org/pdf/1909.11500.pdf
        https://www.stat.cmu.edu/~larry/=sml/Manifolds.pdf
        https://www.mdpi.com/1422-0067/23/14/7775#
        """

        #output = self.softmax(self.conv1(self.dropout(input)))
        output = self.relu(self.conv1(self.dropout(input)))
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.avgpool1(output)
        output = self.relu(self.conv2(self.dropout(output)))
        output = nn.BatchNorm1d(output.size()[1]).to(self.device)(output)
        output = self.avgpool2(output)
        output = output.permute(0,2,1)[:,-1] #Highlight: Does not seem to matter whether to flatten or take the last output
        #output = self.rbf(self.softmax2(self.dropout(self.fc1(output))))
        output = self.fc2(self.fc1(output))
        #output = torch.nn.functional.glu(output, dim=1) #divides by 2 the hidden dimensions
        if self.loss_type != "weighted_bce":
            output = self.sigmoid(output) #TODO: Softmax?
        if mask is not None:
            output = output.masked_fill(mask == 0, 1e10)
        return output
class LetNET5(nn.Module):
    """Adaptation of LetNET5.Following:
         - https://towardsdatascience.com/implementing-yann-lecuns-lenet-5-in-pytorch-5e05a0911320
         - Original paper: http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf"""
    def __init__(self, input_dim, max_len, embedding_dim, num_classes, device, loss_type):
        super(LetNET5, self).__init__()
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.loss_type = loss_type
        self.max_len = max_len
        self.ksize = 3
        self.hidden_sizes = [int(self.embedding_dim/2),self.embedding_dim,int(self.embedding_dim*2)]
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(in_channels=self.input_dim, out_channels=self.hidden_sizes[0], kernel_size=self.ksize, stride=1,padding=int((self.ksize-1)/2)),
            nn.BatchNorm1d(self.hidden_sizes[0]),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=self.ksize),
            nn.Conv1d(in_channels=self.hidden_sizes[0], out_channels=self.hidden_sizes[1], kernel_size=self.ksize, stride=1,padding=int((self.ksize-1)/2)),
            nn.BatchNorm1d(self.hidden_sizes[1]),
            nn.Tanh(),
            nn.AvgPool1d(kernel_size=self.ksize),
            nn.Conv1d(in_channels=self.hidden_sizes[1], out_channels=self.hidden_sizes[2], kernel_size=self.ksize, stride=1,padding=int((self.ksize-1)/2)),
            nn.Tanh()
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.hidden_sizes[2], out_features=self.hidden_sizes[1]),
            nn.Tanh(),
            #nn.Linear(in_features=self.hidden_sizes[1], out_features=self.num_classes),
            RBF(in_features=self.hidden_sizes[1],out_features=self.num_classes,basis_func="linear")
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, input, mask):
        """
        TODO: CNN practices
        CNN visualizer: https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        IMPORTANT : https://towardsdatascience.com/a-guide-for-building-convolutional-neural-networks-e4eefd17f4fd
        https://www.jeremyjordan.me/convnet-architectures/
        https://www.upgrad.com/blog/basic-cnn-architecture/#:~:text=LeNet%2D5%20CNN%20Architecture,-In%201998%2C%20the&text=It%20is%20one%20of%20the,resulting%20in%20dimension%20of%2028x28x6.
        TODO: manifold inspection:
        https://arxiv.org/pdf/1909.11500.pdf
        https://www.stat.cmu.edu/~larry/=sml/Manifolds.pdf
        https://www.mdpi.com/1422-0067/23/14/7775#
        """
        x = self.feature_extractor(input)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        probs = self.softmax(logits) #replaces Euclidean RBF
        return logits, probs
class RNN_layers(nn.Module):
    def __init__(self,input_dim,max_len,gru_hidden_dim,num_classes,device,loss_type):
        super(RNN_layers, self).__init__()
        self.loss_type = loss_type
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.gru_hidden_dim = gru_hidden_dim
        self.loss_type = loss_type
        self.max_len = max_len
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=1) #I choose this dimension to get the max from all aa
        # def softmax(x):
        #     """Compute softmax values for each sets of scores in x."""
        #     e_x = np.exp(x - np.max(x))
        #     return e_x / e_x.sum(axis=0)
        self.num_layers = 1
        self.rnn = nn.GRU(input_size=int(input_dim),
                          hidden_size=gru_hidden_dim,
                          batch_first=True,
                          num_layers=self.num_layers,
                          dropout=0.,
                          bidirectional=True
                          )
        self.h = self.gru_hidden_dim
        self.fc1 = nn.Linear(self.h,int(self.h/2),bias=False)
        self.fc2 = nn.Linear(int(self.h/2),int(self.h/4),bias=False)
        self.fc3 = nn.Linear(int(self.h/4),self.num_classes,bias=False)

    def forward1(self,input,init_h_0,init_c_0,mask):
        """For LSTM"""
        output, out_hidden = self.rnn(input,(init_h_0,init_c_0))

        forward_out,backward_out = output[:,:,:self.gru_hidden_dim],output[:,:,:self.gru_hidden_dim]
        output = self.softmax(forward_out + backward_out)
        output = output[:,-1]

        output = self.softmax(self.fc1(output))
        output = self.softmax(self.fc2(output))
        output = self.softmax(self.fc3(output))

        if self.loss_type != "weighted_bce":
            output = self.sigmoid(output)

        return output
    def forward(self,input,init_h_0,mask):
        "For GRU"
        output, out_hidden = self.rnn(input,init_h_0)

        forward_out,backward_out = output[:,:,:self.gru_hidden_dim],output[:,:,:self.gru_hidden_dim]
        output = self.softmax(forward_out + backward_out)
        output = output[:,-1]

        output = self.softmax(self.fc1(output))
        output = self.softmax(self.fc2(output))
        output = self.softmax(self.fc3(output))

        if self.loss_type != "weighted_bce":
            output = self.sigmoid(output)

        return output
class RBF(nn.Module):
    """
    Funtion from : https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        in_features: size of each input sample
        out_features: size of each output sample

    Shape:
        - Input: (N, in_features) where N is an arbitrary batch size
        - Output: (N, out_features) where N is an arbitrary batch size

    Attributes:
        centres: the learnable centres of shape (out_features, in_features).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.

        log_sigmas: logarithm of the learnable scaling factors of shape (out_features).

        basis_func: the radial basis function used to transform the scaled
            distances.
    """

    def __init__(self, in_features, out_features, basis_func):
        super(RBF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.log_sigmas = nn.Parameter(torch.Tensor(out_features))
        self.basis_func = self.basis_func_choice(basis_func)
        self.reset_parameters()

    def gaussian(self,alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi

    def linear(self,alpha):
        phi = alpha
        return phi

    def quadratic(self,alpha):
        phi = alpha.pow(2)
        return phi

    def inverse_quadratic(self,alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
        return phi

    def multiquadric(self,alpha):
        phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def inverse_multiquadric(self,alpha):
        phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
        return phi

    def spline(self,alpha):
        phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
        return phi

    def poisson_one(self,alpha):
        phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
        return phi

    def poisson_two(self,alpha):
        phi = ((alpha - 2 * torch.ones_like(alpha)) / 2 * torch.ones_like(alpha)) \
              * alpha * torch.exp(-alpha)
        return phi

    def matern32(self,alpha):
        phi = (torch.ones_like(alpha) + 3 ** 0.5 * alpha) * torch.exp(-3 ** 0.5 * alpha)
        return phi

    def matern52(self,alpha):
        phi = (torch.ones_like(alpha) + 5 ** 0.5 * alpha + (5 / 3) * alpha.pow(2)) * torch.exp(-5 ** 0.5 * alpha)
        return phi
    def basis_func_choice(self,name):
        func_dict = {"gaussian":self.gaussian,
                     "linear":self.linear,
                     "quadratic":self.quadratic,
                     "inverse_quadratic":self.inverse_quadratic,
                     "spline":self.spline,
                     "poisson_one":self.poisson_one,
                     "poisson_two":self.poisson_two,
                     "matern32":self.matern32,
                     "matern52":self.matern52}

        return func_dict[name]



    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.log_sigmas, 0)

    def forward(self, input):
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size) #optimizable parameter
        distances = (x - c).pow(2).sum(-1).pow(0.5) / torch.exp(self.log_sigmas).unsqueeze(0)
        return self.basis_func(distances)
class AutoEncoder(nn.Module):
    def __init__(self,input_dim,max_len,embedding_dim,num_classes,device,loss_type):
        super(AutoEncoder,self).__init__()
        self.input_dim = input_dim
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.device = device
        self.loss_type = loss_type
        self.k_size = 3
        self.padding_0 = int((self.k_size - 1) / 2)
        self.padding_1 = 0
        self.dilation = 1
        self.stride = 1
        self.encoder = nn.Sequential(nn.Conv1d(in_channels=self.input_dim, #highlight: the input has shape [N,feats-size,max-len]
                               out_channels=self.embedding_dim,
                               kernel_size=self.k_size,
                               stride=self.stride,
                               dilation=self.dilation,
                               padding=self.padding_0),
                            nn.LeakyReLU(), #PReLU
                            nn.BatchNorm1d(self.embedding_dim),
                            nn.AvgPool1d(kernel_size=self.k_size, stride=1, padding=int((self.k_size) / 2)),
                            nn.Conv1d(in_channels=self.embedding_dim,# highlight: the input has shape [N,feats-size,max-len]
                                      out_channels=int(self.embedding_dim / 2),
                                      kernel_size=self.k_size,
                                      stride=self.stride,
                                      dilation=self.dilation,
                                      padding=self.padding_0),
                                      #padding=int((self.k_size - 1) / 2)),
                            nn.LeakyReLU(),
                            nn.BatchNorm1d(int(self.embedding_dim / 2)),
                            nn.AvgPool1d(kernel_size=self.k_size, stride=1,padding=int((self.k_size) / 2))
                            )

        """https://towardsdatascience.com/what-are-transposed-convolutions-2d43ac1a0771
        https://medium.com/@santi.pdp/how-pytorch-transposed-convs1d-work-a7adac63c4a5"""
        #self.padding = math.ceil(((self.max_len - 1) * self.stride + self.dilation * (self.k_size - 1) + 1 - self.k_size * 2)/2)
        #self.padding = math.ceil((self.stride * (self.max_len/2) - self.max_len + self.dilation * (self.k_size - 1)-1)/2)
        self.decoder = nn.Sequential(nn.ConvTranspose1d(in_channels = int(self.embedding_dim/2),
                                               out_channels=self.embedding_dim,
                                               kernel_size = self.k_size,
                                               stride =1,
                                               padding=1,#only works because stride = 1, i have not calculated a general formula
                                               output_padding=0),
                            nn.LeakyReLU(),
                            nn.BatchNorm1d(self.embedding_dim), #compensates the issues with ReLU handling negative values
                            nn.ConvTranspose1d(in_channels = self.embedding_dim,
                                               out_channels=self.input_dim,
                                               kernel_size = self.k_size,
                                               stride =1,padding=1,output_padding=0),
                            nn.LeakyReLU(),
                            nn.BatchNorm1d(self.input_dim),
                            nn.LogSoftmax(dim=-1)
                            )


        classifier_input_dim = int(self.embedding_dim / 2)*self.max_len
        self.classifier = nn.Sequential(nn.Linear(classifier_input_dim,int(classifier_input_dim/4)),
                                   nn.LeakyReLU(),
                                   nn.Linear(int(classifier_input_dim/4),self.num_classes),
                                   )

    def forward(self,input):
        enc_out = self.encoder(input)
        reconstructed_sequences = self.decoder(enc_out)
        class_output = self.classifier(enc_out.flatten(1))
        return reconstructed_sequences.permute(0,2,1),class_output