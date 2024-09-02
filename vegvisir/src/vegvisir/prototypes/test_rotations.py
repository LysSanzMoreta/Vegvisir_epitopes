import numpy as np
import matplotlib.pyplot as plt
import torch
def rotate_blosum(v1,cosine_sim_mask):
    """

    :return:

    Notes:
        -https://www.rollpie.com/post/311
        -https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
        -https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
        -https://math.stackexchange.com/questions/209768/transformation-matrix-to-go-from-one-vector-to-another
    """


    # input vectors
    #v1 = np.array([1, 1, 1, 1, 1, 1])
    v2 = np.ones_like(v1)
    #v1 = v2
    #v2 = np.array([2, 3, 4, 5, 6, 7])

    plt.plot(v1,v2,c="blue")


    # Gram-Schmidt orthogonalization
    n1 = v1 / np.linalg.norm(v1)
    v2 = v2 - np.dot(n1, v2) * n1
    n2 = v2 / np.linalg.norm(v2)


    # rotation by pi/2 (np.pi = 180)
    #sign = np.random.randn() > 0
    sign = True
    sign_dict ={True:-1,False:1}
    a = sign_dict[sign]*(np.pi*0.8)

    I = np.identity(v1.shape[0])

    R = I + (np.outer(n2, n1.T) - np.outer(n1, n2.T)) * np.sin(a) + (np.outer(n1, n1.T) + np.outer(n2, n2.T)) * (np.cos(a) - 1)

    # check result
    print(np.matmul(R, n1))
    print(n2)
    n1_r = np.matmul(R,n1)
    print("----------------------")
    print(n1_r*np.linalg.norm(v1))
    print(n2*np.linalg.norm(v2))

    plt.plot(n1_r,n2,c="green")
    plt.show()
    exit()

    return n1_r

def rotate_blosum_peptide(data,data_mask):
    """

    :return:

    Notes:
        -https://www.rollpie.com/post/311
        -https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
        -https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
        -https://math.stackexchange.com/questions/209768/transformation-matrix-to-go-from-one-vector-to-another
    """


    # input vectors
    v2 = torch.ones_like(data)

    # Gram-Schmidt orthogonalization
    n1 = data/torch.linalg.norm(data,dim=1)[:,None]
    v2 = v2 - torch.matmul(n1,v2[0])[:,None]*n1 #works [L,feat_dim]
    n2 = v2 / torch.linalg.norm(v2,dim=1)[:,None]


    # rotation by pi/2 (np.pi = 180)
    #sign = torch.randn(1) > 0
    sign = torch.Tensor([True])

    sign_dict ={True:torch.tensor([-1]),False:torch.tensor([1])}
    a = sign_dict[sign.item()]*(torch.pi*0.8) #TODO: Also randomly change degrees of rotation
    #a = torch.rand(-1,1,(1))*torch.pi #degrees
    I = torch.eye(data.shape[1])

    #print("BMM-----------------------------")
    #print(torch.bmm(n2[:,:,None], n1[:,None,:]))
    # print("---------------n2-------------------")
    # print(n2)
    # print("---------------n1-------------------")
    # print(n1)
    # print("n1@n2")
    # print(torch.bmm(n2[:,:,None], n1[:,None,:]).shape)
    #
    # print(torch.bmm(n2[:,:,None], n1[:,None,:]))
    # print("-----------------------")
    R = I + (torch.bmm(n2[:,:,None], n1[:,None,:]) - torch.bmm(n1[:,:,None], n2[:,None,:])) * torch.sin(a) + (torch.bmm(n1[:,:,None], n1[:,None,:]) + torch.bmm(n2[:,:,None], n2[:,None,:])) * (torch.cos(a) - 1)

    # check result
    data_rotated = torch.matmul(R,n1[:,:,None]).squeeze(-1)
    # print("data rotated """"""""""""""")
    # print(data_rotated)

    # print("geerere")
    # print(torch.linalg.norm(data,dim=1)[:,None])
    data_rotated_unnormalized = data_rotated*torch.linalg.norm(data,dim=1)[:,None]
    # print("data rotated unormalized 22222222222222222222")
    # print(data_rotated_unnormalized)

    data_mask = torch.tile(data_mask[:,None],(1,data.shape[-1]))

    data[~data_mask] = 0
    data_rotated_unnormalized[data_mask] = 0

    data_transformed = data + data_rotated_unnormalized

    return data_transformed

def batch_multiplication(a,b,n_data,L,feat_dim):
    """Inspired by https://github.com/pytorch/pytorch/issues/3172"""
    c = torch.bmm(a[:,:,:,None].view(n_data,-1,1),b[:,None,:,:].view(n_data,1,-1))
    c = c.view(n_data,L,feat_dim,L,feat_dim)
    c = c.permute(0,1,3,2,4)[:,torch.arange(L),torch.arange(L)]

    return c

def rotate_blosum_batch(data,data_mask):
    """

    :return:

    Notes:
        -https://www.rollpie.com/post/311
        -https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
        -https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
        -https://math.stackexchange.com/questions/209768/transformation-matrix-to-go-from-one-vector-to-another
    """
    n_data,L,feat_dim = data.shape

    # input vectors
    v2 = torch.ones_like(data) #[N,L,feat_dim]

    # Gram-Schmidt orthogonalization

    n1 = data/torch.linalg.norm(data,dim=2)[:,:,None] #[N,L,feat_dim]
    v2 = v2 - torch.matmul(n1,v2[0,0])[:,:,None]*n1 #works [N,L,feat_dim]
    n2 = v2 / torch.linalg.norm(v2,dim=2)[:,:,None]

    # rotation by pi/2 (np.pi = 180)
    sign = torch.randn(1) > 0
    #sign = torch.Tensor([True])
    degree = torch.rand(1)  #A degree 0 will not rotate the vector

    sign_dict ={True:torch.tensor([-1]),False:torch.tensor([1])}
    a = sign_dict[sign.item()]*(torch.pi*degree) #TODO: Also randomly change degrees of rotation
    #a = torch.rand(-1,1,(1))*torch.pi #degrees
    I = torch.eye(feat_dim)


    one = batch_multiplication(n2,n1,n_data,L,feat_dim)
    two = batch_multiplication(n1,n2,n_data,L,feat_dim)
    three = batch_multiplication(n1,n1,n_data,L,feat_dim)
    four = batch_multiplication(n2,n2,n_data,L,feat_dim)

    R = I + (one - two) * torch.sin(a) + (three + four) * (torch.cos(a) - 1)

    # check result
    data_rotated = torch.matmul(R,n1[:,:,:,None]).squeeze(-1)
    data_rotated_unnormalized = data_rotated*torch.linalg.norm(data,dim=2)[:,:,None]
    data_mask = torch.tile(data_mask[:,:,None],(1,1,data.shape[-1]))

    data[~data_mask] = 0
    data_rotated_unnormalized[data_mask] = 0
    data_transformed = data + data_rotated_unnormalized

    return data_transformed


def batch_multiplication_new(self,a, b, n_data, L, feat_dim):
    """Inspired by https://github.com/pytorch/pytorch/issues/3172"""
    c = torch.bmm(a[:, :, :, None].view(n_data, -1, 1), b[:, None, :, :].view(n_data, 1, -1))
    c = c.view(n_data, L, feat_dim, L, feat_dim)
    c = c.permute(0, 1, 3, 2, 4)[:, torch.arange(L), torch.arange(L)]

    return c

def rotate_blosum_batch_new(self,data, data_mask):
    """

    :return:

    Notes:
        -https://www.rollpie.com/post/311
        -https://math.stackexchange.com/questions/2144153/n-dimensional-rotation-matrix
        -https://analyticphysics.com/Higher%20Dimensions/Rotations%20in%20Higher%20Dimensions.htm
    """
    n_data, L, feat_dim = data.shape

    # input vectors
    v2 = torch.ones_like(data)  # [N,L,feat_dim]

    # Gram-Schmidt orthogonalization

    n1 = data / torch.nan_to_num(torch.linalg.norm(data, dim=2)[:, :, None], nan=1e-6,posinf=1e-6,neginf=-1e-6) # [N,L,feat_dim]
    n1 = torch.nan_to_num(n1, nan=1e-6,posinf=1e-6,neginf=-1e-6) #Calculating the norm and dividing by 0 generates nan values

    v2 = v2 - torch.matmul(n1, v2[0, 0])[:, :, None] * n1  # works [N,L,feat_dim]
    v2 = torch.nan_to_num(v2, nan=1e-6,posinf=1e-6,neginf=-1e-6)


    n2 = v2 / torch.nan_to_num(torch.linalg.norm(v2, dim=2)[:, :, None], nan=1e-6,posinf=1e-6,neginf=-1e-6)
    n2  = torch.nan_to_num(n2, nan=1e-6,posinf=1e-6,neginf=-1e-6)
    # rotation by pi/2 (np.pi = 180)
    sign = torch.randn(1) > 0
    # sign = torch.Tensor([True])
    degree = torch.rand(1)  # A degree 0 will not rotate the vector
    #degree = 0.8 #approx 40 degrees
    #degree = 0
    sign_dict = {True: torch.tensor([-1]), False: torch.tensor([1])}
    a = sign_dict[sign.item()] * (torch.pi * degree)  # TODO: Also randomly change degrees of rotation
    # a = torch.rand(-1,1,(1))*torch.pi #degrees
    I = torch.eye(feat_dim)

    one = self.batch_multiplication(n2, n1, n_data, L, feat_dim)
    two = self.batch_multiplication(n1, n2, n_data, L, feat_dim)
    three = self.batch_multiplication(n1, n1, n_data, L, feat_dim)
    four = self.batch_multiplication(n2, n2, n_data, L, feat_dim)

    R = I + (one - two) * torch.sin(a) + (three + four) * (torch.cos(a) - 1)

    # check result
    data_rotated = torch.matmul(R, n1[:, :, :, None]).squeeze(-1)
    data_rotated_unnormalized = data_rotated * torch.nan_to_num(torch.linalg.norm(data, dim=2)[:, :, None], nan=1e-6,posinf=1e-6,neginf=-1e-6)
    data_rotated_unnormalized = torch.nan_to_num(data_rotated_unnormalized, nan=1e-6,posinf=1e-6,neginf=-1e-6)
    data_mask = torch.tile(data_mask[:, :, None], (1, 1, data.shape[-1]))
    data[~data_mask] = 0
    data_rotated_unnormalized[data_mask] = 0
    data_transformed = data  + data_rotated_unnormalized

    return data_transformed

def rotate_conserved( dataset, dataset_mask):
    """Calculates a frequency for each of the aa & gap at each position.The number of bins (of size 1) is one larger than the largest value in x. This is done for torch tensors
    :param tensor dataset
    :param int freq_bins
    """

    #Highlight: Testing for calculation of one peptide at the time
    results = []
    for i,d_i in enumerate(torch.unbind(dataset.float(), dim=0), 0):
        d_mask = dataset_mask[i]
        d_rotated=rotate_blosum_peptide(d_i,d_mask)
        results.append(d_rotated)
    results = torch.stack(results)
    # print("----------------RESULTS peptide-------------------------")
    # print(results)
    #Highlight: Testing for calculation of all peptides simultaneously
    results = rotate_blosum_batch(dataset.float(), dataset_mask)
    # print("result batch")
    #print(results)


if __name__ == '__main__':
    a = torch.tensor([[[-2,3,4,5,1,9,8],[3,4,-6,5,0,7,1],[4,-5,2,3,1,0,-6]],
                  [[3,2,1,8,1,-7,8],[2,1,-1,0,3,8,1],[7,-3,1,7,2,4,-7]]])
    a_mask = torch.tensor([[True,True,False],
                       [False,True,True]])
    rotate_conserved(a,a_mask)