import torch
def squeeze_tensor(curr_bsize,tensor):
    """RSqueezes a tensor to match the given first_dim size"""
    size = torch.tensor(tensor.shape)
    idx, = torch.where(size == curr_bsize)

    done = False
    while done != True:
        if idx != 0:
            tensor = tensor.squeeze(0)
            size = torch.tensor(tensor.shape)
            idx, = torch.where(size == curr_bsize)
        else:
            done = True
    return tensor

def squeeze_tensor2(required_ndims,tensor):
    """Squeezes a tensor to match the given first_dim size"""
    size = torch.tensor(tensor.shape)
    ndims = len(size)
    if ndims > required_ndims:

        while ndims > required_ndims:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
                size = torch.tensor(tensor.shape)
                ndims = len(size)
            else:
                ndims = required_ndims

        return tensor
    else:
        return tensor


t1 = torch.ones((1,1,3,4))

t2 = torch.ones((1,3,4))

t3 = torch.ones((3,4))

frst_dim = 3

r = squeeze_tensor2(frst_dim,t2)

print(r.shape)

