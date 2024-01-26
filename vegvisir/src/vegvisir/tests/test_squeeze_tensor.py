import torch
def squeeze_tensor(required_ndims,tensor):
    """Squeezes a tensor to match ndim"""
    size = torch.tensor(tensor.shape)
    ndims = len(size)
    idx_ones = (size == 1)
    ones_pos = size.tolist().index(1)

    if ndims > required_ndims:
        while ndims > required_ndims:
            if tensor.shape[0] == 1:
                tensor = tensor.squeeze(0)
                size = torch.tensor(tensor.shape)
                ndims = len(size)
            elif True in idx_ones:
                tensor = tensor.squeeze(ones_pos)
                size = torch.tensor(tensor.shape)
                ndims = len(size)
            else:
                ndims = required_ndims

        return tensor
    else:
        return tensor

t = torch.randn((20,10,1,5))
#t = torch.randn((1,20,10,5))


o = squeeze_tensor(3,t)

print(o.shape)