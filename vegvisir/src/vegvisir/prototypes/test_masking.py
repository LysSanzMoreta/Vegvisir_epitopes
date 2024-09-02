import torch
from torch import tensor
import pyro
from pyro import sample,plate
import pyro.distributions as dist
from pyro.poutine import trace,mask


def model(x,obs_mask):
    with plate("outer",dim=-2):
        with plate("inner",dim=-1):
            with mask(mask= obs_mask):
                return sample("x",dist.Normal(0,1),obs=x)

def model2(x,obs_mask):
    with plate("inner", dim=-1):
        z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(1)) #dim -1 is the number of sequences
        with mask(mask=obs_mask):
            c = sample("c",dist.Categorical(logits= torch.Tensor([[3,5],[10,8]])).to_event(1))
        with plate("outer",dim=-2):
            with mask(mask= obs_mask):
                aa=  sample("x",dist.Categorical(logits= torch.Tensor([[10,2,3],[8,2,1]])).to_event(1).mask(obs_mask),obs=x) #dim -1 is the length and dim -2 is the sequences
                return z,aa,c




if __name__ == "__main__":
    x = tensor([[0.,1.,-1.],[0.,1.,-1]])
    all_mask = tensor([1,1,1],dtype=bool)
    some_mask = tensor([1,0,0],dtype=bool)
    #
    # s = trace(model).get_trace(x,all_mask)
    # print("without masking")
    #
    # s = trace(model).get_trace(x,some_mask)
    # print("with masking")

    # s = trace(model2).get_trace(x,all_mask)
    # print("without masking")
    #
    # s = trace(model2).get_trace(x,some_mask)
    # print("with masking")




