import torch
from torch import tensor
from pyro import sample,plate
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI,Trace_ELBO
from pyro.optim import ClippedAdam
import pyro


def model1(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    :return:
    """
    z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(2))
    logits =  torch.Tensor([[[10,2,3],[8,2,1],[3,6,1]],
                            [[1,2,7],[0,2,1],[2,7,8]]])
    aa = sample("x",dist.Categorical(logits= logits),obs=x)
    with pyro.poutine.mask(mask=class_mask):
        c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).to_event(1), obs=x_class)
    return z,c,aa

def model2(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    :return:
    """
    z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(2))
    logits =  torch.Tensor([[[10,2,3],[8,2,1],[3,6,1]],
                            [[1,2,7],[0,2,1],[2,7,8]]])
    aa = sample("x",dist.Categorical(logits= logits).mask(obs_mask).to_event(1),obs=x)
    c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).to_event(1), obs=x_class)
    return z,c


def model3(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    :return:
    """
    z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(2))
    logits =  torch.Tensor([[[10,2,3],[8,2,1],[3,6,1]],
                            [[1,2,7],[0,2,1],[2,7,8]]])
    aa = sample("x",dist.Categorical(logits= logits).to_event(1),obs=x)
    c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).to_event(1), obs=x_class)
    return z,c,aa

def model4(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    :return:
    """
    z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(2))
    logits =  torch.Tensor([[[10,2,3],[8,2,1],[3,6,1]],
                            [[1,2,7],[0,2,1],[2,7,8]]])
    aa = sample("x",dist.Categorical(logits= logits),obs=x,obs_mask=obs_mask) #partial observations is what i am looking for here
    c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).mask(class_mask), obs=x_class) #in the fully supervised approach no mask here, but in the semi-supervised i would need to mask fully some observations
    return z,c,aa

def model5(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    :return:
    """
    with pyro.plate("plate_batch",dim=-1):
        z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(1))
        logits =  torch.Tensor([[[10,2,3],[8,2,1],[3,6,1]],
                                [[1,2,7],[0,2,1],[2,7,8]]])
        aa = sample("x",dist.Categorical(logits= logits),obs=x,obs_mask=obs_mask) #partial observations is what i am looking for here
        c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).mask(class_mask), obs=x_class)
    return z,c,aa

def guide(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    """
    z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(2))

    return z


if __name__ == "__main__":
    pyro.enable_validation(False)

    x = tensor([[0,2,1],
                [0,1,1]])
    obs_mask = tensor([[1,0,0],[1,1,0]],dtype=bool) #Partial observations
    x_class = tensor([0,1])
    class_mask = tensor([True,False],dtype=bool) #keep/skip some observations

    models_dict = {"model1":model1,
                   "model2":model2,
                   "model3":model3,
                   "model4":model4,
                   "model5":model5,
                   }

    for model in models_dict.keys():
        print("Using {}".format(model))
        guide_tr = poutine.trace(guide).get_trace(x,obs_mask,x_class,class_mask)
        model_tr = poutine.trace(poutine.replay(models_dict[model], trace=guide_tr)).get_trace(x,obs_mask,x_class,class_mask)
        monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()
        print("MC ELBO estimate: {}".format(monte_carlo_elbo))
        try:
            pyro.clear_param_store()
            svi = SVI(models_dict[model],guide,loss=Trace_ELBO(),optim=ClippedAdam(dict()))
            svi.step(x,obs_mask,x_class,class_mask)
            print("Test passed")
        except:
            print("Test failed")
