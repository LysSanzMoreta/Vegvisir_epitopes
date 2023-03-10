import torch
from torch import tensor
from pyro import sample,plate
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.infer import SVI,TraceEnum_ELBO
from pyro.optim import ClippedAdam
def model(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    :return:
    """
    with plate("inner", dim=-1):
        z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(1))
        #Highlight: Class inference
        if learning_type == "unsupervised":
            #c = sample("c",dist.Categorical(logits= torch.Tensor([[3,5],[10,8]])).to_event(1))
            class_logits = torch.Tensor([[3, 5], [10, 8]])
            for t, y in enumerate(x_class):
                c = sample(f"c_{t}", dist.Categorical(class_logits[t]))
        elif learning_type == "semisupervised":
            #c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).to_event(1),obs=x_class,obs_mask=class_mask)
            class_logits = torch.Tensor([[3, 5], [10, 8]])
            for t, y in enumerate(x_class):
                c = sample(f"c_{t}", dist.Categorical(class_logits[t]),obs=x_class[t],obs_mask=class_mask[t])
        else:
            c = sample("c",dist.Categorical(logits= torch.Tensor([[3,5],[10,8]])).to_event(1),obs=x_class)
        #Highlight: Sequence reconstruction
        with plate("outer",dim=-2):
            logits =  torch.Tensor([[[10,2,3],[8,2,1],[3,6,1]],
                                    [[1,2,7],[0,2,1],[2,7,8]]])
            aa = sample("x",dist.Categorical(logits= logits),obs=x,obs_mask=obs_mask)

        return z,c,aa

def guide(x,obs_mask,x_class,class_mask):
    """
    :param x: Data [N,L,feat_dim]
    :param obs_mask: Data sites to mask [N,L]
    :param x_class: Target values [N,]
    :param class_mask: Target values mask [N,]
    """
    with plate("inner", dim=-1):
        z = sample("z",dist.Normal(torch.zeros((2,5)),torch.ones((2,5))).to_event(1))
        if learning_type == "unsupervised":
            class_logits = torch.Tensor([[3, 5], [10, 8]])
            #c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).to_event(1),infer={'enumerate': 'parallel'})
            for t, y in enumerate(x_class):
                c = sample(f"c_{t}_unobserved", dist.Categorical(class_logits[t]),infer={"enumerate": "parallel"})
        elif learning_type == "semisupervised":
            #c = sample("c_unobserved",dist.Categorical(logits= torch.Tensor([[3,5],[10,8]])).to_event(1),infer={'enumerate': 'parallel'})
            class_logits = torch.Tensor([[3, 5], [10, 8]])
            #c = sample("c", dist.Categorical(logits=torch.Tensor([[3, 5], [10, 8]])).to_event(1),infer={'enumerate': 'parallel'})
            for t, y in enumerate(x_class):
                c = sample(f"c_{t}_unobserved", dist.Categorical(class_logits[t]).mask(~class_mask[t]),infer={"enumerate": "parallel"})
        else: #supervised
            c = None
        # #Highlight: Sequence reconstruction
        with plate("outer",dim=-2):
            logits =  torch.Tensor([[[10,2,3],[8,2,1],[3,6,1]],
                                    [[1,2,7],[0,2,1],[2,7,8]]])
            aa = sample("x_unobserved",dist.Categorical(logits= logits).mask(~obs_mask),infer={'enumerate': 'parallel'})


        return z,c,aa


if __name__ == "__main__":
    learning_ops = {0:"supervised",
                     1:"unsupervised",
                    2:"semisupervised"}
    learning_type = learning_ops[2]
    print(learning_type)
    x = tensor([[0,2,1],
                [0,1,1]])
    obs_mask = tensor([[1,0,0],[1,1,0]],dtype=bool) #I need a mask like this
    x_class = tensor([0,1])
    class_mask = tensor([1,0],dtype=bool)

    guide_tr = poutine.trace(guide).get_trace(x,obs_mask,x_class,class_mask)
    model_tr = poutine.trace(poutine.replay(model, trace=guide_tr)).get_trace(x,obs_mask,x_class,class_mask)
    monte_carlo_elbo = model_tr.log_prob_sum() - guide_tr.log_prob_sum()
    print(monte_carlo_elbo)

    svi = SVI(model,guide,loss=TraceEnum_ELBO(),optim=ClippedAdam(dict()))
    svi.step(x,obs_mask,x_class,class_mask)
