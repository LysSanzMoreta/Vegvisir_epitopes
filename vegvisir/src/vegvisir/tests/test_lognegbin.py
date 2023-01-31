import matplotlib.pyplot as plt
import pyro.distributions as dist
import torch
import pyro.distributions.transforms as T
import numpy as np
#https://stackoverflow.com/questions/20385964/generate-random-number-between-0-and-1-with-negativeexponential-distribution
n = 10000
#samples = dist.LogNormalNegativeBinomial(total_count=1,logits=torch.rand((n,)),multiplicative_noise_scale=torch.tensor([0.1]),num_quad_points=8).sample()
#Smaller values of λ give a flatter distribution, larger values show a more rapid exponential drop off.


# rate = torch.ones((n,))*8
# samples_exp = dist.Exponential(rate=rate).sample()
# samples = -torch.log(1 - (1 - samples_exp) * dist.Uniform(0,1).sample((n,))) / rate
#

dist_x = torch.distributions.HalfNormal(torch.ones(1)*0.2)
#dist_x = dist.Normal(torch.zeros(1), torch.ones(1))
exp_transform = T.ExpTransform()
dist_y = dist.TransformedDistribution(dist_x, [exp_transform])

dist_x = torch.distributions.Beta(0.5,0.7)
#plt.hist(dist_y.sample([n]).numpy(),density=True,bins=50)
plt.hist(dist_x.sample([n]).numpy(),bins=50)
#plt.hist(np.abs(1- torch.exp(dist_x.sample([n])).numpy()),density=True,bins=50)

plt.show()
