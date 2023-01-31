
import numpy as np
import torch
def bec(y,yhat):
    beta = (y.shape[0] - y.sum())/y.shape[0]
    #return -(y * np.log(yhat) + (1 - y) * np.log(1 - yhat)).mean()
    return -(y * np.log(yhat) + (1-beta)*(1 - y) * np.log(1 - yhat)).mean()

def bec2(y,yhat):
    loss = torch.nn.BCELoss()
    return loss(yhat,y)


y = np.array([1,0])
yhat = np.array([0.8,0.8])
error =bec(y,yhat)
print(error)
y = torch.tensor([1.,0.])
yhat = torch.tensor([0.8,0.8])
error =bec2(y,yhat)
print(error)
print("----------------------")
y = np.array([1,0])
yhat = np.array([0.51,0.8])
error =bec(y,yhat)
print(error)
y = torch.tensor([1.,0.])
yhat = torch.tensor([0.51,0.8])
error =bec2(y,yhat)
print(error)
print("----------------------")
y = np.array([1,0])
yhat = np.array([0.8,0.51])
error =bec(y,yhat)
print(error)
y = torch.tensor([1.,0.])
yhat = torch.tensor([0.8,0.51])
error =bec2(y,yhat)
print(error)
print("----------------------")
y = np.array([1,0])
yhat = np.array([0.51,0.51])
error =bec(y,yhat)
print(error)
y = torch.tensor([1.,0.])
yhat = torch.tensor([0.51,0.51])
error =bec2(y,yhat)
print(error)
print("----------------------")



