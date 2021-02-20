import torch
import torchvision
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn


x=torch.randn(10,1)
y=2*x+3
class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.linear1=nn.Linear(1,3)
        self.linear2=nn.Linear(3,1)
    def forward(self,x):
        x=self.linear1(x)
        x=self.linear2(x)
        return x
net=Net()
print(net)
for param in net.parameters():
    print(param)
print('================================')

for param in net.linear1.parameters():
    print(param)
    param.requires_grad=False
print('================================')

for name,param in net.named_parameters():
    print(name,'\n',param)

criterion=torch.nn.MSELoss()
optimizer=torch.optim.SGD(net.parameters(),lr=1e-2,momentum=0.9)

for i in range(1000):
    y_pred=net(x)
    loss=criterion(y_pred,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #print('loss:{:.6f}'.format(loss.item()))
print('================================')
for name,param in net.named_parameters():
    print(name,'\n',param)
