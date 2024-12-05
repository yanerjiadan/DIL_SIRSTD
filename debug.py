import torch
x = torch.randn(256,256)
y= torch.randint(low=0, high=1, size=(256,256))
loss = torch.nn.BCELoss(x, y)
print(loss)