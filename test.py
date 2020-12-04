import torch
torch.cuda.set_device(0)
print(torch.rand(10).cuda())
