import torch
from mobilenet import mobilenet_v2


img = torch.rand(2, 3, 256, 256)
model = mobilenet_v2()
out = model(img)


