import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

# model = models.vgg19(pretrained=True, progress=True)
# model = models.densenet161(pretrained=True, progress=True)
# model = models.squeezenet1_0(pretrained=True)
model = models.resnet101(pretrained=True, progress=True)
#model = models.wide_resnet50_2(pretrained=True)
# modal.cuda()
model.eval()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
class Database(Dataset):
    def __init__(self, path, transforms=None):
        self.path = path
        self.img_paths = os.listdir(path)
        self.transforms = transforms
    def __len__(self):
        return len(self.img_paths)
    def __getitem__(self, index):
        img = Image.open(os.path.join(self.path, self.img_paths[index]))
        if self.transforms:
            img = self.transforms(img)
        return img

transforms = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])

database = Database("./data", transforms)
val_loader = torch.utils.data.DataLoader(
    database,
    batch_size=1, shuffle=False,
    num_workers=0, pin_memory=True)

features = []
with torch.no_grad():
    for batch in val_loader:
        # features.append(model.features(batch.cuda()).flatten(start_dim=1))
        # features.append(model.features(batch).flatten(start_dim=1))
        features.append(model(batch).flatten(start_dim=1))#for fucking resnet
features = torch.cat(features, dim=0)

print(features.shape)

with torch.no_grad():
    img = Image.open("./query/bigben.jpg")
    img = transforms(img)
    # query = model.features(img.unsqueeze(0).cuda()).flatten(start_dim=1)
    # query = model.features(img.unsqueeze(0))
    query = model(img.unsqueeze(0)) #for fucking resnet
    query = query.flatten(start_dim=1)
print(query.size())
#for cosine metric
cosine = torch.nn.CosineSimilarity(1)
results = cosine(query, features)
_, idx = results.topk(5)
for i in range(5):  
  img = Image.open(os.path.join("./data", os.listdir("./data")[idx[i]]))
  print(os.path.join("./data", os.listdir("./data")[idx[i]]))
#   plt.imshow(img)
#   plt.show()