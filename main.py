import torch
import numpy as np
import os
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

from signal import signal, SIGPIPE, SIG_DFL
signal(SIGPIPE, SIG_DFL) 

data_path = './dataset'
data_name = 'vn_food'
batch_size = 16
backbone_type = 'wide_resnet50_2'
gd_config = 'SG'
feature_dim = 512
num_epochs = 35
smoothing = 0.1
temperature = 0.5
margin = 0.1
recalls = 1,2,4,8
from model import Model, set_bn_eval
# from utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler
from my_utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler

if __name__ == '__main__':

    query_url = './dataset/vn_food/database/61.png'
    data_path = './dataset/vn_food/database'
    try:
        model = Model(backbone_type, gd_config, feature_dim, num_classes= 22)
        model.load_state_dict(torch.load('./results/vnfood_model_final.pth', map_location=torch.device('cpu')))
    except RuntimeError:
        print('nah')

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
            img = Image.open(os.path.join(self.path, self.img_paths[index])).convert('RGB')
            if self.transforms:
                img = self.transforms(img)
            return img

    transforms = transforms.Compose([
            transforms.Scale(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

    database = Database(data_path, transforms)
    val_loader = torch.utils.data.DataLoader(
        database,
        batch_size=8, shuffle=False,
        num_workers = 1, pin_memory=True)

    features = []
    with torch.no_grad():
        for batch in val_loader:
            features.append(model(batch)[0].flatten(start_dim=1))#for fucking resnet
    features = torch.cat(features, dim=0)

    print(features.shape)

    with torch.no_grad():
        img = Image.open("query_url")
        img = transforms(img)
        query = model(img.unsqueeze(0))
        query = query[0].flatten(start_dim=1)
    #for cosine metric
    cosine = torch.nn.CosineSimilarity(1)
    results = cosine(query, features)
    _, idx = results.topk(5)
    for i in range(5):  
        img = Image.open(os.path.join(data_path, os.listdir(data_path)[idx[i]]))
        print(os.path.join(data_path, os.listdir(data_path)[idx[i]]))
    #   plt.imshow(img)
    #   plt.show()