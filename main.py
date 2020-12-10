import torch
import numpy as np
import os
from PIL import Image
from thop import profile, clever_format
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


from model import Model, set_bn_eval
from my_utils import recall, LabelSmoothingCrossEntropyLoss, BatchHardTripletLoss, ImageReader, MPerClassSampler

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
        normalize])


if __name__ == '__main__':
    data_path = './dataset'
    data_name = 'vn_food'
    batch_size = 4
    backbone_type = 'wide_resnet50_2'
    gd_config = 'SG'
    feature_dim = 512
    num_epochs = 35
    smoothing = 0.1
    temperature = 0.5
    margin = 0.1
    recalls = 1,2,4,8
    num_retrieval = 10
    query_url = './dataset/vn_food/test/203.png'
    data_path = './dataset/vn_food/database'
    model = Model(backbone_type, gd_config, feature_dim, num_classes=22)
    flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224),))
    model.apply(set_bn_eval)
    model.load_state_dict(torch.load('./results/vnfood_model_final.pth', map_location=torch.device('cpu')))
    model.eval()
    database = Database(data_path, transforms)
    val_loader = torch.utils.data.DataLoader(
        database,
        batch_size=16, shuffle=False,
        num_workers = 4, pin_memory=True)

    features = []
    with torch.no_grad():
        for batch in val_loader:
            features.append(model(batch)[0].flatten(start_dim=1))#for fucking resnet
    features = torch.cat(features, dim=0)

    print(features.shape)

    with torch.no_grad():
        img = Image.open(query_url).convert('RGB')
        img = transforms(img)
        query = model(img.unsqueeze(0))[0].flatten(start_dim=1)
    #for cosine metric
    cosine = torch.nn.CosineSimilarity(1)
    results = cosine(query, features)
    _, idx = results.topk(num_retrieval)
    for i in range(num_retrieval):  
        img = Image.open(os.path.join(data_path, os.listdir(data_path)[idx[i]]))
        print(os.path.join(data_path, os.listdir(data_path)[idx[i]]))
    #   plt.imshow(img)
    #   plt.show()