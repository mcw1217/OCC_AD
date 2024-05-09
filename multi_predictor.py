import torch
import utils
from tqdm import tqdm
from torchvision.transforms import transforms
from PIL import Image
import main

def Inference_def(distance):
    count = sum(1 for num in distance if num >=0.4)
    print(count)
        

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model =utils.Model(152)
model.load_state_dict(torch.load("./result/model.pth"))
model = model.to(device)
model.eval()
print("로드 완료!")

train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset="plant", label_class=0, batch_size=64, backbone=152)

train_feature = './result/train_feature_space.npy'

test_feature_space = []
test_labels = []
with torch.no_grad():
    for (imgs, labels) in tqdm(test_loader, desc='Test set feature extracting'):
        imgs = imgs.to(device)
        features = model(imgs)
        print(features.shape)
        test_feature_space.append(features)
        test_labels.append(labels)
    test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
    print(test_feature_space.shape)
    test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

import numpy as np

train_feature_np = np.load(train_feature)
distance = utils.knn_score(train_feature_np, test_feature_space)
print(distance)
Inference_def(distance)

    
        