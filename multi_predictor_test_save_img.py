import torch
import utils
from tqdm import tqdm
from torchvision.transforms import transforms
from PIL import Image
import os
import main
from torchvision.transforms import ToPILImage

save_path = './result/ad_img'
to_pil = ToPILImage()

def Inference_def(distance, test_loader, save_path):
    anomaly_indices = [i for i, num in enumerate(distance) if num <= 0.4]
    for idx in anomaly_indices:
        # DataLoader에서 원본 이미지를 추출 (수정 필요)
        original_img, _ = test_loader.dataset[idx]
        original_img_pil = to_pil(original_img)
        img_path = os.path.join(save_path, f'anomaly_{idx}.png')        
        original_img_pil.save(img_path)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model =utils.Model(152)
model.load_state_dict(torch.load("./result/model.pth"))
model = model.to(device)
model.eval()
print("로드 완료!")

train_loader, test_loader, train_loader_1 = utils.get_loaders(dataset="plant", label_class=0, batch_size=64, backbone=152)

train_feature = './result/train_feature_space.npy'
test_image_indices = []
test_feature_space = []
test_labels = []
with torch.no_grad():
    for idx, (imgs, labels) in enumerate(tqdm(test_loader, desc='Test set feature extracting')):
        imgs = imgs.to(device)
        features = model(imgs)
        print(features.shape)
        test_feature_space.append(features)
        test_labels.append(labels)
        for i in range(imgs.size(0)):
            test_image_indices.append(idx * test_loader.batch_size + i )
    test_feature_space = torch.cat(test_feature_space, dim=0).contiguous().cpu().numpy()
    print(test_feature_space.shape)
    test_labels = torch.cat(test_labels, dim=0).cpu().numpy()

import numpy as np

train_feature_np = np.load(train_feature)
distance = utils.knn_score(train_feature_np, test_feature_space)
print(distance)
Inference_def(distance, test_loader, save_path)

    
        