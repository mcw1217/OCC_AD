import torch
import utils
from tqdm import tqdm
from torchvision.transforms import transforms
from PIL import Image
import main
import numpy as np


def Inference_def(distance, threshold,img_path):
    if distance >= threshold:
        print(f"[System] 해당 이미지는 이상 이미지입니다! ( 이상 수치: {distance[0]:.4f} )")
    else:
        print(f"[System] 해당 이미지는 정상 이미지입니다! ( 이상 수치: {distance[0]:.4f} )")
        img = Image.open(img_path)
        img.save(f'./result/ad_img/{img_path.split("/")[-1]}')
        
def transform_def(x):
    transform_color = transforms.Compose([transforms.Resize(256),  #256
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    return transform_color(x)        
        
def input_img(x):
    img = Image.open(x)
    img = transform_def(img)
    img = img.unsqueeze(0)
    return img

def Start_Inference(img_path, train_feature_space_path, model_path, threshold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[System] Device: {device}")
    print("[System] 모델 로드중...")
    model =utils.Model(152)
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    model.eval()
    train_feature_np = np.load(train_feature_space_path) #Knn score 계산을 위한 학습된 분포
    print("[System] 모델 로드 완료!")
    if img_path.lower().endswith(('.jpg','.png')):
        with torch.no_grad():
            print("[System] 이미지의 이상 여부 확인 중...")
            img = input_img(img_path)
            img = img.to(device)
            features = model(img)
            test_feature_space = torch.cat([features], dim=0).contiguous().cpu().numpy()

        distance = utils.knn_score(train_feature_np, test_feature_space) #학습된 분포와 input img의 분포 거리 계산 ( 0에 가까울수록 학습된 분포와 유사 )
        Inference_def(distance, threshold, img_path)
    
