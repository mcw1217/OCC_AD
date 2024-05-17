from inference_util import Start_Inference
import os

# 파라미터
train_feature_space_path = './result/train_feature_space.npy'
model_path = './result/model.pth'
# img_path = './img/17.png'
img_path = './dataset/plant/test/1'
threshold = 0.4

img_path_list = [path for path in os.listdir(img_path) ]
    
for i in img_path_list:
    path = img_path + "/" + i
    Start_Inference(path, train_feature_space_path, model_path, threshold)
    
        