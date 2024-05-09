from inference_util import Start_Inference
    

# 파라미터
train_feature_space_path = './result/train_feature_space.npy'
model_path = './result/model.pth'
img_path = './img/17.png'
threshold = 0.5

    
    
Start_Inference(img_path, train_feature_space_path, model_path, threshold)
    
        