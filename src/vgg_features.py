from torchvision import models
from PIL import Image
import torch.nn as nn
import torchvision.transforms as T

pretrained_vgg_model = models.vgg16(pretrained=True)
# Extract the features with 1x4096 size (remove the last layers in classifier)
pretrained_vgg_model.classifier = nn.Sequential(*list(pretrained_vgg_model.classifier.children())[:-3])


def extract_vgg_features(img):
    input_tensor = T.ToTensor()(T.Resize((512, 512))(Image.fromarray(img))).unsqueeze(dim=0)
    features = pretrained_vgg_model(input_tensor)
    print("FEATURES shape", features.shape)
    return features

# Example for extracting vgg features 
# img = Image.open('/Users/iamariyap/Desktop/sem3/PredictiveML/Project/code/PMLProject/src/city2_hazy.png')
# features = extract_vgg_features(img)
# print(features)
# print(features.shape)
# print(img.size)
