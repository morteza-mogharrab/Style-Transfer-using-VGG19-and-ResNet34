import torch
import torch.optim as optim
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, models
from torchvision.models.feature_extraction import create_feature_extractor

# Load pre-trained ResNet34 model
resnet = models.resnet34(weights="IMAGENET1K_V1", progress=True)

# Freeze ResNet parameters
for param in resnet.parameters():
    param.requires_grad_(False)

# Set device
device = torch.device("cuda" if torch.backends.cuda.is_built() else "cpu")
resnet.to(device)

# Function to load and preprocess images
def load_image(img_path, max_size=100, shape=None):
    if "http" in img_path:
        response = requests.get(img_path)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(img_path).convert('RGB')

    if max(image.size) > max_size:
        size = max_size
    else:
        size = max(image.size)

    if shape is not None:
        size = shape

    in_transform = transforms.Compose([
                        transforms.Resize(size),
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406),
                                             (0.229, 0.224, 0.225))])

    image = in_transform(image)[:3,:,:].unsqueeze(0)

    return image.to(device)

# Load content and style images
content = load_image('Content.jpg')
style = load_image('Style.jpg', shape=content.shape[-2:])

# Function to convert tensor to image
def tensor_to_image(tensor):
    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze()
    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
    image = image.clip(0, 1)
    return image

# Display content and style images
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
ax1.imshow(tensor_to_image(content))
ax2.imshow(tensor_to_image(style))

# Function to extract features from model
def get_features(image, model, return_nodes=None):
    if return_nodes is None:
        return_nodes = {
            'conv1': 'conv1_1',
            'layer1.2.conv2': 'conv2_1',
            'layer2.3.conv2': 'conv3_1',
            'layer3.0.conv2': 'conv4_1',
            'layer3.5.conv2': 'conv4_2',
            'layer4.0.conv2': 'conv5_1'
        }
    feature_extractor = create_feature_extractor(model, return_nodes=return_nodes)
    features = feature_extractor(image)
    return features

# Function to calculate Gram matrix
def gram_matrix(tensor):
    b, d, h, w = tensor.size()
    tensor = tensor.view(b * d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram

# Get content and style features
content_features = get_features(content, resnet)
style_features = get_features(style, resnet)
style_grams = {layer: gram_matrix(style_features[layer]) for layer in style_features}

# Define weights
style_weights = {'conv1_1': 1., 'conv2_1': 0.75, 'conv3_1': 0.2, 'conv4_1': 0.2, 'conv5_1': 0.2}
content_weight = 1
style_weight = 1

# Optimization setup
target = content.clone().requires_grad_(True).to(device)
optimizer = optim.Adam([target.requires_grad_()], lr=0.003)
show_every = 400
steps = 2000

# Main optimization loop
for ii in range(1, steps+1):
    target_features = get_features(target, resnet)
    content_loss = torch.mean((target_features['conv4_2'] - content_features['conv4_2'])**2)
    style_loss = 0

    for layer in style_weights:
        target_feature = target_features[layer]
        target_gram = gram_matrix(target_feature)
        _, d, h, w = target_feature.shape
        style_gram = style_grams[layer]
        layer_style_loss = style_weights[layer] * torch.mean((target_gram - style_gram)**2)
        style_loss += layer_style_loss / (d * h * w)

    total_loss = content_weight * content_loss + style_weight * style_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    if ii % show_every == 0:
        print('Total loss: ', total_loss.item())
        plt.imshow(tensor_to_image(target))
        plt.show()

# Display content, style, and final target image
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 10))
ax1.imshow(tensor_to_image(content))
ax2.imshow(tensor_to_image(style))
ax3.imshow(tensor_to_image(target))
plt.show()
