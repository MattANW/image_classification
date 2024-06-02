import os
import json
import torch
import numpy as np
from torch import Tensor, optim, nn
from model import SimpleCNN
from typing import Iterator, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import random
from PIL import Image
import cv2

def get_folders(source_dir: str) -> list[str]:
    """ Returns list of folders in the source directory """
    return os.listdir(source_dir)

def get_dict(json_dir: str) -> dict[str, int]:
    """ Returns dictionary of given json file """
    
    with open(json_dir, 'r') as source:
        return json.load(source)

def get_dict_len(json_dir: str) -> int:
    """ Returns length of dictionary in json file """
    
    with open(json_dir, 'r') as source:
        try:
            return len(json.load(source))
        except Exception:
            return 0
        
def inverse_list(json_dir: str, listX: list[str]) -> list[int]:
    """ Returns list of inversed values from given list """
    
    dictionary = get_dict(json_dir)
    return [dictionary.get(label, 0) for label in listX]

def load_and_resize_image(source_dir: str, image_dir: str, resize: Tuple[int, int]):
    """ Returns resized image """
    image_path = source_dir + '/' + image_dir
    image = cv2.imread(image_path)
    image = cv2.resize(image, resize)
    return image

def load_images(source_dir: str, resize: Tuple[int, int]) -> list:
    """ Returns np_list of images from the source directory with a given size """
    
    source = os.listdir(source_dir)
    images = []
    def process_image(image):
        img = load_and_resize_image(source_dir, image, resize)
        return img
    
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(process_image, source))
    
    # Filter out any None images that failed to load
    images = [img for img in images if img is not None]
    
    return images

def create_array(source_dir: str, resize: Tuple[int, int]) -> list:
    """ Returns a list with each image without bounds """
    
    final_list = []
    folders = get_folders(source_dir)
    
    for folder in folders:
        images = load_images(source_dir + "/" + folder, resize)
        final_list.append(images)
        
    return final_list

def add_to_dict(source_dir: str, json_dir: str, x: int = 0):
    """ Checks for new classes and adds new ones to the dictionary """
    
    folders = get_folders(source_dir)
    
    if os.path.exists(json_dir):
        if os.stat(json_dir).st_size == 0:
            existing_dict = {}
        else:
            with open(json_dir, 'r') as reader:
                existing_dict = json.load(reader)
    else:
        existing_dict = {}
        
    for folder in folders:
        if folder not in existing_dict:
            existing_dict[folder] = x
            x += 1
    
    with open(json_dir, "w") as writer:
        json.dump(existing_dict, writer)

def create_features_and_labels(source_dir: str, json_dir: str, resize: Tuple[int, int]) -> Tuple:
    """ Returns tuple: features and labels for given source directory and dictionary (json directory) """
    
    image_list = create_array(source_dir, resize)
    folders = get_folders(source_dir)
    
    features = []
    labels = []
    
    for idx, images in enumerate(image_list):
        for image in images:
            features.append(image)
            labels.append(folders[idx])
    
    labels = inverse_list(json_dir, labels)
    
    
    labels = torch.tensor(labels, dtype=torch.long, device=device)
    features = torch.tensor(features, dtype=torch.float32, device=device).permute(0, 3, 1, 2)
    
    return features, labels

def generate_features_and_labels(features_dir: str, labels_dir: str, batch_size: int, image_size: Tuple[int, int] = (512, 512), shuffle_index: bool = True) -> Iterator[Tuple[Tensor, Tensor]]:
    labels_dict: dict[str, int] = get_dict(labels_dir)
    
    index: List[Tuple[str, str]] = []
    for folder in os.listdir(features_dir):
        label = labels_dict[folder]
        index.extend([(os.path.join(folder, image_path), label) for image_path in os.listdir(os.path.join(features_dir, folder))])
    
    if shuffle_index:
        random.shuffle(index)
    
    num_batches = len(index) // batch_size
    for batch_num in range(num_batches):
        feature_paths, labels = zip(*index[batch_num*batch_size:(batch_num+1)*batch_size])
        features = [load_and_resize_image(features_dir, path, image_size) for path in feature_paths]
        
        yield torch.tensor(features, dtype=torch.float32, device=device).permute(0, 3, 1, 2), torch.tensor(labels, dtype=torch.long, device=device)
        
    feature_paths, labels = zip(*index[num_batches*batch_size:])
    features = [load_and_resize_image(os.path.join(features_dir, path) for path in feature_paths)]
        
    yield  torch.tensor(features, dtype=torch.float32, device=device).permute(0, 3, 1, 2), torch.tensor(labels, dtype=torch.long, device=device)

def classify_image(model: nn.Module, image_path: str, json_dir: str, resize: Tuple[int, int], device: str) -> str:
    """ Classifies a single image and returns the predicted class label. """
    
    image = Image.open(image_path)
    image = image.resize(resize)
    image = np.array(image).astype(np.float32)
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        output = model(image)
        predicted_idx = output.argmax(dim=1).item()
    
    label_dict = {v: k for k, v in get_dict(json_dir).items()}
    predicted_label = label_dict.get(predicted_idx, "Unknown")
    
    return predicted_label

def save_model(model, path: str):
    """ Saves the model to the specified path. """
    
    torch.save(model.state_dict(), path)

image_size = (256, 256)
_images_folder = 'old_images'
_json_directory = 'labels.json'
_input_size = 3
_learning_rate = 3e-6

device = 'cuda' if torch.cuda.is_available() else 'cpu'
epochs = 100
batch_size = 28
num_classes = get_dict_len(_json_directory)

if __name__ == '__main__':
    #add_to_dict(_images_folder, _json_directory, get_dict_len(_json_directory))
    features, labels = create_features_and_labels(_images_folder, _json_directory, image_size)
    model = SimpleCNN(_input_size, num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=_learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    epochs = epochs
    batch_size = batch_size
    steps = features.shape[0] // batch_size
    
    for epoch in range(epochs):
        perm = torch.randperm(features.shape[0])
        train_features, train_labels = features[perm], labels[perm]
        #for step, batch in enumerate(generate_features_and_labels(_images_folder, _json_directory, batch_size, image_size=image_size)):
        for step in range(steps):
            opt.zero_grad()
            batch_features = train_features[step * batch_size:(step+1) * batch_size].to(device)
            batch_labels = train_labels[step * batch_size:(step+1) * batch_size].to(device)
            
            #batch_features, batch_labels = batch
            
            output = model(batch_features)
            
            loss = criterion(output, batch_labels) 
            
            loss.backward()
            opt.step()
            
            accuracy = (output.argmax(dim=-1) == batch_labels).float().mean().item() * 100
            
            print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.detach()}, Accuracy: {accuracy}")
        
        if epoch % 10 == 0:
            print(classify_image(model, 'old_images/chimpamzee/andrey-tikhonovskiy-QQj9477Apog-unsplash.jpg', _json_directory, image_size, device))
            