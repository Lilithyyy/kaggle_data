import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import random
import sys
import time
from torchvision import datasets
import torch
import torchvision
import torchvision.transforms as T   
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import os
from PIL import Image
from torch.utils.data import DataLoader, random_split, Dataset

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device

class AnimalDataset(Dataset):
    def __init__(self, root_dir, labels_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.labels_df = labels_df
        
        # Create a mapping from image path to category
        self.image_to_category = {}
        for idx, row in self.labels_df.iterrows():
            img_path = row['image_name']
            category = row['category']
            self.image_to_category[img_path] = category
        
        # Get all image files
        self.image_files = []
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for img_file in os.listdir(folder_path):
                    if img_file.endswith('.jpg') or img_file.endswith('.png'):
                        img_path = os.path.join(folder, img_file)
                        if img_path in self.image_to_category:
                            self.image_files.append(img_path)
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        full_path = os.path.join(self.root_dir, img_path)
        image = Image.open(full_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Get the category label
        category = self.image_to_category[img_path]
        
        return image, category
    
# Define the preprocessing transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

############################################################
dataset_path = "/animals/animals/animals" #-- treba specifikovat cestu k datam
labels_df = pd.read_csv("/labels.csv") # -||-

# Encode categorical labels
encoder = LabelEncoder()
labels_df['category'] = encoder.fit_transform(labels_df['category'])

# Create your custom dataset
custom_dataset = AnimalDataset(dataset_path, labels_df, transform=transform)

labels_df['category'].nunique() # 11 unique broad categories

train_size = int(0.8 * len(custom_dataset))
val_size = int(0.1 * len(custom_dataset))
test_size = len(custom_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(custom_dataset, [train_size, val_size, test_size])

batch_size = 32

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

for images, labels in train_loader:
    print(f"Labels in batch: {labels}")
    print(f"Min label: {labels.min()}, Max label: {labels.max()}")
    break

# Check batch shapes
for X, y in train_loader:
    print("Train Loader:")
    print("Batch images shape:", X.shape)  
    print("Batch labels shape:", y.shape, '\n')  
    break  
    
for X, y in val_loader:
    print("Validation Loader:")
    print("Batch images shape:", X.shape)  
    print("Batch labels shape:", y.shape, '\n')  
    break  

for X, y in test_loader:
    print("Test Loader:")
    print("Batch images shape:", X.shape)  
    print("Batch labels shape:", y.shape, '\n')  
    break

# Print dataset sizes
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of testing samples: {len(test_dataset)}")

weights = torchvision.models.ResNet50_Weights.DEFAULT
resnet = torchvision.models.resnet50(weights=weights)

summary(resnet.to(device),(3,224,224))

# Freeze all layers
for p in resnet.parameters():
    p.requires_grad = False

# Update final layer to match the 11 categories
num_classes = 11  
resnet.fc = nn.Linear(resnet.fc.in_features, num_classes)

resnet.to(device)

lr = 0.001
l2 = 1e-3
lossfun = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=lr, weight_decay=l2)

def function2traintestTheModel():
    numepochs = 10
     
    # Initialize losses / accuracies
    trainLoss = torch.zeros(numepochs)
    valLoss = torch.zeros(numepochs)
    testLoss = torch.zeros(numepochs)
    trainAcc = torch.zeros(numepochs)
    valAcc = torch.zeros(numepochs)
    testAcc = torch.zeros(numepochs)
    
    start_time = time.time()
    for epochi in range(numepochs):
        epoch_start_time = time.time()
        
        # Train mode
        resnet.train()
        
        # Training
        batch_train_loss = []
        batch_train_acc = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            yHat = resnet(X)
            loss = lossfun(yHat, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_train_loss.append(loss.item())
            batch_train_acc.append(torch.mean((torch.argmax(yHat, axis=1) == y).float()).item())
        
        trainLoss[epochi] = np.mean(batch_train_loss)
        trainAcc[epochi] = 100 * np.mean(batch_train_acc)
        
        # Eval mode
        resnet.eval()
        
        with torch.no_grad():
            # Validation
            batch_val_loss = []
            batch_val_acc = []
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                
                yHat = resnet(X)
                loss = lossfun(yHat, y)
                
                batch_val_loss.append(loss.item())
                batch_val_acc.append(torch.mean((torch.argmax(yHat, axis=1) == y).float()).item())
            
            valLoss[epochi] = np.mean(batch_val_loss)
            valAcc[epochi] = 100 * np.mean(batch_val_acc)
            
            # Test
            batch_test_loss = []
            batch_test_acc = []
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                
                yHat = resnet(X)
                loss = lossfun(yHat, y)
                
                batch_test_loss.append(loss.item())
                batch_test_acc.append(torch.mean((torch.argmax(yHat, axis=1) == y).float()).item())
            
            testLoss[epochi] = np.mean(batch_test_loss)
            testAcc[epochi] = 100 * np.mean(batch_test_acc)
        
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - start_time
        
        # Print progress
        sys.stdout.write(f"\rFinished epoch: {epochi+1}/{numepochs}, time: {epoch_time:.2f}s, total time: {total_time:.2f}s")
    
    print("\nTraining completed!")
    return trainLoss, valLoss, testLoss, trainAcc, valAcc, testAcc, resnet

trainLoss, valLoss, testLoss, trainAcc, valAcc, testAcc, resnet = function2traintestTheModel()

fig, ax = plt.subplots(1, 2, figsize=(16, 5))

ax[0].plot(trainLoss, 's-', label='Train')
ax[0].plot(valLoss, 'o-', label='Validation')
ax[0].plot(testLoss, '^-', label='Test')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss (CrossEntropy)')
ax[0].set_title('Model loss')
ax[0].legend()
ax[0].grid(True)

ax[1].plot(trainAcc, 's-', label='Train')
ax[1].plot(valAcc, 'o-', label='Validation')
ax[1].plot(testAcc, '^-', label='Test')
ax[1].set_xlabel('Epochs')
ax[1].set_ylabel('Accuracy (%)')
ax[1].set_title(f'Final model train / val / test accuracy: {trainAcc[-1]:.2f}% / {valAcc[-1]:.2f}% / {testAcc[-1]:.2f}%')
ax[1].legend()
ax[1].grid(True)

fig.suptitle('Pretrained ResNet-50 on Animal data',fontweight='bold',fontsize=14)
plt.show()

def denormalize(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = image.cpu().numpy().transpose(1, 2, 0)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    return image

# Function to predict and display multiple random images
def predict_multiple_random_images(model, test_dataset, encoder, num_images=5, cols=3):
    model.eval()
    
    # Calculate number of rows needed
    rows = (num_images + cols - 1) // cols
    
    plt.figure(figsize=(cols * 5, rows * 5))
    
    for i in range(num_images):
        # Get a random index
        random_idx = random.randint(0, len(test_dataset) - 1)
        
        # Get the image and label
        image, label = test_dataset[random_idx]
        
        # Get the original image path to extract animal name (folder name)
        img_path = test_dataset.dataset.image_files[test_dataset.indices[random_idx]]
        animal_name = img_path.split('/')[0]  # Getting folder name which is the animal type
        
        # Add batch dimension
        image_batch = image.unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            output = model(image_batch)
            _, predicted_idx = torch.max(output, 1)
            
        # Get the predicted and actual class names
        predicted_class = encoder.inverse_transform([predicted_idx.item()])[0]
        actual_class = encoder.inverse_transform([label])[0]
        
        plt.subplot(rows, cols, i+1)
        plt.imshow(denormalize(image))
        plt.title(f'Animal: {animal_name}\nPredicted: {predicted_class}\nActual: {actual_class}', fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def predict_from_each_category(model, test_dataset, encoder, num_per_category=1):
    model.eval()
    
    # Get all unique animal types in the test dataset
    animal_types = {}
    for idx in test_dataset.indices:
        img_path = test_dataset.dataset.image_files[idx]
        animal_name = img_path.split('/')[0]
        if animal_name not in animal_types:
            animal_types[animal_name] = []
        animal_types[animal_name].append(idx)
    
    # Sort animal types for consistent display
    sorted_animal_types = sorted(animal_types.keys())
    
    # Calculate grid dimensions
    cols = 3
    rows = (len(sorted_animal_types) + cols - 1) // cols
    
    plt.figure(figsize=(cols * 5, rows * 5))
    
    for i, animal_type in enumerate(sorted_animal_types):
        if animal_types[animal_type]:
            # Get a random image of this animal type
            idx = random.choice(animal_types[animal_type])
            image, label = test_dataset[idx]
            
            # Add batch dimension
            image_batch = image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_batch)
                _, predicted_idx = torch.max(output, 1)
                
            # Get the predicted and actual class names
            predicted_class = encoder.inverse_transform([predicted_idx.item()])[0]
            actual_class = encoder.inverse_transform([label])[0]
            
            plt.subplot(rows, cols, i+1)
            plt.imshow(denormalize(image))
            plt.title(f'Animal: {animal_type}\nPredicted: {predicted_class}\nActual: {actual_class}', fontsize=10)
            plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Display random images side by side
predict_multiple_random_images(resnet, test_dataset, encoder, num_images=9, cols=3)