# The adverserial images were stored in google drive.

import os
from datasets import load_dataset
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split


# importing resnet model and default weights
weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights)


# importing the adverserial_tensors stored under the results dir
dv_image_tensors = None
adv_image_labels = None
dir = '/content/drive/MyDrive/results/'
for fl in os.listdir(dir):
  current_tensor = torch.load(dir+fl)
  current_t_im = current_tensor['adv_images']
  current_t_labels = current_tensor['labels']
  if adv_image_tensors is None:
    adv_image_tensors = current_t_im
  else:
    adv_image_tensors = torch.cat((adv_image_tensors, current_t_im), 0)
  if adv_image_labels is None:
    adv_image_labels = current_t_labels
  else:
    adv_image_labels = torch.cat((adv_image_labels, current_t_labels), 0)
  print(f"Done with {fl}")


# defining optimizer and loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
cuda = torch.cuda.is_available()
if cuda:
  model = model.cuda()
  criterion = criterion.cuda()



# creating a dataset of the loaded adverserial images and labels

class CustomDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        return image, label

dataset = CustomDataset(images=adv_image_tensors, labels=adv_image_labels)

#Splitting the dataset into training and testing sets
dataset_size = len(dataset)
test_size = int(0.3 * dataset_size)  # 30% for testing
train_size = dataset_size - test_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])


batch_size = 32 

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)



# training the model with adverserial images

num_epochs = 5 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        
        optimizer.zero_grad()

       
        outputs = model(images).softmax(1)
        loss = criterion(outputs, labels)

        #backprop
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch Loss: {epoch_loss:.4f}")

# Testing the model
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images).to(device)
        predictions = outputs.argmax(dim=1)
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = 100 * correct / total
print(f"Accuracy on test data: {accuracy:.2f}%")