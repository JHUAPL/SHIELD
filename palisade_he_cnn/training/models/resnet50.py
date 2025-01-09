# (c) 2021-2024 The Johns Hopkins University Applied Physics Laboratory LLC (JHU/APL).

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import ImageFile
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES = True


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_ids = [0, 1, 2, 3, 4, 5, 6, 7]

# Set hyperparameters
num_epochs = 1
batch_size = 128
learning_rate = 0.001

# Initialize transformations for data augmentation
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(degrees=45),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the ImageNet Object Localization Challenge dataset
train_dataset = torchvision.datasets.ImageFolder(
    root='~/ImageNet/ILSVRC/Data/CLS-LOC/train', 
    transform=transform
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8)

# Load the ResNet50 model
model = torchvision.models.resnet50(weights='DEFAULT')

# Parallelize training across multiple GPUs
model = torch.nn.DataParallel(model, device_ids = device_ids)

# Set the model to run on the device
model = model.to(f'cuda:{model.device_ids[0]}')

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model...
for epoch in range(num_epochs):
    for inputs, labels in tqdm(train_loader):
        # Move input and label tensors to the device
        inputs = inputs.to(f'cuda:{model.device_ids[0]}')
        labels = labels.to(f'cuda:{model.device_ids[0]}')

        # Zero out the optimizer
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        optimizer.step()

    # Print the loss for every epoch
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

print(f'Finished Training, Loss: {loss.item():.4f}')
