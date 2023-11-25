import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim

# Splitting the dataset
total_size = len(fire_spread_data)
train_size = int(0.8 * total_size)  # 80% for training
val_size = total_size - train_size  # 20% for validation
train_dataset, val_dataset = random_split(fire_spread_data, [train_size, val_size])

# Creating DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Training function
def train_model(model, train_loader, val_loader, loss_criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()  # Training mode
        for i, (features, labels) in enumerate(train_loader):
            # Forward pass
            outputs = model(features)
            loss = loss_criterion(outputs, labels.unsqueeze(1))

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()  # Validation mode
        with torch.no_grad():
            val_loss = 0
            for features, labels in val_loader:
                outputs = model(features)
                val_loss += loss_criterion(outputs, labels.unsqueeze(1)).item()

        val_loss /= len(val_loader)
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')

# Initialize the model
model = BushfireModel(n=4, hidden_layers=[500,100])
loss_criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Train the model
num_epochs = 10
train_model(model, train_loader, val_loader, loss_criterion, optimizer, num_epochs)
