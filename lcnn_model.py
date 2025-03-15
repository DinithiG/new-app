import torch
import torch.nn as nn

class LCNNModel(nn.Module):
    def __init__(self):
        super(LCNNModel, self).__init__()
        
        # Define LCNN architecture
        self.conv1 = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm1d(512)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        # Feature dimension will depend on input length
        self.adaptive_pool = nn.AdaptiveAvgPool1d(16)  # Ensure fixed output size
        self.fc1 = nn.Linear(512 * 16, 512)
        self.fc2 = nn.Linear(512, 1)
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # Forward pass through the network
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Adaptive pooling to ensure fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        features = self.relu(self.fc1(x))
        output = self.sigmoid(self.fc2(features))
        
        return output, features
    
    def extract_features(self, x):
        # Forward pass to extract features before the final classification layer
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Adaptive pooling to ensure fixed size
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Get features from the penultimate layer
        features = self.relu(self.fc1(x))
        
        return features