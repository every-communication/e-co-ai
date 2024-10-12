import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18

# CNN-LSTM Model for Keypoints Data
class CNN_LSTM(nn.Module):
    def __init__(self, input_size=246 * 21 * 2, hidden_size=256, num_classes=60, num_layers=2):
        super(CNN_LSTM, self).__init__()
        
        self.conv1d_1 = nn.Conv1d(in_channels=input_size, out_channels=64, kernel_size=3, padding=1)
        self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        self.fc1 = nn.Linear(hidden_size, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x.shape은 (batch_size, seq_length, num_keypoints, num_joints, coord_dim)
        # keypoints를 (batch_size, seq_length, num_features)로 변환
        x = x.view(x.size(0), x.size(1), -1)  # (batch_size, seq_length, num_features)

        x = x.permute(0, 2, 1)  # (batch_size, num_features, seq_length)
        
        x = F.relu(self.conv1d_1(x))
        x = F.relu(self.conv1d_2(x))
        
        x = x.permute(0, 2, 1)  # (batch_size, seq_length, num_features)
        
        _, (hidden, _) = self.lstm(x)
        
        x = hidden[-1]  # LSTM의 마지막 hidden state 사용
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return F.log_softmax(x, dim=1)
