import torch.nn as nn

class SignClassifier(nn.Module):
    def __init__(self, input_size=225, hidden_size=128, num_classes=9, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                             batch_first=True, dropout=0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]        
        return self.fc(out)        