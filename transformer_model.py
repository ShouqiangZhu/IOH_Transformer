# model/transformer_model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Injects position information into the input tensor."""
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TransformerVitalSigns(nn.Module):
    """Transformer model for predicting intraoperative hypotension from vital signs."""
    def __init__(self, input_dim, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.conv1d = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.pe = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        x = x.permute(0, 2, 1)          # [batch_size, input_dim, seq_len]
        x = self.conv1d(x)              # [batch_size, d_model, seq_len]
        x = x.permute(0, 2, 1)          # [batch_size, seq_len, d_model]
        x = self.pe(x)                  # Apply positional encoding
        x = self.transformer_encoder(x) # [batch_size, seq_len, d_model]
        x = x.permute(0, 2, 1)          # [batch_size, d_model, seq_len]
        x = self.pool(x).squeeze(-1)    # Global average pooling
        return self.fc(x)               # Output: [batch_size, 1]
