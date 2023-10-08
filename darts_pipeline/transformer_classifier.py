import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length):
        super(PositionalEncoding, self).__init__()

        position = torch.arange(0, max_seq_length, dtype=torch.float64).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))

        pe = torch.zeros(max_seq_length, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe


class TransformerClassifier(nn.Module):
    """
    Text classifier based on a pytorch TransformerEncoder. Give (T, N, C) shape!
    """

    def __init__(
            self,
            num_classes,
            input_dim,
            input_chunk_length,
            d_model=64,
            nhead=8,
            dim_feedforward=2048,
            num_layers=6,
            dropout=0.1,
            activation="relu",
            classifier_dropout=0.1
    ):
        super().__init__()
        self.d_model = d_model
        # num_classes = num_classes - 1
        assert d_model % nhead == 0, "nheads must divide evenly into d_model"

        self.embedder = nn.Linear(input_dim, d_model, bias=False, dtype=torch.float64)
        self.pos_encoder = PositionalEncoding(
            d_model=d_model,
            max_seq_length=input_chunk_length,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,
            dtype=torch.float64
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )
        self.classifier = nn.Linear(d_model * input_chunk_length, num_classes, dtype=torch.float64)
        self.classifier_dropout = nn.Dropout1d(classifier_dropout)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        N, T, C = x.size()
        x = x * math.sqrt(self.d_model)
        x = self.embedder(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = torch.reshape(x, (N, -1))
        x = self.classifier(x)
        x = self.classifier_dropout(x)
        x = self.softmax(x)

        return x

    def simple_ordinal(self, x):
        bigger = F.pad(x, (1, 0), "constant", 1)
        smaller = F.pad(x, (0, 1), "constant", 0)
        return bigger - smaller
