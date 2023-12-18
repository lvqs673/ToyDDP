import torch
import torch.nn as nn
from torch import Tensor
from config import *


class Model(nn.Module):
    def __init__(
        self,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        input_size=INPUT_SIZE,
        output_size=OUTPUT_SIZE,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers, batch_first=True, bidirectional=True
        )
        self.dense = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x: Tensor) -> Tensor:
        out, (h, c) = self.lstm(x)
        out = self.dense(out)
        return out

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = Model()
    xs = torch.randn(3, INPUT_SIZE)
    ys = model.forward(xs)
    print(ys.shape)
