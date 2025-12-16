import torch.nn as nn
import torch


class FFN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super(FFN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs, identity = None) -> torch.Tensor:
        x = self.ffn(inputs)

        if identity is None:
            identity = inputs

        return x + identity

class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int,output_dim: int) -> None:
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, inputs) -> torch.Tensor:
        x = self.mlp(inputs)
        return x

if __name__ == "__main__":
    B = 4
    N = 2
    T = 4
    C = 256
    state_dim = 3 # (x, y, heading)
    hidden_dim = 256

    model  = MLP(input_dim=C, output_dim=state_dim, hidden_dim=hidden_dim)

    queries = torch.randn(B, N, T, C)

    out = model(queries)

    print(out.shape)

