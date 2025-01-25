from src.models.modules import *
from src.models.loss import L1_epsilon_lambda
from dataclasses import dataclass
import torch


@dataclass
class SDFTransformerConfig:
    dim_context: int = 4
    dim_input: int = 3
    num_outputs: int = 1
    dim_output: int = 1
    delta: float = 0.1
    dim_hidden: int = 128
    num_hidden_seeds: int = 32
    num_heads: int = 1


class SDFTransformer(nn.Module):
    def __init__(self, config: SDFTransformerConfig):
        super(SDFTransformer, self).__init__()
        self.config = config
        self.epsilon = None
        self.lambdaa = None
        self.proj_x = nn.Linear(config.dim_input, config.dim_hidden)
        self.proj_ctx = nn.Linear(config.dim_context, config.dim_hidden)
        self.pool_ctx = PMA(
            config.dim_hidden, config.num_heads, config.num_hidden_seeds
        )
        self.cross = MAB(
            config.dim_hidden, config.dim_hidden, config.dim_hidden, config.num_heads
        )
        self.dec = nn.Sequential(
            SAB(config.dim_hidden, config.dim_hidden, config.num_heads),
            nn.SiLU(),
            SAB(config.dim_hidden, config.dim_hidden, config.num_heads),
            nn.SiLU(),
            PMA(config.dim_hidden, config.num_heads, config.num_outputs),
            nn.SiLU(),
            nn.Linear(config.dim_hidden, config.dim_hidden),
            nn.ReLU(),
            nn.Linear(config.dim_hidden, config.dim_output),
            nn.Tanh(),
        )

    def forward(
        self, context: torch.Tensor, x: torch.Tensor, labels: torch.Tensor = None
    ):
        x = self.proj_x(x)  # [B, 1, H]
        x = x.expand(-1, self.config.num_hidden_seeds, -1)  # [B, X, H]
        y = self.pool_ctx(self.proj_ctx(context))  # [B, Y, H]
        xy = self.cross(x, y)  # [B, X, H]
        xy = xy + x  # [B, X, H]
        xy = self.dec(xy)

        loss = None
        if labels is not None:
            loss = L1_epsilon_lambda(
                xy, labels, self.epsilon, self.lambdaa, self.config.delta
            )
        return {"loss": loss, "logits": xy}
