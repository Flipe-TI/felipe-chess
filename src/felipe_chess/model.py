"""S2 — rede de política pequena (CNN residual) para behavior cloning.

Contrato de I/O congelado (não mudar sem alinhar com o site):
  input  board  : float32 [N, 18, 8, 8]
  output policy : float32 [N, 4672]  (logits; sem softmax)

O índice de saída segue o mesmo contrato do encoding:
  index = (rank*8 + file) * 73 + moveType   (fromSquare = rank*8+file)
A ordem do flatten (flatten_policy) é contract-crítica — se errar, o modelo
joga lixo silenciosamente, igual a uma divergência de encoding.

Alvo: rede pequena (<5MB fp32) para rodar no navegador via ONNX. Ver
BRAINSTORM §7.3.
"""
from __future__ import annotations

import torch
import torch.nn as nn

POLICY_SIZE = 4672
INPUT_PLANES = 18
MOVE_TYPES = 73


def flatten_policy(head: torch.Tensor) -> torch.Tensor:
    """[N, 73, 8, 8] -> [N, 4672] na ordem do contrato.

    head[n, moveType, rank, file] -> índice (rank*8 + file)*73 + moveType.
    Permuta para (N, rank, file, moveType) e achata em ordem C.
    """
    n = head.shape[0]
    return head.permute(0, 2, 3, 1).reshape(n, POLICY_SIZE)


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return torch.relu(out + x)


class PolicyNet(nn.Module):
    """CNN residual pequena. input [N,18,8,8] -> policy logits [N,4672]."""

    def __init__(self, channels: int = 64, blocks: int = 5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_PLANES, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[_ResidualBlock(channels) for _ in range(blocks)])
        # Policy head: 1x1 conv para 73 planos (moveType) sobre o grid 8x8.
        self.policy_conv = nn.Conv2d(channels, MOVE_TYPES, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        head = self.policy_conv(x)  # [N, 73, 8, 8]
        return flatten_policy(head)  # [N, 4672]
