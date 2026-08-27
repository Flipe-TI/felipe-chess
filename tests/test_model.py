"""Testes do PolicyNet: contrato de I/O, ordem do flatten da policy head
(contract-critical), budget de parâmetros e sanidade de aprendizado (overfit
de um mini-batch, BRAINSTORM §8).
"""
import numpy as np
import torch

from felipe_chess.model import PolicyNet, flatten_policy


def test_output_shape_matches_contract():
    net = PolicyNet(channels=32, blocks=2).eval()
    x = torch.zeros(3, 18, 8, 8)
    out = net(x)
    assert out.shape == (3, 4672)


def test_flatten_policy_matches_index_contract():
    # head[n, moveType, rank, file] -> índice (rank*8+file)*73 + moveType.
    head = torch.zeros(1, 73, 8, 8)
    rank, file, move_type = 5, 3, 42
    head[0, move_type, rank, file] = 1.0
    flat = flatten_policy(head)
    assert flat.shape == (1, 4672)
    expected_index = (rank * 8 + file) * 73 + move_type
    assert flat[0, expected_index].item() == 1.0
    assert flat[0].sum().item() == 1.0  # nada mais aceso


def test_param_budget_under_5mb_fp32():
    net = PolicyNet(channels=64, blocks=5)
    n_params = sum(p.numel() for p in net.parameters())
    assert n_params * 4 < 5_000_000, f"{n_params} params -> {n_params*4/1e6:.1f}MB fp32"


def test_forward_is_deterministic_in_eval():
    torch.manual_seed(0)
    net = PolicyNet(channels=16, blocks=1).eval()
    x = torch.randn(2, 18, 8, 8)
    with torch.no_grad():
        a = net(x)
        b = net(x)
    assert torch.allclose(a, b)
    assert torch.isfinite(a).all()


def test_can_overfit_tiny_batch():
    # Sanidade: com poucos exemplos e passos, a rede memoriza (gradientes fluem,
    # capacidade existe). top-1 deve chegar a 100%.
    torch.manual_seed(0)
    net = PolicyNet(channels=32, blocks=2).train()
    x = torch.randn(8, 18, 8, 8)
    y = torch.randint(0, 4672, (8,))
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(net(x), y)
        loss.backward()
        opt.step()
    net.eval()
    with torch.no_grad():
        pred = net(x).argmax(dim=1)
    acc = (pred == y).float().mean().item()
    assert acc == 1.0, f"nao memorizou o mini-batch (acc={acc})"
