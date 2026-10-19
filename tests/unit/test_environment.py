import numpy as np
import torch


def test_numpy_basic_operations():
    values = np.array([1, 2, 3])
    assert values.sum() == 6


def test_torch_tensor_operations():
    values = torch.tensor([1.0, 2.0, 3.0])
    assert torch.isclose(values.sum(), torch.tensor(6.0))


def test_mps_tensor_operations():
    if not torch.backends.mps.is_available():
        return

    device = torch.device("mps")
    values = torch.tensor([1.0, 2.0, 3.0], device=device)
    result = values.sum()

    torch.mps.synchronize()

    assert result.device.type == "mps"
    assert torch.isclose(result.cpu(), torch.tensor(6.0))