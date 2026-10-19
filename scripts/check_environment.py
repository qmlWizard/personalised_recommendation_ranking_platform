import platform
import sys

import numpy as np
import pandas as pd
import sklearn
import torch


def main() -> None:
    print("Python:", sys.version.split()[0])
    print("Architecture:", platform.machine())
    print("PyTorch:", torch.__version__)
    print("NumPy:", np.__version__)
    print("Pandas:", pd.__version__)
    print("Scikit-learn:", sklearn.__version__)

    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())

    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple GPU acceleration is unavailable. Check the Mac architecture, "
            "macOS version, PyTorch installation, and Xcode command-line tools."
        )

    device = torch.device("mps")

    # Verify that tensor operations execute on the MPS device.
    x = torch.randn(2048, 2048, device=device)
    y = x @ x.T
    loss = y.mean()
    loss.backward() if loss.requires_grad else None

    torch.mps.synchronize()

    print("Device:", x.device)
    print("Tensor shape:", tuple(y.shape))
    print("Result finite:", torch.isfinite(y).all().item())
    print("MPS smoke test: PASS")


if __name__ == "__main__":
    main()