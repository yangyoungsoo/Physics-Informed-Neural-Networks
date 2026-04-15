# Physics-Informed Neural Networks

A modular, PyTorch-based implementation of Physics-Informed Neural Networks (PINNs) with flexible architecture options and built-in training utilities.

## Features

| Feature | Description |
|---|---|
| Flexible architecture | Define any depth/width via a `layers` list |
| Activation functions | `tanh`, `relu`, `gelu`, `swish/silu`, `sin` |
| Fourier Feature Embedding | Random Fourier Features `[sin(Bx), cos(Bx)]` to capture high-frequency patterns |
| Positional Encoding | NeRF-style multi-frequency encoding `[sin(2^k πx), cos(2^k πx)]` |
| Residual connections | Skip-add between same-width hidden layers |
| Skip connections | Encoded input projected and concatenated to every hidden layer |
| NTK initialization | Weight scaling by `1/sqrt(fan_in)` at init |
| NTK adaptive weighting | Automatic loss-weight tuning via gradient norms (Wang et al. 2022) |
| `fit()` method | Built-in training loop with BC / IC / IC-velocity support |
| Loss history | `model.loss_history` dict for post-training analysis |
| Best-model checkpointing | Saves and auto-restores the lowest-loss weights |

## Quickstart

```python
import torch
from pinns import PINN

# Define model: input dim=2 (x, t), 4 hidden layers of 128, output dim=1
model = PINN(
    layers=[2, 128, 128, 128, 128, 1],
    activation="tanh",
    skip=True,
    use_fourier=False,
)

# Define PDE residual (e.g. heat equation: u_t - u_xx = 0)
def pde_fn(model, xt):
    u = model(xt)
    grads = torch.autograd.grad(u, xt, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = grads[:, 0:1]
    u_t = grads[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
    return u_t - u_xx

# Train
model.fit(
    pde_fn=pde_fn,
    x_pde=x_pde,       # collocation points [N, 2]
    x_bc=x_bc,         # boundary points [M, 2]
    u_bc=u_bc,         # boundary values [M, 1]
    x_ic=x_ic,         # initial condition points [K, 2]
    u_ic=u_ic,         # initial condition values [K, 1]
    epochs=10000,
    lr=1e-3,
    save_path="best_model.pt",
)
```

## PINN Constructor Options

```python
PINN(
    layers,                     # e.g. [2, 64, 64, 64, 1]
    activation="tanh",          # tanh | relu | gelu | swish | sin
    residual=False,             # residual connections
    skip=False,                 # skip connections (input concat to every layer)
    use_fourier=False,          # Fourier Feature Embedding
    fourier_dim=128,            # features per input dim (output = 2x)
    fourier_scale=1.0,          # std of random matrix B
    fourier_trainable=False,    # make B a learnable parameter
    use_positional=False,       # NeRF-style Positional Encoding
    positional_frequencies=6,   # number of frequency bands
    use_ntk=False,              # NTK weight initialization
)
```

> `use_fourier` and `use_positional` are mutually exclusive.

## `fit()` Parameters

| Parameter | Type | Description |
|---|---|---|
| `pde_fn` | callable | `(model, x) → residual` — PDE residual function |
| `x_pde` | Tensor `[N, d]` | Collocation points inside the domain |
| `x_bc`, `u_bc` | Tensor | Boundary condition points and targets |
| `x_ic`, `u_ic` | Tensor (optional) | Initial displacement condition |
| `x_ic_ut`, `ut_ic` | Tensor (optional) | Initial velocity condition — for 2nd-order time PDEs (e.g. wave equation) |
| `w_pde/w_bc/w_ic/w_ic_ut` | float | Manual loss weights |
| `ntk_adaptive` | bool | Enable NTK adaptive loss weighting |
| `ntk_update_every` | int | Recompute adaptive weights every N epochs |
| `epochs`, `lr` | int, float | Training epochs and learning rate |
| `log_every` | int | Print interval |
| `save_path` | str | Path to save best model weights |
| `optimizer_cls` | class | Custom optimizer (default: `Adam`) |

## Example Notebooks

### [reaction_1d.ipynb](reaction_1d.ipynb) — 1D Reaction Equation

$$\frac{\partial u}{\partial t} - \rho\, u(1-u) = 0, \quad x \in [0, 2\pi],\ t \in [0,1], \quad \rho = 5$$

- Architecture: `[2, 128, 128, 128, 128, 1]` with skip connections
- Periodic BC enforced via supervised boundary points
- Compares PINN prediction to the analytical solution via heatmap and time slices

### [wave_1d.ipynb](wave_1d.ipynb) — 1D Wave Equation

$$\frac{\partial^2 u}{\partial t^2} - \beta\, \frac{\partial^2 u}{\partial x^2} = 0, \quad x \in [0,1],\ t \in [0,1], \quad \beta = 3$$

- Architecture: `[2, 256, 256, 256, 256, 256, 1]` with skip connections + Fourier features
- Two initial conditions: displacement `u(x,0)` and velocity `∂u/∂t(x,0) = 0`
- Second derivatives computed with `torch.autograd.grad`

## Requirements

```
torch
numpy
matplotlib
```

Install with:

```bash
pip install torch numpy matplotlib
```

## References

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks. *Journal of Computational Physics*, 378, 686–707.
- Wang, S., Teng, Y., & Perdikaris, P. (2021). Understanding and mitigating gradient pathologies in physics-informed neural networks. *SIAM Journal on Scientific Computing*.
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. *Journal of Computational Physics*.

## License

MIT License — see [LICENSE](LICENSE) for details.

