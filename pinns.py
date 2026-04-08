import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Input Encodings
# ---------------------------------------------------------------------------

class FourierFeatureEmbedding(nn.Module):
    """Random Fourier Feature embedding: x -> [sin(Bx), cos(Bx)]"""

    def __init__(self, input_dim: int, embedding_dim: int, scale: float = 1.0, trainable: bool = False):
        super().__init__()
        B = torch.randn(input_dim, embedding_dim) * scale
        if trainable:
            self.B = nn.Parameter(B)
        else:
            self.register_buffer("B", B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        proj = x @ self.B                          # (..., embedding_dim)
        return torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)  # (..., 2*embedding_dim)

    @property
    def out_dim(self) -> int:
        return self.B.shape[1] * 2


class PositionalEncoding(nn.Module):
    """NeRF-style positional encoding: x -> [x, sin(2^0 pi x), cos(2^0 pi x), ..., sin(2^(L-1) pi x), cos(2^(L-1) pi x)]"""

    def __init__(self, input_dim: int, num_frequencies: int = 6, include_input: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.include_input = include_input

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [x] if self.include_input else []
        for k in range(self.num_frequencies):
            freq = (2 ** k) * math.pi
            parts.append(torch.sin(freq * x))
            parts.append(torch.cos(freq * x))
        return torch.cat(parts, dim=-1)

    @property
    def out_dim(self) -> int:
        base = self.input_dim if self.include_input else 0
        return base + 2 * self.num_frequencies * self.input_dim


# ---------------------------------------------------------------------------
# Activation helpers
# ---------------------------------------------------------------------------

def get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "tanh":
        return nn.Tanh()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "swish" or name == "silu":
        return nn.SiLU()
    elif name == "sin":
        return Sin()
    else:
        raise ValueError(f"Unknown activation: '{name}'. Choose from tanh, relu, gelu, swish, sin.")


class Sin(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x)


# ---------------------------------------------------------------------------
# PINN Model
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    """
    MLP-based Physics-Informed Neural Network.

    Parameters
    ----------
    layers : list[int]
        Network architecture including input and output dims.
        e.g. [3, 64, 64, 64, 64, 64, 1]
    activation : str
        Activation function name. One of: tanh, relu, gelu, swish, sin.
    residual : bool
        Enable residual connections between hidden layers of the same width.
    skip : bool
        Enable skip connections: encoded input is concatenated to every hidden layer input.
    use_fourier : bool
        Enable Fourier Feature Embedding on the input.
    fourier_dim : int
        Number of Fourier features per input dimension (output will be 2x this).
    fourier_scale : float
        Standard deviation of random Fourier matrix B.
    fourier_trainable : bool
        Whether B is a learnable parameter.
    use_positional : bool
        Enable NeRF-style Positional Encoding on the input.
    positional_frequencies : int
        Number of frequency bands for positional encoding.
    use_ntk : bool
        Apply NTK-style layer-wise weight normalization (scales weights by 1/sqrt(fan_in)).
    """

    def __init__(
        self,
        layers: list,
        activation: str = "tanh",
        residual: bool = False,
        skip: bool = False,
        use_fourier: bool = False,
        fourier_dim: int = 128,
        fourier_scale: float = 1.0,
        fourier_trainable: bool = False,
        use_positional: bool = False,
        positional_frequencies: int = 6,
        use_ntk: bool = False,
    ):
        super().__init__()
        assert len(layers) >= 2, "layers must have at least input and output dims."
        assert not (use_fourier and use_positional), "use_fourier and use_positional cannot both be True."

        self.residual = residual
        self.skip = skip
        self.use_ntk = use_ntk
        self.layers_cfg = layers
        self.loss_history: dict = {"total": [], "pde": [], "bc": [], "ic": []}

        input_dim = layers[0]

        # --- Input encoding ---
        self.encoder = None
        if use_fourier:
            self.encoder = FourierFeatureEmbedding(input_dim, fourier_dim, fourier_scale, fourier_trainable)
            first_dim = self.encoder.out_dim
        elif use_positional:
            self.encoder = PositionalEncoding(input_dim, positional_frequencies)
            first_dim = self.encoder.out_dim
        else:
            first_dim = input_dim

        # --- Skip connection projection (encode input -> hidden width) ---
        hidden_dim = layers[1]
        if skip:
            self.skip_proj = nn.Linear(first_dim, hidden_dim)

        # --- MLP layers ---
        self.act = get_activation(activation)
        self.linears = nn.ModuleList()

        in_dim = first_dim
        for out_dim in layers[1:-1]:
            if skip and in_dim != first_dim:
                # hidden layers receive concatenation of previous output + skip
                linear_in = out_dim + hidden_dim  # prev hidden + skip projected
            else:
                linear_in = in_dim
            self.linears.append(nn.Linear(linear_in if (skip and in_dim != first_dim) else in_dim, out_dim))
            in_dim = out_dim

        self.output_layer = nn.Linear(layers[-2], layers[-1])

        if use_ntk:
            self._apply_ntk_init()

    def _apply_ntk_init(self):
        """Scale weights by 1/sqrt(fan_in) for NTK parameterization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fan_in = m.weight.shape[1]
                nn.init.normal_(m.weight, mean=0.0, std=1.0 / math.sqrt(fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input encoding
        if self.encoder is not None:
            z = self.encoder(x)
        else:
            z = x

        # Skip connection: project encoded input once
        if self.skip:
            skip_vec = self.act(self.skip_proj(z))

        h = z
        for i, linear in enumerate(self.linears):
            if self.skip and i > 0:
                h = torch.cat([h, skip_vec], dim=-1)
            h_new = self.act(linear(h))

            # Residual: only when dimensions match
            if self.residual and h.shape == h_new.shape:
                h_new = h_new + h

            h = h_new

        return self.output_layer(h)

    # -----------------------------------------------------------------------
    # fit()
    # -----------------------------------------------------------------------

    def _compute_ntk_weights(
        self,
        loss_pde: torch.Tensor,
        loss_bc: torch.Tensor,
        loss_ic: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        has_ic: bool,
    ) -> tuple:
        """
        Compute adaptive loss weights based on gradient norms (Wang et al. 2022).
        w_bc = mean|grad(L_pde)| / mean|grad(L_bc)|
        w_ic = mean|grad(L_pde)| / mean|grad(L_ic)|
        PDE is used as the reference (w_pde = 1.0).
        """
        def grad_norm(loss: torch.Tensor) -> torch.Tensor:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            norms = [p.grad.detach().norm() for p in self.parameters() if p.grad is not None]
            optimizer.zero_grad()
            return torch.stack(norms).mean() if norms else torch.tensor(1.0)

        norm_pde = grad_norm(loss_pde)
        norm_bc = grad_norm(loss_bc)
        w_bc = (norm_pde / (norm_bc + 1e-8)).item()

        if has_ic:
            norm_ic = grad_norm(loss_ic)
            w_ic = (norm_pde / (norm_ic + 1e-8)).item()
        else:
            w_ic = 1.0

        return 1.0, w_bc, w_ic

    def fit(
        self,
        pde_fn,
        x_pde: torch.Tensor,
        x_bc: torch.Tensor,
        u_bc: torch.Tensor,
        x_ic: torch.Tensor = None,
        u_ic: torch.Tensor = None,
        x_ic_ut: torch.Tensor = None,
        ut_ic: torch.Tensor = None,
        epochs: int = 10000,
        lr: float = 1e-3,
        w_pde: float = 1.0,
        w_bc: float = 1.0,
        w_ic: float = 1.0,
        w_ic_ut: float = 1.0,
        ntk_adaptive: bool = False,
        ntk_update_every: int = 1000,
        log_every: int = 500,
        save_path: str = "./best_model.pt",
        optimizer_cls=None,
    ):
        """
        Train the PINN.

        Parameters
        ----------
        pde_fn : callable
            Function that takes (model, x) and returns the PDE residual tensor.
            x requires grad. Example: lambda model, x: u_xx + u_yy  (Laplacian)
        x_pde : Tensor [N, d]
            Collocation points inside the domain.
        x_bc : Tensor [M, d]
            Boundary condition input points.
        u_bc : Tensor [M, out]
            Boundary condition target values.
        x_ic : Tensor [K, d], optional
            Initial displacement condition points (t=0).
        u_ic : Tensor [K, out], optional
            Initial displacement target values.
        x_ic_ut : Tensor [K, d], optional
            Points for initial velocity condition (t=0). Required for 2nd-order
            time PDEs (e.g. wave equation) where du/dt(x,0) is prescribed.
        ut_ic : Tensor [K, out], optional
            Initial velocity target values. Typically zeros for rest initial state.
        epochs : int
            Number of training iterations.
        lr : float
            Learning rate.
        w_pde, w_bc, w_ic, w_ic_ut : float
            Initial loss weights (overridden by NTK adaptive if enabled).
        ntk_adaptive : bool
            Enable NTK-based adaptive loss weighting (Wang et al. 2022).
        ntk_update_every : int
            How often (in epochs) to recompute adaptive weights.
        log_every : int
            Print loss every n epochs.
        save_path : str
            Path to save best model weights.
        optimizer_cls : optional
            Custom optimizer class. Defaults to Adam.
        """
        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        optimizer = optimizer_cls(self.parameters(), lr=lr)

        best_loss = float("inf")
        has_ic    = x_ic is not None and u_ic is not None
        has_ic_ut = x_ic_ut is not None and ut_ic is not None
        self.loss_history = {
            "total": [], "pde": [], "bc": [], "ic": [], "ic_ut": [],
            "w_bc": [], "w_ic": [],
        }

        for epoch in range(1, epochs + 1):
            self.train()

            # --- PDE loss ---
            x_pde_ = x_pde.clone().requires_grad_(True)
            res = pde_fn(self, x_pde_)
            loss_pde = (res ** 2).mean()

            # --- BC loss ---
            u_bc_pred = self(x_bc)
            loss_bc = nn.functional.mse_loss(u_bc_pred, u_bc)

            # --- IC displacement loss ---
            if has_ic:
                u_ic_pred = self(x_ic)
                loss_ic = nn.functional.mse_loss(u_ic_pred, u_ic)
            else:
                loss_ic = torch.tensor(0.0, device=x_pde.device)

            # --- IC velocity loss (2nd-order time PDEs) ---
            if has_ic_ut:
                x_ic_ut_ = x_ic_ut.clone().requires_grad_(True)
                u_at_ic  = self(x_ic_ut_)
                # du/dt at t=0: gradient w.r.t. the time column
                u_t_at_ic = torch.autograd.grad(
                    u_at_ic, x_ic_ut_,
                    grad_outputs=torch.ones_like(u_at_ic),
                    create_graph=True,
                )[0][:, -1:]                          # last column = t
                loss_ic_ut = nn.functional.mse_loss(u_t_at_ic, ut_ic)
            else:
                loss_ic_ut = torch.tensor(0.0, device=x_pde.device)

            # --- NTK adaptive weight update ---
            if ntk_adaptive and epoch % ntk_update_every == 0:
                w_pde, w_bc, w_ic = self._compute_ntk_weights(
                    loss_pde, loss_bc, loss_ic, optimizer, has_ic
                )

            optimizer.zero_grad()
            total = (w_pde * loss_pde
                     + w_bc * loss_bc
                     + w_ic * loss_ic
                     + w_ic_ut * loss_ic_ut)
            total.backward()
            optimizer.step()

            # --- History ---
            self.loss_history["total"].append(total.item())
            self.loss_history["pde"].append(loss_pde.item())
            self.loss_history["bc"].append(loss_bc.item())
            self.loss_history["ic"].append(loss_ic.item())
            self.loss_history["ic_ut"].append(loss_ic_ut.item())
            self.loss_history["w_bc"].append(w_bc)
            self.loss_history["w_ic"].append(w_ic)

            # --- Best model save ---
            if total.item() < best_loss:
                best_loss = total.item()
                torch.save(self.state_dict(), save_path)

            # --- Logging ---
            if epoch % log_every == 0 or epoch == 1:
                ic_str    = f"  IC: {loss_ic.item():.3e}"    if has_ic    else ""
                ic_ut_str = f"  IC_ut: {loss_ic_ut.item():.3e}" if has_ic_ut else ""
                ntk_str   = f"  [w_bc={w_bc:.2f} w_ic={w_ic:.2f}]" if ntk_adaptive else ""
                print(
                    f"[{epoch:>6}/{epochs}]  "
                    f"Total: {total.item():.3e}  "
                    f"PDE: {loss_pde.item():.3e}  "
                    f"BC: {loss_bc.item():.3e}"
                    f"{ic_str}{ic_ut_str}{ntk_str}"
                )

        # --- Restore best weights ---
        self.load_state_dict(torch.load(save_path, weights_only=True))
        print(f"\nTraining complete. Best total loss: {best_loss:.3e} | weights restored from '{save_path}'")
