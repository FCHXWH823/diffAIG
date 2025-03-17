# difflogic.py
import torch
import difflogic_cuda
import numpy as np
from .functional import get_unique_connections, GradFactor
from .packbitstensor import PackBitsTensor

########################################################################################################################
class LogicLayer(torch.nn.Module):
    """
    AIG-style LogicLayer where every gate is an AND gate with two input edges that can be inverted.
    Each gate learns two parameters (negation_logits) that decide whether to invert each edge.
    """
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            device: str = 'cuda',
            grad_factor: float = 1.,
            implementation: str = None,
            connections: str = 'random',
    ):
        super().__init__()
        # Learn 2 parameters per neuron instead of 16 weights.
        self.negation_logits = torch.nn.Parameter(torch.randn(out_dim, 2, device=device))
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.grad_factor = grad_factor

        # Set up implementation (default to CUDA if available).
        self.implementation = implementation
        if self.implementation is None and device == 'cuda':
            self.implementation = 'cuda'
        elif self.implementation is None and device == 'cpu':
            self.implementation = 'python'
        assert self.implementation in ['cuda', 'python'], self.implementation

        self.connections = connections
        assert self.connections in ['random', 'unique'], self.connections
        self.indices = self.get_connections(self.connections, device)

        if self.implementation == 'cuda':
            # Precompute indices for a fast backward pass.
            given_x_indices_of_y = [[] for _ in range(in_dim)]
            indices_0_np = self.indices[0].cpu().numpy()
            indices_1_np = self.indices[1].cpu().numpy()
            for y in range(out_dim):
                given_x_indices_of_y[indices_0_np[y]].append(y)
                given_x_indices_of_y[indices_1_np[y]].append(y)
            self.given_x_indices_of_y_start = torch.tensor(
                np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(), device=device, dtype=torch.int64)
            self.given_x_indices_of_y = torch.tensor(
                [item for sublist in given_x_indices_of_y for item in sublist],
                dtype=torch.int64, device=device)
        self.num_neurons = out_dim

    def forward(self, x):
        if isinstance(x, PackBitsTensor):
            assert not self.training, 'PackBitsTensor is not supported in training mode.'
            assert self.device == 'cuda', 'PackBitsTensor is only supported on CUDA.'
        else:
            if self.grad_factor != 1.:
                x = GradFactor.apply(x, self.grad_factor)

        if self.implementation == 'cuda':
            if isinstance(x, PackBitsTensor):
                return self.forward_cuda_eval(x)
            return self.forward_cuda(x)
        elif self.implementation == 'python':
            return self.forward_python(x)
        else:
            raise ValueError(self.implementation)

    def forward_python(self, x):
        # x shape: [batch, in_dim]
        a = x[..., self.indices[0]]  # shape: [batch, out_dim]
        b = x[..., self.indices[1]]  # shape: [batch, out_dim]

        # Compute negation probabilities
        p = torch.sigmoid(self.negation_logits)  # shape: [out_dim, 2]
        p_hard = (p > 0.5).float()
        if self.training:
            # Use straight-through estimator: forward uses hard decision but gradients flow.
            p_used = p + (p_hard - p).detach()
        else:
            p_used = p_hard

        p_used = p_used.unsqueeze(0)  # shape: [1, out_dim, 2]
        # Conditionally invert: if flag is 1, invert the bit: 1-a, else a.
        a_mod = torch.where(p_used[..., 0] == 1, 1 - a, a)
        b_mod = torch.where(p_used[..., 1] == 1, 1 - b, b)

        # AND gate: elementwise multiplication.
        out = a_mod * b_mod
        return out

    def forward_cuda(self, x):
        # Similar to Python but calling CUDA kernels.
        if self.training:
            assert x.device.type == 'cuda', x.device
        assert x.ndim == 2, x.ndim
        x = x.transpose(0, 1).contiguous()  # x: [in_dim, batch]
        a, b = self.indices

        # Compute negation probabilities.
        p = torch.sigmoid(self.negation_logits)
        p_hard = (p > 0.5).float()
        if self.training:
            p_used = p + (p_hard - p).detach()
        else:
            p_used = p_hard

        # For the CUDA kernel, pack these as uint8 (0 or 1).
        neg_flags = (p_used > 0.5).to(torch.uint8)
        return LogicLayerCudaFunction.apply(
            x, a, b, neg_flags, self.given_x_indices_of_y_start, self.given_x_indices_of_y
        ).transpose(0, 1)

    def forward_cuda_eval(self, x: PackBitsTensor):
        # For evaluation using PackBitsTensor on CUDA.
        assert not self.training
        assert isinstance(x, PackBitsTensor)
        a, b = self.indices
        neg_flags = (torch.sigmoid(self.negation_logits) > 0.5).to(torch.uint8)
        x.t = difflogic_cuda.eval(x.t, a, b, neg_flags)
        return x

    def extra_repr(self):
        return '{}, {}, {}'.format(self.in_dim, self.out_dim, 'train' if self.training else 'eval')

    def get_connections(self, connections, device='cuda'):
        assert self.out_dim * 2 >= self.in_dim, (
            'Neurons ({}) must not be smaller than half of inputs ({}).'.format(self.out_dim, self.in_dim)
        )
        if connections == 'random':
            c = torch.randperm(2 * self.out_dim) % self.in_dim
            c = torch.randperm(self.in_dim)[c]
            c = c.reshape(2, self.out_dim)
            a, b = c[0], c[1]
            a, b = a.to(torch.int64).to(device), b.to(torch.int64).to(device)
            return a, b
        elif connections == 'unique':
            return get_unique_connections(self.in_dim, self.out_dim, device)
        else:
            raise ValueError(connections)

########################################################################################################################
class GradFactor(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, f):
        ctx.f = f
        return x

    @staticmethod
    def backward(ctx, grad_y):
        return grad_y * ctx.f, None

########################################################################################################################
class LogicLayerCudaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a, b, neg_flags, given_x_indices_of_y_start, given_x_indices_of_y):
        ctx.save_for_backward(x, a, b, neg_flags, given_x_indices_of_y_start, given_x_indices_of_y)
        return difflogic_cuda.forward(x, a, b, neg_flags)
    @staticmethod
    def backward(ctx, grad_y):
        x, a, b, neg_flags, given_x_indices_of_y_start, given_x_indices_of_y = ctx.saved_tensors
        grad_y = grad_y.contiguous()
        grad_neg = grad_x = None
        if ctx.needs_input_grad[0]:
            grad_x = difflogic_cuda.backward_x(x, a, b, neg_flags, grad_y, given_x_indices_of_y_start, given_x_indices_of_y)
        if ctx.needs_input_grad[3]:
            grad_neg = difflogic_cuda.backward_neg(x, a, b, grad_y)
        return grad_x, None, None, grad_neg, None, None
