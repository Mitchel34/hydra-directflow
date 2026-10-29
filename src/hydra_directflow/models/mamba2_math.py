"""Exact attributed Mamba-2 math copied from the prior qualified local implementation.
No model, job or dependency installation occurs on import.
HF initialized mixer parameters are supplied by the model wrapper.
"""
import torch
from torch.nn import functional as F

def segmented_sum(a):
    n = a.shape[-1]
    expanded = a.unsqueeze(-1).expand(*a.shape, n)
    lower = torch.tril(torch.ones(n, n, dtype=torch.bool, device=a.device), diagonal=-1)
    sums = expanded.masked_fill(~lower, 0).cumsum(-2)
    return sums.masked_fill(~torch.tril(torch.ones_like(lower)), -torch.inf)

def ssd(x, dt, A, B, C, D, chunk=32):
    """Exact chunk-factorized selective recurrence; returns outputs and last state."""
    b, length, heads, dim = x.shape
    pad = (-length) % chunk
    # Padding AFTER discretization gives identity transitions for artificial steps.
    v = F.pad(x * dt[..., None], (0, 0, 0, 0, 0, pad))
    a = F.pad(dt * A, (0, 0, 0, pad))
    bb = F.pad(B, (0, 0, 0, 0, 0, pad))
    cc = F.pad(C, (0, 0, 0, 0, 0, pad))
    chunks = v.shape[1] // chunk
    v = v.reshape(b, chunks, chunk, heads, dim)
    a = a.reshape(b, chunks, chunk, heads).permute(0, 3, 1, 2)
    bb = bb.reshape(b, chunks, chunk, heads, -1)
    cc = cc.reshape(b, chunks, chunk, heads, -1)
    cumulative = a.cumsum(-1)
    decay = segmented_sum(a).exp().permute(0, 2, 3, 4, 1)
    interactions = torch.einsum("bclhn,bcshn->bclsh", cc, bb) * decay
    diagonal = torch.einsum("bclsh,bcshp->bclhp", interactions, v)
    to_end = (cumulative[..., -1:] - cumulative).exp().permute(0, 2, 3, 1)
    additions = torch.einsum("bclhn,bclhp->bchpn", bb * to_end[..., None], v)
    state = x.new_zeros(b, heads, dim, B.shape[-1])
    ys = []
    for k in range(chunks):
        from_start = cumulative[:, :, k].exp().transpose(1, 2)
        off = torch.einsum("blhn,bhpn->blhp", cc[:, k], state) * from_start[..., None]
        ys.append(diagonal[:, k] + off)
        state = state * cumulative[:, :, k, -1].exp()[..., None, None] + additions[:, k]
    y = torch.stack(ys, dim=1).reshape(b, -1, heads, dim)[:, :length]
    return y + x * D[None, None, :, None], state

def sequential(x, dt, A, B, C, D):
    state = x.new_zeros(x.shape[0], x.shape[2], x.shape[3], B.shape[-1])
    values = []
    for t in range(x.shape[1]):
        step = dt[:, t]
        state = state * (step * A).exp()[..., None, None] + (
            step[..., None, None] * x[:, t, :, :, None] * B[:, t, :, None, :])
        values.append((state * C[:, t, :, None, :]).sum(-1) + x[:, t] * D[None, :, None])
    return torch.stack(values, dim=1), state

def mixer_math(mixer, u, reference=False):
    """Use all initialized HF mixer parameters, with exact SSD or float64 oracle."""
    dtype = torch.float64 if reference else u.dtype
    u = u.to(dtype)
    linear = lambda inp, mod: F.linear(inp, mod.weight.to(dtype),
                                      None if mod.bias is None else mod.bias.to(dtype))
    z, bc, delta = linear(u, mixer.in_proj).split(
        [mixer.intermediate_size, mixer.conv_dim, mixer.num_heads], dim=-1)
    conv = F.conv1d(bc.transpose(1, 2), mixer.conv1d.weight.to(dtype),
                    None if mixer.conv1d.bias is None else mixer.conv1d.bias.to(dtype),
                    padding=mixer.conv_kernel_size - 1, groups=mixer.conv_dim)
    conv = F.silu(conv[:, :, :u.shape[1]].transpose(1, 2))
    x, B, C = conv.split([mixer.intermediate_size, mixer.n_groups * mixer.ssm_state_size,
                         mixer.n_groups * mixer.ssm_state_size], dim=-1)
    shape = (u.shape[0], u.shape[1], mixer.num_heads, mixer.head_dim)
    x = x.reshape(shape)
    B = B.reshape(u.shape[0], u.shape[1], mixer.n_groups, mixer.ssm_state_size)
    C = C.reshape_as(B)
    B = B.repeat_interleave(mixer.num_heads // mixer.n_groups, dim=2)
    C = C.repeat_interleave(mixer.num_heads // mixer.n_groups, dim=2)
    delta = F.softplus(delta + mixer.dt_bias.to(dtype)).clamp(*mixer.time_step_limit)
    A = -mixer.A_log.to(dtype).exp()
    method = sequential if reference else ssd
    if reference:
        y, state = method(x, delta, A, B, C, mixer.D.to(dtype))
    else:
        y, state = method(x, delta, A, B, C, mixer.D.to(dtype), mixer.chunk_size)
    y = y.flatten(2) * F.silu(z)
    y = y * torch.rsqrt(y.square().mean(-1, keepdim=True) + mixer.norm.variance_epsilon)
    y = y * mixer.norm.weight.to(dtype)
    return linear(y, mixer.out_proj), state
