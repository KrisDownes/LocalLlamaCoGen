import torch
import torch.nn as nn

device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else
                      "cpu")

class KVCache(nn.Module):
    def __init__(self, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int):
        super().__init__()
        shape = (layers, bsz, max_seq_len, kv_heads, head_dim)
        self.register_buffer('k', torch.zeros(shape, dtype=torch.bfloat16, device=device))
        self.register_buffer('v', torch.zeros(shape, dtype=torch.bfloat16, device=device))
    
    @classmethod
    def new(cls, layers: int, bsz: int, max_seq_len: int, kv_heads: int, head_dim: int) -> 'KVCache':
        return cls(layers, bsz, max_seq_len, kv_heads, head_dim)
    def update(self, xk: torch.Tensor, xv: torch.Tensor, layer_idx: int, cur_pos: int, n_rep: int):
        insert_len = xk.size(1)
        self.k[layer_idx, :, cur_pos:cur_pos+insert_len] = xk.to(self.k.dtype)
        self.v[layer_idx, :, cur_pos:cur_pos+insert_len] = xv.to(self.v.dtype)

        if cur_pos == 0:
            keys = xk.repeat_interleave(n_rep, dim=2)
            values = xv.repeat_interleave(n_rep, dim=2)
        else:
            keys = self.k[layer_idx].repeat_interleave(n_rep, dim=2)
            values = self.v[layer_idx].repeat_interleave(n_rep, dim=2)

        # Ensure xk and xv have the correct device and dtype
        keys = keys.to(self.k.dtype)
        values = values.to(self.v.dtype)
        return keys, values, self

    def clear(self):
        self.k.zero_()
        self.v.zero_()
