from typing import List, NamedTuple
import torch
from pathlib import Path

# Device selection
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LayerWeights(NamedTuple):
    wq: torch.tensor
    wk: torch.tensor
    wv: torch.tensor
    wo: torch.tensor
    w1: torch.tensor
    w2: torch.tensor
    w3: torch.tensor
    ffn_norm: torch.tensor
    attention_norm: torch.tensor

class XfmrWeights(NamedTuple):
    tok_embeddings: torch.tensor
    norm: torch.tensor
    output: torch.tensor
    layer_weights: List[LayerWeights]

def load_weights(ckpt_path: Path = Path('checkpoints\Llama3.2-1B-Instruct\consolidated.00.pth'), n_layers: int = 16):
    with torch.inference_mode():
        # Load the entire state dict
        state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)        
        layer_weights = []
        for i in range(n_layers):
            layer_weights.append(LayerWeights(
                wq=state_dict[f'layers.{i}.attention.wq.weight'].to(torch.bfloat16),
                wk=state_dict[f'layers.{i}.attention.wk.weight'].to(torch.bfloat16),
                wv=state_dict[f'layers.{i}.attention.wv.weight'].to(torch.bfloat16),
                wo=state_dict[f'layers.{i}.attention.wo.weight'].to(torch.bfloat16),
                w1=state_dict[f'layers.{i}.feed_forward.w1.weight'].to(torch.bfloat16),
                w2=state_dict[f'layers.{i}.feed_forward.w2.weight'].to(torch.bfloat16),
                w3=state_dict[f'layers.{i}.feed_forward.w3.weight'].to(torch.bfloat16),
                ffn_norm=state_dict[f'layers.{i}.ffn_norm.weight'].to(torch.bfloat16),
                attention_norm=state_dict[f'layers.{i}.attention_norm.weight'].to(torch.bfloat16),
            ))
        
        xfmr_weights = XfmrWeights(
            tok_embeddings=state_dict['tok_embeddings.weight'].to(torch.bfloat16),
            norm=state_dict['norm.weight'].to(torch.bfloat16),
            output=state_dict['output.weight'].to(torch.bfloat16),
            layer_weights=layer_weights
        )
        return xfmr_weights
