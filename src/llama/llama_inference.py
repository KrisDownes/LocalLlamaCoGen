import argparse
import sys
import torch
from pathlib import Path
from typing import Optional

from llama import xfmr, LLAMA_1B_PARAMS
from weights import load_weights
from tokenizer import Tokenizer
from kvcache import KVCache
from sampler import sample
from prompts import create_prompt_template

def precompute_freqs_cis(dim: int, end: int, theta: float = 500000.0, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=dtype, device=device)[: (dim // 2)] / dim))
    t = torch.arange(end, dtype=dtype, device=device).unsqueeze(1)
    freqs = freqs.unsqueeze(0)
    freqs = t * freqs
    return torch.exp(1j * freqs)

def build_attn_mask(seqlen: int, start_pos: int, device: torch.device) -> Optional[torch.Tensor]:
    if seqlen > 1:
        mask = torch.full((seqlen, seqlen), float("-inf"))
        mask = torch.triu(mask, diagonal=1)
        mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(device)
        return mask
    return None

def main():
    parser = argparse.ArgumentParser(description='Run Llama model inference')
    parser.add_argument('--model-path', required=True, help='Path to model directory')
    parser.add_argument('--prompt', required=True, help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum number of tokens')
    parser.add_argument('--system-prompt', default='', help='System prompt')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        with torch.inference_mode():
            # Load model weights and tokenizer
            model_params = LLAMA_1B_PARAMS
            xfmr_weights = load_weights(Path(args.model_path) / 'consolidated.00.pth')
            tokenizer = Tokenizer(Path(args.model_path) / 'tokenizer.model')

            # Prepare prompt
            if args.system_prompt:
                full_prompt = create_prompt_template(f"{args.system_prompt}",args.prompt)
            else:
                full_prompt = create_prompt_template(f"{args.system_prompt}",args.prompt)
                
            raw_tokens = tokenizer.encode(full_prompt, bos=False, eos=False, allowed_special='all')
            gen_tokens = None
            tokens = torch.tensor([raw_tokens], dtype=torch.long).to(device)
            
            # Initialize generation
            bsz, seqlen = tokens.shape
            cur_pos = 0
            attn_mask = build_attn_mask(seqlen, cur_pos, device)
            freqs_cis = precompute_freqs_cis(
                model_params.head_dim,
                model_params.max_seq_len,
                model_params.rope_theta,
            )
            
            kvcache = KVCache.new(
                model_params.n_layers,
                bsz,
                model_params.max_seq_len,
                model_params.n_local_kv_heads,
                model_params.head_dim
            ).to(device)

            # Initial forward pass
            logits, kvcache, scores, _ = xfmr(
                xfmr_weights,
                model_params,
                tokens,
                cur_pos,
                freqs_cis[:seqlen],
                kvcache,
                attn_mask=attn_mask
            )
            
            next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
            gen_tokens = next_token
            
            # Print first token
            sys.stdout.write(tokenizer.decode([next_token.item()]))
            sys.stdout.flush()
            
            cur_pos = seqlen
            stop = torch.tensor([128001, 128008, 128009], device=device, dtype=torch.int32)
            stop_tokens = torch.tensor([tokenizer.encode(token, bos=False, eos=False)[0] for token in ['<|endoftext|>', '</s>']], device=device)
            
            # Generate tokens
            while cur_pos < args.max_tokens:
                cur_pos += 1
                logits, kvcache, scores, _ = xfmr(
                    xfmr_weights,
                    model_params,
                    next_token,
                    cur_pos,
                    freqs_cis[cur_pos:cur_pos+1],
                    kvcache
                )
                
                next_token = sample(gen_tokens, logits, scores)
                
                if torch.isin(next_token, stop).any():
                    break
                    
                gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                
                # Print token
                sys.stdout.write(tokenizer.decode([next_token.item()]))
                sys.stdout.flush()

    except torch.cuda.OutOfMemoryError as e:
        print(f"CUDA out of memory: {str(e)}", file=sys.stderr)
        sys.exit(2)
    except ImportError as e:
        print(f"Missing dependency: {str(e)}", file=sys.stderr)
        sys.exit(3)

    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == '__main__':
    main()