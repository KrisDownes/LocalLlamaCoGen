import argparse
import sys
import signal
import torch
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass
from contextlib import contextmanager

from llama import xfmr, LLAMA_1B_PARAMS
from weights import load_weights
from tokenizer import Tokenizer
from kvcache import KVCache
from sampler import sample
from prompts import create_prompt_template

@dataclass
class GenerationState:
    """Tracks the state of text generation"""
    cursor_position: Tuple[int, int] = (0, 0)  # (row, col)
    is_generating: bool = False
    current_output: List[str] = None
    
    def __post_init__(self):
        self.current_output = []

class GenerationManager:
    def __init__(self, model_path: str, max_tokens: int = 4096):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = Path(model_path)
        self.max_tokens = max_tokens
        self.state = GenerationState()
        
        # Initialize model components
        self.model_params = LLAMA_1B_PARAMS
        self.xfmr_weights = None
        self.tokenizer = None
        self.setup_model()

    def setup_model(self):
        """Initialize model weights and tokenizer"""
        try:
            self.xfmr_weights = load_weights(self.model_path / 'consolidated.00.pth')
            self.tokenizer = Tokenizer(self.model_path / 'tokenizer.model')
        except Exception as e:
            print(f"Error loading model: {str(e)}", file=sys.stderr)
            sys.exit(1)

    def update_cursor_position(self, token_text: str):
        """Update cursor position based on generated token"""
        row, col = self.state.cursor_position
        if '\n' in token_text:
            lines = token_text.split('\n')
            row += len(lines) - 1
            col = len(lines[-1])
        else:
            col += len(token_text)
        self.state.cursor_position = (row, col)

    def precompute_freqs_cis(self, dim: int, end: int, theta: float = 500000.0) -> torch.Tensor:
        """Precompute frequency cis"""
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device=self.device)[: (dim // 2)] / dim))
        t = torch.arange(end, dtype=torch.float32, device=self.device).unsqueeze(1)
        freqs = freqs.unsqueeze(0)
        freqs = t * freqs
        return torch.exp(1j * freqs)

    def build_attn_mask(self, seqlen: int, start_pos: int) -> Optional[torch.Tensor]:
        """Build attention mask"""
        if seqlen > 1:
            mask = torch.full((seqlen, seqlen), float("-inf"))
            mask = torch.triu(mask, diagonal=1)
            mask = torch.hstack([torch.zeros((seqlen, start_pos)), mask]).to(torch.float32).to(self.device)
            return mask
        return None

    @contextmanager
    def generation_context(self):
        """Context manager for generation state"""
        self.state.is_generating = True
        try:
            yield
        finally:
            self.state.is_generating = False
            self.state.current_output = []

    def stream_token(self, token_text: str):
        """Stream token to output and update state"""
        sys.stdout.write(token_text)
        sys.stdout.flush()
        self.state.current_output.append(token_text)
        self.update_cursor_position(token_text)

    def generate(self, prompt: str, system_prompt: str = ''):
        """Generate text with streaming output"""
        with self.generation_context():
            try:
                with torch.inference_mode():
                    # Prepare prompt
                    full_prompt = create_prompt_template(system_prompt, prompt) if system_prompt else prompt
                    raw_tokens = self.tokenizer.encode(full_prompt, bos=False, eos=False, allowed_special='all')
                    gen_tokens = None
                    tokens = torch.tensor([raw_tokens], dtype=torch.long).to(self.device)
                    
                    # Initialize generation
                    bsz, seqlen = tokens.shape
                    cur_pos = 0
                    attn_mask = self.build_attn_mask(seqlen, cur_pos)
                    freqs_cis = self.precompute_freqs_cis(
                        self.model_params.head_dim,
                        self.model_params.max_seq_len,
                        self.model_params.rope_theta,
                    )
                    
                    kvcache = KVCache.new(
                        self.model_params.n_layers,
                        bsz,
                        self.model_params.max_seq_len,
                        self.model_params.n_local_kv_heads,
                        self.model_params.head_dim
                    ).to(self.device)

                    # Initial forward pass
                    logits, kvcache, scores, _ = xfmr(
                        self.xfmr_weights,
                        self.model_params,
                        tokens,
                        cur_pos,
                        freqs_cis[:seqlen],
                        kvcache,
                        attn_mask=attn_mask
                    )
                    
                    next_token = torch.argmax(logits[:, -1], dim=-1, keepdim=True).to(torch.int32)
                    gen_tokens = next_token
                    
                    # Stream first token
                    token_text = self.tokenizer.decode([next_token.item()])
                    self.stream_token(token_text)
                    
                    cur_pos = seqlen
                    stop = torch.tensor([128001, 128008, 128009], device=self.device, dtype=torch.int32)
                    
                    # Generate and stream remaining tokens
                    while cur_pos < self.max_tokens and self.state.is_generating:
                        cur_pos += 1
                        logits, kvcache, scores, _ = xfmr(
                            self.xfmr_weights,
                            self.model_params,
                            next_token,
                            cur_pos,
                            freqs_cis[cur_pos:cur_pos+1],
                            kvcache
                        )
                        
                        next_token = sample(gen_tokens, logits, scores)
                        
                        if torch.isin(next_token, stop).any():
                            break
                            
                        gen_tokens = torch.cat((gen_tokens, next_token), dim=1)
                        token_text = self.tokenizer.decode([next_token.item()])
                        self.stream_token(token_text)

            except torch.cuda.OutOfMemoryError as e:
                print(f"\nCUDA out of memory: {str(e)}", file=sys.stderr)
                sys.exit(2)
            except Exception as e:
                print(f"\nError during generation: {str(e)}", file=sys.stderr)
                sys.exit(1)

def handle_interrupt(signum, frame):
    """Handle interrupt signal"""
    print("\nGeneration interrupted by user")
    sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description='Run Llama model inference with streaming')
    parser.add_argument('--model-path', required=True, help='Path to model directory')
    parser.add_argument('--prompt', required=True, help='Input prompt')
    parser.add_argument('--max-tokens', type=int, default=4096, help='Maximum number of tokens')
    parser.add_argument('--system-prompt', default='', help='System prompt')
    parser.add_argument('--completion-mode', type=bool, default=False,
                   help='Enable completion mode with shorter responses')
    args = parser.parse_args()

    # Set up interrupt handling
    signal.signal(signal.SIGINT, handle_interrupt)
    
    # Initialize and run generation
    generator = GenerationManager(args.model_path, args.max_tokens)
    generator.generate(args.prompt, args.system_prompt)

if __name__ == '__main__':
    main()