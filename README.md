# Decoder-Only MoE

A from-scratch PyTorch implementation of a **decoder-only Transformer with a Mixture-of-Experts (MoE) feed-forward layer**, built for learning and experimentation. It implements causal self-attention, a noisy top-k router, capacity-limited expert dispatch, batched expert MLPs, and the auxiliary losses used to train MoE language models (load-balancing loss and router z-loss).

## Architecture

Each decoder block follows the standard pre-norm Transformer pattern, but replaces the dense feed-forward sub-layer with a sparse MoE layer:

```
x = x + CausalSelfAttention(LayerNorm(x))
x = x + MoELayer(LayerNorm(x))
```

**MoE layer pipeline:**
1. **Router** — a linear gate scores each token against every expert (with optional learned noise for exploration), selects the top-k experts per token, and assigns tokens to fixed-capacity expert buffers (dropping tokens that overflow capacity).
2. **Experts** — a batch of independent two-layer MLPs (`Linear → GELU → Linear`) implemented with batched matrix multiplication (`torch.bmm`) over all experts at once.
3. **Combine** — expert outputs are weighted by the router's softmax probabilities and scattered back to their original token positions.

## Repository Structure

| File | Description |
|---|---|
| `attention_mecanism.py` | `CasualSelfAttention` — multi-head causal self-attention with a triangular mask. |
| `router.py` | `Router` — noisy top-k gating, expert capacity computation, and token-to-expert dispatch. |
| `experts_layer.py` | `MLPExperts` — batched per-expert feed-forward networks. |
| `moe_layer.py` | `MOELayer` — wires the router and experts together into a single sparse FFN layer. |
| `moe_block.py` | `MOEBlock` — a full decoder block (attention + MoE layer with residuals and layer norm). |
| `tokenization_setup.py` | Loads a Hugging Face tokenizer and sets up the token embedding layer. |
| `load_balancing_losses.py` | Standalone example computing the MoE load-balancing auxiliary loss. |
| `moe_demo.py` | End-to-end demo: tokenization → embeddings → attention/router/expert components → MoE block → a minimal language model → forward pass → text generation → auxiliary losses. |
| `main.py` | CLI entry point that runs the demo or an interactive component-testing menu. |
| `combined/Decoder_only_MOE.py` / `combined/Decoder_Only_MOE.ipynb` | The full implementation collected into a single script / notebook. |

## Requirements

- Python 3.10+
- [PyTorch](https://pytorch.org/)
- [Hugging Face Transformers](https://github.com/huggingface/transformers)

```bash
pip install torch transformers
```

> Note: `tokenization_setup.py` loads the `CohereLabs/North-Mini-Code-1.0` tokenizer from the Hugging Face Hub. You'll need internet access (and possibly to be logged in via `huggingface-cli login` if the model is gated) the first time it runs, or you can swap in any other tokenizer of your choice.

## Usage

### Run the full demo

```bash
python main.py
# or
python main.py --demo
```

This walks through tokenization, embeddings, each MoE component in isolation, a full MoE block, auxiliary loss computation, and a sample forward pass / text generation with a minimal language model wrapper.

### Interactive component testing

```bash
python main.py --interact
```

Opens a menu to test individual pieces (tokenizer, embeddings, attention, router, experts, MoE layer, MoE block, full model, generation) one at a time with dummy inputs.

### Using the components directly

```python
import torch
from moe_block import MOEBlock

block = MOEBlock(
    embedding_dim=768,
    num_head=8,
    max_sequence_length=768,
    num_expert=8,
    top_k=2,
    use_noisy_top_k=True,
    capacity_factor=1.25,
    bias=False,
    dropout=0.2,
)

x = torch.randn(2, 16, 768)  # [batch, sequence_length, embedding_dim]
out = block(x)                # -> [2, 16, 768]
```

## Key Concepts Implemented

- **Causal self-attention** with a registered triangular buffer mask for autoregressive decoding.
- **Noisy top-k routing**: softplus-scaled Gaussian noise added to router logits before top-k selection, encouraging better expert utilization during training.
- **Expert capacity**: each expert can only process a fixed number of tokens per batch (`capacity_factor * top_k * num_tokens / num_experts`); tokens beyond capacity are dropped, mirroring how production MoE systems (e.g., Switch Transformer, GShard) handle load.
- **Batched expert computation**: all experts' weights are stored as a single `[num_expert, ...]` parameter tensor and processed with one `torch.bmm` call instead of a Python loop.
- **Auxiliary losses**: a load-balancing loss (encourages uniform expert usage) and a router z-loss (penalizes large router logits) for stabilizing MoE training.

## Status

This is an educational / experimental implementation — there is no training loop, dataset pipeline, or pretrained checkpoint included. It's intended as a clear, readable reference for how the pieces of a decoder-only MoE Transformer fit together.

## License

Released under the [MIT License](LICENSE).
