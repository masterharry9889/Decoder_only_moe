"""
MOE Demo Script - Standalone Example

This script demonstrates the complete MOE model pipeline including:
1. Tokenization setup
2. Model initialization
3. Text generation
4. Forward pass and loss computation
5. Load balancing and auxiliary losses
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

# Import all components
from tokenization_setup import VOCABULARY_SIZE, EMBEDDING_DIM, tokenizer, text
from attention_mecanism import CasualSelfAttention
from moe_block import MOEBlock
from router import Router
from moe_layer import MOELayer
from experts_layer import MLPExperts
from load_balancing_losses import load_balance_loss


def generate_next_token(model, tokenizer, input_text: str, max_new_tokens: int = 1, device: str = 'cpu') -> str:
    """
    Generates the next token(s) using a given LanguageModel and tokenizer.

    Args:
        model: The language model to use for generation.
        tokenizer: The tokenizer object (e.g., from Hugging Face) corresponding to the model.
        input_text (str): The input text to condition the generation on.
        max_new_tokens (int): The maximum number of new tokens to generate.
        device (str): The device to run the model on ('cpu' or 'cuda').

    Returns:
        str: The generated text, including the original input and the new token(s).
    """
    model.eval()
    model.to(device)

    # Encode the input text to token IDs
    input_ids = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(device)

    # Generate tokens using the model's generate method
    for _ in range(max_new_tokens):
        idx_cond = input_ids if input_ids.shape[1] <= EMBEDDING_DIM else input_ids[:, -EMBEDDING_DIM:]
        logits = model(idx_cond)  # Get logits only
        logits = logits[:, -1, :]
        probs = F.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        input_ids = torch.cat((input_ids, idx_next), dim=1)

    # Decode the generated token IDs back to text
    generated_text = tokenizer.decode(input_ids[0].tolist())
    return generated_text


def compute_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Computes the cross-entropy loss given model output (logits) and target labels.

    Args:
        logits (torch.Tensor): The raw logits from the model,
                               shape (batch_size, sequence_length, vocab_size).
        targets (torch.Tensor): The target token IDs,
                                shape (batch_size, sequence_length).

    Returns:
        torch.Tensor: The computed cross-entropy loss.
    """
    batch_size, sequence_length, vocab_size = logits.shape
    main_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
    return main_loss


def demonstrate_load_balancing_losses():
    """Demonstrates load balancing and auxiliary losses"""
    print("\n=== Load Balancing Losses ===")
    print(f"Load Balance Loss: {load_balance_loss:.4f}")


def demonstrate_router_losses():
    """Demonstrates router Z-loss"""
    print("\n=== Router Z-Loss ===")
    Batch_size = 16
    sequence_length = 256
    num_expert = 8

    router_logits = torch.rand(Batch_size, sequence_length, num_expert)
    router_z_loss = torch.logsumexp(router_logits, dim=-1).mean()
    print(f"Router Z-Loss: {router_z_loss:.4f}")
    return router_z_loss


def demonstrate_total_loss():
    """Demonstrates combining auxiliary losses with main loss"""
    print("\n=== Total Loss (Main + Aux) ===")

    main_loss = torch.tensor(2.5)  # Example main loss
    aux_loss_weight = 0.1
    router_z_loss = demonstrate_router_losses()
    total_loss = main_loss + aux_loss_weight * (load_balance_loss + router_z_loss)
    print(f"Main Loss: {main_loss:.4f}")
    print(f"Auxiliary Loss (load_balance + router_z): {load_balance_loss + router_z_loss:.4f}")
    print(f"Total Loss: {total_loss:.4f}")


def demonstrate_tokenization():
    """Demonstrates the tokenization setup"""
    print("\n=== Tokenization Setup ===")
    print("Text tokens generation successful")
    print("Token IDs generated successfully")
    print("Max token ID:", max(tokenizer.encode(text)))
    print("Vocab size for embedding layer:", VOCABULARY_SIZE)


def demonstrate_embedding_layer():
    """Demonstrates token embedding layer"""
    print("\n=== Embedding Layer ===")
    token_embedding_layer = torch.nn.Embedding(
        num_embeddings=VOCABULARY_SIZE,
        embedding_dim=EMBEDDING_DIM
    )

    token_id = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    token_emb = token_embedding_layer(token_id)
    print("Token Embeddings shape:", token_emb.shape)
    print("Token Embeddings size:", token_emb.numel())


def demonstrate_components():
    """Demonstrates individual components"""
    print("\n=== Component Demonstration ===")

    # Create a simple input tensor
    dummy_input = torch.randn(2, 4, EMBEDDING_DIM)

    # Demonstrate CasualSelfAttention
    attn = CasualSelfAttention(
        embedding_dim=EMBEDDING_DIM,
        num_heads=8,
        max_token_length=EMBEDDING_DIM,
        bias=False,
        dropout=0.1
    )
    attn_output = attn(dummy_input)
    print(f"Attention Output Shape: {attn_output.shape}")

    # Demonstrate Router
    router = Router(
        embedding_dim=EMBEDDING_DIM,
        num_expert=8,
        top_k=2,
        user_noisy_top_k=True,
        capacity_factor=1.25
    )
    exp_weights, exp_mask, expert_batches = router(dummy_input)
    print(f"Router - Expert Weights Shape: {exp_weights.shape}")
    print(f"Router - Expert Batches Shape: {expert_batches.shape}")

    # Demonstrate MLPExperts
    experts = MLPExperts(
        embedding_dim=EMBEDDING_DIM,
        num_expert=8,
        bias=False,
        dropout=0.2
    )
    expert_output = experts(expert_batches)
    print(f"Experts Output Shape: {expert_output.shape}")


def demonstrate_moe_layer():
    """Demonstrates MOE Layer"""
    print("\n=== MOE Layer Demonstration ===")
    moe_layer = MOELayer(
        embedding_dim=EMBEDDING_DIM,
        num_expert=8,
        top_k=2,
        use_noisy_top_k=True,
        capacity_factor=1.25,
        bias=False,
        dropout=0.2
    )

    # Create a simple input tensor
    dummy_input = torch.randn(2, 4, EMBEDDING_DIM)
    moe_output = moe_layer(dummy_input)
    print(f"MOE Layer Output Shape: {moe_output.shape}")
    print(f"MOE Layer Output (first few values): {moe_output[0, 0, :3]}")


def demonstrate_moe_block():
    """Demonstrates MOE Block"""
    print("\n=== MOE Block Demonstration ===")
    moe_block = MOEBlock(
        embedding_dim=EMBEDDING_DIM,
        num_head=8,
        max_sequence_length=EMBEDDING_DIM,
        num_expert=8,
        top_k=2,
        use_noisy_top_k=True,
        capacity_factor=1.25,
        bias=False,
        dropout=0.2
    )

    # Create a simple input tensor
    dummy_input = torch.randn(2, 4, EMBEDDING_DIM)
    moe_block_output = moe_block(dummy_input)
    print(f"MOE Block Output Shape: {moe_block_output.shape}")


def main():
    """Main function to run all demonstrations"""
    print("=" * 60)
    print("Decoder-Only MOE Model - Standalone Demo")
    print("=" * 60)

    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. Demonstrate tokenization
    demonstrate_tokenization()

    # 2. Demonstrate embedding layer
    demonstrate_embedding_layer()

    # 3. Demonstrate individual components
    demonstrate_components()

    # 4. Demonstrate MOE Layer
    demonstrate_moe_layer()

    # 5. Demonstrate MOE Block
    demonstrate_moe_block()

    # 6. Demonstrate load balancing losses
    demonstrate_load_balancing_losses()

    # 7. Demonstrate router losses
    demonstrate_router_losses()

    # 8. Demonstrate total loss computation
    demonstrate_total_loss()

    # 9. Create a simple model for demonstration
    print("\n=== Language Model Setup ===")
    from torch import nn

    class SimpleLanguageModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.token_embedding_table = nn.Embedding(VOCABULARY_SIZE, EMBEDDING_DIM)
            self.position_embedding_table = nn.Embedding(EMBEDDING_DIM, EMBEDDING_DIM)
            self.moe_block = MOEBlock(
                embedding_dim=EMBEDDING_DIM,
                num_head=8,
                max_sequence_length=EMBEDDING_DIM,
                num_expert=8,
                top_k=2,
                use_noisy_top_k=True,
                capacity_factor=1.25,
                bias=False,
                dropout=0.2
            )
            self.ln_f = nn.LayerNorm(EMBEDDING_DIM)
            self.lm_head = nn.Linear(EMBEDDING_DIM, VOCABULARY_SIZE)

        def forward(self, idx):
            B, T = idx.shape
            if T > EMBEDDING_DIM:
                idx = idx[:, -EMBEDDING_DIM:]
                T = EMBEDDING_DIM

            token_emb = self.token_embedding_table(idx)
            pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = token_emb + pos_emb

            x = self.moe_block(x)
            x = self.ln_f(x)
            logits = self.lm_head(x)

            return logits

    # Initialize model
    model = SimpleLanguageModel().to(device)
    print(f"Model initialized and moved to {device}")

    # 10. Demonstrate forward pass
    print("\n=== Forward Pass Demonstration ===")
    input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)
    logits = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Logits shape: {logits.shape}")

    # 11. Demonstrate text generation
    print("\n=== Text Generation Demonstration ===")
    print("Input text length: {} characters".format(len(text)))
    generated_output = generate_next_token(model, tokenizer, text, max_new_tokens=3, device=device)
    print("Generated text length: {} characters".format(len(generated_output)))

    # Safe printing to avoid encoding issues
    def safe_print(text):
        try:
            print(text)
        except UnicodeEncodeError:
            cleaned = ''.join(char for char in text if ord(char) < 128)
            print(f"Generated (cleaned): {cleaned}...")

    print("Sample generated text (first 100 chars):")
    safe_print(generated_output[:100])

    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()