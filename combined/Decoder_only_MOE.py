# -*- coding: utf-8 -*-
"""MOE Model Demo and Training Setup"""

import torch
from transformers import AutoTokenizer
torch.cuda.is_available()
torch.device('cuda')

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('CohereLabs/North-Mini-Code-1.0')

# Raw text
text = "This raw text will be tokenized"

# Create tokenize using tokenizer
tokens = tokenizer.tokenize(text)
token_id = tokenizer.encode(text)  # directly create token ids

print(f"Tokens: {tokens}")
print(f"Token IDs: {token_id}")

# create token embedding layer
VOCABULARY_SIZE: int = tokenizer.vocab_size
EMBEDDING_DIM: int = 768

# Define model constants
MAX_SEQUENCE_LENGTH = 1024  # Maximum sequence length the model can handle
NUM_LAYERS = 6             # Number of MOEBlock layers in the model
NUM_HEADS = 8              # Number of attention heads in CasualSelfAttention
NUM_EXPERT = 8             # Number of experts in MOELayer
TOP_K_EXPERT = 2           # Number of top experts chosen by the router

# Import components
from tokenization_setup import token_embedding_layer
from moe_block import MOEBlock
from torch import nn

# Create a simple language model for demonstration
class SimpleLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCABULARY_SIZE, EMBEDDING_DIM)
        self.position_embedding_table = nn.Embedding(MAX_SEQUENCE_LENGTH, EMBEDDING_DIM)
        self.blocks = nn.Sequential(*[
            MOEBlock(EMBEDDING_DIM, NUM_HEADS, MAX_SEQUENCE_LENGTH, NUM_EXPERT, TOP_K_EXPERT)
            for _ in range(NUM_LAYERS)
        ])
        self.ln_f = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCABULARY_SIZE)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        B, T = idx.shape

        if T > MAX_SEQUENCE_LENGTH:
            idx = idx[:, -MAX_SEQUENCE_LENGTH:]
            T = MAX_SEQUENCE_LENGTH
            if targets is not None:
                targets = targets[:, -MAX_SEQUENCE_LENGTH:]

        token_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = token_emb + pos_emb

        x = self.blocks(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            logits_reshaped = logits.view(-1, logits.shape[-1])
            targets_reshaped = targets.view(-1)
            loss = torch.nn.functional.cross_entropy(logits_reshaped, targets_reshaped)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= MAX_SEQUENCE_LENGTH else idx[:, -MAX_SEQUENCE_LENGTH:]

            logits, _ = self(idx_cond)

            logits = logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            idx = torch.cat((idx, idx_next), dim=1)
        return idx
def generate_next_token(model: SimpleLanguageModel, tokenizer, input_text: str, max_new_tokens: int = 1, device: str = 'cpu') -> str:
    """
    Generates the next token(s) using a given LanguageModel and tokenizer.
    """
    model.eval()
    model.to(device)

    input_ids = torch.tensor(tokenizer.encode(input_text), dtype=torch.long).unsqueeze(0).to(device)

    generated_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

    generated_text = tokenizer.decode(generated_ids[0].tolist())

    return generated_text


print("Initializing Language Model...")
model = SimpleLanguageModel().to(device)

print(f"Model initialized and moved to {device}.")
print(f"Input text: '{text}'")

# Generate Next Token
print("\n--- Generating next token(s) ---")
generated_output = generate_next_token(model, tokenizer, text, max_new_tokens=5, device=device)
print(f"Generated output (input + 5 new tokens): '{generated_output}'")

# Compute Logits and Loss for a sequence
print("\n--- Computing Logits and Loss ---")
input_ids = torch.tensor(token_id, dtype=torch.long).unsqueeze(0).to(device)
target_ids = torch.cat((input_ids[:, 1:], torch.tensor([[tokenizer.pad_token_id or 0]], dtype=torch.long, device=device)), dim=1)

current_sequence_length = min(input_ids.shape[1], MAX_SEQUENCE_LENGTH)
input_ids_for_forward = input_ids[:, :current_sequence_length]
target_ids_for_forward = target_ids[:, :current_sequence_length]

model.train()

logits, loss = model(input_ids_for_forward, target_ids_for_forward)

print(f"Input sequence length: {input_ids_for_forward.shape[1]}")
print(f"Logits shape: {logits.shape}")
print(f"Computed Loss: {loss.item():.4f}")

print("\nExample of logit values (first token, first 5 vocab entries):\n", logits[0, 0, :5].detach().cpu().numpy())

# Using the compute_model_loss function
from moe_demo import compute_model_loss

manual_loss = compute_model_loss(logits, target_ids_for_forward)
print(f"Loss computed manually using compute_model_loss function: {manual_loss.item():.4f}")
