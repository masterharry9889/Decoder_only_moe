import torch
from transformers import AutoTokenizer

# load the tokenizer
tokenizer = AutoTokenizer.from_pretrained('CohereLabs/North-Mini-Code-1.0')

# raw text
text = "This raw text will be tokenized"

# create tokenize using tokenizer
tokens = tokenizer.tokenize(text)
token_id = tokenizer.encode(text)  # directly create token ids

# create token embedding layer
VOCABULARY_SIZE: int = tokenizer.vocab_size
EMBEDDING_DIM: int = 768
token_embedding_layer = torch.nn.Embedding(
    num_embeddings=VOCABULARY_SIZE,
    embedding_dim=EMBEDDING_DIM
)
