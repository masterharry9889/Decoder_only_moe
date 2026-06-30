#!/usr/bin/env python3
"""
Decoder-Only MOE - Quick Start Script

This is the main entry point for running the Decoder-Only MOE model.
It provides easy access to all the core functionality with a simple command.

Run this script to:
1. Run the complete MOE demo
2. Test individual components
3. See the model in action
"""

import argparse
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

from moe_demo import main as run_demo
def check_dependencies():
    """Check if all required dependencies are available"""
    try:
        import torch
        import transformers
        import torch.nn.functional as F
        print("✓ All dependencies are available")
        return True
    except ImportError as e:
        print(f"✗ Missing dependency: {e.name}")
        return False
def run_demo_mode():
    """Run the complete MOE demo"""
    print("\n" + "=" * 60)
    print("Running Decoder-Only MOE Demo")
    print("=" * 60 + "\n")
    run_demo()
def interactive_mode():
    """Interactive mode for testing individual components"""
    print("\n" + "=" * 60)
    print("Interactive Component Testing Mode")
    print("=" * 60 + "\n")

    try:
        import torch
        from torch import nn

        from tokenization_setup import VOCABULARY_SIZE, EMBEDDING_DIM, tokenizer, text
        from attention_mecanism import CasualSelfAttention
        from moe_block import MOEBlock
        from router import Router
        from moe_layer import MOELayer
        from experts_layer import MLPExperts

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")

        while True:
            print("\nAvailable commands:")
            print("1. tokenization - Show tokenization setup")
            print("2. embedding - Show embedding layer")
            print("3. attention - Test attention mechanism")
            print("4. router - Test router component")
            print("5. experts - Test expert layer")
            print("6. moe_layer - Test MOE layer")
            print("7. moe_block - Test MOE block")
            print("8. full_model - Create and test full model")
            print("9. generate - Generate text")
            print("10. help - Show this help")
            print("0. exit - Exit interactive mode")

            choice = input("\nEnter command (0-10): ").strip()

            if choice == "0":
                break
            elif choice == "1":
                print(f"\nText: {text}")
                print(f"Tokens: {tokenizer.tokenize(text)}")
                print(f"Token IDs: {tokenizer.encode(text)}")
            elif choice == "2":
                token_embedding_layer = nn.Embedding(
                    num_embeddings=VOCABULARY_SIZE,
                    embedding_dim=EMBEDDING_DIM
                )
                token_id = torch.tensor(tokenizer.encode(text), dtype=torch.long)
                token_emb = token_embedding_layer(token_id)
                print(f"\nToken Embeddings shape: {token_emb.shape}")
                print(f"Token Embeddings: {token_emb}")
            elif choice == "3":
                attn = CasualSelfAttention(
                    embedding_dim=EMBEDDING_DIM,
                    num_heads=8,
                    max_token_length=EMBEDDING_DIM,
                    bias=False,
                    dropout=0.1
                )
                dummy_input = torch.randn(1, 4, EMBEDDING_DIM)
                output = attn(dummy_input)
                print(f"\nAttention Output shape: {output.shape}")
            elif choice == "4":
                router = Router(
                    embedding_dim=EMBEDDING_DIM,
                    num_expert=8,
                    top_k=2,
                    user_noisy_top_k=True,
                    capacity_factor=1.25
                )
                dummy_input = torch.randn(1, 4, EMBEDDING_DIM)
                exp_weights, exp_mask, expert_batches = router(dummy_input)
                print(f"\nRouter - Expert Weights shape: {exp_weights.shape}")
                print(f"Router - Expert Batches shape: {expert_batches.shape}")
            elif choice == "5":
                experts = MLPExperts(
                    embedding_dim=EMBEDDING_DIM,
                    num_expert=8,
                    bias=False,
                    dropout=0.2
                )
                dummy_input = torch.randn(8, 2, EMBEDDING_DIM)
                output = experts(dummy_input)
                print(f"\nExperts Output shape: {output.shape}")
            elif choice == "6":
                moe_layer = MOELayer(
                    embedding_dim=EMBEDDING_DIM,
                    num_expert=8,
                    top_k=2,
                    use_noisy_top_k=True,
                    capacity_factor=1.25,
                    bias=False,
                    dropout=0.2
                )
                dummy_input = torch.randn(1, 4, EMBEDDING_DIM)
                output = moe_layer(dummy_input)
                print(f"\nMOE Layer Output shape: {output.shape}")
            elif choice == "7":
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
                dummy_input = torch.randn(1, 4, EMBEDDING_DIM)
                output = moe_block(dummy_input)
                print(f"\nMOE Block Output shape: {output.shape}")
            elif choice == "8":
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

                model = SimpleLanguageModel().to(device)
                input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)
                logits = model(input_ids)
                print(f"\nModel Input shape: {input_ids.shape}")
                print(f"Model Logits shape: {logits.shape}")
            elif choice == "9":
                from moe_demo import generate_next_token

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

                model = SimpleLanguageModel().to(device)

                input_ids = torch.tensor(tokenizer.encode(text), dtype=torch.long).unsqueeze(0).to(device)
                generated_output = generate_next_token(model, tokenizer, text, max_new_tokens=3, device=device)
                print(f"\nInput text: {text}")
                print(f"Generated text: {generated_output}")
            elif choice == "10":
                print("\nInteractive Component Testing Commands:")
                print("1. tokenization - Show tokenization setup")
                print("2. embedding - Show embedding layer")
                print("3. attention - Test attention mechanism")
                print("4. router - Test router component")
                print("5. experts - Test expert layer")
                print("6. moe_layer - Test MOE layer")
                print("7. moe_block - Test MOE block")
                print("8. full_model - Create and test full model")
                print("9. generate - Generate text")
                print("10. help - Show this help")
                print("0. exit - Exit interactive mode")
            else:
                print(f"\nUnknown command: {choice}")
                print("Type 'help' for available commands")

    except Exception as e:
        print(f"Error in interactive mode: {e}")
        import traceback
        traceback.print_exc()
def main():
    """Parse arguments and run the appropriate mode"""
    parser = argparse.ArgumentParser(
        description="Decoder-Only MOE - Quick Start Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python main.py              # Run the complete demo
    python main.py --demo      # Run the complete demo
    python main.py --interact  # Start interactive component testing
    python main.py -h          # Show this help message
        """
    )

    parser.add_argument(
        "--demo", "-d",
        action="store_true",
        help="Run the complete MOE demo"
    )

    parser.add_argument(
        "--interact", "-i",
        action="store_true",
        help="Start interactive component testing mode"
    )

    args = parser.parse_args()

    # Check dependencies
    if not check_dependencies():
        print("\n✗ Please install missing dependencies")
        print("Run: pip install torch transformers")
        sys.exit(1)

    # Run appropriate mode
    if args.interact:
        interactive_mode()
    elif args.demo or not (args.interact or args.demo):
        run_demo_mode()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()