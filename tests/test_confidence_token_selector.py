import sys
import os
import traceback
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def test_selector():
    try:
        import torch
    except ImportError:
        print("Input token shape             : N/A")
        print("Confidence score shape        : N/A")
        print("Selected token shape          : N/A")
        print("Tokens before                 : N/A")
        print("Tokens after                  : N/A")
        print("Token reduction               : N/A")
        print("Gradient test                 : FAIL (PyTorch missing)")
        sys.exit(1)
        
    try:
        from models.confidence_token_selector import ConfidenceTokenSelector
        
        B = 2
        N = 32
        D = 512
        K = 16
        
        tokens = torch.randn(B, N, D, requires_grad=True)
        
        selector = ConfidenceTokenSelector(embed_dim=D, top_k=K)
        
        selected_tokens, conf_scores = selector(tokens)
        
        print(f"Original number of tokens : {N}")
        print(f"Selected number of tokens : {K}")
        print(f"Reduction ratio           : {K/N:.2f}")
        print(f"Confidence score range    : [{conf_scores.min().item():.4f}, {conf_scores.max().item():.4f}]")
        print(f"Output shape              : {list(selected_tokens.shape)}")
        
        assert selected_tokens.shape == (B, K, D)
        assert conf_scores.shape == (B, N)
        
        # Test gradients
        loss = selected_tokens.sum()
        loss.backward()
        
        assert tokens.grad is not None
        print("Gradient test                 : PASS")
        return True
    except Exception as e:
        print(f"Test failed: {e}")
        return False

if __name__ == '__main__':
    test_selector()
