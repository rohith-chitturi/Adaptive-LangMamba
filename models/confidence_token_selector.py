import torch
import torch.nn as nn

class ConfidenceTokenSelector(nn.Module):
    def __init__(self, embed_dim, selection_ratio=0.5, top_k=None):
        """
        Confidence-Aware Semantic Token Selection.
        Args:
            embed_dim (int): The dimensionality of the token embeddings.
            selection_ratio (float): Ratio of tokens to select (if top_k is None).
            top_k (int): Exact number of tokens to select. Overrides selection_ratio.
        """
        super(ConfidenceTokenSelector, self).__init__()
        self.embed_dim = embed_dim
        self.selection_ratio = selection_ratio
        self.top_k = top_k
        
        # Learnable confidence scorer (MLP)
        self.confidence_scorer = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, tokens, k=None):
        """
        Args:
            tokens (torch.Tensor): Shape [B, N, D] where B is batch size, N is number of tokens, 
                                   D is embedding dimension.
            k (int): Optional override for the number of tokens to select.
        Returns:
            selected_tokens (torch.Tensor): Shape [B, K, D]
            confidence_scores (torch.Tensor): Shape [B, N]
        """
        B, N, D = tokens.shape
        
        # Determine K
        if k is not None:
            K = min(k, N)
        elif self.top_k is not None:
            K = min(self.top_k, N)
        else:
            K = max(1, int(N * self.selection_ratio))
            
        # 1. Compute confidence score for each token
        raw_scores = self.confidence_scorer(tokens)
        confidence_scores = raw_scores.squeeze(-1) # [B, N]
        
        # 2. Select top-K tokens based on confidence
        topk_scores, topk_indices = torch.topk(confidence_scores, k=K, dim=1)
        
        # 3. Gather the selected tokens
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, D)
        selected_tokens = torch.gather(tokens, 1, expanded_indices)
        
        # 4. Multiply selected tokens by their confidence scores so gradients flow to the scorer
        selected_scores = torch.gather(confidence_scores, 1, topk_indices)
        selected_tokens = selected_tokens * selected_scores.unsqueeze(-1)
        
        return selected_tokens, confidence_scores
