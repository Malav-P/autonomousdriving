import torch

def mon_loss(proposals: torch.Tensor,
             target: torch.Tensor,
             lambda_val: float) -> torch.Tensor:
    """
    Compute Minimum over N (MoN) loss based on the formula:
    L = sum_k( lambda^(K-1-k) * min_n( || P_k_n - P_hat ||_1 ) )

    Args:
        proposals : (B, K, N, T, 3) tensor of proposals, coming straight from the pytorch module
        target : (B, T, 3) tensor of targets, NOTE that these values should be normalized and in the rig frame
        lambda_val : Decay factor lambda
    
    Returns:
       torch.Tensor: Scalar MoN loss (averaged over batch)
    """
    # Unpack shapes
    # B: Batch, K: Stages, N: Modes/Proposals, T: Time, D: Spatial Dim (3)
    B, K, N, T, D = proposals.shape
    
    # 1. Align Target Shape
    # Reshape target from (B, T, 3) to (B, 1, 1, T, 3) to broadcast against proposals
    target_expanded = target.unsqueeze(1).unsqueeze(1)
    
    # 2. Compute L1 Norm Error
    # Formula term: || P - P_hat ||_1
    # We calculate absolute difference and sum over Time (T) and Spatial (3) dimensions
    # Shape becomes: (B, K, N)
    l1_error = torch.abs(proposals - target_expanded).sum(dim=(-2, -1))
    
    # 3. Compute Min over N
    # Formula term: min_{n=1...N}
    # We take the minimum error along the N dimension (dim 2)
    # Shape becomes: (B, K)
    min_error, _ = l1_error.min(dim=2)
    
    # 4. Apply Lambda Weights
    # Formula term: lambda^(K-1-k)
    # Create a range for k: [0, 1, ..., K-1]
    k_indices = torch.arange(K, device=proposals.device, dtype=torch.float32)
    
    # Calculate weights: [lambda^(K-1), lambda^(K-2), ..., lambda^0]
    weights = lambda_val ** (K - 1 - k_indices)
    
    # Expand weights to match batch dimension: (1, K)
    weights = weights.unsqueeze(0)
    
    # 5. Sum over K and Average over Batch
    # Formula term: sum_{k=0}^{K-1}
    weighted_loss = (min_error * weights).sum(dim=1) # Sum over K -> Shape (B,)
    
    return weighted_loss.mean() # Average over batch