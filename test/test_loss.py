import torch
import pytest

from my_av.loss import mon_loss

# --- The Tests ---

def test_perfect_match_yields_zero_loss():
    """
    If one of the proposals (N) exactly matches the target for every stage (K),
    loss should be 0.0.
    """
    B, K, N, T, D = 2, 3, 5, 10, 3
    target = torch.randn(B, T, D)
    
    # Initialize proposals with random noise
    proposals = torch.randn(B, K, N, T, D)
    
    # Force the first proposal (n=0) to be exactly the target for all K stages
    # Note: target is (B, T, D), need to broadcast to (B, K, 1, T, D) assignment
    for k in range(K):
        proposals[:, k, 0, :, :] = target
        
    loss = mon_loss(proposals, target, lambda_val=0.5)
    
    assert torch.isclose(loss, torch.tensor(0.0)), f"Loss should be 0, got {loss}"

def test_manual_calculation():
    """
    Manually construct simple tensors to verify the arithmetic.
    B=1, K=2, N=2, T=1, D=1 (simplified dims for math)
    """
    lambda_val = 0.5
    target = torch.tensor([[[0.0, 0.0, 0.0]]]) # Shape (1, 1, 3) representing origin
    
    # Proposals: Shape (1, 2, 2, 1, 3) -> (B, K, N, T, D)
    # Stage k=0 (Weight = lambda^1 = 0.5)
    #   n=0: Error = 2.0
    #   n=1: Error = 10.0
    #   Min = 2.0
    # Stage k=1 (Weight = lambda^0 = 1.0)
    #   n=0: Error = 10.0
    #   n=1: Error = 4.0
    #   Min = 4.0
    
    # Expected Loss = (0.5 * 2.0) + (1.0 * 4.0) = 1.0 + 4.0 = 5.0
    
    proposals = torch.zeros(1, 2, 2, 1, 3)
    
    # k=0
    proposals[0, 0, 0, 0, :] = torch.tensor([2.0, 0.0, 0.0]) # L1 sum = 2
    proposals[0, 0, 1, 0, :] = torch.tensor([10.0, 0.0, 0.0])
    
    # k=1
    proposals[0, 1, 0, 0, :] = torch.tensor([10.0, 0.0, 0.0])
    proposals[0, 1, 1, 0, :] = torch.tensor([0.0, 4.0, 0.0]) # L1 sum = 4

    loss = mon_loss(proposals, target, lambda_val=lambda_val)
    
    assert torch.isclose(loss, torch.tensor(5.0)), f"Expected 5.0, got {loss}"

def test_gradient_propagation():
    """
    Ensure gradients flow back to the proposals tensor.
    """
    B, K, N, T, D = 1, 2, 2, 5, 3
    target = torch.randn(B, T, D)
    proposals = torch.randn(B, K, N, T, D, requires_grad=True)
    
    loss = mon_loss(proposals, target, lambda_val=0.5)
    loss.backward()
    
    assert proposals.grad is not None, "Gradients were not computed for proposals"
    assert not torch.isnan(proposals.grad).any(), "Gradients contain NaNs"

def test_batch_averaging():
    """
    Ensure the loss is correctly averaged across the batch dimension.
    """
    B = 2
    K, N, T, D = 2, 2, 1, 3
    lambda_val = 1.0 # Set to 1 to ignore weighting complexity
    
    target = torch.zeros(B, T, D)
    proposals = torch.zeros(B, K, N, T, D)
    
    # Batch 0: Perfect match (Loss should be 0)
    # proposals are already 0, target is 0.
    
    # Batch 1: Constant error of 1 per stage
    # Target is 0. Make best proposal have distance 1.
    proposals[1, :, :, 0, 0] = 1.0 
    # Because spatial dim is 3, if we set index 0 to 1.0, L1 is 1.0 per proposal.
    # Sum over K=2 stages = 1.0 + 1.0 = 2.0
    
    # Expected: (Loss_B0 + Loss_B1) / 2 = (0.0 + 2.0) / 2 = 1.0
    
    loss = mon_loss(proposals, target, lambda_val=lambda_val)
    
    assert torch.isclose(loss, torch.tensor(1.0)), f"Expected 1.0, got {loss}"

def test_device_consistency():
    """
    If inputs are on GPU, ensure internal tensors (weights) are created on GPU too.
    Skipped if CUDA is not available.
    """
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    device = torch.device("cuda")
    B, K, N, T, D = 2, 3, 4, 5, 3
    
    target = torch.randn(B, T, D).to(device)
    proposals = torch.randn(B, K, N, T, D).to(device)
    
    try:
        loss = mon_loss(proposals, target, lambda_val=0.5)
        # Just checking it doesn't crash due to device mismatch
        assert loss.device.type == "cuda"
    except RuntimeError as e:
        pytest.fail(f"Device mismatch error: {e}")