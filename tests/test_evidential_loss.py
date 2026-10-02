import pytest
import torch

from egasn.losses.evidential import EvidentialLoss, TemporalConsistencyLoss

def test_loss_computes():
    loss_fn = EvidentialLoss(num_classes=19, annealing_epochs=10)
    
    # Dummy outputs
    alpha = torch.ones(2, 19, 32, 32, requires_grad=True)
    probs = alpha / alpha.sum(dim=1, keepdim=True)
    uncertainty = 19.0 / alpha.sum(dim=1, keepdim=True)
    
    output = {'alpha': alpha, 'probs': probs, 'uncertainty': uncertainty}
    label = torch.randint(0, 19, (2, 32, 32))
    
    loss = loss_fn(output, label)
    
    assert loss.dim() == 0
    assert loss.requires_grad

def test_kl_uniform_prior():
    loss_fn = EvidentialLoss(num_classes=19, annealing_epochs=10)
    
    alpha = torch.ones(2, 19, 32, 32) # Uniform prior
    label = torch.zeros(2, 32, 32, dtype=torch.long)
    
    # Convert label to one hot to check KL
    # the KL for uniform alpha with itself is 0, but kl is computed on (alpha - 1) * (1 - y)
    kl_div = loss_fn.kl_divergence(alpha, label)
    
    assert torch.allclose(kl_div, torch.zeros_like(kl_div))

def test_beta_annealing():
    loss_fn = EvidentialLoss(num_classes=19, annealing_epochs=10, beta_max=1.0)
    
    loss_fn.set_epoch(0)
    assert loss_fn.beta == 0.0
    
    loss_fn.set_epoch(5)
    assert loss_fn.beta == 0.5
    
    loss_fn.set_epoch(10)
    assert loss_fn.beta == 1.0

def test_ignore_index():
    loss_fn = EvidentialLoss(num_classes=19, ignore_index=255)
    
    alpha = torch.rand(2, 19, 32, 32, requires_grad=True) + 1.0
    output = {'alpha': alpha}
    
    # All pixels ignored
    label = torch.ones(2, 32, 32, dtype=torch.long) * 255
    
    loss = loss_fn(output, label)
    
    # Loss should be 0 or effectively disconnected if all pixels are ignored
    # Usually it returns 0.0
    assert loss.item() == 0.0

def test_temporal_consistency_loss():
    tc_loss = TemporalConsistencyLoss()
    
    probs = torch.rand(2, 19, 32, 32)
    probs = probs / probs.sum(dim=1, keepdim=True)
    
    prev_probs_warped = torch.rand(2, 19, 32, 32)
    prev_probs_warped = prev_probs_warped / prev_probs_warped.sum(dim=1, keepdim=True)
    
    valid_mask = torch.ones(2, 1, 32, 32)
    
    loss = tc_loss(probs, prev_probs_warped, valid_mask)
    
    assert loss.dim() == 0
    assert loss >= 0.0
