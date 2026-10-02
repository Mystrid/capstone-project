import pytest
import torch

from egasn.models.cross_attention import CrossAttentionGating


def test_output_shape():
    attention = CrossAttentionGating(channels=256, num_heads=4)
    x_current = torch.randn(2, 256, 16, 16)
    x_warped = torch.randn(2, 256, 16, 16)

    out = attention(x_current, x_warped)
    assert out.shape == (2, 256, 16, 16)


def test_lambda_learnable():
    attention = CrossAttentionGating(channels=256)
    assert attention.lambda_scale.requires_grad


def test_none_warped():
    attention = CrossAttentionGating(channels=256)
    x_current = torch.randn(2, 256, 16, 16)

    # Should return x_current unchanged when warped is None
    out = attention(x_current, None)
    assert torch.allclose(out, x_current)


def test_attention_no_nan():
    attention = CrossAttentionGating(channels=256, num_heads=4)
    x_current = torch.randn(2, 256, 8, 8)
    x_warped = torch.randn(2, 256, 8, 8)

    out = attention(x_current, x_warped)

    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()
