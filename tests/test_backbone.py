import pytest
import torch

from egasn.models.backbone import MobileNetV2Encoder, PrunedASPP


def test_mobilenet_encoder_output_shapes():
    encoder = MobileNetV2Encoder(pretrained=False)
    x = torch.randn(1, 3, 128, 128)  # Small input for fast test
    low_level, high_level = encoder(x)

    # MobileNetV2 features[:4] -> stride 4, 24 channels
    assert low_level.shape[0] == 1
    assert low_level.shape[1] == 24
    assert low_level.shape[2] == 128 // 4  # stride 4
    assert low_level.shape[3] == 128 // 4

    # MobileNetV2 features[4:] -> stride 32, 1280 channels
    assert high_level.shape[0] == 1
    assert high_level.shape[1] == 1280
    assert high_level.shape[2] == 128 // 32  # stride 32
    assert high_level.shape[3] == 128 // 32


def test_aspp_output_shape():
    aspp = PrunedASPP(in_channels=1280, out_channels=256, dilations=[6, 12, 18])
    aspp.eval()  # BatchNorm requires >1 samples in training mode
    x = torch.randn(1, 1280, 4, 4)
    out = aspp(x)

    assert out.shape == (1, 256, 4, 4)


def test_aspp_dilations():
    rates = [6, 12, 18]
    aspp = PrunedASPP(in_channels=1280, out_channels=256, dilations=rates)

    # Check that branches contain convolutions with the correct dilation rates
    dilations = []
    for m in aspp.modules():
        if isinstance(m, torch.nn.Conv2d) and m.kernel_size == (3, 3):
            dilations.append(m.dilation[0])

    for rate in rates:
        assert rate in dilations, f"Dilation rate {rate} not found in ASPP branches"


def test_encoder_pretrained():
    encoder = MobileNetV2Encoder(pretrained=True)

    # Verify weights are not all zeros (pretrained loaded)
    total_abs_sum = sum(torch.sum(torch.abs(p)).item() for p in encoder.parameters())
    assert total_abs_sum > 0
