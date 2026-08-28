import torch

from src.segmentation.model import (
    UNet,
    build_model,
)


def test_model_output_shape():
    model = build_model(
        base_channels=16,
    )

    x = torch.randn(
        2,
        3,
        256,
        256,
    )

    output = model(x)

    assert output.shape == (
        2,
        1,
        256,
        256,
    )


def test_model_outputs_logits():
    model = build_model(
        base_channels=16,
    )

    x = torch.randn(
        2,
        3,
        256,
        256,
    )

    output = model(x)

    assert output.dtype == torch.float32
    assert torch.isfinite(output).all()


def test_model_handles_non_power_of_two_dimensions():
    model = build_model(
        base_channels=8,
    )

    x = torch.randn(
        1,
        3,
        250,
        310,
    )

    output = model(x)

    assert output.shape == (
        1,
        1,
        250,
        310,
    )


def test_model_backward():
    model = build_model(
        base_channels=8,
    )

    x = torch.randn(
        1,
        3,
        128,
        128,
    )

    output = model(x)

    loss = output.mean()
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert all(
        gradient is not None
        for gradient in gradients
    )
