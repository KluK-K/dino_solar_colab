"""Merge and parameter-isolation checks with a tiny random test fixture only."""
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel
from elresearch.adapt import inject_lora, merge_lora

def test_lora_gradient_isolation_and_merged_output():
    torch.manual_seed(11)
    config = DINOv3ViTConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=4,
                            intermediate_size=64, patch_size=16, num_register_tokens=4)
    model = DINOv3ViTModel(config).eval()
    image = torch.randn(1, 3, 32, 32)
    before = model(pixel_values=image).last_hidden_state.detach()
    count = inject_lora(model)
    assert count == 4 * 2 * 2 * 32 * 8
    initial = model(pixel_values=image).last_hidden_state
    assert torch.allclose(initial, before, atol=1e-6)
    initial.square().mean().backward()
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
    assert any(p.grad is not None for p in model.parameters() if p.requires_grad)
    with torch.no_grad():
        for layer in model.layer:
            layer.attention.q_proj.b.normal_(std=.01)
            layer.attention.v_proj.b.normal_(std=.01)
    adapted = model(pixel_values=image).last_hidden_state.detach()
    merge_lora(model)
    merged = model(pixel_values=image).last_hidden_state.detach()
    assert torch.allclose(adapted, merged, atol=2e-6, rtol=2e-5)
    assert not any(p.requires_grad for p in model.parameters())
