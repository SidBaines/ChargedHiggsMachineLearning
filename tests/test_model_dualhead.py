import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.models import TestNetwork


B = 4
N_OBJ = 15
NUM_CLASSES = 3
NUM_PARTICLE_TYPES = 6
FEATURE_SET = ["phi", "eta", "pt", "m", "tag"]


def _model_kwargs(**overrides):
    kwargs = {
        "hidden_dim": 16,
        "num_attention_blocks": 2,
        "num_heads": 2,
        "embedding_size": 6,
        "num_particle_types": NUM_PARTICLE_TYPES,
        "feature_set": FEATURE_SET,
        "num_classes": NUM_CLASSES,
    }
    kwargs.update(overrides)
    return kwargs


def _inputs():
    torch.manual_seed(123)
    object_features = torch.randn(B, N_OBJ, len(FEATURE_SET))
    object_types = torch.randint(0, NUM_PARTICLE_TYPES, (B, N_OBJ), dtype=torch.long)
    object_types[:, 0] = 0
    object_types[:, -3:] = NUM_PARTICLE_TYPES - 1
    return object_features, object_types


def test_default_unchanged_shapes():
    object_features, object_types = _inputs()

    reco_model = TestNetwork(**_model_kwargs(is_reconstruction_model=True))
    reco_output = reco_model(object_features, object_types)
    assert isinstance(reco_output, torch.Tensor)
    assert reco_output.shape == (B, N_OBJ, NUM_CLASSES)

    event_model = TestNetwork(**_model_kwargs(is_reconstruction_model=False))
    event_output = event_model(object_features, object_types)
    assert isinstance(event_output, torch.Tensor)
    assert event_output.shape == (B, NUM_CLASSES)


def test_dualhead_shapes():
    object_features, object_types = _inputs()

    for num_event_classes in (3, 2):
        model = TestNetwork(
            **_model_kwargs(
                add_event_head=True,
                num_event_classes=num_event_classes,
            )
        )
        output = model(object_features, object_types)
        assert set(output.keys()) == {"reco", "event"}
        assert output["reco"].shape == (B, N_OBJ, NUM_CLASSES)
        assert output["event"].shape == (B, num_event_classes)


def test_reco_head_unperturbed_by_event_head():
    object_features, object_types = _inputs()

    torch.manual_seed(0)
    model_a = TestNetwork(**_model_kwargs(add_event_head=False))
    torch.manual_seed(0)
    model_b = TestNetwork(**_model_kwargs(add_event_head=True))

    output_a = model_a(object_features, object_types)
    output_b = model_b(object_features, object_types)["reco"]
    assert torch.allclose(output_a, output_b, atol=0, rtol=0)


def test_checkpoint_loads_into_dualhead():
    base_model = TestNetwork(**_model_kwargs(add_event_head=False))
    state_dict = base_model.state_dict()

    dualhead_model = TestNetwork(**_model_kwargs(add_event_head=True))
    load_result = dualhead_model.load_state_dict(state_dict, strict=False)

    assert load_result.missing_keys == [
        "event_classifier.0.weight",
        "event_classifier.0.bias",
    ]
    assert load_result.unexpected_keys == []


if __name__ == "__main__":
    failures = 0
    for name, test_fn in sorted(globals().items()):
        if name.startswith("test_") and callable(test_fn):
            try:
                test_fn()
            except Exception:
                failures += 1
                print(f"{name}: FAIL")
                traceback.print_exc()
            else:
                print(f"{name}: PASS")
    if failures:
        raise SystemExit(1)
