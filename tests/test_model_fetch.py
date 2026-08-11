import os, pytest

@pytest.mark.gpu
def test_model_weights_present():
    cfg = "models/Muse-Glimmer-30B/config.json"
    assert os.path.exists(cfg), "model not fetched; run scripts/02-fetch-model.sh"
    import json
    assert json.load(open(cfg))["model_type"] == "muse_glimmer"
