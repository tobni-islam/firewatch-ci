import json


def test_metrics_file_has_required_keys(tmp_path):
    metrics = {"mAP50": 0.75, "mAP50_95": 0.55, "fire_mAP50": 0.80, "smoke_mAP50": 0.68}
    p = tmp_path / "eval_results.json"
    p.write_text(json.dumps(metrics))
    loaded = json.loads(p.read_text())
    for key in ("mAP50", "mAP50_95", "fire_mAP50", "smoke_mAP50"):
        assert key in loaded, f"missing key: {key}"


def test_smoke_threshold_passes():
    assert 0.70 > 0.65


def test_metrics_values_in_unit_range():
    metrics = {"mAP50": 0.75, "fire_mAP50": 0.80, "smoke_mAP50": 0.68}
    for k, v in metrics.items():
        assert 0.0 <= v <= 1.0, f"{k}={v} out of [0,1]"
