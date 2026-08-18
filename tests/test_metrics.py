from camkit_ai.metrics import classification_metrics_at_threshold, discrimination_metrics


def test_classification_metrics_at_threshold_counts() -> None:
    y_true = [0, 0, 1, 1]
    y_prob = [0.1, 0.7, 0.8, 0.2]
    metrics = classification_metrics_at_threshold(y_true, y_prob, threshold=0.5)
    assert metrics["true_positive"] == 1
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 1
    assert metrics["ppv"] == 0.5
    assert metrics["npv"] == 0.5


def test_discrimination_metrics_returns_expected_keys() -> None:
    y_true = [0, 0, 0, 1, 1, 1]
    y_prob = [0.1, 0.2, 0.3, 0.8, 0.9, 0.7]
    metrics = discrimination_metrics(y_true, y_prob)
    assert metrics["auprc"] > 0.9
    assert metrics["auroc"] == 1.0
    assert "calibration_slope" in metrics
    assert "calibration_intercept" in metrics

