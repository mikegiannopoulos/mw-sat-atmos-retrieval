from __future__ import annotations


def aggregate_metrics(results: list[dict]) -> dict:
    """Aggregate bias and RMSE across a batch of experiment results."""
    if not results:
        raise ValueError("Results list must not be empty")

    biases = []
    rmses = []

    for result in results:
        metrics = result.get("metrics")
        if not isinstance(metrics, dict):
            raise ValueError("Each result must contain a 'metrics' dictionary")
        if "bias" not in metrics or "rmse" not in metrics:
            raise ValueError("Each result metrics dictionary must contain 'bias' and 'rmse'")

        biases.append(metrics["bias"])
        rmses.append(metrics["rmse"])

    return {
        "n_profiles": len(results),
        "mean_bias": sum(biases) / len(biases),
        "mean_rmse": sum(rmses) / len(rmses),
    }
