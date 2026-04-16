"""Tests for contract-driven reporting metrics."""

from __future__ import annotations

import unittest

import numpy as np

from src.task_runtime import (
    flatten_task_metrics_scalars,
    summarize_task_metrics,
    summarize_task_metrics_from_chunks,
)
from src.topologies import resolve_topology_spec
from src.train import _format_yaw_metric_log_line


def _yaw_sin_cos(deg: float) -> tuple[float, float]:
    rad = np.deg2rad(float(deg))
    return float(np.sin(rad)), float(np.cos(rad))


class TaskRuntimeReportingTests(unittest.TestCase):
    def test_multitask_yaw_reporting_uses_declared_thresholds(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_dual_stream_yaw",
            topology_variant="dual_stream_yaw_v0_1",
            topology_params={},
        )

        pred_0 = _yaw_sin_cos(3.0)
        pred_1 = _yaw_sin_cos(12.0)
        true = _yaw_sin_cos(0.0)

        metrics = summarize_task_metrics(
            prediction_heads={
                "distance": np.asarray([[1.0], [2.0]], dtype=np.float32),
                "orientation": np.asarray([pred_0, pred_1], dtype=np.float32),
            },
            target_heads={
                "distance": np.asarray([[1.0], [2.0]], dtype=np.float32),
                "orientation": np.asarray([true, true], dtype=np.float32),
            },
            task_contract=spec.task_contract,
            tolerance_values=(0.10, 0.25, 0.50),
            primary_tolerance=0.10,
            rows=[
                {"yaw_deg": 0.0},
                {"yaw_deg": 0.0},
            ],
            collect_predictions=False,
        )

        orientation = metrics.task_metrics["orientation"]
        self.assertAlmostEqual(float(orientation["yaw_mean_error_deg"]), 7.5, places=4)
        self.assertAlmostEqual(float(orientation["yaw_median_error_deg"]), 7.5, places=4)
        self.assertAlmostEqual(float(orientation["yaw_acc@5deg"]), 0.5, places=4)
        self.assertAlmostEqual(float(orientation["yaw_acc@10deg"]), 0.5, places=4)
        self.assertAlmostEqual(float(orientation["yaw_acc@15deg"]), 1.0, places=4)

        flattened = flatten_task_metrics_scalars(metrics.task_metrics)
        self.assertIn("yaw_mean_error_deg", flattened)
        self.assertIn("yaw_acc@15deg", flattened)

        yaw_log_line = _format_yaw_metric_log_line(metrics.task_metrics, spec.task_contract)
        self.assertTrue(yaw_log_line.startswith("        "))
        self.assertIn("yaw_acc@5deg", yaw_log_line)
        self.assertIn("yaw_acc@15deg", yaw_log_line)

    def test_multitask_chunk_summary_matches_direct_summary(self) -> None:
        spec = resolve_topology_spec(
            topology_id="distance_regressor_dual_stream_yaw",
            topology_variant="dual_stream_yaw_v0_1",
            topology_params={},
        )

        direct = summarize_task_metrics(
            prediction_heads={
                "distance": np.asarray([[1.05], [2.10]], dtype=np.float32),
                "orientation": np.asarray(
                    [_yaw_sin_cos(2.0), _yaw_sin_cos(11.0)],
                    dtype=np.float32,
                ),
            },
            target_heads={
                "distance": np.asarray([[1.00], [2.00]], dtype=np.float32),
                "orientation": np.asarray(
                    [_yaw_sin_cos(0.0), _yaw_sin_cos(0.0)],
                    dtype=np.float32,
                ),
            },
            task_contract=spec.task_contract,
            tolerance_values=(0.10, 0.25, 0.50),
            primary_tolerance=0.10,
            collect_predictions=False,
        )

        from_chunks = summarize_task_metrics_from_chunks(
            prediction_head_chunks={
                "distance": [
                    np.asarray([[1.05]], dtype=np.float32),
                    np.asarray([[2.10]], dtype=np.float32),
                ],
                "orientation": [
                    np.asarray([[_yaw_sin_cos(2.0)[0], _yaw_sin_cos(2.0)[1]]], dtype=np.float32),
                    np.asarray([[_yaw_sin_cos(11.0)[0], _yaw_sin_cos(11.0)[1]]], dtype=np.float32),
                ],
            },
            target_head_chunks={
                "distance": [
                    np.asarray([[1.00]], dtype=np.float32),
                    np.asarray([[2.00]], dtype=np.float32),
                ],
                "orientation": [
                    np.asarray([[_yaw_sin_cos(0.0)[0], _yaw_sin_cos(0.0)[1]]], dtype=np.float32),
                    np.asarray([[_yaw_sin_cos(0.0)[0], _yaw_sin_cos(0.0)[1]]], dtype=np.float32),
                ],
            },
            task_contract=spec.task_contract,
            tolerance_values=(0.10, 0.25, 0.50),
            primary_tolerance=0.10,
            collect_predictions=False,
        )

        self.assertAlmostEqual(from_chunks.mae, direct.mae, places=6)
        self.assertAlmostEqual(from_chunks.rmse, direct.rmse, places=6)
        self.assertEqual(
            flatten_task_metrics_scalars(from_chunks.task_metrics),
            flatten_task_metrics_scalars(direct.task_metrics),
        )


if __name__ == "__main__":
    unittest.main()
