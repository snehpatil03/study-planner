"""
monitor.py
-----------

Utilities for recording inference metrics to Amazon CloudWatch.  SageMaker
automatically captures a number of invocation metrics for your endpoints and
sends them to CloudWatch:contentReference[oaicite:1]{index=1}.  This module allows the application to publish
additional custom metrics such as request latency, number of retrieved
documents or model confidence scores.

To use these functions you must configure AWS credentials via environment
variables or IAM roles.  Custom metrics will be sent to a namespace
``RAG/Inference`` by default.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import boto3

logger = logging.getLogger(__name__)


def push_metric(
    metric_name: str,
    value: float,
    *,
    unit: str = "Milliseconds",
    dimensions: Optional[List[Dict[str, str]]] = None,
    namespace: str = "RAG/Inference",
) -> None:
    """Send a custom metric to Amazon CloudWatch.

    Parameters
    ----------
    metric_name: str
        Name of the metric, e.g. "Latency" or "RetrievedCount".
    value: float
        The metric value.
    unit: str, default "Milliseconds"
        Unit for the metric.  See the CloudWatch documentation for valid values.
    dimensions: list of dicts, optional
        Additional dimensions to associate with the metric.  Each dict must
        contain the keys ``"Name"`` and ``"Value"``.
    namespace: str, default "RAG/Inference"
        CloudWatch namespace under which to publish the metric.
    """
    try:
        cw = boto3.client("cloudwatch")
        cw.put_metric_data(
            Namespace=namespace,
            MetricData=[
                {
                    "MetricName": metric_name,
                    "Value": value,
                    "Unit": unit,
                    **({"Dimensions": dimensions} if dimensions else {}),
                }
            ],
        )
    except Exception as e:
        # Log errors but do not crash the application
        logger.warning("Failed to push metric %s: %s", metric_name, e)
