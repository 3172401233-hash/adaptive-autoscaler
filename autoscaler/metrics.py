"""指标数据模型与缺失/延迟处理工具。"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class MetricSample:
    """单个采样点；所有字段为 None 表示该指标在本点缺失。"""

    t: int
    cpu_utilization: Optional[float] = None
    request_queue: Optional[float] = None
    p95_latency: Optional[float] = None
    current_replicas: Optional[int] = None

    def usable_fields(self) -> List[str]:
        fields = []
        if self.cpu_utilization is not None:
            fields.append("cpu_utilization")
        if self.request_queue is not None:
            fields.append("request_queue")
        if self.p95_latency is not None:
            fields.append("p95_latency")
        if self.current_replicas is not None:
            fields.append("current_replicas")
        return fields


def is_stale(sample: Optional[MetricSample], now: int, max_age: int) -> bool:
    """指标延迟/中断检测：采样时间过旧即视为不可用。"""

    if sample is None:
        return True
    if sample.t > now:
        return True
    return now - sample.t > max_age


def interpolate_missing(values: List[Optional[float]]) -> List[Optional[float]]:
    """短缺口线性插值，用于统计/测试；首尾缺失保持 None。

    控制器本身不依赖插值做决策（缺失时保持 HOLD），这里仅提供给
    预测器评估历史连续性等离线场景使用。
    """

    out: List[Optional[float]] = list(values)
    n = len(out)
    i = 0
    while i < n:
        if out[i] is not None:
            i += 1
            continue
        start = i
        while i < n and out[i] is None:
            i += 1
        end = i
        if start == 0 or end == n:
            continue
        left = out[start - 1]
        right = out[end]
        span = end - start + 1
        for j in range(start, end):
            out[j] = left + (right - left) * (j - start + 1) / span
    return out
