"""离散时间仿真器：驱动控制器面对可重复的负载场景。

每个 tick：
1. 工作负载给出归一化总 CPU 压力（1.0 = 一个满载副本）、队列、p95；
2. 处理缺失/延迟注入；
3. 控制器决策并提交副本，副本经过 startup_delay 后就绪；
4. 按就绪容量计算利用率、过载与队列堆积。
"""

import random
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .controller import Action, AutoscalerController, AutoScalerConfig
from .metrics import MetricSample


@dataclass
class Workload:
    """负载曲线：load[t] 为 t 时刻的总 CPU 压力（副本单位）。"""

    load: List[float]
    queue: List[float]
    p95: List[float]


@dataclass
class TickRecord:
    t: int
    offered_load: float
    ready_replicas: int
    target_replicas: int
    cpu_utilization: Optional[float]
    queue: Optional[float]
    p95: Optional[float]
    overloaded: bool
    action: Action
    reason: str
    sample_skipped: bool


@dataclass
class SimulationResult:
    records: List[TickRecord] = field(default_factory=list)

    @property
    def overload_ticks(self) -> int:
        return sum(1 for r in self.records if r.overloaded)

    @property
    def scale_up_count(self) -> int:
        return sum(1 for r in self.records if r.action == Action.SCALE_UP)

    @property
    def scale_down_count(self) -> int:
        return sum(1 for r in self.records if r.action == Action.SCALE_DOWN)

    @property
    def scale_event_count(self) -> int:
        return self.scale_up_count + self.scale_down_count

    @property
    def reversal_count(self) -> int:
        """方向反复次数：扩容后立即缩容，或缩容后立即扩容。"""

        events = [r.action for r in self.records
                  if r.action in (Action.SCALE_UP, Action.SCALE_DOWN)]
        return sum(1 for a, b in zip(events, events[1:]) if a != b)

    @property
    def skipped_ticks(self) -> int:
        return sum(1 for r in self.records if r.sample_skipped)

    def max_ready_replicas(self) -> int:
        return max((r.ready_replicas for r in self.records), default=0)


class Simulator:
    def __init__(self,
                 config: AutoScalerConfig,
                 workload: Workload,
                 missing_fn: Optional[Callable[[int], bool]] = None,
                 delay_ticks: int = 0,
                 overload_cpu: float = 0.95,
                 seed: int = 7) -> None:
        self.config = config
        self.workload = workload
        self.missing_fn = missing_fn or (lambda t: False)
        self.delay_ticks = max(0, delay_ticks)
        self.overload_cpu = overload_cpu
        config.expected_metric_age_ticks = self.delay_ticks

    def run(self) -> SimulationResult:
        cfg = self.config
        controller = AutoscalerController(cfg)
        result = SimulationResult()
        ready_replica_count = cfg.min_replicas
        pending_at: List[int] = []  # 在途副本的就绪 tick
        pending_queue: List[MetricSample] = []
        duration = len(self.workload.load)

        for t in range(duration):
            arriving = sum(1 for tick in pending_at if tick <= t)
            ready_replica_count += arriving
            pending_at = [tick for tick in pending_at if tick > t]
            ready = ready_replica_count

            load = self.workload.load[t]
            queue = self.workload.queue[t]
            p95 = self.workload.p95[t]
            # 与真实平台一致：CPU 利用率只统计就绪副本，在途副本
            # 由控制器的预测窗口和在途容量判断覆盖
            cpu = load / ready if ready > 0 else float("inf")

            overloaded = cpu > self.overload_cpu or queue > ready

            sample: Optional[MetricSample] = None
            skipped = False
            if not self.missing_fn(t):
                sample = MetricSample(
                    t=t,
                    cpu_utilization=cpu,
                    request_queue=queue,
                    p95_latency=p95,
                    current_replicas=ready,
                )

            delivered: Optional[MetricSample] = None
            if sample is not None:
                if self.delay_ticks == 0:
                    delivered = sample
                else:
                    pending_queue.append(sample)
                    while (pending_queue
                           and pending_queue[0].t <= t - self.delay_ticks):
                        delivered = pending_queue.pop(0)
            if sample is None:
                skipped = True

            if delivered is not None:
                decision = controller.observe(delivered, now=t)
            else:
                # 本 tick 没有可用样本：用时间戳过旧的空样本触发 staleness/HOLD
                stale_age = cfg.metric_max_age_ticks + self.delay_ticks + 1
                stale_sample = MetricSample(t=t - stale_age,
                                            current_replicas=ready)
                decision = controller.observe(stale_sample, now=t)

            if decision.action == Action.SCALE_UP:
                in_flight = len(pending_at)
                to_submit = decision.desired_replicas - ready - in_flight
                for _ in range(max(0, to_submit)):
                    pending_at.append(t + cfg.startup_delay_ticks)
                    controller.note_scale_event(t + cfg.startup_delay_ticks)

            if decision.action == Action.SCALE_DOWN:
                ready_replica_count = min(ready_replica_count,
                                          decision.desired_replicas)

            result.records.append(TickRecord(
                t=t,
                offered_load=load,
                ready_replicas=ready,
                target_replicas=controller.target_replicas,
                cpu_utilization=cpu if delivered is not None else None,
                queue=queue if delivered is not None else None,
                p95=p95 if delivered is not None else None,
                overloaded=overloaded,
                action=decision.action,
                reason=decision.reason,
                sample_skipped=skipped or delivered is None,
            ))

        return result


def make_sine_workload(period: int,
                       duration: int,
                       base: float,
                       amplitude: float,
                       noise: float = 0.0,
                       seed: int = 11) -> Workload:
    """正弦周期负载，队列/p95 随利用率派生。"""

    rng = random.Random(seed)
    import math

    load: List[float] = []
    queue: List[float] = []
    p95: List[float] = []
    for t in range(duration):
        value = base + amplitude * math.sin(2 * math.pi * t / period)
        value += rng.uniform(-noise, noise)
        value = max(0.1, value)
        load.append(value)
        queue.append(max(0.0, value - 4.0) * 2.0)
        p95.append(0.1 + 0.25 * value)
    return Workload(load=load, queue=queue, p95=p95)


def make_ramp_workload(duration: int, start: float, end: float,
                       noise: float = 0.0, seed: int = 13) -> Workload:
    rng = random.Random(seed)
    load: List[float] = []
    queue: List[float] = []
    p95: List[float] = []
    for t in range(duration):
        value = start + (end - start) * t / max(1, duration - 1)
        value += rng.uniform(-noise, noise)
        value = max(0.1, value)
        load.append(value)
        queue.append(max(0.0, value - 4.0) * 2.0)
        p95.append(0.1 + 0.25 * value)
    return Workload(load=load, queue=queue, p95=p95)


def make_fall_workload(high: float, low: float, high_ticks: int,
                       low_ticks: int, noise: float = 0.0,
                       seed: int = 17) -> Workload:
    rng = random.Random(seed)
    duration = high_ticks + low_ticks
    load: List[float] = []
    queue: List[float] = []
    p95: List[float] = []
    for t in range(duration):
        level = high if t < high_ticks else low
        value = max(0.1, level + rng.uniform(-noise, noise))
        load.append(value)
        queue.append(max(0.0, value - 4.0) * 2.0)
        p95.append(0.1 + 0.25 * value)
    return Workload(load=load, queue=queue, p95=p95)
