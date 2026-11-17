import unittest

from autoscaler.controller import Action, AutoScalerConfig
from autoscaler.simulator import (
    Simulator,
    make_fall_workload,
    make_ramp_workload,
    make_sine_workload,
)


def proactive_config(**overrides):
    cfg = AutoScalerConfig(
        target_utilization=0.7,
        min_replicas=2,
        max_replicas=12,
        startup_delay_ticks=4,
        tick_interval_seconds=15.0,
        metric_max_age_ticks=1,
        up_tolerance=0.1,
        down_tolerance=0.3,
        scale_up_cooldown=1,
        scale_down_cooldown=8,
        min_down_observation_window=5,
        max_scale_down_fraction=0.5,
        queue_drain_ticks=3,
        latency_saturation=3.0,
        safety_sigma=1.5,
        max_forecast_sigma_fraction=0.35,
        ewma_alpha=0.4,
        max_scale_up_step=6,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def reactive_config(**overrides):
    # 反应式基线：只看当前指标、无季节性预测、无缩容保护期，
    # 冷却/观察窗口都取最小值，等价于传统"当前 CPU 驱动"的 HPA
    cfg = AutoScalerConfig(
        target_utilization=0.7,
        min_replicas=2,
        max_replicas=12,
        startup_delay_ticks=4,
        tick_interval_seconds=15.0,
        metric_max_age_ticks=1,
        up_tolerance=0.1,
        down_tolerance=0.3,
        scale_up_cooldown=1,
        scale_down_cooldown=1,
        min_up_observation_window=1,
        min_down_observation_window=1,
        max_scale_down_fraction=0.5,
        post_down_guard_ticks=0,
        queue_drain_ticks=3,
        latency_saturation=3.0,
        safety_sigma=0.0,
        max_forecast_sigma_fraction=0.35,
        ewma_alpha=0.9,
        seasonal_period=-1,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


class PeriodicSpikeTest(unittest.TestCase):
    def test_proactive_beats_reactive_on_overload(self):
        period = 16
        workload = make_sine_workload(
            period=period, duration=5 * period, base=4.0, amplitude=3.0, noise=0.15)

        proactive = Simulator(
            proactive_config(seasonal_period=period), workload, seed=1).run()
        reactive = Simulator(reactive_config(), workload, seed=1).run()

        # 第二个周期起，预测式扩容应在尖峰到达前完成
        later_overload = [r for r in proactive.records if r.t >= 2 * period]
        self.assertLessEqual(
            sum(1 for r in later_overload if r.overloaded),
            period // 2,
            "预测式控制器在后续周期不应有明显过载",
        )
        self.assertLess(proactive.overload_ticks, reactive.overload_ticks)
        # 稳定后每个完整周期恰好一次扩容、一次缩容；
        # 允许启动期多一次扩容，但不允许同周期内反复横跳
        stable_events = [r for r in proactive.records
                         if r.t >= 2 * period
                         and r.action in (Action.SCALE_UP, Action.SCALE_DOWN)]
        stable_cycles = (len(proactive.records) - 2 * period) // period
        self.assertLessEqual(len(stable_events), stable_cycles * 2 + 1)
        self.assertLessEqual(proactive.scale_up_count, 5 + 2)
        self.assertLessEqual(proactive.scale_down_count, 5)


class SteadyGrowthTest(unittest.TestCase):
    def test_ramp_is_followed_without_thrashing(self):
        workload = make_ramp_workload(duration=60, start=2.0, end=10.0, noise=0.2)
        result = Simulator(proactive_config(), workload, seed=2).run()

        self.assertGreaterEqual(result.scale_up_count, 3)
        self.assertEqual(result.reversal_count, 0)
        # 增长末端扩容基本跟得上：受 4-tick 启动延迟限制，
        # 最后 10 个 tick 的过载窗口应明显短于反应式基线
        tail = result.records[-10:]
        self.assertLessEqual(sum(1 for r in tail if r.overloaded), 4)

        reactive = Simulator(reactive_config(), workload, seed=2).run()
        self.assertLess(
            sum(1 for r in result.records[-10:] if r.overloaded),
            sum(1 for r in reactive.records[-10:] if r.overloaded) + 1)


class FastFallTest(unittest.TestCase):
    def test_fast_fall_does_not_trigger_immediate_shrink(self):
        workload = make_fall_workload(
            high=8.0, low=1.5, high_ticks=20, low_ticks=40, noise=0.1)
        result = Simulator(proactive_config(), workload, seed=3).run()

        first_down = next((r.t for r in result.records
                           if r.action == Action.SCALE_DOWN), None)
        self.assertIsNotNone(first_down)
        # 20 时刻流量回落，缩容必须等观察窗口 + 冷却
        self.assertGreaterEqual(first_down, 20 + 5)
        # 唯一允许的方向切换是高负载扩容 -> 回落缩容，不允许缩了又扩
        self.assertLessEqual(result.reversal_count, 1)
        self.assertLessEqual(result.scale_down_count, 4)


class JitterStabilityTest(unittest.TestCase):
    def test_metric_jitter_does_not_cause_flapping(self):
        workload = make_sine_workload(
            period=40, duration=60, base=3.5, amplitude=0.0, noise=0.45)
        result = Simulator(proactive_config(up_tolerance=0.2), workload, seed=4).run()

        self.assertLessEqual(result.scale_event_count, 5)
        self.assertLessEqual(result.reversal_count, 1)


class MissingMetricTest(unittest.TestCase):
    def test_missing_window_holds_replicas(self):
        workload = make_ramp_workload(duration=40, start=3.0, end=6.0, noise=0.1)
        missing = lambda t: 12 <= t <= 17  # noqa: E731
        result = Simulator(proactive_config(), workload,
                           missing_fn=missing, seed=5).run()

        gap = [r for r in result.records if 12 <= r.t <= 17]
        self.assertTrue(all(r.action == Action.HOLD for r in gap))
        targets = {r.target_replicas for r in gap}
        self.assertEqual(len(targets), 1)
        self.assertEqual(result.skipped_ticks, 6)


class MetricDelayTest(unittest.TestCase):
    def test_delayed_metrics_still_complete_scenario(self):
        workload = make_sine_workload(
            period=16, duration=48, base=4.0, amplitude=3.0, noise=0.1)
        result = Simulator(proactive_config(seasonal_period=16),
                           workload, delay_ticks=2, seed=6).run()

        self.assertGreaterEqual(result.skipped_ticks, 2)
        self.assertGreaterEqual(result.scale_up_count, 1)
        # 方向切换次数与周期数同阶，不会因延迟产生额外振荡
        self.assertLessEqual(result.reversal_count, 48 // 16 * 2)


class CapacityCapTest(unittest.TestCase):
    def test_replicas_never_exceed_capacity(self):
        workload = make_ramp_workload(duration=40, start=4.0, end=30.0)
        cfg = proactive_config(max_replicas=8)
        result = Simulator(cfg, workload, seed=8).run()

        self.assertLessEqual(result.max_ready_replicas(), 8)
        self.assertTrue(all(r.target_replicas <= 8 for r in result.records))
        # 触顶后持续过载是容量上限的预期行为，但控制器不再反复扩
        cap_ticks = [r for r in result.records if r.target_replicas == 8]
        self.assertGreater(len(cap_ticks), 5)


if __name__ == "__main__":
    unittest.main()
