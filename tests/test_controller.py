import unittest

from autoscaler.controller import Action, AutoscalerController, AutoScalerConfig
from autoscaler.metrics import MetricSample


def make_config(**overrides):
    defaults = dict(
        target_utilization=0.7,
        min_replicas=2,
        max_replicas=10,
        startup_delay_ticks=3,
        tick_interval_seconds=15.0,
        metric_max_age_ticks=1,
        up_tolerance=0.1,
        down_tolerance=0.25,
        scale_up_cooldown=1,
        min_up_observation_window=1,
        scale_down_cooldown=6,
        min_down_observation_window=3,
        max_scale_down_fraction=0.5,
        queue_drain_ticks=3,
        latency_saturation=2.0,
        safety_sigma=1.2,
        max_forecast_sigma_fraction=1.0,
        ewma_alpha=0.45,
        seasonal_period=None,
    )
    defaults.update(overrides)
    return AutoScalerConfig(**defaults)


def sample(t, cpu, queue=0.0, p95=0.2, replicas=2):
    return MetricSample(t, cpu_utilization=cpu, request_queue=queue,
                        p95_latency=p95, current_replicas=replicas)


class StaleMetricTest(unittest.TestCase):
    def test_missing_metric_holds_and_never_scales_down(self):
        cfg = make_config()
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5, replicas=4))

        missing = MetricSample(t=0, current_replicas=4)
        decision = controller.observe(missing, now=2)
        self.assertEqual(decision.action, Action.HOLD)
        self.assertEqual(decision.desired_replicas, 4)

    def test_delayed_sample_is_treated_as_stale(self):
        cfg = make_config()
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5))
        delayed = sample(1, 0.1, replicas=2)
        decision = controller.observe(delayed, now=3)
        self.assertEqual(decision.action, Action.HOLD)
        self.assertIsNone(decision.forecast)

    def test_future_timestamp_is_stale(self):
        cfg = make_config()
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5))
        future = sample(5, 0.1, replicas=2)
        decision = controller.observe(future, now=1)
        self.assertEqual(decision.action, Action.HOLD)


class ScaleUpTest(unittest.TestCase):
    def test_sustained_load_scales_up_quickly(self):
        cfg = make_config()
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5, replicas=2))

        actions = []
        for t in range(1, 6):
            decision = controller.observe(sample(t, 0.98, replicas=2))
            actions.append(decision.action)
        self.assertIn(Action.SCALE_UP, actions)
        self.assertLessEqual(actions.index(Action.SCALE_UP), 3)

    def test_max_replicas_is_hard_cap(self):
        cfg = make_config(max_replicas=5)
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5, replicas=5))
        for t in range(1, 12):
            decision = controller.observe(sample(t, 1.0, replicas=5))
            self.assertLessEqual(decision.desired_replicas, 5)
        self.assertEqual(controller.target_replicas, 5)

    def test_pending_replicas_prevent_duplicate_scale_up(self):
        cfg = make_config(scale_up_cooldown=1)
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5, replicas=2))
        first = controller.observe(sample(1, 0.99, replicas=2))
        self.assertEqual(first.action, Action.SCALE_UP)
        for replica in range(first.desired_replicas - 2):
            controller.note_scale_event(1 + cfg.startup_delay_ticks)
        second = controller.observe(sample(2, 0.99, replicas=2))
        self.assertEqual(second.action, Action.HOLD)


class ScaleDownTest(unittest.TestCase):
    def _warm_up(self, controller):
        controller.observe(sample(0, 0.55, replicas=4))
        for t in range(1, 5):
            controller.observe(sample(t, 0.6, replicas=4))

    def _drive_first_scale_down(self, controller, start=5):
        # 先满足观察窗口完成一次缩容，作为后续冷却判断的起点
        t = start
        while True:
            decision = controller.observe(sample(t, 0.12, replicas=4))
            t += 1
            if decision.action == Action.SCALE_DOWN:
                return decision, t

    def test_down_requires_observation_window_and_cooldown(self):
        cfg = make_config(min_down_observation_window=3, scale_down_cooldown=6)
        controller = AutoscalerController(cfg)
        self._warm_up(controller)

        early = [controller.observe(sample(t, 0.12, replicas=4))
                 for t in range(5, 7)]
        self.assertTrue(all(d.action == Action.HOLD for d in early))

        decision = controller.observe(sample(7, 0.12, replicas=4))
        self.assertEqual(decision.action, Action.SCALE_DOWN)

        # 冷却期内即使持续低负载也不能再次缩容
        first_down_tick = 7
        cooldown_holds = [
            controller.observe(sample(t, 0.12, replicas=decision.desired_replicas))
            for t in range(first_down_tick + 1,
                           first_down_tick + cfg.scale_down_cooldown)
        ]
        self.assertTrue(all(d.action != Action.SCALE_DOWN for d in cooldown_holds))

    def test_single_jitter_tick_resets_observation_window(self):
        cfg = make_config(min_down_observation_window=3, scale_down_cooldown=1)
        controller = AutoscalerController(cfg)
        self._warm_up(controller)

        for t in range(5, 7):
            controller.observe(sample(t, 0.12, replicas=4))
        # 一次抖动把负载拉回容忍带内，观察窗口应重置
        controller.observe(sample(8, 0.6, replicas=4))
        # 抖动后只重新累计了 t=9,10 两个 tick，窗口未满足
        controller.observe(sample(9, 0.12, replicas=4))
        decision = controller.observe(sample(10, 0.12, replicas=4))
        self.assertEqual(decision.action, Action.HOLD)
        # 第三个连续 tick（t=11）才允许缩容
        decision = controller.observe(sample(11, 0.12, replicas=4))
        self.assertEqual(decision.action, Action.SCALE_DOWN)

    def test_scale_down_is_fraction_capped(self):
        cfg = make_config(min_down_observation_window=2,
                          scale_down_cooldown=2,
                          max_scale_down_fraction=0.25)
        controller = AutoscalerController(cfg)
        self._warm_up(controller)
        decision, _ = self._drive_first_scale_down(controller)
        self.assertEqual(decision.action, Action.SCALE_DOWN)
        # 4 副本最多缩 25% -> 至少保留 3 个
        self.assertGreaterEqual(decision.desired_replicas, 3)

    def test_never_scales_below_min_replicas(self):
        cfg = make_config(min_replicas=2, min_down_observation_window=2,
                          scale_down_cooldown=2)
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5, replicas=2))
        for t in range(1, 10):
            decision = controller.observe(sample(t, 0.05, replicas=2))
            self.assertGreaterEqual(decision.desired_replicas, 2)


class SignalFusionTest(unittest.TestCase):
    def test_queue_backlog_drives_demand_even_if_cpu_low(self):
        cfg = make_config(target_utilization=0.7, queue_drain_ticks=3)
        controller = AutoscalerController(cfg)
        controller.observe(sample(0, 0.5, queue=0.0, replicas=4))
        decision = controller.observe(sample(1, 0.5, queue=20.0, replicas=4))
        self.assertIsNotNone(decision.observed_demand)
        self.assertGreater(decision.observed_demand, 4.0)


if __name__ == "__main__":
    unittest.main()
