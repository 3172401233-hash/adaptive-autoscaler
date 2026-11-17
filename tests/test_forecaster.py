import unittest

from autoscaler.forecaster import DemandForecaster, detect_period


class PeriodDetectionTest(unittest.TestCase):
    def test_detects_sine_period(self):
        period = 8
        values = [2 + 1.5 * _sin(t, period) for t in range(48)]
        self.assertEqual(detect_period(values), period)

    def test_flat_series_has_no_period(self):
        self.assertIsNone(detect_period([3.0] * 20))

    def test_short_history_returns_none(self):
        self.assertIsNone(detect_period([1.0, 2.0]))


class ForecastTest(unittest.TestCase):
    def test_growth_trend_forecasts_above_last_value(self):
        history = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        forecast = DemandForecaster(horizon=2, enable_seasonality=False).forecast(history)
        self.assertGreater(forecast.point, history[-1])
        self.assertGreaterEqual(forecast.trend_slope, 0.7)
        self.assertGreater(forecast.upper, forecast.point)

    def test_periodic_peak_anticipated_ahead_of_time(self):
        period = 8
        horizon = 3
        history = [4 + 3 * _sin(t, period) for t in range(32)]
        forecast = DemandForecaster(horizon=horizon, period=period).forecast(history)
        future_phase_value = 4 + 3 * _sin(32 - 1 + horizon, period)
        # 未来相位处于上升沿：预测应明显高于当前水平并接近同相位历史值
        self.assertAlmostEqual(forecast.point, future_phase_value, delta=1.2)
        self.assertGreater(forecast.point, history[-1])
        self.assertEqual(forecast.period, period)

    def test_upper_bound_is_conservative(self):
        history = [2.0, 2.1, 1.9, 2.05, 1.95, 2.0]
        forecast = DemandForecaster(horizon=2).forecast(history)
        self.assertGreaterEqual(forecast.upper, forecast.point)
        self.assertGreaterEqual(forecast.lower, 0.0)

    def test_empty_history_raises(self):
        with self.assertRaises(ValueError):
            DemandForecaster(horizon=1).forecast([])


def _sin(t: int, period: int) -> float:
    import math

    return math.sin(2 * math.pi * t / period)


if __name__ == "__main__":
    unittest.main()
