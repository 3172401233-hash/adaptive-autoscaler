"""负载（副本等价需求）预测器。

控制器每个 tick 把多指标负载折算成"副本等价需求"序列，本模块对该
序列做短期预测，预测窗口由外部按启动延迟传入（见 controller.py）。

方法：
- 跨 tick 保持状态的双指数平滑（Holt EWMA）同时给出平滑水平与斜率，
  对单点噪声不敏感且不会因窗口滑动产生跳变；
- 显式配置周期时用季节性 naive 项（历史同相位需求）捕捉周期尖峰；
- 残差标准差形成保守上界（含比例上限），吸收指标抖动与预测不确定性。

未配置周期时只做水平 + 趋势外推；数据不足时退化为水平预测。
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple


def detect_period(values: Sequence[float], min_period: int = 3,
                  max_period: Optional[int] = None,
                  min_correlation: float = 0.5) -> Optional[int]:
    """用自相关检测最强周期；不显著时返回 None。"""

    n = len(values)
    if n < 4:
        return None
    upper = max_period or n // 2
    upper = min(upper, n - 2)
    if upper < min_period:
        return None

    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values)
    if variance <= 1e-12:
        return None

    best_period: Optional[int] = None
    best_score = min_correlation
    for lag in range(min_period, upper + 1):
        pairs = [(values[i], values[i + lag]) for i in range(n - lag)]
        m = len(pairs)
        mx = sum(p[0] for p in pairs) / m
        my = sum(p[1] for p in pairs) / m
        cov = sum((a - mx) * (b - my) for a, b in pairs)
        vx = sum((a - mx) ** 2 for a, _ in pairs)
        vy = sum((b - my) ** 2 for _, b in pairs)
        denom = (vx * vy) ** 0.5
        if denom <= 1e-12:
            continue
        score = cov / denom
        if score > best_score:
            best_score = score
            best_period = lag
    return best_period


def _ewma(values: Sequence[float], alpha: float) -> Tuple[float, float]:
    """双指数平滑，返回 (当前水平, 每步斜率)。

    相比滚动 OLS，EWMA 对单点噪声不敏感且对近期变化响应更快，
    是伸缩控制器抑制指标抖动的标准手段。
    """

    level = values[0]
    trend = 0.0
    if len(values) >= 2:
        trend = values[1] - values[0]
    for value in values[1:]:
        new_level = alpha * value + (1 - alpha) * (level + trend)
        trend = alpha * (new_level - level) + (1 - alpha) * trend
        level = new_level
    return level, trend


@dataclass
class Forecast:
    horizon: int
    point: float
    upper: float
    lower: float
    smoothed: float
    period: Optional[int]
    trend_slope: float


class DemandForecaster:
    """对副本等价需求序列做趋势 + 季节性预测。"""

    def __init__(self,
                 horizon: int,
                 period: Optional[int] = None,
                 safety_sigma: float = 1.5,
                 min_history: int = 4,
                 enable_seasonality: bool = True,
                 max_sigma_fraction: float = 1.0,
                 ewma_alpha: float = 0.45) -> None:
        if horizon < 1:
            raise ValueError("horizon 必须 >= 1 个采样周期")
        self.horizon = horizon
        self.period = period
        self.safety_sigma = safety_sigma
        self.min_history = min_history
        self.enable_seasonality = enable_seasonality
        self.max_sigma_fraction = max_sigma_fraction
        self.ewma_alpha = ewma_alpha
        self._level: Optional[float] = None
        self._trend: float = 0.0
        self._last_n: int = 0

    def reset(self) -> None:
        self._level = None
        self._trend = 0.0
        self._last_n = 0

    def _stateful_ewma(self, values: List[float]) -> Tuple[float, float]:
        """跨调用保持状态的 EWMA：只对新到达的样本做增量更新，
        避免滚动窗口滑动时重新初始化造成的平滑值跳变。"""

        if self._level is None or self._last_n > len(values):
            self.reset()
            start = 0
        else:
            start = self._last_n

        if self._level is None:
            self._level = values[0]
            self._trend = 0.0
            start = 1

        for value in values[start:]:
            new_level = (self.ewma_alpha * value
                         + (1 - self.ewma_alpha) * (self._level + self._trend))
            self._trend = (self.ewma_alpha * (new_level - self._level)
                           + (1 - self.ewma_alpha) * self._trend)
            self._level = new_level
        self._last_n = len(values)
        return self._level, self._trend

    def forecast(self, history: Sequence[float]) -> Forecast:
        values = list(history)
        n = len(values)
        if n == 0:
            raise ValueError("历史序列为空，无法预测")

        ewma_level, slope = self._stateful_ewma(values)
        period = self.period
        if period is not None and period <= 0:
            period = None
        elif period is None and self.enable_seasonality:
            period = detect_period(values)

        use_seasonal = period is not None and n >= period
        if use_seasonal:
            # 季节性 naive：取同一相位的最近观测。EWMA 斜率主要由季节
            # 波动本身产生，不能再线性叠加，否则会把尖峰预测成低谷
            phase = (n - 1 + self.horizon) % period
            point = values[phase]
            fitted = [values[i % period] for i in range(n)]
            season_ewma = ewma_level
        else:
            point = ewma_level + slope * self.horizon
            # 用 EWMA 平滑值构造残差，让 sigma 反映去噪后的真实波动
            fitted = []
            level = values[0]
            for value in values:
                level = self.ewma_alpha * value + (1 - self.ewma_alpha) * level
                fitted.append(level)

        # 非季节性场景下，点预测不得低于当前平滑水平：避免强平滑 +
        # 长窗口下趋势外推把增长中的需求预测成下降
        if not use_seasonal:
            point = max(point, ewma_level)

        if n < self.min_history:
            sigma = max(0.1 * point, 0.25)
        else:
            if use_seasonal:
                # 季节性残差只取同相位内的波动，避免整段季节振幅
                # 被当成不确定性而把安全边际撑得过大
                phase_groups = {}
                for i, value in enumerate(values):
                    phase_groups.setdefault(i % period, []).append(value)
                residuals = []
                for i, value in enumerate(values):
                    group = phase_groups[i % period]
                    phase_mean = sum(group) / len(group)
                    residuals.append(value - phase_mean)
            else:
                residuals = [v - f for v, f in zip(values, fitted)]
            variance = sum(r * r for r in residuals) / n
            # 上界只吸收合理波动：残差超过 point 一定比例的部分视为
            # 结构性变化（由趋势/季节性项解释），不重复计入安全边际
            reference = max(ewma_level, point)
            sigma_floor = max(0.05 * reference, 0.1)
            sigma = min(max(variance ** 0.5, sigma_floor),
                        max(sigma_floor, self.max_sigma_fraction * reference))

        margin = self.safety_sigma * sigma
        # 上界至少覆盖当前平滑水平 + 安全边际，保证增长期不会被压回当前值
        upper_base = season_ewma if use_seasonal else ewma_level
        upper = max(point, upper_base) + margin
        lower = max(0.0, point - margin)
        smoothed_for_decision = point if use_seasonal else ewma_level
        return Forecast(
            horizon=self.horizon,
            point=max(0.0, point),
            upper=max(0.0, upper),
            lower=lower,
            smoothed=max(0.0, smoothed_for_decision),
            period=period,
            trend_slope=slope,
        )

    def smoothed_level(self, history: Sequence[float]) -> float:
        """返回 EWMA 平滑后的当前需求水平，供缩容判断使用。"""

        values = list(history)
        if not values:
            raise ValueError("历史序列为空，无法平滑")
        level, _ = self._stateful_ewma(values)
        return max(0.0, level)
