"""预测式自动伸缩控制器。

设计原则：
- 扩容快：只要预测上界超过目标 + 上行容忍度，立即扩，冷却短；
- 缩容慢：点预测持续低于目标 - 下行容忍度，满足最小观察窗口
  且经过缩容冷却后才缩，并限制单次缩容幅度（hysteresis）；
- 指标缺失/延迟/乱序时保持 HOLD，绝不基于猜测缩容；
- 预测窗口 = 启动延迟对应的采样周期数，让新副本在尖峰到达时就绪。
"""

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .forecaster import DemandForecaster, Forecast
from .metrics import MetricSample, is_stale


class Action(str, Enum):
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    HOLD = "hold"


@dataclass
class AutoScalerConfig:
    target_utilization: float = 0.7
    min_replicas: int = 1
    max_replicas: int = 20
    startup_delay_ticks: int = 4
    tick_interval_seconds: float = 15.0
    # 指标最大可接受年龄（采样周期数）；超过即视为延迟/缺失
    metric_max_age_ticks: int = 1
    # 已知的指标管道延迟：该年龄内的样本仍视为新鲜
    expected_metric_age_ticks: int = 0
    # hysteresis：目标两侧的容忍带（占目标副本比例）
    up_tolerance: float = 0.1
    down_tolerance: float = 0.25
    # 冷却（tick 数）
    scale_up_cooldown: int = 1
    scale_down_cooldown: int = 8
    # 扩容确认窗口：上行条件连续满足的 tick 数（仍远短于缩容窗口）
    min_up_observation_window: int = 2
    # 缩容最小观察窗口：条件需连续满足这么多 tick
    min_down_observation_window: int = 4
    # 单次最多缩掉当前目标的该比例，避免一步缩穿
    max_scale_down_fraction: float = 0.5
    # 单次最多扩容的副本数（None 表示不限）
    max_scale_up_step: Optional[int] = 4
    # 缩容后的反向保护期：期间禁止再扩容（吸收指标抖动）
    post_down_guard_ticks: int = 4
    # 队列与延迟折算参数
    queue_drain_ticks: int = 3
    latency_saturation: float = 1.0
    # 预测保守性（残差标准差倍数）
    safety_sigma: float = 1.5
    # 残差安全边际占预测值的上限，防止抖动把上界无限放大
    max_forecast_sigma_fraction: float = 1.0
    # EWMA 平滑系数：越大越跟随近期变化，越小越抗抖
    ewma_alpha: float = 0.45
    seasonal_period: Optional[int] = None
    # 小于该副本数的预测偏差不触发伸缩（抖动死区）
    min_replica_delta: float = 0.75


@dataclass
class ScaleDecision:
    action: Action
    t: int
    desired_replicas: int
    current_target: int
    ready_replicas: int
    observed_demand: Optional[float]
    forecast: Optional[Forecast]
    reason: str


class AutoscalerController:
    def __init__(self, config: AutoScalerConfig) -> None:
        self.config = config
        self._forecaster = DemandForecaster(
            horizon=max(1, config.startup_delay_ticks),
            period=config.seasonal_period,
            safety_sigma=config.safety_sigma,
            max_sigma_fraction=config.max_forecast_sigma_fraction,
            ewma_alpha=config.ewma_alpha,
            # 仅当显式配置周期时启用季节性；自动检测容易把
            # 噪声/短序列误判成周期，伸缩场景宁可不预测也不误判
            enable_seasonality=(config.seasonal_period is not None
                                and config.seasonal_period > 0),
        )
        self._demand_history: List[float] = []
        self._target_replicas: Optional[int] = None
        self._pending: List[int] = []  # 每个在途副本的就绪 tick
        self._last_scale_up_tick: Optional[int] = None
        self._last_scale_down_tick: Optional[int] = None
        self._down_since: Optional[int] = None
        self._up_since: Optional[int] = None
        self.last_decision: Optional[ScaleDecision] = None

    # ---- 副本生命周期（由外部平台/模拟器回调） ----

    def note_scale_event(self, ready_tick: int) -> None:
        """记录一个已提交、将于 ready_tick 就绪的新副本。"""

        self._pending.append(ready_tick)

    def _advance_pending(self, now: int) -> None:
        self._pending = [tick for tick in self._pending if tick > now]

    @property
    def target_replicas(self) -> int:
        return self._target_replicas if self._target_replicas is not None else 0

    @property
    def pending_replicas(self) -> int:
        return len(self._pending)

    # ---- 主入口 ----

    def observe(self, sample: MetricSample, now: Optional[int] = None) -> ScaleDecision:
        now = sample.t if now is None else now
        cfg = self.config

        acceptable_age = cfg.metric_max_age_ticks + cfg.expected_metric_age_ticks
        if is_stale(sample, now, acceptable_age):
            self._down_since = None
            self._up_since = None
            decision = ScaleDecision(
                action=Action.HOLD,
                t=now,
                desired_replicas=self.target_replicas,
                current_target=self.target_replicas,
                ready_replicas=sample.current_replicas or 0,
                observed_demand=None,
                forecast=None,
                reason="指标缺失或延迟，保持当前副本（禁止猜测性缩容）",
            )
            self.last_decision = decision
            return decision

        self._advance_pending(now)
        ready = sample.current_replicas or 0

        if self._target_replicas is None:
            self._target_replicas = max(cfg.min_replicas, ready)
            self._demand_history.append(self._observed_demand(sample, ready))
            decision = ScaleDecision(
                action=Action.HOLD,
                t=now,
                desired_replicas=self._target_replicas,
                current_target=self._target_replicas,
                ready_replicas=ready,
                observed_demand=self._demand_history[-1],
                forecast=None,
                reason="首个有效采样，以当前副本为基准",
            )
            self.last_decision = decision
            return decision

        demand = self._observed_demand(sample, ready)
        self._demand_history.append(demand)
        forecast = self._forecaster.forecast(self._demand_history)
        return self._decide(sample, now, ready, demand, forecast)

    # ---- 负载折算：CPU 绝对负载为基线，队列/p95 压力做加法叠加 ----

    def _observed_demand(self, sample: MetricSample, ready: int) -> float:
        cfg = self.config
        # 绝对负载估算：当前利用率 * 当前副本数，得到"满载副本等价数"，
        # 再归一到目标利用率。该值只取决于真实外部负载，不会因为控制器
        # 自己扩容（摊薄 CPU）而跳水，是抑制扩缩振荡的关键。
        cpu_demand: Optional[float] = None

        if sample.cpu_utilization is not None and ready > 0:
            cpu_demand = ready * sample.cpu_utilization / cfg.target_utilization

        extra = 0.0
        if sample.request_queue is not None and ready > 0:
            per_replica_capacity = 1.0 / cfg.target_utilization
            if sample.request_queue > 0.0:
                extra += sample.request_queue / (
                    per_replica_capacity * cfg.queue_drain_ticks)

        if sample.p95_latency is not None:
            # 延迟超过饱和阈值的部分，折算为需要追加的容量（加法信号，
            # 避免低负载时延迟基线被误当成容量需求）
            base = ready if ready > 0 else cfg.min_replicas
            excess = max(0.0, sample.p95_latency - cfg.latency_saturation)
            if excess > 0.0:
                extra += base * excess / cfg.latency_saturation

        if cpu_demand is None:
            return float(ready) + extra
        return cpu_demand + extra

    # ---- 决策 ----

    def _decide(self, sample: MetricSample, now: int, ready: int,
                demand: float, forecast: Forecast) -> ScaleDecision:
        cfg = self.config
        target = self._target_replicas or cfg.min_replicas
        up_threshold = target * (1.0 + cfg.up_tolerance)
        down_threshold = target * (1.0 - cfg.down_tolerance)

        # 扩容用预测上界（保守），且计入已在途副本，避免重复扩容
        effective_capacity = target + len(self._pending)
        post_down_guard = (
            self._last_scale_down_tick is not None
            and now - self._last_scale_down_tick < cfg.post_down_guard_ticks
        )
        # 上行触发：预测上界超过已提交容量一个最小步长，
        # 且越过目标的上行容忍带（双重确认，避免在带内抖动）
        need_more = forecast.upper - effective_capacity > cfg.min_replica_delta
        beyond_band = forecast.upper > up_threshold
        if not post_down_guard and need_more and beyond_band:
            self._down_since = None
            if self._up_since is None:
                self._up_since = now
            up_confirmed = (now - self._up_since + 1
                            >= cfg.min_up_observation_window)
            if (up_confirmed and self._cooldown_ok(
                    now, self._last_scale_up_tick, cfg.scale_up_cooldown)):
                wanted = max(target + 1, int(-(-forecast.upper // 1)))
                if cfg.max_scale_up_step is not None:
                    wanted = min(wanted, target + cfg.max_scale_up_step)
                desired = min(cfg.max_replicas, wanted)
                if desired > target:
                    self._up_since = None
                    return self._scale(Action.SCALE_UP, desired, now, ready,
                                       demand, forecast,
                                       "预测上界 %.2f 超过上行阈值 %.2f"
                                       % (forecast.upper, up_threshold))
            return self._hold(now, ready, demand, forecast,
                              "需要扩容但处于扩容冷却/在途副本已覆盖")

        # 缩容看点预测（不带上界），且必须持续满足观察窗口
        # 季节性模式下点预测已包含同相位信息，直接用点预测；
        # 非季节性模式用平滑水平，避免趋势外推噪声触发误缩
        down_signal = forecast.point if forecast.period else forecast.smoothed
        rising_market = forecast.period is None and forecast.trend_slope > 0.02
        # 双保险：增长中即使斜率瞬时变小，只要当前平滑需求本身
        # 没有跌破下行阈值，就不允许缩容
        demand_collapsing = forecast.smoothed < down_threshold
        if (not rising_market
                and down_signal < down_threshold
                and demand_collapsing
                and (target - down_signal) > cfg.min_replica_delta):
            if self._down_since is None:
                self._down_since = now
            observed_for = now - self._down_since + 1
            window_ok = observed_for >= cfg.min_down_observation_window
            cooldown_ok = self._cooldown_ok(
                now, self._last_scale_down_tick, cfg.scale_down_cooldown)
            if window_ok and cooldown_ok:
                floor = max(cfg.min_replicas,
                            int(target * (1.0 - cfg.max_scale_down_fraction)))
                # 只收缩确认富余的部分：保留目标利用率余量，避免一步缩穿
                desired = max(floor,
                              target - max(1, int((target - down_signal)
                                                  * cfg.target_utilization)))
                if desired < target:
                    return self._scale(Action.SCALE_DOWN, desired, now, ready,
                                       demand, forecast,
                                       "平滑需求 %.2f 持续低于下行阈值 %.2f"
                                       % (down_signal, down_threshold))
            return self._hold(now, ready, demand, forecast,
                              "缩容条件成立但观察窗口/冷却未满足")

        # 进入 hysteresis 容忍带或压力回升：重置缩容观察
        self._down_since = None
        self._up_since = None
        return self._hold(now, ready, demand, forecast,
                          "预测位于 hysteresis 容忍带内")

    def _cooldown_ok(self, now: int, last_tick: Optional[int],
                     cooldown: int) -> bool:
        return last_tick is None or now - last_tick >= cooldown

    def _scale(self, action: Action, desired: int, now: int, ready: int,
               demand: float, forecast: Forecast, reason: str) -> ScaleDecision:
        self._target_replicas = desired
        if action == Action.SCALE_UP:
            self._last_scale_up_tick = now
        else:
            self._last_scale_down_tick = now
            self._down_since = None
            self._up_since = None
        decision = ScaleDecision(
            action=action,
            t=now,
            desired_replicas=desired,
            current_target=desired,
            ready_replicas=ready,
            observed_demand=demand,
            forecast=forecast,
            reason=reason,
        )
        self.last_decision = decision
        return decision

    def _hold(self, now: int, ready: int, demand: float,
              forecast: Forecast, reason: str) -> ScaleDecision:
        decision = ScaleDecision(
            action=Action.HOLD,
            t=now,
            desired_replicas=self.target_replicas,
            current_target=self.target_replicas,
            ready_replicas=ready,
            observed_demand=demand,
            forecast=forecast,
            reason=reason,
        )
        self.last_decision = decision
        return decision
