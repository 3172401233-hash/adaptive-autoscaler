# adaptive-autoscaler

预测式 Python 自动伸缩控制器。针对“业务流量有明显周期、副本从创建到可用
需要几分钟”的场景，用负载预测替代“只看当前 CPU”的反应式伸缩，并通过
冷却、hysteresis 和最小观察窗口抑制指标抖动导致的反复扩缩。

## 为什么不是普通 HPA

只看当前 CPU 的反应式控制器有两个固有问题：

- **扩得太晚**：副本需要 `startup_delay_ticks` 个采样周期才就绪，等到
  CPU 打满再扩容，尖峰期间必然持续过载。
- **反复横跳**：流量回落时指标抖动会不断触发扩容/缩容，缩完又扩。

本控制器把“副本等价需求”作为预测对象，在尖峰到达前提前扩容，缩容则
刻意保守且缓慢。

## 模块结构

- `autoscaler/metrics.py`：指标样本模型、staleness（缺失/延迟/乱序）判定。
- `autoscaler/forecaster.py`：需求预测器（Holt 双指数 EWMA + 季节性
  naive + 残差安全边际）。
- `autoscaler/controller.py`：多信号负载折算、扩缩容决策与状态机。
- `autoscaler/simulator.py`：离散时间仿真器与负载曲线生成器，用于离线
  评估过载时间、扩缩次数和稳定性。
- `tests/`：预测器/控制器单元测试 + 六类端到端场景测试。

## 预测窗口

预测窗口（`horizon`）直接等于启动延迟：

```
horizon = max(1, startup_delay_ticks)
```

含义：控制器在每个 tick 预测“距今一个启动延迟之后”的需求，从而现在就
提交扩容请求，让新副本恰好在预测尖峰落地时就绪。`tick_interval_seconds`
只用于把 tick 换算成真实时间，不参与决策。

预测器输出三个值：

- `point`：点预测。非季节性模式为 `EWMA 水平 + 斜率 × horizon`，且
  不低于当前平滑水平（防止增长期被平滑成下降）；显式配置周期时取
  历史同相位需求（季节性 naive）。
- `upper`：保守上界，`point + safety_sigma × σ`。σ 来自历史残差，
  并被 `max_forecast_sigma_fraction` 限幅，避免抖动把上界无限放大。
- `smoothed`：当前 EWMA 平滑水平，供缩容判断使用。

季节性**只在显式配置 `seasonal_period` 时启用**，不做自动周期检测——
短序列上的自相关检测极易把噪声误判成周期，伸缩场景宁可不预测也不误判。

## 保守性设计

扩容和缩容刻意不对称：扩容快、缩容慢。

**扩容（偏激进但有确认）**

- 触发信号用预测**上界**（含安全边际），宁可多备一点容量。
- 计入在途副本（`target + pending`），避免启动延迟内重复扩容。
- 仅需 `min_up_observation_window`（默认 2 tick）确认 + 短冷却。
- 单次扩容受 `max_scale_up_step` 限制，防止一步跳变。

**缩容（多重保守闸门）**

- 用 EWMA **平滑需求**而不是瞬时值，单点抖动不触发。
- 必须连续满足 `min_down_observation_window`（默认 4 tick）。
- 必须通过 `scale_down_cooldown`（默认 8 tick）缩容冷却。
- hysteresis 容忍带 `down_tolerance`（默认 25%）+ `min_replica_delta`
  死区，带内一律 HOLD。
- 单次最多缩 `max_scale_down_fraction`（默认 50%），且只收缩“确认
  富余”的副本，不按点预测一步缩穿。
- 非季节性模式下检测到上升趋势（`trend_slope`）时禁止缩容。
- 缩容后 `post_down_guard_ticks` 内禁止反向扩容，吸收抖动。

**指标缺失/延迟**

- 样本年龄超过 `metric_max_age_ticks + expected_metric_age_ticks`
  即判定 stale，控制器只输出 HOLD，不更新需求历史，**绝不基于猜测缩容**。
- 管道延迟通过 `expected_metric_age_ticks` 显式告知，延迟但在预算内
  的样本仍视为新鲜。

**负载折算的抗振荡关键**

观测需求用绝对量 `ready × cpu / target_utilization` 估算，它只取决于
外部真实负载，不会因为控制器自己扩容摊薄了 CPU 就“假性跳水”。队列
积压和超阈值 p95 延迟作为额外容量需求做加法叠加，无积压/无延迟超阈
时不贡献信号，避免低负载基线误触发缩容。

## 容量边界

所有决策都被 `min_replicas` / `max_replicas` 硬钳制；达到上限后即使
持续过载也不会反复提交扩容（见 `tests/test_scenarios.py` 的
`CapacityCapTest`）。

## 运行测试

```bash
python -m unittest
```

共 25 个测试，覆盖：

| 场景 | 验证点 |
| --- | --- |
| 周期尖峰 | 预测式过载 tick 数显著少于同启动延迟的反应式基线，稳定后每周期至多一扩一缩 |
| 持续增长 | 无方向反复，尾部过载窗口短于反应式基线 |
| 快速回落 | 回落后必须等观察窗口 + 冷却才首次缩容，不允许缩了又扩 |
| 指标抖动 | 扩缩事件与方向反复次数受限 |
| 缺失指标 | 缺口内全部 HOLD 且目标副本不变 |
| 指标延迟 | 延迟预算内样本仍可用，不产生额外振荡 |
| 容量上限 | 副本数永不超过 `max_replicas` |

仿真结果通过 `SimulationResult` 暴露 `overload_ticks`、
`scale_up_count`、`scale_down_count`、`reversal_count` 等指标，
便于量化过载时间与稳定性。

## 最小用法

```python
from autoscaler import AutoscalerController, AutoScalerConfig, MetricSample

controller = AutoscalerController(AutoScalerConfig(
    target_utilization=0.7,
    min_replicas=2,
    max_replicas=20,
    startup_delay_ticks=4,   # 副本 4 个采样周期才就绪
    seasonal_period=16,      # 业务周期为 16 个采样周期；无周期设 None
))

decision = controller.observe(MetricSample(
    t=10,
    cpu_utilization=0.82,
    request_queue=12.0,
    p95_latency=0.45,
    current_replicas=6,
))
print(decision.action, decision.desired_replicas, decision.reason)
```
