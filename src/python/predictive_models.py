#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测性维护 - 模型核心库
========================
  - 时序特征工程（滑动窗口统计 / 趋势 / 差分 / 异常计数 / 时间特征）
  - 模型 A：无监督异常检测（IsolationForest）
  - 模型 B：有监督风险预测（XGBoost / LightGBM / sklearn），预测"未来 N 步是否异常"
  - RUL 回归（距下次异常的步数）
  - 可解释性（SHAP 优先 → feature_importances_ → 排列重要性）
  - 在线预测接口（供 FastAPI / DeepSeek Agent 工具调用）

数据输入约定（与 main.py 传感器读数一致）：
  readings: [{timestamp, sensor_id, device_type, status, metrics:{...}}]
"""
import os
import json
import numpy as np
import pandas as pd

# ---- 可选依赖（缺失自动降级）----
try:
    import joblib
except ImportError:
    joblib = None
try:
    import xgboost as xgb
    HAVE_XGB = True
except ImportError:
    HAVE_XGB = False
try:
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:
    HAVE_LGB = False
try:
    import shap
    HAVE_SHAP = True
except ImportError:
    HAVE_SHAP = False

from sklearn.ensemble import (IsolationForest, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score,
                             precision_score, recall_score, mean_absolute_error, r2_score)
try:
    from sklearn.inspection import permutation_importance
    HAVE_PERM = True
except ImportError:
    HAVE_PERM = False

# 指标列（与 main.py 的 SENSOR_METRIC_DEFS 一致）
METRICS = ["pressure", "flow", "temperature", "gas_concentration", "level", "vibration"]

# RUL 回归目标的右删失上限 = horizon * RUL_CAP_FACTOR。
# 必须与 online_predict 里 risk_score 的 min(rul/(horizon*2), 1) 用同一个倍数：风险分在该
# 上限之上完全无法区分 24 步和 120 步，让回归器去学这段差别等于把容量浪费在产品丢掉的
# 信息上。原来取 10 倍（horizon=12 → cap=120），实测 88% 的测试帧被删失在 cap 上，回归器
# 学成「一律预测 ~cap」，于是大屏出现「异常概率 99.9% 但 RUL 75 步」这种自相矛盾的输出。
RUL_CAP_FACTOR = 2

# 阈值 / 分档 / 建单线 / 权重的**唯一定义处**。
# 这些常量原来在三个文件里各写了一遍（本文件的 RUL 投影硬编码 0.5、main.py 的风险档
# 硬编码 80/60/40、自动建单线硬编码 60 分与兜底 0.7、agent_brain 又各写一套），于是同一张
# 预警卡片会自相矛盾（"12 步内必然异常"与"还要 75 步"并存），"危急"档几乎不可达导致紧急工单
# 永远升不上去。现在只在这里定义一份：训练时写进 predictive_meta.json，推理端、台账、Agent
# 与前端一律读 meta，代码不再各写死一份。
DEFAULT_CONFIG = {
    # 兜底判正线：只在模型 meta 里没有阈值时使用（模型太老或阈值扫描失败）
    "risk_threshold": 0.7,
    # critical 线 = warn + (1 - warn) * crit_midpoint
    "crit_midpoint": 0.5,
    # risk_score = (w_prob * prob + w_anomaly * anomaly_score + w_rul * rul项) * 100
    "risk_score_weights": {"prob": 0.5, "anomaly": 0.3, "rul": 0.2},
    # 风险分档（用于台账/看板配色与分级），键名与前端一致
    "risk_levels": {"critical": 80, "warning": 60, "attention": 40},
    # 风险分达到该线才自动建工单
    "workorder_min_score": 60,
    # RUL 回归目标的右删失上限倍数，必须与 risk_score 的截断点一致
    "rul_cap_factor": RUL_CAP_FACTOR,
    # 扫描节拍（帧）。必须 <= horizon：horizon=12 时每 20 帧扫一次意味着每轮约 40% 的时间段
    # 没有任何预测覆盖，落在那段的异常既不计 hit 也不计 miss，漏报率看起来比真实情况好。
    "scan_every_frames": 5,
}


def get_config(overrides=None):
    """返回一份配置副本；overrides 里非 None 的键覆盖默认值（供读 meta / 环境变量用）。"""
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULT_CONFIG.items()}
    for k, v in (overrides or {}).items():
        if v is not None and k in cfg:
            cfg[k] = v
    return cfg


def readings_to_frame(readings):
    """读数列表 -> DataFrame（含指标列 + status + timestamp + sensor_id），按时间升序。"""
    rows = []
    for r in readings or []:
        m = r.get("metrics") or {}
        row = {
            "timestamp": r.get("timestamp", r.get("t", "")),
            "sensor_id": r.get("sensor_id", ""),
            "device_type": r.get("device_type", r.get("pipe_type", "")),
            "status": r.get("status", "normal"),
        }
        for mk in METRICS:
            row[mk] = m.get(mk) if m.get(mk) is not None else np.nan
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["timestamp", "sensor_id", "device_type", "status"] + METRICS)
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    # 必须用稳定排序：同一 tick 的所有传感器 timestamp 完全相同，默认 quicksort 不稳定，
    # 相同时间戳的行序每次都不一样，会让训练结果无法复现
    df = df.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
    return df


def build_features(df, window=15):
    """构造滑动窗口统计特征。返回特征 DataFrame（丢弃前 window 行的 NaN）。"""
    feats = {}
    for m in METRICS:
        if m not in df.columns:
            continue
        col = df[m]
        if col.isna().all():
            continue
        feats[f"{m}_val"] = col
        feats[f"{m}_mean{window}"] = col.rolling(window, min_periods=1).mean()
        feats[f"{m}_std{window}"] = col.rolling(window, min_periods=1).std()
        feats[f"{m}_min{window}"] = col.rolling(window, min_periods=1).min()
        feats[f"{m}_max{window}"] = col.rolling(window, min_periods=1).max()
        feats[f"{m}_diff"] = col.diff()
        feats[f"{m}_slope{window}"] = col.rolling(window, min_periods=2).apply(
            lambda x: float(np.polyfit(np.arange(len(x)), np.asarray(x, dtype=float), 1)[0]), raw=True)
        # 跨管网类型 pooled 训练时原始量纲不可比：燃气压力 range=0.10 MPa，供暖流量 range=50 m³/h，
        # 同一个阈值在不同管型上含义完全不同；而全网劣化管段只有约 8 个，模型没有足够正样本
        # 为每种管型各学一套阈值。用因果 expanding z-score 把每个传感器每个指标拉到同一量纲，
        # 劣化趋势就能跨管型复用（仿真实测劣化管段的 slope AUC 0.88 → 0.93）。
        # 必须用 expanding 而非整列 mean/std：整列统计含未来信息，会让离线指标虚高、上线掉分。
        e_mu = col.expanding(min_periods=1).mean()
        e_sd = col.expanding(min_periods=2).std()
        z = ((col - e_mu) / e_sd.replace(0, np.nan)).fillna(0.0)
        feats[f"{m}_z"] = z
        feats[f"{m}_zslope{window}"] = z.rolling(window, min_periods=2).apply(
            lambda x: float(np.polyfit(np.arange(len(x)), np.asarray(x, dtype=float), 1)[0]), raw=True)
    feats["window_anomaly_cnt"] = (df["status"] != "normal").astype(float).rolling(window, min_periods=1).sum()
    if not df.empty:
        feats["hour"] = df["timestamp"].dt.hour
        feats["dow"] = df["timestamp"].dt.dayofweek
    out = pd.DataFrame(feats, index=df.index)
    return out.iloc[window - 1:].dropna(axis=1, how="all").fillna(0)


def future_anomaly_labels(anomaly_series, horizon):
    """label[i]=1 表示 i+1..i+horizon 步内将出现异常；-1 表示**未知**。

    尾部 horizon 帧的未来窗口会被数据末尾截断。原实现把"看不全"的情况一律写成 0，
    而每个传感器的测试集恰好是末尾 20%——120 帧时最后 12 帧（horizon）必然被标 0，
    占测试集（≈21 帧）约 57%。测试集基准率因此被系统性压低，写进 meta 的判正阈值、
    precision / recall / 命中率全部建立在这个截尾偏差上。

    现在三态：
      · 窗口内**确实观察到**异常           → 1（确定正，截断也算，因为已经看到了）
      · 窗口完整且无异常                    → 0（确定负）
      · 窗口被截断且可见区间内无异常        → -1（未知，训练/阈值扫描/评估三处都必须剔除）
    """
    arr = anomaly_series.values
    n = len(arr)
    out = np.full(n, -1.0)
    for i in range(n):
        lo = i + 1
        hi = min(lo + horizon, n)
        seen = bool(arr[lo:hi].any()) if lo < n else False
        if seen:
            out[i] = 1.0
        elif lo + horizon <= n:
            out[i] = 0.0
        # else: 窗口被截断且未见异常，保持 -1.0（未知）
    return pd.Series(out, index=anomaly_series.index)


def rul_targets(anomaly_series, cap=200):
    """距*下一次*异常的步数，超出 cap 或数据末尾再无异常时记为 cap（右删失）。

    只看 i+1 之后、不看当前帧 arr[i]，这一点必须和 future_anomaly_labels 保持一致：
    两者都排除当前帧，才有「label==1 ⟺ rul < horizon」这个恒等关系（实测 12000 帧
    上成立比例 1.0000）。若改成「当前异常即 0」，正在异常的帧会得到 rul=0 而 label=0，
    恒等关系破裂，分类器与 RUL 回归器的输出就无法互相校验。
    """
    arr = anomaly_series.values
    n = len(arr)
    out = np.full(n, float(cap))
    for i in range(n):
        j = i + 1
        while j < n and arr[j] == 0:
            j += 1
        out[i] = float(min(j - i - 1, cap)) if j < n else float(cap)
    return pd.Series(out, index=anomaly_series.index)


def consistent_rul(prob, rul, horizon, threshold=0.5):
    """把 RUL 预测投影到分类器的判定在逻辑上允许的区间，消除两个头互相打脸的输出。

    model_b 的正类定义就是「未来 horizon 步内出现异常」，rul_targets 数的是同一
    anomaly_series 上到下次异常的步数，两者都排除当前帧，因此恒有
        真值 label==1  ⟺  真值 rul < horizon
    （实测 12000 帧成立比例 1.0000）。两个模型各自独立输出时违反它的比例实测高达
    57.4%（151/263），大屏于是把「12 步内必然异常」和「还要 75 步才异常」印在同一张
    卡片上。投影后违反数为 0，代价是未删失子集 MAE 5.16→5.26、R² -1.105→-1.211。

    要清楚这是在强制与*分类器的判定*一致，不是让 RUL 变得更准：分类器判错时，两个数
    会一起错，而不是一个说快出事、一个说还早。对运维和大屏来说，一致地错比自相矛盾
    可用——后者会让人无法判断该信哪个。

    标量与数组皆可，返回同形状结果。

    threshold 必须传**当前生效的判正线**（bundle.threshold），默认 0.5 只是兼容旧调用的
    兜底。原实现把边界硬编码成 0.5，而线上判正线实测是 0.80，于是 prob ∈ [0.5, 0.8) 的点位
    得到 predicted_status="normal" 却同时输出 rul < horizon，前端渲染出"正常 + 约 11 步内
    可能异常"——自相矛盾只是从"两个头之间"搬到了"状态与 RUL 之间"。
    """
    p = np.asarray(prob, dtype=float)
    r = np.asarray(rul, dtype=float)
    h = float(max(horizon, 1))
    thr = float(threshold) if threshold is not None else 0.5
    out = np.where(p >= thr, np.minimum(r, h - 1.0), np.maximum(r, h))
    return float(out) if np.ndim(prob) == 0 and np.ndim(rul) == 0 else out


def make_classifier():
    if HAVE_LGB:
        return lgb.LGBMClassifier(n_estimators=250, learning_rate=0.05, max_depth=6,
                                  num_leaves=63, random_state=42, verbose=-1)
    if HAVE_XGB:
        return xgb.XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.1,
                                 random_state=42, eval_metric="logloss")
    return HistGradientBoostingClassifier(max_iter=250, random_state=42)


def make_regressor():
    if HAVE_LGB:
        return lgb.LGBMRegressor(n_estimators=250, learning_rate=0.05, max_depth=6,
                                 random_state=42, verbose=-1)
    if HAVE_XGB:
        return xgb.XGBRegressor(n_estimators=250, max_depth=6, learning_rate=0.1, random_state=42)
    return HistGradientBoostingRegressor(max_iter=250, random_state=42)


class PredictiveBundle:
    """打包训练好的 A/B/RUL 模型、特征列、window/horizon，支持保存/加载/在线预测。"""

    def __init__(self, model_a=None, model_b=None, model_rul=None, features=None,
                 window=15, horizon=12, y_mean=None, threshold=None,
                 threshold_source=None, config=None, a_score_mean=None, a_score_std=None):
        self.model_a = model_a
        self.model_b = model_b
        self.model_rul = model_rul
        self.features = features or []
        self.window = window
        self.horizon = horizon
        self.y_mean = y_mean
        # A 头 score_samples 在**训练集**上的均值/标准差。IsolationForest 的 score_samples
        # 典型区间只有约 (-0.8, -0.3)，直接 sigmoid 会把它压成 0.31~0.43 的近似常数，
        # risk_score 里 0.3 的权重形同虚设。按训练分布标准化后再 sigmoid，方向正确且量纲稳定。
        self.a_score_mean = a_score_mean
        self.a_score_std = a_score_std
        # 判正阈值：**在训练段内部切出的验证集**上 F1 最优的那个点，而不是拍脑袋的常数，
        # 更不是测试集上扫出来的（那会让报告的 precision/recall/F1 全部带乐观偏差）。
        # 为 None 时表示没有可信阈值，online_predict 退回 config 里的 risk_threshold 默认值。
        self.threshold = threshold
        # 阈值来源：validation_scan / train_only_fallback / none。只有光看阈值数字分不清
        # 它是扫出来的还是兜底的，落进 meta 后线上也能核对。
        self.threshold_source = threshold_source
        # 阈值/分档/建单线/权重/扫描节拍的统一配置（见 DEFAULT_CONFIG）
        self.config = get_config(config)

    def save(self, out_dir):
        if joblib is None:
            raise RuntimeError("未安装 joblib，无法保存模型：pip install joblib")
        os.makedirs(out_dir, exist_ok=True)
        meta = {"features": self.features, "window": self.window, "horizon": self.horizon,
                "y_mean": self.y_mean, "threshold": self.threshold,
                # 阈值来源与统一配置一起落盘：main.py / agent_brain / 前端都读这里，
                # 不再各自硬编码（原来 RUL 投影 0.5、风险档 80/60/40、建单线 60 分各写一份）
                "threshold_source": self.threshold_source,
                "config": self.config,
                "rul_cap_factor": self.config.get("rul_cap_factor", RUL_CAP_FACTOR),
                "a_score_mean": self.a_score_mean,
                "a_score_std": self.a_score_std,
                "engine": {"lightgbm": HAVE_LGB, "xgboost": HAVE_XGB, "shap": HAVE_SHAP}}
        with open(os.path.join(out_dir, "predictive_meta.json"), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        joblib.dump({"model_a": self.model_a, "model_b": self.model_b,
                     "model_rul": self.model_rul}, os.path.join(out_dir, "predictive_models.joblib"))

    @classmethod
    def load(cls, out_dir):
        with open(os.path.join(out_dir, "predictive_meta.json"), "r", encoding="utf-8") as f:
            meta = json.load(f)
        data = joblib.load(os.path.join(out_dir, "predictive_models.joblib"))
        threshold = meta.get("threshold")
        threshold_source = meta.get("threshold_source")
        if threshold is None:
            # meta 里没有就是阈值上线前存的模型。best_f1_threshold 一直有写进
            # 同目录的 predictive_report.json，捞回来即可，不必逼人重训一遍。
            # 注意：那种旧阈值是在测试集上扫的，来源标记为 legacy_test_scan，口径偏乐观。
            try:
                with open(os.path.join(out_dir, "predictive_report.json"), "r", encoding="utf-8") as f:
                    threshold = json.load(f).get("best_f1_threshold")
                    if threshold is not None:
                        threshold_source = "legacy_test_scan"
            except Exception:
                threshold = None
        # 旧模型 meta 里没有 config：用默认值补齐，保证新代码能直接跑旧 checkpoint。
        cfg = get_config(meta.get("config"))
        if meta.get("rul_cap_factor") is not None:
            cfg["rul_cap_factor"] = meta["rul_cap_factor"]
        return cls(model_a=data["model_a"], model_b=data["model_b"], model_rul=data["model_rul"],
                   features=meta["features"], window=meta["window"], horizon=meta["horizon"],
                   y_mean=meta.get("y_mean"), threshold=threshold,
                   threshold_source=threshold_source, config=cfg,
                   a_score_mean=meta.get("a_score_mean"), a_score_std=meta.get("a_score_std"))


def _build_training_frames(df, window, horizon):
    """按 sensor_id 分组做特征工程与打标，返回 (feat, y_future, y_rul, is_test, gids)。

    gids 是逐行的传感器标识，供 train_bundle 按传感器切验证集用（阈值只能在验证集上扫，
    且不能跨传感器混切）。y_future 为三态：1 确定正 / 0 确定负 / -1 未知（尾部窗口被截断）。

    分组是必须的，不能图省事直接对 pooled df 做 rolling：同一 tick 里 100 个传感器的
    timestamp 完全相同，排序后彼此相邻，rolling(15) 算出来的是「同一时刻 15 个不同
    传感器」的统计量，「未来 12 步」也退化成「接下来 12 个传感器是否异常」。特征和
    要预测的未来之间没有因果关系，AUC 必然≈0.5，正样本率还会被推到 85% 以上。
    """
    if "sensor_id" in df.columns and df["sensor_id"].nunique() > 1:
        groups = [g for _, g in df.groupby("sensor_id", sort=True)]
    else:
        groups = [df]

    feats, yfs, yrs, tests, gids = [], [], [], [], []
    for g in groups:
        g = g.sort_values("timestamp", kind="mergesort").reset_index(drop=True)
        f = build_features(g, window)
        if f.empty:
            continue
        anomaly = (g["status"] != "normal").astype(int)
        feats.append(f)
        yfs.append(future_anomaly_labels(anomaly, horizon).loc[f.index].values)
        yrs.append(rul_targets(anomaly, cap=horizon * RUL_CAP_FACTOR).loc[f.index].values)
        # 每个传感器各自留出末尾 20% 作测试集：既保住「用过去预测未来」的时序语义，
        # 又让劣化管段在训练集与测试集里都有分布——整体按位置切分会把它们全甩到一侧。
        t = np.zeros(len(f), dtype=bool)
        t[int(len(f) * 0.8):] = True
        tests.append(t)
        # 逐行记录所属传感器：阈值必须在**按传感器分组**的验证集上扫，跨传感器的行绝不能
        # 混进同一个 split——同一 tick 的行没有因果先后，混着切会让"未来"泄漏进特征。
        sid = str(g["sensor_id"].iloc[0]) if ("sensor_id" in g.columns and len(g)) else ""
        gids.append(np.full(len(f), sid, dtype=object))

    if not feats:
        return (pd.DataFrame(), np.array([]), np.array([]), np.array([], dtype=bool),
                np.array([], dtype=object))
    # 各管网类型的指标集不同，分组 dropna 后列不一致；concat 取并集，缺口留 NaN，下游按
    # 模型能力处理（树的三个引擎都支持 NaN 分裂，但 sklearn 的 IsolationForest 不支持，
    # 所以 A 头单独喂一份填 0 的矩阵，见 train_bundle）
    return (pd.concat(feats, ignore_index=True).fillna(0),
            np.concatenate(yfs), np.concatenate(yrs), np.concatenate(tests),
            np.concatenate(gids))


def train_bundle(df, window=15, horizon=12, anomaly_contamination="auto"):
    """训练 A/B/RUL，返回 (bundle, report)。df 为 readings_to_frame 输出的时序 DataFrame（可 pooled）。"""
    if len(df) < 30:
        raise ValueError("样本太少（需累积至少 window 长度的传感器历史），请先采集数据再训练")
    feat, y_future, y_rul, is_test, gids = _build_training_frames(df, window, horizon)
    if feat.empty or len(feat) < 30:
        raise ValueError("特征样本太少（需累积至少 window 长度的传感器历史），请先采集数据再训练")
    feature_cols = list(feat.columns)
    X = feat[feature_cols].values
    cfg = get_config()
    rul_cap = horizon * int(cfg["rul_cap_factor"])

    errors = []

    # 标签里有 -1（尾部窗口被截断且未见异常，见 future_anomaly_labels）。这些帧必须从
    # 训练、阈值扫描、评估三处**全部剔除**：原实现把它们一律写成 0，而测试集恰是每个
    # 传感器的末尾 20%，于是测试集基准率被系统性压低，阈值与 precision/recall 全带偏差。
    known = y_future >= 0
    n_unknown = int((~known).sum())

    reading_anomaly_rate = float((df["status"] != "normal").mean()) if "status" in df.columns else 0.0
    y_known = y_future[known]
    pos_all = float(np.mean(y_known == 1)) if len(y_known) else 0.0
    if len(set(y_known.tolist())) < 2:
        # 早失败：单一类别会让 sklearn 的 predict_proba 只返回一列，后面 [:, 1] 直接 IndexError，
        # 报错信息完全看不出是数据问题。重训前若跑的是纯 fault 场景（或还没产生过异常）就会这样。
        raise ValueError(
            f"标签全是同一类（正样本率 {pos_all:.0%}，异常读数占比 {reading_anomaly_rate:.0%}），"
            f"无法训练判别模型。若为 0%：请把场景切到 mixed 并多跑几十帧；"
            f"若接近 100%：异常太密集，horizon={horizon} 会把几乎每一帧都判成「未来会异常」，"
            f"请改用 mixed 场景（异常应为稀有事件）")

    tr_all, te_all = ~is_test, is_test
    tr = tr_all & known          # 未知标签不进训练
    te = te_all & known          # 未知标签不进评估

    # 阈值必须在**训练段内部、按传感器切出的验证集**上扫，测试集只用于最终评估一次。
    # 原实现直接拿测试集扫 F1 并把该点当成模型成绩上报，报告的 precision/recall/F1 全部带
    # 乐观偏差、换个测试集最优阈值还会漂移。切法：每个传感器的训练段按时间取尾 25% 作验证，
    # 既保住「用过去预测未来」的时序语义，又不跨传感器混切（同一 tick 的行没有因果先后）。
    val = np.zeros(len(y_future), dtype=bool)
    if len(gids):
        for g in pd.unique(gids[tr]):
            idx = np.where(tr & (gids == g))[0]
            if len(idx) >= 20:
                val[idx[int(len(idx) * 0.75):]] = True
    fit = tr & ~val
    threshold_source = "validation_scan"
    if int(fit.sum()) < 30 or int(val.sum()) < 20:
        # 样本太薄（传感器太少或历史太短）：切不出可信验证集。此时**宁可不要阈值**
        # （推理端走 config 的兜底线），也绝不回到"在测试集上扫"的老路。
        fit, val = tr, np.zeros(len(y_future), dtype=bool)
        threshold_source = "train_only_no_threshold"

    X_fit = X[fit]
    # IsolationForest 虽是无监督的，也只能见训练段：它的 anomaly_score 直接占 risk_score
    # 的 0.3 权重，而下面又要拿测试段评估，用全量拟合等于让被评估样本参与自己的评分。
    # 另外 sklearn 的 IsolationForest **不接受 NaN**（三个引擎里只有它不是树模型的不完整
    # 数据实现），所以 A 头单独喂一份填 0 的矩阵；填 0 对 z 类特征是中性值（= 自身历史均值）。
    model_a = IsolationForest(n_estimators=200, contamination=anomaly_contamination, random_state=42)
    model_a.fit(np.nan_to_num(X_fit, nan=0.0))

    model_b = make_classifier()
    # 三个引擎的概率口径必须一致：原来只有 LightGBM 设了 class_weight="balanced"，
    # XGBoost / HistGB 没有，于是换引擎后扫出的阈值不可迁移、prob 的含义随引擎漂移。
    try:
        if HAVE_LGB:
            model_b.set_params(class_weight="balanced")
        elif HAVE_XGB:
            pos_fit = float(np.mean(y_future[fit] == 1))
            if 0 < pos_fit < 1:
                model_b.set_params(scale_pos_weight=(1 - pos_fit) / pos_fit)
        else:
            model_b.set_params(class_weight="balanced")
    except Exception as e:
        errors.append(f"class_weight: {type(e).__name__}: {e}")
    model_b.fit(X_fit, y_future[fit])

    # RUL 目标是右删失的：rul == rul_cap 只表示「至少还有 cap 步」，不是「恰好 cap 步」。
    # 把删失值当真值拟合，回归器会被占绝对多数的 cap 拉平（实测 cap=horizon*10 时 88% 的
    # 测试帧删失在 cap 上，模型学成一律输出 ~cap，于是大屏念出「异常概率 99.9%、RUL 75 步」）。
    # 标准做法是丢弃删失观测，只用真正观测到「下次异常何时来」的帧拟合。
    # 实测对比（12000 帧 / 100 传感器 / horizon=12，矛盾 = prob≥0.5 却 rul≥horizon 的帧数）：
    #   丢弃前  矛盾 33/143，92% 的预测 ≥ 2*horizon（risk_score 的 RUL 项恒为 0，权重白白浪费）
    #   丢弃后  矛盾  1/143， 0% 的预测 ≥ 2*horizon
    yr_fit = y_rul[fit]
    keep = yr_fit < rul_cap
    model_rul = make_regressor()
    if int(keep.sum()) >= 30:
        model_rul.fit(X_fit[keep], yr_fit[keep])
    else:
        # 删失后所剩无几（异常极稀有的数据），退回全量拟合：宁可 RUL 偏大，
        # 也不能让它在这里抛异常拖垮整条在线预测链路
        model_rul.fit(X_fit, yr_fit)

    y_mean = float(np.mean(yr_fit[keep])) if int(keep.sum()) else float(np.mean(yr_fit))
    # A 头分数在训练集上的分布：推理端按它标准化，否则 0.3 的权重退化成近似常数
    try:
        a_scores_fit = model_a.score_samples(np.nan_to_num(X_fit, nan=0.0))
        a_mean = float(np.mean(a_scores_fit))
        a_std = float(np.std(a_scores_fit)) or 1.0
    except Exception as e:
        errors.append(f"a_score_stats: {type(e).__name__}: {e}")
        a_mean, a_std = None, None
    bundle = PredictiveBundle(model_a=model_a, model_b=model_b, model_rul=model_rul,
                              features=feature_cols, window=window, horizon=horizon,
                              y_mean=y_mean, config=cfg,
                              a_score_mean=a_mean, a_score_std=a_std)

    pos_te = float(np.mean(y_future[te] == 1)) if int(te.sum()) else 0.0
    pos_tr = float(np.mean(y_future[fit] == 1)) if int(fit.sum()) else 0.0
    report = {"window": window, "horizon": horizon, "n_samples": int(len(X)),
              "n_sensors": int(df["sensor_id"].nunique()) if "sensor_id" in df.columns else 1,
              "engine": {"lightgbm": HAVE_LGB, "xgboost": HAVE_XGB, "shap": HAVE_SHAP},
              "config": cfg,
              "threshold_source": threshold_source,
              # 基准率必须和指标一起看：avg_precision≈正样本率 就说明模型只是背了基准率。
              # 训练集与测试集基准率要并报：实测过一次「场景中途被切换」的数据（前 91 tick 跑
              # normal、后 29 tick 跑异常场景），按位置切分后测试集整段落在全异常区，
              # h=6 正样本率 88.12% 而训练集只有 5.99%。只看测试集基准率认不出这是切分漂移。
              "label": {"reading_anomaly_rate": round(reading_anomaly_rate, 4),
                        "positive_rate": round(pos_all, 4),
                        "positive_rate_train": round(pos_tr, 4),
                        "positive_rate_test": round(pos_te, 4),
                        "n_train": int(np.sum(fit)), "n_val": int(np.sum(val)),
                        "n_test": int(np.sum(te)),
                        "n_positive": int(np.sum(y_known == 1)),
                        # 被截断而标为未知的帧数：读报告的人必须知道丢掉了多少标签，
                        # 否则无法判断测试集基准率是怎么来的
                        "label_truncated_frames": n_unknown}}

    # ---- B 模型：阈值只在验证集上扫；测试集只做最终评估一次 ----
    prob_te, y_te = None, y_future[te]
    if hasattr(model_b, "predict_proba"):
        try:
            prob_te = model_b.predict_proba(X[te])[:, 1]
        except Exception as e:
            errors.append(f"model_b.eval: {type(e).__name__}: {e}")
    if prob_te is not None and len(y_te):
        if len(set(y_te.tolist())) > 1:
            report["auc"] = round(float(roc_auc_score(y_te, prob_te)), 4)
        report["avg_precision"] = round(float(average_precision_score(y_te, prob_te)), 4)

    best_t = None
    if int(val.sum()) >= 20 and hasattr(model_b, "predict_proba"):
        try:
            prob_val = model_b.predict_proba(X[val])[:, 1]
            y_val = y_future[val]
            best_f1, cand, scan = 0.0, None, {}
            for t in np.arange(0.1, 0.91, 0.05):
                f1 = f1_score(y_val, (prob_val >= t).astype(int), zero_division=0)
                scan[round(float(t), 2)] = round(float(f1), 4)
                if f1 > best_f1:
                    best_f1, cand = float(f1), round(float(t), 2)
            # 只有真正扫到有效 F1 才写阈值。原实现用 `best_f1, best_t = 0.0, 0.5` 初始化，
            # 测试集只有单一类别、或所有阈值 F1 都为 0 时，0.5 会被静默写进 meta；推理端
            # 因为 meta 里"有阈值"就不再退回默认 0.7，判正线被悄悄从 0.8 放宽到 0.5，
            # 预警量暴增而无人知晓。
            if cand is not None and best_f1 > 0:
                best_t = cand
                report["val_f1"] = round(best_f1, 4)
                report["val_threshold_scan"] = scan
                near = [v for k, v in scan.items() if abs(k - best_t) <= 0.0501]
                if near:
                    # 稳健性：阈值 ±0.05 内 F1 的波动范围。波动很小说明这个阈值不敏感，
                    # 成绩不是"碰巧挑到某个点"的结果。
                    report["threshold_robustness"] = {
                        "window": "±0.05", "f1_min": round(min(near), 4),
                        "f1_max": round(max(near), 4)}
            else:
                threshold_source = "validation_scan_failed"
        except Exception as e:
            errors.append(f"threshold_scan: {type(e).__name__}: {e}")
            threshold_source = "validation_scan_failed"

    bundle.threshold = best_t
    bundle.threshold_source = threshold_source
    report["threshold_source"] = threshold_source
    report["best_f1_threshold"] = best_t

    # 用验证集选出的阈值在**测试集**上评估一次——这才是可以对外报的数字
    if prob_te is not None and best_t is not None and len(y_te):
        pred_te = (prob_te >= best_t).astype(int)
        report["test_f1"] = round(float(f1_score(y_te, pred_te, zero_division=0)), 4)
        report["precision"] = round(float(precision_score(y_te, pred_te, zero_division=0)), 4)
        report["recall"] = round(float(recall_score(y_te, pred_te, zero_division=0)), 4)
        # 兼容旧字段名：原来的 f1 就是"在测试集上扫出的最优 F1"，现在它是在测试集上、
        # 用**验证集选的**阈值算出来的，口径更严格
        report["f1"] = report["test_f1"]

    # ---- A 头（IsolationForest）单独评估 ----
    # 原来 A 头从未被评估过，而它的 anomaly_score 直接占 risk_score 的 0.3 权重。sklearn 的
    # score_samples 越**低**越异常（典型区间约 (-0.8, -0.3)），原实现写 sigmoid(raw) 方向是反的：
    # 越正常的点分越高，等于给异常点减分。这里把 A 头分数分布与 AUC 打进报告，
    # 判据是"正样本组 mean 应显著低于负样本组 mean"（因为 anomaly_score = sigmoid(-raw)）。
    try:
        if int(te.sum()):
            ra = model_a.score_samples(np.nan_to_num(X[te], nan=0.0))
            report["anomaly_head"] = {
                "score_samples_p05": round(float(np.percentile(ra, 5)), 4),
                "score_samples_p50": round(float(np.percentile(ra, 50)), 4),
                "score_samples_p95": round(float(np.percentile(ra, 95)), 4),
                "mean_positive": round(float(np.mean(ra[y_te == 1])), 4) if (y_te == 1).any() else None,
                "mean_negative": round(float(np.mean(ra[y_te == 0])), 4) if (y_te == 0).any() else None,
                # 取负号后算 AUC：>0.5 才说明"越异常分越低"这个方向成立
                "auc_score_samples_negated": (round(float(roc_auc_score(y_te, -ra)), 4)
                                              if len(set(y_te.tolist())) > 1 else None),
            }
    except Exception as e:
        errors.append(f"anomaly_head_eval: {type(e).__name__}: {e}")

    # RUL 回归评估。必须在*未删失*子集上算：rul == cap 的帧只表示「至少还有 cap 步」，
    # 把它当真值会让 MAE/R² 被大量「预测 cap、真值也 cap」的平凡命中撑起来。虚高到什么
    # 程度有实测——同一份数据同一套特征，全量口径 R² 报 0.7731，未删失子集上 R² 为负，
    # 也就是说这个头其实没有绝对量级的预测力，只是把「远/近」分开了。删失率与评估样本数
    # 一并写进报告，读的人才知道这两个指标是在什么基础上算出来的。
    try:
        rul_raw = np.maximum(model_rul.predict(X[te]), 0.0)
        # 评估的必须是投影后、真正会上屏的那个数。否则报告会写着 20 条矛盾而大屏一条
        # 都看不到，两边对不上，读报告的人会以为投影没生效。
        # 投影边界必须用**当前生效的判正线**，原来硬编码 0.5 而线上是 0.80，
        # 于是 prob∈[0.5,0.8) 的点位被判 normal 却输出 rul<horizon。
        rul_pred = (consistent_rul(prob_te, rul_raw, horizon, threshold=best_t)
                    if prob_te is not None else rul_raw)
        y_rul_te = y_rul[te]
        obs = y_rul_te < rul_cap
        report["rul_censored_rate"] = round(float(1 - obs.mean()), 4) if len(obs) else 0.0
        report["rul_n_eval"] = int(obs.sum())
        if int(obs.sum()) >= 10:
            report["rul_mae"] = round(float(mean_absolute_error(y_rul_te[obs], rul_pred[obs])), 2)
            report["rul_r2"] = round(float(r2_score(y_rul_te[obs], rul_pred[obs])), 4)
        # 分类器与 RUL 头学的是同一 anomaly_series 的两个切面，且 label==1 ⟺ rul<horizon
        # 恒成立，因此 prob≥阈值 的帧其 RUL 必须 < horizon。违反条数直接暴露「12 步内必然
        # 异常」与「还要 75 步才异常」同时出现在一张卡片上的自相矛盾输出。
        if prob_te is not None and len(prob_te) and best_t is not None:
            hi = prob_te >= best_t
            report["n_high_risk"] = int(hi.sum())
            report["rul_contradictions"] = int(np.sum(rul_pred[hi] >= horizon))
            # 残余矛盾单独统计：prob 落在 [0.5, 阈值) 的点位被判 normal，但投影只按阈值切，
            # 它们的 RUL 仍可能 < horizon（"状态正常 + 约 11 步内可能异常"）。原来
            # rul_contradictions 的计数边界与投影边界同源（都是 0.5），结构上恒为 0，
            # 那是自证不是校验——这一条才是真实的残余矛盾量。
            band = (prob_te >= 0.5) & (prob_te < best_t)
            report["rul_contradictions_band"] = int(np.sum(rul_pred[band] < horizon))
    except Exception as e:
        errors.append(f"rul_eval: {type(e).__name__}: {e}")

    # ---- 规则基线对比（3σ 阈值）----
    # 注意口径：这条基线用的是 15 帧滚动 z>3，而**在线**台账里的 rule_baseline 用的是
    # SENSOR_METRIC_DEFS 的固定阈值表，两者不是同一个检测器。对外只能说"与滚动 3σ 基线
    # 在前向窗口口径下对比"，不能说"和线上规则完全同口径"。
    try:
        base_pred = rule_baseline_predict(feat, window).values[te]
        bp, br, bf = precision_recall_f1(y_te, base_pred)
        report["baseline_precision"] = round(float(bp), 4)
        report["baseline_recall"] = round(float(br), 4)
        report["baseline_f1"] = round(float(bf), 4)
    except Exception as e:
        errors.append(f"baseline: {type(e).__name__}: {e}")

    # 全局重要度只在**训练段**上算：原来传的是 train+test 拼起来的全量 X，让被评估样本
    # 参与自己的解释，口径混乱。
    report["importance"] = feature_importance(model_b, X_fit, feature_cols, y=y_future[fit])
    # 静默吞异常是这套代码的老毛病：一旦某段抛出，report 里直接没有 rul_* / baseline_*
    # 字段，读的人以为"这次没算"，线上则表现为"模型一直不预警"而日志一片干净。全部显式记录。
    if errors:
        report["errors"] = errors
    report["warnings"] = label_warnings(report)
    bundle.report = report
    return bundle, report


def label_warnings(report):
    """把「指标好看但模型其实没学到东西」的几种情况显式写进报告，省得每次另写脚本诊断。"""
    lab = report.get("label") or {}
    pos_tr = lab.get("positive_rate_train", 0.0)
    pos_te = lab.get("positive_rate_test", 0.0)
    reading_anomaly_rate = lab.get("reading_anomaly_rate", 0.0)
    n_test = lab.get("n_test", 0)
    horizon = report.get("horizon")
    auc = report.get("auc")
    w = []

    # 训练/测试基准率背离 = 数据在时间轴上被切换过场景。实测案例：前 91 tick 跑 normal、
    # 后 29 tick 跑异常场景，按位置切分后测试集（tick 97-120）整段在全异常区，
    # h=6 正样本率 88.12% 而训练集 5.99%，于是 precision 0.86 / recall 0.96 配上 auc 0.4988。
    # 这种数据上任何指标都没有意义，必须先修采集过程，所以排在最前面。
    if n_test >= 100 and abs(pos_te - pos_tr) > 0.35:
        w.append(f"训练/测试基准率严重背离（训练集 {pos_tr:.1%} vs 测试集 {pos_te:.1%}）：采集期间场景被切换过，"
                 f"测试集整段落在单一区间里，precision/recall/avg_precision 全部失真。"
                 f"请只用一个场景连续采集，或改用带周期劣化的 mixed 场景让异常在时间轴上均匀分布")

    if pos_te > 0.6:
        w.append(f"标签退化：测试集 {pos_te:.0%} 为正样本。avg_precision 的基准线就是 {pos_te:.2f}，"
                 f"此时只有 AUC / F1 可信。异常读数占比 {reading_anomaly_rate:.1%}，"
                 f"horizon={horizon} 会把它放大成 1-(1-p)^{horizon}")
    elif pos_te < 0.02 and n_test >= 100:
        w.append(f"正样本过少（测试集 {pos_te:.1%}）：模型看不到足够的异常模式，"
                 f"请把模拟场景切到 mixed 多采一些数据再重训")

    if auc is not None and auc < 0.55 and n_test >= 200:
        w.append(f"AUC={auc}≈0.5，模型没有排序能力。两种可能：①异常是逐帧独立的随机尖峰，"
                 f"「未来会不会异常」与历史本质上无关，任何模型都学不到，需要持续劣化型数据"
                 f"（趋势可被滑动窗口捕捉）才有可学习信号；②特征与标签错位（如滑动窗口跨了传感器）。"
                 f"若上面同时报了基准率背离，先解决那个再看 AUC")
    contra = report.get("rul_contradictions")
    nhi = report.get("n_high_risk") or 0
    if contra and nhi:
        w.append(f"RUL 自洽约束被破坏：{contra}/{nhi} 个 prob≥阈值 的测试帧仍预测出 RUL≥horizon({horizon})。"
                 f"consistent_rul 的投影本应让该计数恒为 0，出现非零说明投影被绕过、"
                 f"或 label 与 rul 的定义已不再互为充要条件——两者会被印在同一张卡片上，务必先查")
    # 残余矛盾：prob 落在 [0.5, 阈值) 的帧被判 normal 却输出 rul<horizon。投影按阈值切，
    # 这段无法消除，只能如实统计出来；它非零说明"正常 + 约 N 步内可能异常"仍会少量出现。
    band = report.get("rul_contradictions_band")
    if band:
        w.append(f"残余矛盾 {band} 帧：prob∈[0.5, 阈值) 被判 normal 但 RUL<horizon，"
                 f"大屏会显示「正常 + 约 {horizon} 步内可能异常」。这是判正线高于 0.5 的必然代价，"
                 f"若要彻底消除得把判正线降回 0.5（代价是精确率大幅下降）")

    # 标签截尾：测试集里被丢掉的未知帧占比。占比越高，说明本版指标与旧版（把尾部一律标 0）
    # 的差距越大，两者不可直接对比。
    lab_trunc = (report.get("label") or {}).get("label_truncated_frames", 0)
    n_all = report.get("n_samples", 0)
    if lab_trunc and n_all:
        w.append(f"标签截尾：{lab_trunc}/{n_all} 帧（{lab_trunc / n_all:.0%}）因未来窗口被数据末尾"
                 f"截断而标为未知，已从训练/阈值扫描/评估中剔除。旧版本把这些帧一律标 0，"
                 f"压低了测试集基准率，因此本版指标与旧版不可直接对比")

    # 阈值来源不是"训练段内的验证集扫描"时，本次的 precision/recall 不能当成绩报。
    src = report.get("threshold_source")
    if src and src != "validation_scan":
        w.append(f"判正阈值来源 = {src}：未能在训练段内切出可信验证集，本次**不写阈值**，"
                 f"推理端退回配置兜底线（{report.get('config', {}).get('risk_threshold')}）。"
                 f"不要对外报本次的 precision/recall 作为模型成绩")

    # A 头方向自检：anomaly_score = sigmoid(-score_samples)，因此正样本组 mean 必须更低。
    ah = report.get("anomaly_head") or {}
    mp, mn = ah.get("mean_positive"), ah.get("mean_negative")
    if mp is not None and mn is not None and mp > mn:
        w.append(f"A 头方向可能反了：正样本组 score_samples 均值 {mp} > 负样本组 {mn}。"
                 f"anomaly_score 取的是 sigmoid(-raw)，正样本必须更低；若确为反向，"
                 f"risk_score 里 0.3 的权重等于在给异常点加分，务必先查")

    r2 = report.get("rul_r2")
    if r2 is not None and r2 < 0:
        w.append(f"RUL 头没有绝对量级的预测力（未删失子集 R²={r2}，n={report.get('rul_n_eval')}，"
                 f"删失率 {report.get('rul_censored_rate', 0):.0%}）：它能分出「快出事」和「还早」，"
                 f"但报出的步数不能当精确倒计时念。演示时应说「约 N 步内」而非「还剩 N 步」")
    return w


def rule_baseline_predict(feat, window):
    """规则基线：任一指标 |z-score|>3 即判异常（近似现有阈值逻辑）。"""
    pred = pd.Series(0, index=feat.index)
    for m in METRICS:
        vcol, mcol, scol = feat.get(f"{m}_val"), feat.get(f"{m}_mean{window}"), feat.get(f"{m}_std{window}")
        if vcol is None or mcol is None or scol is None:
            continue
        z = (vcol - mcol) / (scol.replace(0, np.nan))
        pred = pred | (z.abs() > 3).fillna(False).astype(int)
    return pred


def precision_recall_f1(y_true, y_pred):
    if len(set(np.asarray(y_true).tolist())) < 2:
        return 0.0, 0.0, 0.0
    return (precision_score(y_true, y_pred, zero_division=0),
            recall_score(y_true, y_pred, zero_division=0),
            f1_score(y_true, y_pred, zero_division=0))


# SHAP explainer 缓存。**必须同时持有 model 引用**：原来只用 id(model) 作键，
# 模型对象被回收后内存地址会被复用，后来者拿到的是上一个模型的解释器 → 归因张冠李戴。
# 存成 (model, explainer) 既让缓存持有强引用（对象不会被回收，id 自然不会被复用），
# 也便于命中时校验是不是同一个模型。代价是缓存会留住模型，故只缓存少量对象。
_SHAP_EXPLAINER_CACHE = {}


def _get_shap_explainer(model):
    """按模型取（并缓存）TreeExplainer；同一 id 但不同对象时重建。"""
    key = id(model)
    hit = _SHAP_EXPLAINER_CACHE.get(key)
    if hit is not None and hit[0] is model:
        return hit[1]
    explainer = shap.TreeExplainer(model)
    _SHAP_EXPLAINER_CACHE[key] = (model, explainer)
    return explainer


def feature_importance(model, X, feature_cols, top_k=8, return_method=False, y=None):
    """可解释性：SHAP 优先 → feature_importances_ → 排列重要性。返回 [{feature, importance}]。

    return_method=True 时额外返回方法名。三条分支的语义并不等价，调用方必须知道走的是哪条：
    SHAP 传单行 X 是局部归因，传多行 X 求 mean(|值|) 是**全局平均幅度**（这里如实区分标注），
    而 feature_importances_ 完全忽略 X、永远是训练集的全局分裂增益——拿它解释单个点位是把
    全局结论冒充成局部结论。

    y：排列重要度的真实标签。**必传**才走排列分支——排列重要度是相对 y 的 scoring 变化，
    原来传 np.zeros(len(X)) 当标签（常数），该分支输出的是没有统计意义的噪声，在既无 SHAP
    又无 feature_importances_ 的模型上会静默给出错误的重要度排序。宁可不给重要度，也不给错的。
    """
    X = np.asarray(X, dtype=float)
    n_rows = X.shape[0] if X.ndim > 1 else 1
    scores = None
    method = ""
    # 1) SHAP
    if HAVE_SHAP:
        try:
            sv = _get_shap_explainer(model).shap_values(X)
            arr = np.asarray(sv, dtype=float)
            if arr.ndim == 3:
                arr = arr[..., 1]          # 二分类取正类
            scores = dict(zip(feature_cols, [float(np.mean(np.abs(arr[:, i]))) for i in range(arr.shape[1])]))
            # 语义必须如实：单行是"这个点位为什么高风险"，多行是"训练集里哪些特征普遍有用"
            method = "SHAP（局部）" if n_rows <= 1 else "SHAP（全局平均|值|）"
        except Exception:
            scores = None
    # 2) feature_importances_
    if not scores:
        try:
            scores = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
            method = "全局特征重要度（非局部）"
        except Exception:
            scores = None
    # 3) permutation importance（必须有真实标签，否则不做）
    if not scores and HAVE_PERM and y is not None and len(y) == n_rows:
        try:
            perm = permutation_importance(model, X, np.asarray(y), n_repeats=5, random_state=42)
            scores = dict(zip(feature_cols, [float(x) for x in perm.importances_mean]))
            method = "排列重要度（全局）"
        except Exception:
            scores = None
    if not scores:
        return ([], "") if return_method else []
    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    out = [{"feature": f, "importance": round(float(v), 4)} for f, v in top]
    return (out, method) if return_method else out


def local_attribution(model, x_row, baseline_row, feature_cols, top_k=5):
    """遮挡归因：把第 j 个特征换成该点位的常态值，看异常概率掉多少。不依赖 shap。

    归因值 = P(异常|实际) − P(异常|遮挡第 j 个)，正数表示该特征正在推高风险。全部特征拼成
    一个 (n_features, n_features) 矩阵，一次批量 predict_proba 算完，开销是毫秒级。

    基线必须取该传感器自己历史的逐特征中位数，不能取全网中位数：模型是跨管型 pooled 训练的，
    燃气压力 range 0.10 MPa 与供暖流量 range 50 m³/h 量纲不可比。用自身常态还有个好性质——
    该管型没有的指标列在 reindex 后恒为 0，遮挡 0→0 归因自然为 0，供水点位就不会再冒出
    「燃气浓度」这种无关因子。

    返回 (top, ok)。top 为空且 ok=True 表示「没有任何特征在推高风险」，这是有效结论，
    不是失败，调用方不要因此降级到全局重要度。

    固有局限——概率饱和时会严重低估：VM 实跑中一个 prob=0.999 的危急点位，遮挡归因给出的
    首要因子只有 +0.0061（0.6 个百分点），而同一行样本的 SHAP 值是 +2.58。原因是概率已经顶到
    天花板，遮挡掉单个特征几乎推不动它；SHAP 工作在对数几率上，不受这个上限约束。
    这正是 explain_local 把 SHAP 排在遮挡之前的原因，不是可以随手调换的偏好。
    """
    if not hasattr(model, "predict_proba"):
        return [], False
    x = np.asarray(x_row, dtype=float).reshape(1, -1)
    base = np.asarray(baseline_row, dtype=float).reshape(-1)
    n = x.shape[1]
    if base.shape[0] != n:
        return [], False
    try:
        p_act = float(model.predict_proba(x)[0][1])
        occ = np.repeat(x, n, axis=0)
        occ[np.arange(n), np.arange(n)] = base      # 对角线逐个替换
        p_occ = np.asarray(model.predict_proba(occ)[:, 1], dtype=float)
    except Exception:
        return [], False
    attr = p_act - p_occ
    out = []
    for j in np.argsort(attr)[::-1]:
        v = float(attr[j])
        if v <= 1e-6:
            break
        out.append({"feature": feature_cols[j], "importance": round(v, 4)})
        if len(out) >= top_k:
            break
    return out, True


def shap_local_signed(model, x_row, feature_cols, top_k=5):
    """单行样本的有符号 SHAP 值，按「推高风险」的方向排序。

    不能复用 feature_importance：那里返回 mean(|shap|)，只有幅度没有方向，一个把风险压下去的
    因子也会显示成正数，前端标注成「推高异常概率 +53%」就是错的。这里保留符号，
    语义与 local_attribution 完全一致，两条路径的数字才可以直接对比。
    训练报告里的全局重要度仍用幅度（那里关心的是特征整体的解释力，不是方向）。
    """
    if not HAVE_SHAP:
        return [], False
    try:
        sv = np.asarray(_get_shap_explainer(model).shap_values(
            np.asarray(x_row, dtype=float).reshape(1, -1)), dtype=float)
        if sv.ndim == 3:
            sv = sv[..., 1]
        vals = sv.reshape(-1)
    except Exception:
        return [], False
    if vals.shape[0] != len(feature_cols):
        return [], False
    out = []
    for j in np.argsort(vals)[::-1]:
        v = float(vals[j])
        if v <= 1e-6:
            break
        out.append({"feature": feature_cols[j], "importance": round(v, 4)})
        if len(out) >= top_k:
            break
    return out, True


def explain_local(model, x_row, baseline_row, feature_cols, top_k=5):
    """单点位可解释性：SHAP → 遮挡归因 → 全局重要度，并把实际用的方法名如实返回。

    前两条路径的 importance 都是「该特征把未来窗口异常概率推高了多少」，**但数值不可互换**：
    SHAP 工作在对数几率尺度且保留符号，遮挡归因在概率尺度上取差值，概率饱和时遮挡归因会
    严重低估（实跑：同一行样本 SHAP +2.58，遮挡归因只有 +0.0061）。所以这里保留 SHAP 优先，
    这不是可以随手调换的偏好；任何"两者量纲一致、可直接比较"的表述都是错的。
    第三条是训练集的全局分裂增益/幅度，语义更远，只能降级使用并明确标注。
    方法名必须跟着结果一起传给前端，否则前端无从判断该用哪种措辞。
    """
    top, ok = shap_local_signed(model, x_row, feature_cols, top_k=top_k)
    if ok:
        return top, "SHAP（局部）"
    if baseline_row is not None:
        top, ok = local_attribution(model, x_row, baseline_row, feature_cols, top_k=top_k)
        if ok:
            return top, "遮挡归因（局部）"
    top, method = feature_importance(model, x_row, feature_cols, top_k=top_k, return_method=True)
    return top, (method or "全局特征重要度（非局部）")


def online_predict(readings, bundle, risk_threshold=0.7):
    """给定某传感器最近若干条历史读数，返回预测结果 dict。"""
    df = readings_to_frame(readings)
    feat = build_features(df, bundle.window)
    if feat.empty:
        # 历史不足 window 帧：绝不能返回一个"看起来正常"的 0 分（见 default_prediction）
        return default_prediction(bundle, n_history=len(df))
    # 必须对齐训练时的特征列。模型是跨管网类型 pooled 训练的，而单个传感器的
    # build_features 会丢掉它没有的指标列（供水管网没有 gas_concentration/level/vibration）。
    # 列数或列序不一致会让下面三个模型全部抛异常，风险分恒为 0 且看不出任何报错。
    if bundle.features:
        feat = feat.reindex(columns=bundle.features, fill_value=0.0)
    cols = list(bundle.features) if bundle.features else list(feat.columns)
    # 注意 iloc[-1] 取出的是 Series，Series.reindex() 只接受标签序列（位置参数），
    # 写成 reindex(columns=...) 会在运行期抛 TypeError——py_compile 查不出来
    X = feat.iloc[-1].reindex(cols).fillna(0).values.reshape(1, -1)
    # 遮挡归因的基线 = 该点位自身常态。必须排除当前帧：中位数若把当前帧算进去，
    # 劣化帧会把基线一起拉高，归因值被稀释甚至变号。历史不足 3 帧时中位数没有意义，
    # 留 None 让 explain_local 自己降级并如实标注方法。
    baseline = None
    if len(feat) >= 3:
        baseline = (feat.iloc[:-1].reindex(columns=cols).median()
                    .fillna(0.0).values.astype(float))

    errors = []
    anomaly_score = 0.0
    try:
        raw = float(bundle.model_a.score_samples(X)[0])
        # sklearn 的 score_samples **越低越异常**（判离群的条件是 decision_function =
        # score_samples - offset_ < 0，offset_ = -0.5）。原实现写 sigmoid(raw) 方向是反的：
        # 越正常的点分越高，risk_score 里 0.3 的权重等于在给异常点加分。
        # 再按训练集分布标准化一次——score_samples 的典型区间只有约 (-0.8, -0.3)，
        # 直接 sigmoid 会被压成 0.31~0.43 的近似常数，权重形同虚设。
        z = raw
        if bundle.a_score_mean is not None:
            z = (raw - float(bundle.a_score_mean)) / (float(bundle.a_score_std) or 1.0)
        anomaly_score = float(1.0 / (1.0 + np.exp(z)))  # = sigmoid(-z)
    except Exception as e:
        errors.append(f"model_a: {type(e).__name__}: {e}")

    prob = 0.0
    if hasattr(bundle.model_b, "predict_proba"):
        try:
            prob = float(bundle.model_b.predict_proba(X)[0][1])
        except Exception as e:
            errors.append(f"model_b: {type(e).__name__}: {e}")

    rul = 0.0
    try:
        rul = float(max(bundle.model_rul.predict(X)[0], 0))
    except Exception as e:
        rul = float(bundle.y_mean or 0)
        errors.append(f"model_rul: {type(e).__name__}: {e}")
    # 概率和 RUL 会印在同一张卡片上，必须一起自洽：投影掉「12 步内必然异常却还要
    # 75 步才异常」这种组合（依据与实测数据见 consistent_rul）。
    # 投影边界必须与线上判正线一致，否则 prob∈[阈值,1) 之外那一段会冒出"正常 + 约 11 步内"。
    cfg = get_config(getattr(bundle, "config", None))
    w = cfg["risk_score_weights"]
    rul = consistent_rul(prob, rul, bundle.horizon, threshold=bundle.threshold)

    # 权重、RUL 截断倍数一律读配置（原来三个文件各写一份，见 DEFAULT_CONFIG）。
    # RUL 项的截断点必须等于训练目标的删失上限：高于它的 RUL 一律记 0 分贡献。
    cap = max(bundle.horizon * int(cfg["rul_cap_factor"]), 1)
    risk_score = float(min(100.0, (w["prob"] * prob + w["anomaly"] * anomaly_score
                                   + w["rul"] * (1 - min(rul / cap, 1))) * 100))
    # 判正线取**训练段内验证集**扫出的 F1 最优点（随模型存进 meta），不再是写死的 0.7 / 0.35。
    # 这个选择直接决定成果面板上模型对规则基线的胜负——同一批 268 条已评估台账记录实测：
    #   阈值 0.35（旧 *0.5 的 warning 线）命中率 62.5% / 精确率 40.3% / 每轮预警 62 条
    #   阈值 0.80（验证集所得）          命中率 45.0% / 精确率 78.3% / 每轮预警 23 条
    # 基线是 30.0% / 75.0% / 提前 0 步，所以只有 0.80 这一档三项全胜；代价是预警变稀。
    # meta 里没有阈值（老模型 / 扫描失败）时才退回配置兜底线，并把这个事实透传出去。
    warn_t = float(bundle.threshold) if bundle.threshold else float(cfg["risk_threshold"])
    crit_t = warn_t + (1.0 - warn_t) * float(cfg["crit_midpoint"])
    status = "critical" if prob >= crit_t else ("warning" if prob >= warn_t else "normal")

    top, imp_method = [], ""
    try:
        top, imp_method = explain_local(bundle.model_b, X, baseline, cols, top_k=5)
    except Exception as e:
        errors.append(f"explain: {type(e).__name__}: {e}")
        imp_method = "归因不可用"

    result = {
        "risk_score": round(risk_score, 1),
        "anomaly_score": round(anomaly_score, 3),
        "future_anomaly_prob": round(prob, 3),
        "rul": round(rul, 1),
        "rul_unit": "steps",
        # 下游要判断「RUL 是否落进预警窗口」，窗口就是 horizon。不带这个字段，
        # agent_brain 只能把 12 写死，而 horizon 是重训时可调的（1-48）
        "horizon": int(bundle.horizon),
        # 生效的判正阈值一并透传：光看 predicted_status 分不清这个工作点是训练扫出来的
        # 还是退回的默认常数，尤其线上没有 shell 可查 meta 时，这是唯一的核对入口。
        "threshold": round(warn_t, 3),
        "predicted_status": status,
        "top_features": top,
        # 前端必须照这个名字标注归因方式：遮挡/SHAP 是「这个点位为什么被判高风险」，
        # 全局重要度是「训练集里哪些特征普遍有用」，两者不能混为一谈
        "top_features_method": imp_method,
        "n_features": int(X.shape[1]),
        # 阈值来源 + 风险分档 + 权重一起透传：光看 predicted_status 分不清阈值是扫出来的
        # 还是兜底的，Agent 与前端需要据此决定措辞（"模型打分" vs "异常概率"）
        "threshold_source": getattr(bundle, "threshold_source", None),
        "risk_levels": cfg["risk_levels"],
        "risk_score_weights": w,
        # prob 是 class_weight="balanced" 下的**原始**输出，没做概率校准，所以它的准确说法是
        # "模型打分"而不是"异常概率"（重加权会系统性抬高正类概率）。isotonic 校准 +
        # 可靠性曲线 + Brier 是后续工作，在那之前对外一律按"打分"表述。
        "prob_calibrated": False,
        "data_insufficient": False,
    }
    if errors:
        result["model_errors"] = errors
    return result


def default_prediction(bundle, n_history=None):
    """历史不足（读入帧数 < window）时的降级结果。

    **必须带 data_insufficient 标记**：原实现只返回 risk_score=0 / prob=0 / status=normal，
    前端与台账无法区分"真的没有风险"和"数据不够没法判断"——这正是"让 bug 伪装成合理的 0"。
    演示前 200 帧的热身期、或某传感器刚上线时都会命中。前端遇到该标记必须灰显"数据不足"，
    台账统计命中/误报时也要把这类记录从分母里剔除。

    降级路径同样要满足自洽约束：prob=0 表示「horizon 步内不会异常」，RUL 就不能小于 horizon。
    """
    rul = consistent_rul(0.0, float(bundle.y_mean or 0), bundle.horizon)
    cfg = get_config(getattr(bundle, "config", None))
    return {"risk_score": 0.0, "anomaly_score": 0.0, "future_anomaly_prob": 0.0,
            "rul": rul, "rul_unit": "steps", "horizon": int(bundle.horizon),
            "predicted_status": "normal", "top_features": [],
            "top_features_method": "历史不足，无法归因",
            "threshold": round(float(bundle.threshold or cfg["risk_threshold"]), 3),
            "threshold_source": getattr(bundle, "threshold_source", None),
            "prob_calibrated": False,
            "data_insufficient": True,
            "n_history": int(n_history) if n_history is not None else None,
            "required_history": int(bundle.window)}
