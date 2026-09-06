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
    """label[i]=1 表示 i+1..i+horizon 步内将出现异常。"""
    arr = anomaly_series.values
    n = len(arr)
    out = np.zeros(n)
    for i in range(n):
        lo, hi = i + 1, min(i + 1 + horizon, n)
        out[i] = 1 if (lo < n and arr[lo:hi].any()) else 0
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


def consistent_rul(prob, rul, horizon):
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
    """
    p = np.asarray(prob, dtype=float)
    r = np.asarray(rul, dtype=float)
    h = float(max(horizon, 1))
    out = np.where(p >= 0.5, np.minimum(r, h - 1.0), np.maximum(r, h))
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
                 window=15, horizon=12, y_mean=None, threshold=None):
        self.model_a = model_a
        self.model_b = model_b
        self.model_rul = model_rul
        self.features = features or []
        self.window = window
        self.horizon = horizon
        self.y_mean = y_mean
        # 判正阈值：测试集上 F1 最优的那个点，而不是拍脑袋的常数。为 None 时
        # online_predict 退回 risk_threshold 默认值（模型太老、没存过阈值的情形）。
        self.threshold = threshold

    def save(self, out_dir):
        if joblib is None:
            raise RuntimeError("未安装 joblib，无法保存模型：pip install joblib")
        os.makedirs(out_dir, exist_ok=True)
        meta = {"features": self.features, "window": self.window, "horizon": self.horizon,
                "y_mean": self.y_mean, "threshold": self.threshold,
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
        if threshold is None:
            # meta 里没有就是阈值上线前存的模型。best_f1_threshold 一直有写进
            # 同目录的 predictive_report.json，捞回来即可，不必逼人重训一遍。
            try:
                with open(os.path.join(out_dir, "predictive_report.json"), "r", encoding="utf-8") as f:
                    threshold = json.load(f).get("best_f1_threshold")
            except Exception:
                threshold = None
        return cls(model_a=data["model_a"], model_b=data["model_b"], model_rul=data["model_rul"],
                   features=meta["features"], window=meta["window"], horizon=meta["horizon"],
                   y_mean=meta.get("y_mean"), threshold=threshold)


def _build_training_frames(df, window, horizon):
    """按 sensor_id 分组做特征工程与打标，返回 (feat, y_future, y_rul, is_test)。

    分组是必须的，不能图省事直接对 pooled df 做 rolling：同一 tick 里 100 个传感器的
    timestamp 完全相同，排序后彼此相邻，rolling(15) 算出来的是「同一时刻 15 个不同
    传感器」的统计量，「未来 12 步」也退化成「接下来 12 个传感器是否异常」。特征和
    要预测的未来之间没有因果关系，AUC 必然≈0.5，正样本率还会被推到 85% 以上。
    """
    if "sensor_id" in df.columns and df["sensor_id"].nunique() > 1:
        groups = [g for _, g in df.groupby("sensor_id", sort=True)]
    else:
        groups = [df]

    feats, yfs, yrs, tests = [], [], [], []
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

    if not feats:
        return pd.DataFrame(), np.array([]), np.array([]), np.array([], dtype=bool)
    # 各管网类型的指标集不同，分组 dropna 后列不一致；concat 取并集，缺口留 NaN 由下游填 0
    return (pd.concat(feats, ignore_index=True).fillna(0),
            np.concatenate(yfs), np.concatenate(yrs), np.concatenate(tests))


def train_bundle(df, window=15, horizon=12, anomaly_contamination="auto"):
    """训练 A/B/RUL，返回 (bundle, report)。df 为 readings_to_frame 输出的时序 DataFrame（可 pooled）。"""
    if len(df) < 30:
        raise ValueError("样本太少（需累积至少 window 长度的传感器历史），请先采集数据再训练")
    feat, y_future, y_rul, is_test = _build_training_frames(df, window, horizon)
    if feat.empty or len(feat) < 30:
        raise ValueError("特征样本太少（需累积至少 window 长度的传感器历史），请先采集数据再训练")
    feature_cols = list(feat.columns)
    X = feat[feature_cols].values

    reading_anomaly_rate = float((df["status"] != "normal").mean()) if "status" in df.columns else 0.0
    pos_all = float(np.mean(y_future))
    if len(set(y_future.tolist())) < 2:
        # 早失败：单一类别会让 sklearn 的 predict_proba 只返回一列，后面 [:, 1] 直接 IndexError，
        # 报错信息完全看不出是数据问题。重训前若跑的是纯 fault 场景（或还没产生过异常）就会这样。
        raise ValueError(
            f"标签全是同一类（正样本率 {pos_all:.0%}，异常读数占比 {reading_anomaly_rate:.0%}），"
            f"无法训练判别模型。若为 0%：请把场景切到 mixed 并多跑几十帧；"
            f"若接近 100%：异常太密集，horizon={horizon} 会把几乎每一帧都判成「未来会异常」，"
            f"请改用 mixed 场景（异常应为稀有事件）")

    tr, te = ~is_test, is_test
    X_tr, X_te = X[tr], X[te]
    yf_tr, yf_te = y_future[tr], y_future[te]
    yr_tr = y_rul[tr]
    rul_cap = horizon * RUL_CAP_FACTOR

    # IsolationForest 虽是无监督的，也只能见训练段：它的 anomaly_score 直接占 risk_score
    # 的 0.3 权重，而下面又要拿测试段评估，用全量拟合等于让被评估样本参与自己的评分。
    model_a = IsolationForest(n_estimators=200, contamination=anomaly_contamination, random_state=42)
    model_a.fit(X_tr)

    model_b = make_classifier()
    try:
        if HAVE_LGB:
            model_b.set_params(class_weight="balanced")
    except Exception:
        pass
    model_b.fit(X_tr, yf_tr)

    # RUL 目标是右删失的：rul == rul_cap 只表示「至少还有 cap 步」，不是「恰好 cap 步」。
    # 把删失值当真值拟合，回归器会被占绝对多数的 cap 拉平（实测 cap=horizon*10 时 88% 的
    # 测试帧删失在 cap 上，模型学成一律输出 ~cap，于是大屏念出「异常概率 99.9%、RUL 75 步」）。
    # 标准做法是丢弃删失观测，只用真正观测到「下次异常何时来」的帧拟合。
    # 实测对比（12000 帧 / 100 传感器 / horizon=12，矛盾 = prob≥0.5 却 rul≥horizon 的帧数）：
    #   丢弃前  矛盾 33/143，92% 的预测 ≥ 2*horizon（risk_score 的 RUL 项恒为 0，权重白白浪费）
    #   丢弃后  矛盾  1/143， 0% 的预测 ≥ 2*horizon
    keep = yr_tr < rul_cap
    model_rul = make_regressor()
    if int(keep.sum()) >= 30:
        model_rul.fit(X_tr[keep], yr_tr[keep])
    else:
        # 删失后所剩无几（异常极稀有的数据），退回全量拟合：宁可 RUL 偏大，
        # 也不能让它在这里抛异常拖垮整条在线预测链路
        model_rul.fit(X_tr, yr_tr)

    y_mean = float(np.mean(yr_tr[keep])) if int(keep.sum()) else float(np.mean(yr_tr))
    bundle = PredictiveBundle(model_a=model_a, model_b=model_b, model_rul=model_rul,
                              features=feature_cols, window=window, horizon=horizon,
                              y_mean=y_mean)

    pos_te = float(np.mean(yf_te)) if len(yf_te) else 0.0
    pos_tr = float(np.mean(yf_tr)) if len(yf_tr) else 0.0
    report = {"window": window, "horizon": horizon, "n_samples": int(len(X)),
              "n_sensors": int(df["sensor_id"].nunique()) if "sensor_id" in df.columns else 1,
              "engine": {"lightgbm": HAVE_LGB, "xgboost": HAVE_XGB, "shap": HAVE_SHAP},
              # 基准率必须和指标一起看：avg_precision≈正样本率 就说明模型只是背了基准率。
              # 训练集与测试集基准率要并报：实测过一次「场景中途被切换」的数据（前 91 tick 跑
              # normal、后 29 tick 跑异常场景），按位置切分后测试集整段落在全异常区，
              # h=6 正样本率 88.12% 而训练集只有 5.99%。只看测试集基准率认不出这是切分漂移。
              "label": {"reading_anomaly_rate": round(reading_anomaly_rate, 4),
                        "positive_rate": round(pos_all, 4),
                        "positive_rate_train": round(pos_tr, 4),
                        "positive_rate_test": round(pos_te, 4),
                        "n_train": int(np.sum(tr)), "n_test": int(np.sum(te)),
                        "n_positive": int(np.sum(y_future))}}

    # B 模型评估
    prob = model_b.predict_proba(X_te)[:, 1] if hasattr(model_b, "predict_proba") else None
    if prob is not None and len(yf_te):
        y_te = yf_te
        if len(set(y_te.tolist())) > 1:
            report["auc"] = round(float(roc_auc_score(y_te, prob)), 4)
        report["avg_precision"] = round(float(average_precision_score(y_te, prob)), 4)
        best_f1, best_t, bp, br = 0.0, 0.5, 0.0, 0.0
        for t in np.arange(0.1, 0.91, 0.05):
            pred = (prob >= t).astype(int)
            f1 = f1_score(y_te, pred, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
                bp = precision_score(y_te, pred, zero_division=0)
                br = recall_score(y_te, pred, zero_division=0)
        report["best_f1_threshold"] = round(float(best_t), 2)
        report["f1"] = round(float(best_f1), 4)
        report["precision"] = round(float(bp), 4)
        report["recall"] = round(float(br), 4)
        # bundle 在上面就构造好了，阈值却要到这一段扫完才得出，只能事后补上。
        # 扫描没跑（测试集只有单一类别）时留 None，推理端退回默认阈值。
        bundle.threshold = round(float(best_t), 2)

    # RUL 回归评估。必须在*未删失*子集上算：rul == cap 的帧只表示「至少还有 cap 步」，
    # 把它当真值会让 MAE/R² 被大量「预测 cap、真值也 cap」的平凡命中撑起来。虚高到什么
    # 程度有实测——同一份数据同一套特征，全量口径 R² 报 0.7731，未删失子集上 R² 为负，
    # 也就是说这个头其实没有绝对量级的预测力，只是把「远/近」分开了。删失率与评估样本数
    # 一并写进报告，读的人才知道这两个指标是在什么基础上算出来的。
    try:
        rul_raw = np.maximum(model_rul.predict(X_te), 0.0)
        # 评估的必须是投影后、真正会上屏的那个数。否则报告会写着 20 条矛盾而大屏一条
        # 都看不到，两边对不上，读报告的人会以为投影没生效。
        rul_pred = consistent_rul(prob, rul_raw, horizon) if prob is not None else rul_raw
        y_rul_te = y_rul[te]
        obs = y_rul_te < rul_cap
        report["rul_censored_rate"] = round(float(1 - obs.mean()), 4) if len(obs) else 0.0
        report["rul_n_eval"] = int(obs.sum())
        if int(obs.sum()) >= 10:
            report["rul_mae"] = round(float(mean_absolute_error(y_rul_te[obs], rul_pred[obs])), 2)
            report["rul_r2"] = round(float(r2_score(y_rul_te[obs], rul_pred[obs])), 4)
        # 分类器与 RUL 头学的是同一 anomaly_series 的两个切面，且 label==1 ⟺ rul<horizon
        # 恒成立，因此 prob≥0.5 的帧其 RUL 必须 < horizon。违反条数直接暴露「12 步内必然
        # 异常」与「还要 75 步才异常」同时出现在一张卡片上的自相矛盾输出。
        if prob is not None and len(prob):
            hi = prob >= 0.5
            report["n_high_risk"] = int(hi.sum())
            report["rul_contradictions"] = int(np.sum(rul_pred[hi] >= horizon))
    except Exception:
        pass

    # 规则基线对比（3σ 阈值）
    try:
        base_pred = rule_baseline_predict(feat, window).values[te]
        bp, br, bf = precision_recall_f1(yf_te, base_pred)
        report["baseline_precision"] = round(float(bp), 4)
        report["baseline_recall"] = round(float(br), 4)
        report["baseline_f1"] = round(float(bf), 4)
    except Exception:
        pass

    report["importance"] = feature_importance(model_b, X, feature_cols)
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
        w.append(f"RUL 自洽约束被破坏：{contra}/{nhi} 个 prob≥0.5 的测试帧仍预测出 RUL≥horizon({horizon})。"
                 f"consistent_rul 的投影本应让该计数恒为 0，出现非零说明投影被绕过、"
                 f"或 label 与 rul 的定义已不再互为充要条件——两者会被印在同一张卡片上，务必先查")

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


_SHAP_EXPLAINER_CACHE = {}


def feature_importance(model, X, feature_cols, top_k=8, return_method=False):
    """可解释性：SHAP 优先 → feature_importances_ → 排列重要性。返回 [{feature, importance}]。

    return_method=True 时额外返回方法名。三条分支的语义并不等价，调用方必须知道走的是哪条：
    SHAP 传入单行 X 时是局部归因，而 feature_importances_ 完全忽略 X、永远是训练集的全局
    分裂增益，拿它解释单个点位是把全局结论冒充成局部结论。
    """
    X = np.asarray(X, dtype=float)
    scores = None
    method = ""
    # 1) SHAP
    if HAVE_SHAP:
        try:
            if id(model) not in _SHAP_EXPLAINER_CACHE:
                _SHAP_EXPLAINER_CACHE[id(model)] = shap.TreeExplainer(model)
            sv = _SHAP_EXPLAINER_CACHE[id(model)].shap_values(X)
            arr = np.asarray(sv, dtype=float)
            if arr.ndim == 3:
                arr = arr[..., 1]          # 二分类取正类
            scores = dict(zip(feature_cols, [float(np.mean(np.abs(arr[:, i]))) for i in range(arr.shape[1])]))
            method = "SHAP（局部）"
        except Exception:
            scores = None
    # 2) feature_importances_
    if not scores:
        try:
            scores = dict(zip(feature_cols, [float(x) for x in model.feature_importances_]))
            method = "全局特征重要度（非局部）"
        except Exception:
            scores = None
    # 3) permutation importance
    if not scores and HAVE_PERM:
        try:
            perm = permutation_importance(model, X, np.zeros(len(X)), n_repeats=5, random_state=42)
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
        if id(model) not in _SHAP_EXPLAINER_CACHE:
            _SHAP_EXPLAINER_CACHE[id(model)] = shap.TreeExplainer(model)
        sv = np.asarray(_SHAP_EXPLAINER_CACHE[id(model)].shap_values(
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

    前两条路径的 importance 都是「该特征把未来窗口异常概率推高了多少」，量纲一致、可互换。
    第三条是训练集的全局分裂增益/幅度，语义完全不同，只能降级使用并明确标注。
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
        return default_prediction(bundle)
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
        anomaly_score = float(1.0 / (1.0 + np.exp(-raw)))  # sigmoid 归一
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
    # 75 步才异常」这种组合（依据与实测数据见 consistent_rul）
    rul = consistent_rul(prob, rul, bundle.horizon)

    # RUL 项的截断点必须等于训练目标的删失上限（见 RUL_CAP_FACTOR）：高于它的 RUL 一律
    # 记 0 分贡献。实测旧配置下 93/100 个点位的 RUL 都落在截断点之上，0.2 的权重形同虚设。
    risk_score = float(min(100.0, (0.5 * prob + 0.3 * anomaly_score + 0.2 * (1 - min(rul / max(bundle.horizon * RUL_CAP_FACTOR, 1), 1))) * 100))
    # 判正线取训练时扫出的测试集 F1 最优点（随模型存进 meta），不再是写死的 0.7 / 0.35。
    # 这个选择直接决定成果面板上模型对规则基线的胜负——同一批 268 条已评估台账记录实测：
    #   阈值 0.35（旧 *0.5 的 warning 线）命中率 62.5% / 精确率 40.3% / 每轮预警 62 条
    #   阈值 0.80（训练所得）           命中率 45.0% / 精确率 78.3% / 每轮预警 23 条
    # 基线是 30.0% / 75.0% / 提前 0 步，所以只有 0.80 这一档三项全胜；代价是预警变稀。
    # critical 线取阈值到 1.0 的中点（0.80 → 0.90），台账「warning 及以上即判正」的口径不变。
    warn_t = float(bundle.threshold) if bundle.threshold else risk_threshold
    crit_t = warn_t + (1.0 - warn_t) / 2.0
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
    }
    if errors:
        result["model_errors"] = errors
    return result


def default_prediction(bundle):
    # 降级路径也要满足同一条自洽约束：prob=0 表示「horizon 步内不会异常」，RUL 就不能
    # 小于 horizon。y_mean 现在是未删失子集的均值（偏小），不投影的话这条分支反而成了
    # 唯一会自相矛盾的出口。
    rul = consistent_rul(0.0, float(bundle.y_mean or 0), bundle.horizon)
    return {"risk_score": 0.0, "anomaly_score": 0.0, "future_anomaly_prob": 0.0,
            "rul": rul, "rul_unit": "steps", "horizon": int(bundle.horizon),
            "predicted_status": "normal", "top_features": [],
            "top_features_method": "历史不足，无法归因"}
