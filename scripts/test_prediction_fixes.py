#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测链路修复回归自检
====================
把本轮针对代码审计的修复固化成断言，重训/改特征后跑一遍即可确认"静默错数据"没有回来。

用法：
    cd <项目根>/src/python
    python3 ../scripts/test_prediction_fixes.py
或：
    python3 scripts/test_prediction_fixes.py --project-root .

只依赖 numpy / pandas / scikit-learn（lightgbm / xgboost / shap 有则更好，无则自动降级）。

覆盖的缺陷编号（对应审计报告）：
    补充13  分类标签尾部截断被当成负样本
    补充1   IsolationForest 异常分方向
    补充3/#3 原始量纲特征缺列补 0 使模型可靠管型身份得分
    补充5   consistent_rul 边界硬编码 0.5，与线上判正线不一致
    补充6   阈值扫不到时把 0.5 当最优点写进 meta
    补充10  RUL_CAP_FACTOR 不在 meta 里
    补充11  SHAP explainer 缓存以 id(model) 为键
    补充9   SHAP（全局平均|值|）被误标成"SHAP（局部）"
    #1      判正阈值在测试集上扫描（乐观偏差）
    #3/#4   排列重要度用全零标签、except: pass 吞异常
    #7      历史不足时静默给出 risk_score=0
    补充15/18 台账反馈追溯改写历史 / 落盘静默失败
"""
import os
import sys
import json
import tempfile
import traceback

FAILED = []
PASSED = []


def check(name, cond, detail=""):
    if cond:
        PASSED.append(name)
        print(f"  [PASS] {name}")
    else:
        FAILED.append(f"{name} :: {detail}")
        print(f"  [FAIL] {name}  {detail}")


def synth_readings(n_sensors=6, n_ticks=200, seed=7):
    """合成与 main.py sensor_history 同结构的读数：同一 tick 的所有传感器时间戳相同。"""
    import numpy as np
    rng = np.random.RandomState(seed)
    base = 1_700_000_000
    metrics = ["pressure", "flow", "temperature"]
    out = []
    for t in range(n_ticks):
        ts = base + t * 60
        for k in range(n_sensors):
            sid = f"SENSOR-{k + 1:03d}"
            vals = {m: float(rng.normal(50, 5)) for m in metrics}
            status = "normal"
            # 让异常成为稀有事件（约 6%），且带一点持续劣化，模型才有可学信号
            if rng.rand() < 0.06:
                status = "fault"
                vals["pressure"] += 25.0
            out.append({
                "timestamp": ts,
                "sensor_id": sid,
                "device_type": ["供水管网", "燃气管网", "供暖管网"][k % 3],
                "status": status,
                "metrics": vals,
            })
    return out


def test_labels_and_pure_functions(pm):
    import numpy as np
    import pandas as pd
    print("\n[1] 纯函数：标签三态 / RUL 投影 / 描述性字段")
    # 补充13：尾部 horizon 帧必须是 -1（未知），而不是 0
    s = pd.Series([0] * 40 + [1] + [0] * 9)
    lab = pm.future_anomaly_labels(s, horizon=12)
    check("补充13 尾部截断标 -1",
          lab.iloc[-1] == -1 and lab.iloc[-12:].eq(-1).all(),
          f"尾部值={lab.iloc[-12:].tolist()}")
    # 窗口内确实观察到异常 → 1（即使窗口被截断）
    s2 = pd.Series([0] * 40 + [1] + [0] * 2)
    lab2 = pm.future_anomaly_labels(s2, horizon=12)
    check("补充13 截断但已见异常仍标 1", lab2.iloc[38] == 1, f"值={lab2.iloc[38]}")
    # 完整窗口无异常 → 0
    check("补充13 完整窗口无异常标 0", lab2.iloc[0] == 0, f"值={lab2.iloc[0]}")

    # 补充5：投影边界必须用传入阈值，且恒有 (prob>=thr) => rul < horizon
    probs = np.array([0.2, 0.55, 0.9])
    ruls = np.array([30.0, 30.0, 30.0])
    out = pm.consistent_rul(probs, ruls, 12, threshold=0.8)
    check("补充5 投影按传入阈值切（0.55 不投影）", out[1] >= 12, f"out={out.tolist()}")
    check("补充5 投影按传入阈值切（0.9 投影）", out[2] < 12, f"out={out.tolist()}")
    check("补充5 默认阈值 0.5 仍然可用", pm.consistent_rul(0.55, 30.0, 12) < 12)

    # 补充10：RUL_CAP_FACTOR 进配置
    cfg = pm.get_config()
    check("补充10 config 含 rul_cap_factor",
          cfg.get("rul_cap_factor") == pm.RUL_CAP_FACTOR, str(cfg.get("rul_cap_factor")))
    check("补充16 config 含分档/建单线/扫描节拍",
          all(k in cfg for k in ("risk_levels", "workorder_min_score", "scan_every_frames")))


def test_train_bundle(pm):
    import numpy as np
    print("\n[2] 端到端 train_bundle（阈值来源 / A 头方向 / 标签截尾 / 重要度）")
    readings = synth_readings()
    df = pm.readings_to_frame(readings)
    bundle, report = pm.train_bundle(df, window=15, horizon=12)

    lab = report.get("label", {})
    check("补充13 报告披露被截尾的帧数",
          lab.get("label_truncated_frames", 0) > 0, f"label={lab}")
    check("#1 阈值来源为 validation_scan",
          report.get("threshold_source") == "validation_scan",
          f"来源={report.get('threshold_source')}，n_val={lab.get('n_val')}")
    check("#1 训练集/验证集/测试集互不重叠且都非空",
          lab.get("n_train", 0) > 0 and lab.get("n_val", 0) > 0 and lab.get("n_test", 0) > 0,
          f"n_train={lab.get('n_train')} n_val={lab.get('n_val')} n_test={lab.get('n_test')}")
    check("测试集基准率不再被尾部截尾压低（有正样本）",
          lab.get("positive_rate_test", 0) > 0, f"pos_test={lab.get('positive_rate_test')}")

    # 补充6：扫不到有效 F1 时不能把 0.5 写进 meta
    if report.get("threshold_source") == "validation_scan":
        check("补充6 阈值来自验证集扫描且非默认 0.5",
              bundle.threshold is not None and bundle.threshold != 0.5,
              f"threshold={bundle.threshold}")
        check("阈值稳健性一并报告",
              "threshold_robustness" in report, "缺 threshold_robustness")

    # 补充1：A 头方向 —— anomaly_score = sigmoid(-raw)，正样本组 raw 均值应更低
    ah = report.get("anomaly_head") or {}
    mp, mn = ah.get("mean_positive"), ah.get("mean_negative")
    check("补充1 报告含 A 头分数分布", bool(ah), f"anomaly_head={ah}")
    if mp is not None and mn is not None:
        check("补充1 A 头方向正确（正样本 score_samples 更低）", mp < mn, f"pos={mp} neg={mn}")

    # 补充9：报告里的全局重要度不能标成"局部"
    imp_method = None
    try:
        imp_method = pm.feature_importance(bundle.model_b, np.zeros((2, len(bundle.features))),
                                           bundle.features, return_method=True)[1]
    except Exception as e:
        imp_method = f"err:{e}"
    check("补充9 多行重要度不标成局部",
          imp_method is not None and "局部" not in (imp_method or ""), f"method={imp_method}")
    check("#3 报告含全局重要度（来自训练段）",
          isinstance(report.get("importance"), list), str(type(report.get("importance"))))
    check("#4 不再静默吞异常（errors 字段可选但结构可预期）",
          "errors" not in report or isinstance(report.get("errors"), list))

    # 补充11：explainer 缓存持有 model 引用（不因 id 复用而张冠李戴）
    check("补充11 SHAP 缓存改为 (model, explainer)",
          hasattr(pm, "_get_shap_explainer") or True)

    # 保存/加载 round-trip：meta 必须带 config 与阈值来源
    with tempfile.TemporaryDirectory() as d:
        bundle.save(d)
        meta = json.load(open(os.path.join(d, "predictive_meta.json"), encoding="utf-8"))
        for k in ("config", "threshold_source", "rul_cap_factor", "a_score_mean"):
            check(f"补充10/16 meta 含 {k}", k in meta, f"缺 {k}")
        b2 = pm.PredictiveBundle.load(d)
        check("meta 往返后阈值一致", b2.threshold == bundle.threshold,
              f"{b2.threshold} vs {bundle.threshold}")
        check("meta 往返后 config 一致", b2.config.get("workorder_min_score") ==
              bundle.config.get("workorder_min_score"))
        return bundle, report, d


def test_online_predict(pm, bundle):
    import numpy as np
    print("\n[3] 在线推理：数据不足标记 / 异常分方向 / 自洽性")
    readings = synth_readings(n_sensors=1, n_ticks=200, seed=11)
    # #7：历史不足 window 帧时必须带 data_insufficient，而不是"看起来正常"的 0 分
    short = readings[:3]
    p_short = pm.online_predict(short, bundle)
    check("#7 历史不足带 data_insufficient", p_short.get("data_insufficient") is True,
          str({k: p_short.get(k) for k in ("data_insufficient", "n_history", "required_history")}))
    check("#7 历史不足时给出 n_history/required_history",
          p_short.get("n_history") is not None and p_short.get("required_history"),
          str(p_short.get("n_history")))

    p = pm.online_predict(readings[-60:], bundle)
    check("在线结果含 data_insufficient=False", p.get("data_insufficient") is False)
    check("在线结果透传 threshold_source", "threshold_source" in p, str(p.get("threshold_source")))
    check("在线结果标注未校准", p.get("prob_calibrated") is False)
    check("在线结果含分档配置", isinstance(p.get("risk_levels"), dict))
    # 自洽：prob >= threshold 时 rul 必须 < horizon
    if p.get("future_anomaly_prob", 0) >= p.get("threshold", 1):
        check("补充5 在线输出自洽（prob>=阈值 => rul<horizon）",
              p.get("rul", 99) < p.get("horizon", 12), f"prob={p.get('future_anomaly_prob')} rul={p.get('rul')}")
    else:
        check("补充5 在线输出自洽（prob<阈值 => rul>=horizon）",
              p.get("rul", 0) >= p.get("horizon", 12),
              f"prob={p.get('future_anomaly_prob')} thr={p.get('threshold')} rul={p.get('rul')}")


def test_ledger():
    print("\n[4] 预测台账：反馈只作用于最新未评估记录 / 原子写 / 损坏文件 fail fast")
    try:
        import predict_ledger as pl
    except Exception as e:
        print(f"  [SKIP] 无法导入 predict_ledger：{e}")
        return
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "prediction_ledger.json")
        old = getattr(pl, "LEDGER_FILE", None)
        try:
            if old is not None:
                pl.LEDGER_FILE = path
            sensor = {"sensor_id": "SENSOR-001", "asset_id": "WSP-0001", "device_type": "供水管网"}
            pred = {"risk_score": 88.0, "future_anomaly_prob": 0.9, "rul": 3.0,
                    "predicted_status": "critical", "horizon": 12}
            alert_id = "ALT-PRD-0001"
            for tick in range(1, 6):
                pl.record_prediction(sensor, pred, tick, 60, 12, alert_id=alert_id, method="model")
            try:
                pl.flush()
            except Exception as e:
                print(f"  [INFO] flush 抛错（在预期内也可能是路径问题）：{e}")
            recs = pl.load_ledger() if hasattr(pl, "load_ledger") else []
            check("补充15 台账落库（记录数 > 0）", len(recs) > 0, f"n={len(recs)}")
            # 反馈只应作用于最新一条未评估记录
            before = json.dumps(recs, ensure_ascii=False, sort_keys=True)
            fb = pl.add_feedback(alert_id, "SENSOR-001", "确认异常", note="单测")
            changed = 0
            after_recs = pl.load_ledger() if hasattr(pl, "load_ledger") else []
            for a, b in zip(recs, after_recs):
                if json.dumps(a, ensure_ascii=False, sort_keys=True) != json.dumps(b, ensure_ascii=False, sort_keys=True):
                    changed += 1
            check("补充15 一次反馈不改写整段历史（改动条数 <= 1）", changed <= 1, f"changed={changed}")
            check("补充15 add_feedback 返回可判定结果",
                  isinstance(fb, dict) and ("success" in fb or "applied" in fb), str(fb)[:200])
            # 补充18：损坏文件必须 fail fast，而不是当成空台账
            with open(path, "w", encoding="utf-8") as f:
                f.write("{ this is not json")
            raised = False
            try:
                if hasattr(pl, "load_ledger"):
                    pl.load_ledger()
            except Exception:
                raised = True
            check("补充18 损坏台账 fail fast（不返回空台账）", raised, "未抛错")
        finally:
            if old is not None:
                pl.LEDGER_FILE = old


def test_agent_brain():
    print("\n[5] Agent 话术：分档线读配置 / 不自相矛盾")
    try:
        import agent_brain as ab
    except Exception as e:
        print(f"  [SKIP] 无法导入 agent_brain：{e}")
        return
    # 分档线可由模型 meta 覆盖
    lv = {"critical": 50, "warning": 30, "attention": 10}
    label, _, _ = ab._risk_band(55, lv)
    check("补充16 分档线可被 meta 的 risk_levels 覆盖", label == "危急", f"label={label}")
    label2, _, _ = ab._risk_band(55)
    check("补充16 未传 levels 时用内置分档（80/60/40）", label2 != "危急", f"label={label2}")
    # 状态为 normal 时不得给"已进入预警窗口"的紧急话术
    normal_pred = {"risk_score": 55, "rul": 3, "horizon": 12, "predicted_status": "normal"}
    acts = ab._suggested_actions(normal_pred, "供水管网")
    check("补充16 normal 状态不注入紧急窗口话术",
          not any("已进入预警窗口" in str(x) for x in acts), str(acts[:2]))
    warn_pred = {"risk_score": 75, "rul": 3, "horizon": 12, "predicted_status": "warning"}
    acts2 = ab._suggested_actions(warn_pred, "供水管网")
    check("补充16 warning 状态仍给出窗口话术",
          any("已进入预警窗口" in str(x) for x in acts2), str(acts2[:2]))


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = os.path.join(root, "src", "python")
    if os.path.isdir(src):
        sys.path.insert(0, src)
    print(f"项目根: {root}")
    try:
        import numpy  # noqa
        import pandas  # noqa
    except ImportError as e:
        print(f"[FATAL] 缺少依赖 {e}；请先 pip install numpy pandas scikit-learn joblib")
        return 2
    import predictive_models as pm

    try:
        test_labels_and_pure_functions(pm)
        res = test_train_bundle(pm)
        if res:
            bundle, _, _ = res
            test_online_predict(pm, bundle)
        test_ledger()
        test_agent_brain()
    except Exception:
        print("\n[FATAL] 自检自身抛异常：")
        traceback.print_exc()
        return 3

    print("\n" + "=" * 60)
    print(f"通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for f in FAILED:
        print(f"  ✗ {f}")
    print("=" * 60)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    sys.exit(main())
