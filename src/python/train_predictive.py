#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测性维护 - 模型训练脚本
==========================
读取传感器历史时序数据（JSONL 或 CSV），训练 异常检测(A) + 风险预测(B) + RUL 回归，
输出模型到 models/ 目录与一份训练评估报告 predictive_report.json。

用法：
    python3 train_predictive.py --input data/sensor_history.jsonl --model-dir models \
        --window 15 --horizon 12

输入数据（JSONL，每行一条，与 main.py 传感器读数一致）：
    {"timestamp":"2026-09-01 19:55:00","sensor_id":"SENSOR-001","device_type":"供水管网",
     "status":"normal","metrics":{"pressure":0.45,"flow":120,"temperature":22.1}}

说明：
  - 若 xgboost/lightgbm/shap 未安装，自动降级到 sklearn + feature_importances_，不影响运行
  - 训练只需 CPU 即可
"""
import argparse
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import predictive_models as pm  # noqa: E402


def load_input(path, limit=None):
    """读取 JSONL 或 CSV 为 DataFrame；同时给出按传感器分组的 reading 列表。"""
    if path.endswith(".csv"):
        df = pd.read_csv(path)
        if "timestamp" in df.columns and "metrics" not in df.columns:
            # 宽表：每行已含各指标列
            df["status"] = df.get("status", "normal")
            return df, None
    readings = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                readings.append(json.loads(line))
            except Exception:
                continue
            if limit and len(readings) >= limit:
                break
    df = pm.readings_to_frame(readings)
    return df, readings


def group_by_sensor(readings):
    groups = {}
    for r in readings or []:
        sid = r.get("sensor_id", "")
        groups.setdefault(sid, []).append(r)
    return groups


def main():
    ap = argparse.ArgumentParser(description="预测性维护模型训练")
    ap.add_argument("--input", required=True, help="输入数据（JSONL 或 CSV）")
    ap.add_argument("--model-dir", default="models", help="模型输出目录")
    ap.add_argument("--window", type=int, default=15, help="滑动窗口长度")
    ap.add_argument("--horizon", type=int, default=12, help="预测步数（未来 N 步内是否异常）")
    ap.add_argument("--limit", type=int, default=0, help="最多读取条数（0=全部）")
    ap.add_argument("--contamination", default="auto", help="IsolationForest 污染率（auto/0.0-0.5）")
    args = ap.parse_args()

    print("=" * 60)
    print("预测性维护模型训练")
    print(f"  输入: {args.input}  窗口: {args.window}  预测步数: {args.horizon}")
    print(f"  引擎: lightgbm={pm.HAVE_LGB} xgboost={pm.HAVE_XGB} shap={pm.HAVE_SHAP}")
    print("=" * 60)

    df, readings = load_input(args.input, args.limit or None)
    if df.empty:
        print("错误: 未读取到数据")
        sys.exit(1)
    print(f"已读取 {len(df)} 条，传感器 {df['sensor_id'].nunique() if 'sensor_id' in df else '?'} 个")

    # 若为宽表 CSV（无 readings），仍可按 sensor_id 用 df；train_bundle 接受 pooled df
    try:
        bundle, report = pm.train_bundle(df, window=args.window, horizon=args.horizon,
                                         anomaly_contamination=args.contamination)
    except Exception as e:
        print(f"训练失败: {e}")
        sys.exit(1)

    bundle.save(args.model_dir)
    report_path = os.path.join(args.model_dir, "predictive_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n【评估报告】")
    label = report.get("label") or {}
    for k, v in report.items():
        if k in ("importance", "label", "warnings"):
            continue
        print(f"  {k}: {v}")
    if label:
        # 基准率必须紧挨着 avg_precision 看：两者接近就说明模型只是背了基准率。
        # 训练集与测试集基准率要并排打：采集期间切换过场景会让测试集整段落在单一区间
        # （实测遇到过 训练集 5.99% vs 测试集 88.12%），只看一个数认不出来。
        print("\n【数据体检】")
        print(f"  异常读数占比:   {label.get('reading_anomaly_rate', 0):.2%}")
        print(f"  正样本占比:     全量 {label.get('positive_rate', 0):.2%}"
              f"   训练集 {label.get('positive_rate_train', 0):.2%}"
              f"   测试集 {label.get('positive_rate_test', 0):.2%}")
        print(f"  样本数:         训练 {label.get('n_train', 0)}  测试 {label.get('n_test', 0)}"
              f"  正样本 {label.get('n_positive', 0)}")
        print(f"  avg_precision:  {report.get('avg_precision', 'n/a')}"
              f"  ← 与测试集正样本占比接近即为无效")
        print(f"  AUC:            {report.get('auc', 'n/a')}"
              f"  ← 唯一不受基准率影响的判别指标")
        print(f"  RUL(未删失):    MAE {report.get('rul_mae', 'n/a')}  R² {report.get('rul_r2', 'n/a')}"
              f"   n={report.get('rul_n_eval', 0)}  删失率 {report.get('rul_censored_rate', 0):.2%}"
              f"  ← 全量口径会把删失帧算成命中，指标虚高")
        print(f"  RUL 自相矛盾:   {report.get('rul_contradictions', 0)}/{report.get('n_high_risk', 0)}"
              f"  ← prob≥0.5 却 RUL≥horizon 的帧数，必须为 0")
        print(f"  传感器数:       {report.get('n_sensors', '?')}")
    print("  特征重要性(全局): " + ", ".join(
        f"{i['feature']}={i['importance']}" for i in (report.get("importance") or [])[:6]))

    warns = report.get("warnings") or []
    if warns:
        print("\n【警告】" + "─" * 50)
        for i, w in enumerate(warns, 1):
            print(f"  {i}. {w}")
    else:
        print("\n【警告】无：标签分布、AUC 判别力与 RUL 一致性检查均通过")

    print(f"\n模型已保存到: {args.model_dir}/")
    print("  - predictive_models.joblib")
    print("  - predictive_meta.json")
    print(f"  - predictive_report.json")


if __name__ == "__main__":
    main()
