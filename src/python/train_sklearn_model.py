#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市管网资产数字化台账 - 资产风险预测模型训练（**演示用**）

训练3个模型：
1. 资产异常风险检测模型（IsolationForest）
2. 剩余寿命预测模型（RandomForestRegressor）
3. 健康等级分类模型（RandomForestClassifier）

⚠️ 定位说明（务必先读）
  本脚本的标签是**特征的确定性函数**，因此模型必然"学得极准"，报出的 R²/AUC 没有预测力含义：
    · remaining_life = design_life - service_years，而这两个都在特征列表里；
    · y_anomaly = risk_score >= 60，而 risk_score 又由 8 个特征线性加权生成；
    · anomaly_model.fit(X_normal) 用标签筛过"正常样本"，所以它不是无监督模型。
  它的定位是**演示用资产风险打分**，产物供 /predict/anomaly、/predict/rul 的演示链路使用。
  真正的预测性维护模型见 predictive_models.py（时序特征 + 右删失 RUL + 前向评估）。

  另外：数据不可用时**默认拒绝**用随机模拟数据训练（需要显式 --allow-mock，且产物写入
  *_mock.pkl，不覆盖生产模型文件），避免一次误运行把线上接口换成随机数据训出来的模型。
"""

import numpy as np
import pandas as pd
import os
import sys


def generate_sample_data(n_samples=10000):
    """生成模拟管网资产数据用于模型训练"""
    np.random.seed(42)

    pipeline_types = ["供水管网", "供暖管网", "燃气管网", "污水管网", "危废输送管网"]
    materials = ["球墨铸铁管", "PE管", "钢管", "混凝土管", "不锈钢管", "HDPE管"]

    data = {
        "service_years": np.random.randint(1, 45, n_samples),
        "design_life": np.random.choice([30, 40, 50], n_samples),
        "segment_length_m": np.random.uniform(20, 500, n_samples),
        "burial_depth_m": np.random.uniform(0.5, 4.0, n_samples),
        "diameter_numeric": np.random.choice([50, 100, 150, 200, 300, 400, 600, 800, 1000, 1200], n_samples),
        "risk_score": np.random.randint(0, 100, n_samples),
        "depreciation_rate": np.random.uniform(0, 0.95, n_samples),
        "inspection_gap_years": np.random.randint(0, 5, n_samples),
        "maintenance_count": np.random.randint(0, 20, n_samples),
        "incident_count": np.random.randint(0, 5, n_samples),
    }

    df = pd.DataFrame(data)
    df["aging_index"] = df["service_years"] / df["design_life"]
    df["aging_index"] = df["aging_index"].clip(0, 2.0)

    df["risk_score"] = (
        df["aging_index"] * 40 +
        df["depreciation_rate"] * 20 +
        df["inspection_gap_years"] * 5 +
        df["incident_count"] * 8 -
        df["maintenance_count"] * 1.5 +
        np.random.normal(0, 5, n_samples)
    ).clip(0, 100).astype(int)

    df["remaining_life"] = (df["design_life"] - df["service_years"]).clip(0, 50)

    return df


def preprocess_data(df):
    """特征工程"""
    feature_cols = [
        "service_years", "design_life", "segment_length_m", "burial_depth_m",
        "diameter_numeric", "depreciation_rate", "inspection_gap_years",
        "maintenance_count", "incident_count", "aging_index",
    ]

    X = df[feature_cols].copy()

    y_anomaly = (df["risk_score"] >= 60).astype(int)
    y_rul = df["remaining_life"].values
    y_health = pd.cut(
        df["risk_score"],
        bins=[-1, 30, 50, 70, 100],
        labels=[0, 1, 2, 3]
    ).astype(int).values

    return X, y_anomaly, y_rul, y_health


def train_anomaly_model(X, y):
    """训练异常风险检测模型"""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    normal_mask = y == 0
    X_normal = X_scaled[normal_mask]

    model = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    model.fit(X_normal)

    return model, scaler


def train_rul_model(X, y_rul):
    """训练剩余寿命预测模型"""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_scaled, y_rul)

    return model, scaler


def train_health_model(X, y_health):
    """训练健康等级分类模型"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
    model.fit(X_scaled, y_health)

    return model, scaler


def save_model(model, name, output_dir="models", name_suffix=""):
    import joblib
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}{name_suffix}.pkl")
    joblib.dump(model, path)
    print(f"  模型保存: {path}")
    return path


def evaluate_models(anomaly_model, anomaly_scaler, rul_model, rul_scaler,
                    health_model, health_scaler, X, y_anomaly, y_rul, y_health,
                    test_ratio=0.2, seed=42):
    """在留出集上给出指标，**并明确这些指标不能当作预测力**。

    本脚本的标签是特征的确定性函数，模型必然"学得极准"，这个 R²/AUC 没有任何意义：
      · remaining_life = design_life - service_years，而两者都在特征列表里；
      · y_anomaly = risk_score >= 60，而 risk_score 又由 8 个特征线性加权生成；
      · anomaly_model.fit(X_normal) 还用标签筛过"正常样本"，所以它不是无监督模型，
        contamination=0.05 与真实异常率也不一致。
    因此本脚本的定位是 **演示用资产风险打分**，产出的 pkl 供 /predict/anomaly、/predict/rul
    演示链路使用，不要对外宣称"模型预测寿命/异常的准确率"。真要评估预测力，得换成有真实
    标签（或等待一段时间后回填标签）的数据。
    """
    from sklearn.metrics import roc_auc_score, mean_absolute_error, r2_score, accuracy_score
    n = len(X)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    cut = int(n * (1 - test_ratio))
    tr, te = idx[:cut], idx[cut:]
    X_tr, X_te = X.iloc[tr], X.iloc[te]
    m = {}
    try:
        a_sc = anomaly_scaler.transform(X_te)
        # 注意方向：score_samples 越低越异常，所以取负号后再算 AUC
        m["anomaly_auc"] = round(float(roc_auc_score(y_anomaly[te], -anomaly_model.score_samples(a_sc))), 4)
    except Exception as e:
        m["anomaly_auc_error"] = f"{type(e).__name__}: {e}"
    try:
        r_sc = rul_scaler.transform(X_te)
        pred = rul_model.predict(r_sc)
        m["rul_mae"] = round(float(mean_absolute_error(y_rul[te], pred)), 3)
        m["rul_r2"] = round(float(r2_score(y_rul[te], pred)), 4)
    except Exception as e:
        m["rul_error"] = f"{type(e).__name__}: {e}"
    try:
        h_sc = health_scaler.transform(X_te)
        m["health_accuracy"] = round(float(accuracy_score(y_health[te], health_model.predict(h_sc))), 4)
    except Exception as e:
        m["health_error"] = f"{type(e).__name__}: {e}"
    m["note"] = ("标签由特征确定性生成（remaining_life=design_life-service_years；"
                 "y_anomaly=risk_score>=60），指标只证明脚本自洽，不代表预测力")
    return m


def _load_from_hive():
    """从 Hive 读 dwd.pipeline_asset_detail；不可用时抛异常（由调用方决定是否降级）。"""
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.appName("AssetModelTraining").enableHiveSupport().getOrCreate()
    try:
        df = spark.sql("SELECT * FROM dwd.pipeline_asset_detail").toPandas()
    finally:
        try:
            spark.stop()
        except Exception:
            pass
    return df


def main():
    import argparse
    import json
    from datetime import datetime

    ap = argparse.ArgumentParser(description="资产风险预测模型训练（演示用）")
    ap.add_argument("--input", default="", help="本地 CSV/Parquet 数据；给了就不连 Hive")
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--allow-mock", action="store_true",
                    help="允许在数据不可用时用随机模拟数据训练。**默认禁止**："
                         "原实现只要 Hive 抛异常就静默改用随机数据训练，并 joblib.dump 覆盖 "
                         "models/anomaly_model.pkl 与 rul_model.pkl —— 而 main.py 正是加载这两个 "
                         "文件供 /predict/anomaly、/predict/rul 使用，一次误运行就能让线上接口换成"
                         "随机数据训出来的模型，退出码还是 0、日志只有一行提示。")
    ap.add_argument("--min-rows", type=int, default=200,
                    help="数据行数下限，低于该值直接非 0 退出（避免用残缺数据静默训出模型）")
    args = ap.parse_args()

    print("=" * 50)
    print("城市管网资产风险预测模型训练（演示用）")
    print("=" * 50)

    source, df = None, None
    if args.input:
        print(f"从文件加载数据: {args.input}")
        df = pd.read_csv(args.input) if args.input.lower().endswith(".csv") else pd.read_parquet(args.input)
        source = f"file:{os.path.basename(args.input)}"
    else:
        try:
            df = _load_from_hive()
            print(f"从Hive加载数据: {len(df)} 条")
            source = "hive:dwd.pipeline_asset_detail"
        except Exception as e:
            print(f"[WARN] Hive 不可用：{type(e).__name__}: {e}")
            if not args.allow_mock:
                print("[ERROR] 未指定 --allow-mock，拒绝用随机数据训练并覆盖生产模型文件。")
                print("        如需演示，请显式运行：python3 train_sklearn_model.py --allow-mock")
                sys.exit(2)
            df = generate_sample_data(10000)
            source = "mock:generate_sample_data"
            print(f"[WARN] 已按 --allow-mock 生成模拟数据: {len(df)} 条（将写入 *_mock.pkl，不覆盖生产模型）")

    if df is None or len(df) < args.min_rows:
        print(f"[ERROR] 有效数据不足（{0 if df is None else len(df)} < {args.min_rows}），退出码非 0，"
              f"避免用残缺数据静默产出模型")
        sys.exit(3)

    is_mock = str(source or "").startswith("mock")
    suffix = "_mock" if is_mock else ""

    X, y_anomaly, y_rul, y_health = preprocess_data(df)

    print(f"\n[1/3] 训练异常风险检测模型...")
    anomaly_model, anomaly_scaler = train_anomaly_model(X, y_anomaly)
    save_model(anomaly_model, "anomaly_model", args.output_dir, suffix)
    save_model(anomaly_scaler, "anomaly_scaler", args.output_dir, suffix)

    print(f"\n[2/3] 训练剩余寿命预测模型...")
    rul_model, rul_scaler = train_rul_model(X, y_rul)
    save_model(rul_model, "rul_model", args.output_dir, suffix)
    save_model(rul_scaler, "rul_scaler", args.output_dir, suffix)

    print(f"\n[3/3] 训练健康等级分类模型...")
    health_model, health_scaler = train_health_model(X, y_health)
    save_model(health_model, "health_model", args.output_dir, suffix)
    save_model(health_scaler, "health_scaler", args.output_dir, suffix)

    metrics = evaluate_models(anomaly_model, anomaly_scaler, rul_model, rul_scaler,
                              health_model, health_scaler, X, y_anomaly, y_rul, y_health)
    # 数据来源与指标必须随模型落盘：不标记来源，就无法判断某个 pkl 是真数据训的还是随机数据训的
    meta = {"source": source, "rows": int(len(df)), "is_mock": bool(is_mock),
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "name_suffix": suffix, "metrics": metrics,
            "demo_only": True}
    os.makedirs(args.output_dir, exist_ok=True)
    meta_path = os.path.join(args.output_dir, f"sklearn_train_meta{suffix}.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n{'=' * 50}")
    print("全部模型训练完成!")
    print(f"数据来源: {source}  行数: {len(df)}")
    print(f"留出集指标: {json.dumps(metrics, ensure_ascii=False)}")
    print(f"来源与指标: {meta_path}")
    if is_mock:
        print("注意：本次为模拟数据训练，产物为 *_mock.pkl，未覆盖生产模型文件")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
