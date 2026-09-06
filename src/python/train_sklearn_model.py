#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市管网资产数字化台账 - 资产风险预测模型训练

训练3个模型：
1. 资产异常风险检测模型（IsolationForest）
2. 剩余寿命预测模型（RandomForestRegressor）
3. 健康等级分类模型（RandomForestClassifier）
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


def save_model(model, name, output_dir="models"):
    import joblib
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.pkl")
    joblib.dump(model, path)
    print(f"  模型保存: {path}")


def main():
    print("=" * 50)
    print("城市管网资产风险预测模型训练")
    print("=" * 50)

    try:
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.appName("AssetModelTraining").enableHiveSupport().getOrCreate()
        df_spark = spark.sql("SELECT * FROM dwd.pipeline_asset_detail")
        df = df_spark.toPandas()
        print(f"从Hive加载数据: {len(df)} 条")
        spark.stop()
    except Exception as e:
        print(f"Hive不可用({e})，使用模拟数据")
        df = generate_sample_data(10000)
        print(f"生成模拟数据: {len(df)} 条")

    X, y_anomaly, y_rul, y_health = preprocess_data(df)

    print(f"\n[1/3] 训练异常风险检测模型...")
    anomaly_model, anomaly_scaler = train_anomaly_model(X, y_anomaly)
    save_model(anomaly_model, "anomaly_model")
    save_model(anomaly_scaler, "anomaly_scaler")

    print(f"\n[2/3] 训练剩余寿命预测模型...")
    rul_model, rul_scaler = train_rul_model(X, y_rul)
    save_model(rul_model, "rul_model")
    save_model(rul_scaler, "rul_scaler")

    print(f"\n[3/3] 训练健康等级分类模型...")
    health_model, health_scaler = train_health_model(X, y_health)
    save_model(health_model, "health_model")
    save_model(health_scaler, "health_scaler")

    print(f"\n{'=' * 50}")
    print("全部模型训练完成!")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
