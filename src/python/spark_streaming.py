#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市管网资产数字化台账 - Spark Streaming 实时计算

数据链路：
    Kafka(sensor-data) → Spark Structured Streaming
        → ODS  ods.sensor_raw          （原始数据，TEXTFILE，按 dt 分区）
        → DWD  dwd.sensor_detail       （展开 metrics JSON 为独立字段，ORC）
        → DWS  dws.sensor_minute_stats （60s 滑动窗口聚合，ORC）
        → ADS  ads.sensor_realtime     （大屏最新快照，ORC，overwrite）

启动（见 scripts/start_spark_streaming.sh）：
    spark-submit --master spark://spark-master:7077 \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 \
      src/python/spark_streaming.py --bootstrap kafka:9093 --metastore-uris thrift://hive-metastore:9083
"""

import argparse
import sys

from pyspark.sql import SparkSession, functions as F
from pyspark.sql.types import (StructType, StructField, StringType,
                               IntegerType, DoubleType)

SENSOR_SCHEMA = StructType([
    StructField("sensor_id", StringType()),
    StructField("asset_id", StringType()),
    StructField("pipe_type", StringType()),
    StructField("metrics", StringType()),
    StructField("status", StringType()),
    StructField("alert_code", IntegerType()),
    StructField("alert_desc", StringType()),
    StructField("timestamp", StringType()),
])

# ODS/DWD/DWS/ADS 列顺序与 config/hive/hive_ddl.sql 保持一致
ODS_COLS = ["sensor_id", "asset_id", "pipe_type", "metrics", "status", "alert_code", "alert_desc", "timestamp"]
DWD_COLS = ["sensor_id", "asset_id", "pipe_type", "status", "alert_code", "alert_desc",
            "pressure", "flow", "temperature", "gas_concentration", "level", "vibration", "timestamp"]
DWS_COLS = ["window_start", "window_end", "pipe_type", "avg_pressure", "avg_temperature",
            "avg_flow", "anomaly_count", "sensor_count", "health_score"]
ADS_COLS = ["window_start", "pipe_type", "avg_pressure", "avg_temperature", "avg_flow",
            "anomaly_count", "sensor_count", "health_score"]

DDL_STATEMENTS = [
    "CREATE DATABASE IF NOT EXISTS ods",
    "CREATE DATABASE IF NOT EXISTS dwd",
    "CREATE DATABASE IF NOT EXISTS dws",
    "CREATE DATABASE IF NOT EXISTS ads",
    """CREATE EXTERNAL TABLE IF NOT EXISTS ods.sensor_raw (
        sensor_id STRING, asset_id STRING, pipe_type STRING, metrics STRING,
        status STRING, alert_code INT, alert_desc STRING, timestamp STRING)
      PARTITIONED BY (dt STRING) STORED AS TEXTFILE
      LOCATION '/user/hive/warehouse/ods.db/sensor_raw'""",
    """CREATE TABLE IF NOT EXISTS dwd.sensor_detail (
        sensor_id STRING, asset_id STRING, pipe_type STRING, status STRING,
        alert_code INT, alert_desc STRING, pressure DOUBLE, flow DOUBLE,
        temperature DOUBLE, gas_concentration DOUBLE, level DOUBLE, vibration DOUBLE, timestamp STRING)
      PARTITIONED BY (dt STRING) STORED AS ORC
      TBLPROPERTIES ('orc.compress'='SNAPPY')""",
    """CREATE TABLE IF NOT EXISTS dws.sensor_minute_stats (
        window_start STRING, window_end STRING, pipe_type STRING,
        avg_pressure DOUBLE, avg_temperature DOUBLE, avg_flow DOUBLE,
        anomaly_count INT, sensor_count INT, health_score DOUBLE)
      PARTITIONED BY (dt STRING) STORED AS ORC
      TBLPROPERTIES ('orc.compress'='SNAPPY')""",
    """CREATE TABLE IF NOT EXISTS ads.sensor_realtime (
        window_start STRING, pipe_type STRING, avg_pressure DOUBLE,
        avg_temperature DOUBLE, avg_flow DOUBLE,
        anomaly_count INT, sensor_count INT, health_score DOUBLE)
      STORED AS ORC TBLPROPERTIES ('orc.compress'='SNAPPY')""",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Spark Streaming 传感器实时计算")
    parser.add_argument("--bootstrap", default="kafka:9093", help="Kafka bootstrap servers（容器内 kafka:9093，宿主机 localhost:9092）")
    parser.add_argument("--topic", default="sensor-data")
    parser.add_argument("--master", default=None, help="Spark master（默认使用 spark-submit 指定）")
    parser.add_argument("--window", type=int, default=60, help="滑动窗口时长（秒）")
    parser.add_argument("--slide", type=int, default=10, help="滑动步长（秒）")
    parser.add_argument("--starting-offsets", default="latest", choices=["latest", "earliest"])
    parser.add_argument("--checkpoint", default=None,
                        help="checkpoint 根目录；每个 sink 自动使用其独立子目录。"
                             "默认取 --warehouse-dir 同级的 checkpoint/sensor-streaming")
    parser.add_argument("--fail-on-data-loss", action="store_true",
                        help="Kafka 分区偏移越界/数据被清理时直接抛错。默认 False（保持原行为：静默跳过，可能丢数据）")
    parser.add_argument("--metastore-uris", default="thrift://hive-metastore:9083",
                        help="Hive Metastore 地址（宿主机直连用 thrift://localhost:9083）")
    parser.add_argument("--warehouse-dir", default="hdfs://hadoop-namenode:8020/user/hive/warehouse")
    parser.add_argument("--fs-default", default="hdfs://hadoop-namenode:8020")
    return parser.parse_args()


def default_checkpoint_root(warehouse_dir):
    """默认 checkpoint 根目录：与 warehouse 同级的 checkpoint/ 子目录（放在 HDFS 上，容器重建不丢进度）。

    hdfs://nn:8020/user/hive/warehouse -> hdfs://nn:8020/user/hive/checkpoint/sensor-streaming
    """
    base = (warehouse_dir or "").rstrip("/")
    if not base:
        return "/tmp/spark-checkpoint/sensor-streaming"
    if "/" not in base:
        return base + "/checkpoint/sensor-streaming"
    return base.rsplit("/", 1)[0] + "/checkpoint/sensor-streaming"


def main():
    args = parse_args()

    # checkpoint 根目录：显式传入优先，否则落在 warehouse 同级的 checkpoint/ 下
    ckpt_root = (args.checkpoint or default_checkpoint_root(args.warehouse_dir)).rstrip("/")
    # 每个 OutputStream 必须独占自己的 checkpoint 目录，
    # 否则第二个流启动会报 "checkpoint directory ... already in use"
    ckpt_ods_dwd = f"{ckpt_root}/ods-dwd"
    ckpt_dws_ads = f"{ckpt_root}/dws-ads"

    builder = SparkSession.builder.appName("SensorStreaming").enableHiveSupport()
    if args.master:
        builder = builder.master(args.master)
    builder = (builder
               .config("spark.sql.warehouse.dir", args.warehouse_dir)
               .config("spark.hadoop.hive.metastore.uris", args.metastore_uris)
               .config("spark.hadoop.fs.defaultFS", args.fs_default))
    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    print("=" * 60)
    print("Spark Streaming - 传感器实时计算")
    print(f"  Kafka: {args.bootstrap}/{args.topic}")
    print(f"  窗口: {args.window}s / 滑动 {args.slide}s")
    print(f"  Metastore: {args.metastore_uris}")
    print(f"  Checkpoint 根目录: {ckpt_root}")
    print(f"    - ODS/DWD 流: {ckpt_ods_dwd}")
    print(f"    - DWS/ADS 流: {ckpt_dws_ads}")
    if args.fail_on_data_loss:
        print("  failOnDataLoss: true（严格模式，偏移越界直接失败）")
    else:
        print("  [警告] failOnDataLoss: false —— 当前允许静默丢数据：")
        print("         Kafka 分区偏移越界或数据被 retention 清理时会被直接跳过且不报错。")
        print("         需要暴露该问题请加 --fail-on-data-loss")
    print("=" * 60)

    # 1) 建库建表（幂等）
    for ddl in DDL_STATEMENTS:
        try:
            spark.sql(ddl)
            print(f"[DDL OK] {ddl.splitlines()[0][:70]}")
        except Exception as e:
            print(f"[DDL WARN] {e}")

    # 2) 读取 Kafka 实时流
    raw = (spark.readStream
           .format("kafka")
           .option("kafka.bootstrap.servers", args.bootstrap)
           .option("subscribe", args.topic)
           .option("startingOffsets", args.starting_offsets)
           .option("failOnDataLoss", "true" if args.fail_on_data_loss else "false")
           .load()
           .selectExpr("CAST(value AS STRING) AS raw_json"))

    parsed = (raw
              .withColumn("s", F.from_json(F.col("raw_json"), SENSOR_SCHEMA))
              .filter(F.col("s").isNotNull())
              .select(
                  F.col("s.sensor_id"), F.col("s.asset_id"), F.col("s.pipe_type"),
                  F.col("s.metrics"), F.col("s.status"),
                  F.col("s.alert_code"), F.col("s.alert_desc"), F.col("s.timestamp"),
                  F.to_timestamp("s.timestamp", "yyyy-MM-dd HH:mm:ss").alias("ts"),
              )
              .filter(F.col("ts").isNotNull())
              .withColumn("dt", F.date_format("ts", "yyyy-MM-dd"))
              # 缺失指标保持 NULL：avg() 自动忽略 NULL，避免用 0 拉低均值
              .withColumn("pressure", F.get_json_object("metrics", "$.pressure").cast("double"))
              .withColumn("flow", F.get_json_object("metrics", "$.flow").cast("double"))
              .withColumn("temperature", F.get_json_object("metrics", "$.temperature").cast("double"))
              .withColumn("gas_concentration", F.get_json_object("metrics", "$.gas_concentration").cast("double"))
              .withColumn("level", F.get_json_object("metrics", "$.level").cast("double"))
              .withColumn("vibration", F.get_json_object("metrics", "$.vibration").cast("double")))

    # 3) 滑动窗口聚合（每 pipe_type 的均值/异常数/健康度）
    agg = (parsed
           .groupBy(F.window(F.col("ts"), f"{args.window} seconds", f"{args.slide} seconds"),
                    F.col("pipe_type"))
           .agg(
               F.round(F.avg("pressure"), 3).alias("avg_pressure"),
               F.round(F.avg("temperature"), 3).alias("avg_temperature"),
               F.round(F.avg("flow"), 3).alias("avg_flow"),
               F.sum(F.when(F.col("status") != "normal", 1).otherwise(0)).alias("anomaly_count"),
               F.count("*").alias("sensor_count"),
           )
           .withColumn("window_start", F.date_format("window.start", "yyyy-MM-dd HH:mm:ss"))
           .withColumn("window_end", F.date_format("window.end", "yyyy-MM-dd HH:mm:ss"))
           .withColumn("dt", F.date_format("window.start", "yyyy-MM-dd"))
           .withColumn("health_score", F.round(100 - F.col("anomaly_count") * 100.0 / F.col("sensor_count"), 1))
           .drop("window"))

    # 4) 批写 Hive：ODS + DWD
    def sink_ods_dwd(batch_df, epoch_id):
        if batch_df.rdd.isEmpty():
            return
        batch_df.select(*ODS_COLS, "dt").write.mode("append").partitionBy("dt").saveAsTable("ods.sensor_raw")
        batch_df.select(*DWD_COLS, "dt").write.mode("append").partitionBy("dt").saveAsTable("dwd.sensor_detail")
        print(f"[{epoch_id}] ODS/DWD 写入完成: {batch_df.count()} 条")

    # 5) 批写 Hive：DWS（追加）+ ADS（覆盖快照）
    def sink_dws_ads(batch_df, epoch_id):
        if batch_df.rdd.isEmpty():
            return
        batch_df.select(*DWS_COLS, "dt").write.mode("append").partitionBy("dt").saveAsTable("dws.sensor_minute_stats")
        ads = (batch_df.groupBy("pipe_type").agg(
            F.round(F.avg("avg_pressure"), 3).alias("avg_pressure"),
            F.round(F.avg("avg_temperature"), 3).alias("avg_temperature"),
            F.round(F.avg("avg_flow"), 3).alias("avg_flow"),
            F.sum("anomaly_count").alias("anomaly_count"),
            F.sum("sensor_count").alias("sensor_count"),
            F.round(F.avg("health_score"), 1).alias("health_score"),
            F.max("window_start").alias("window_start"),
        ).select(*ADS_COLS))
        ads.write.mode("overwrite").saveAsTable("ads.sensor_realtime")
        print(f"[{epoch_id}] DWS/ADS 写入完成: {batch_df.count()} 行")

    q1 = (parsed.writeStream
          .foreachBatch(sink_ods_dwd)
          .option("checkpointLocation", ckpt_ods_dwd)
          .outputMode("append")
          .trigger(processingTime="5 seconds")
          .start())

    q2 = (agg.writeStream
          .foreachBatch(sink_dws_ads)
          .option("checkpointLocation", ckpt_dws_ads)
          .outputMode("update")
          .trigger(processingTime="5 seconds")
          .start())

    print("Streaming 已启动，等待数据... (Ctrl+C 退出)")
    try:
        spark.streams.awaitAnyTermination()
    except KeyboardInterrupt:
        print("正在停止...")
        q1.stop()
        q2.stop()
        spark.stop()


if __name__ == "__main__":
    main()
