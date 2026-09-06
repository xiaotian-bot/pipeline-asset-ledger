#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市管网资产数字化台账 - Spark ETL分析流程

实现 ODS → DWD → DWS → ADS 的完整数据清洗和聚合流程
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, get_json_object, count, sum as spark_sum, avg, max as spark_max, min as spark_min, when, round as spark_round, row_number, concat, lit
from pyspark.sql.window import Window
from datetime import datetime, timedelta
import sys


def create_spark_session(app_name="城市管网资产台账分析"):
    return (SparkSession.builder
            .appName(app_name)
            .enableHiveSupport()
            .getOrCreate())


def clean_asset_data(spark, date):
    """ODS → DWD: 管网资产数据清洗"""
    print(f"[步骤1] 管网资产数据清洗 ODS → DWD ({date})")

    raw_df = spark.sql(f"""
        SELECT raw_json FROM ods.pipeline_asset_raw WHERE dt = '{date}'
    """)

    fields = [
        ("asset_id", "STRING"), ("pipeline_type_code", "STRING"),
        ("pipeline_type_name", "STRING"), ("diameter", "STRING"),
        ("material", "STRING"), ("install_year", "INT"),
        ("service_years", "INT"), ("design_life", "INT"),
        ("remaining_life", "INT"), ("region_code", "STRING"),
        ("region_name", "STRING"), ("longitude", "DOUBLE"),
        ("latitude", "DOUBLE"), ("segment_length_m", "DOUBLE"),
        ("burial_depth_m", "DOUBLE"), ("pressure_level", "STRING"),
        ("asset_status", "STRING"), ("original_value_yuan", "DOUBLE"),
        ("net_value_yuan", "DOUBLE"), ("depreciation_rate", "DOUBLE"),
        ("risk_score", "INT"), ("ownership_unit", "STRING"),
        ("oam_unit", "STRING"), ("supervision_unit", "STRING"),
        ("last_inspection_date", "STRING"), ("qr_code", "STRING"),
    ]

    select_exprs = [f"get_json_object(raw_json, '$.{f[0]}') as {f[0]}" for f in fields]
    select_exprs.append(f"'{date}' as dt")

    result_df = raw_df.selectExpr(*select_exprs)

    result_df.write.mode("overwrite").partitionBy("dt").saveAsTable("dwd.pipeline_asset_detail")
    asset_count = result_df.count()
    print(f"  资产明细写入完成: {asset_count} 条")
    return asset_count


def clean_lifecycle_events(spark, date):
    """ODS → DWD: 生命周期事件清洗"""
    print(f"[步骤2] 生命周期事件清洗 ODS → DWD ({date})")

    raw_df = spark.sql(f"""
        SELECT raw_json FROM ods.lifecycle_event_raw WHERE dt = '{date}'
    """)

    fields = [
        ("event_id", "STRING"), ("asset_id", "STRING"),
        ("pipeline_type_name", "STRING"), ("event_type_code", "STRING"),
        ("event_type_name", "STRING"), ("event_date", "STRING"),
        ("responsible_unit", "STRING"), ("cost_yuan", "DOUBLE"),
        ("description", "STRING"), ("operator", "STRING"),
    ]

    select_exprs = [f"get_json_object(raw_json, '$.{f[0]}') as {f[0]}" for f in fields]
    select_exprs.append(f"'{date}' as dt")

    result_df = raw_df.selectExpr(*select_exprs)
    result_df.write.mode("overwrite").partitionBy("dt").saveAsTable("dwd.lifecycle_event_detail")
    event_count = result_df.count()
    print(f"  生命周期事件写入完成: {event_count} 条")
    return event_count


def clean_inventory_checks(spark, date):
    """ODS → DWD: 盘点记录清洗"""
    print(f"[步骤3] 盘点记录清洗 ODS → DWD ({date})")

    raw_df = spark.sql(f"""
        SELECT raw_json FROM ods.inventory_check_raw WHERE dt = '{date}'
    """)

    fields = [
        ("check_id", "STRING"), ("batch_id", "STRING"),
        ("asset_id", "STRING"), ("pipeline_type_name", "STRING"),
        ("region_name", "STRING"), ("check_method", "STRING"),
        ("check_date", "STRING"), ("diff_status", "STRING"),
        ("diff_description", "STRING"), ("checker", "STRING"),
        ("checker_unit", "STRING"),
    ]

    select_exprs = [f"get_json_object(raw_json, '$.{f[0]}') as {f[0]}" for f in fields]
    select_exprs.append(f"'{date}' as dt")

    result_df = raw_df.selectExpr(*select_exprs)
    result_df.write.mode("overwrite").partitionBy("dt").saveAsTable("dwd.inventory_check_detail")
    check_count = result_df.count()
    print(f"  盘点记录写入完成: {check_count} 条")
    return check_count


def build_asset_category_summary(spark, date):
    """DWD → DWS: 资产分类统计"""
    print(f"[步骤4] 资产分类统计 DWD → DWS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE dws.asset_category_summary PARTITION (dt = '{date}')
        SELECT
            pipeline_type_code,
            pipeline_type_name,
            diameter,
            material,
            CONCAT(CAST((install_year DIV 10) * 10 AS STRING), 's') AS decade,
            region_name,
            COUNT(*) AS asset_count,
            SUM(segment_length_m) AS total_length_m,
            SUM(original_value_yuan) AS total_original_value,
            SUM(net_value_yuan) AS total_net_value,
            AVG(risk_score) AS avg_risk_score,
            SUM(CASE WHEN asset_status = '在用' THEN 1 ELSE 0 END) AS in_service_count,
            SUM(CASE WHEN asset_status = '报废' THEN 1 ELSE 0 END) AS retired_count
        FROM dwd.pipeline_asset_detail
        WHERE dt = '{date}'
        GROUP BY pipeline_type_code, pipeline_type_name, diameter, material,
                 CONCAT(CAST((install_year DIV 10) * 10 AS STRING), 's'), region_name
    """)
    print("  资产分类统计完成")


def build_lifecycle_summary(spark, date):
    """DWD → DWS: 生命周期事件统计"""
    print(f"[步骤5] 生命周期事件统计 DWD → DWS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE dws.asset_lifecycle_summary PARTITION (dt = '{date}')
        SELECT
            pipeline_type_name,
            event_type_code,
            event_type_name,
            COUNT(*) AS event_count,
            SUM(cost_yuan) AS total_cost,
            AVG(cost_yuan) AS avg_cost,
            MAX(cost_yuan) AS max_cost,
            MIN(cost_yuan) AS min_cost
        FROM dwd.lifecycle_event_detail
        WHERE dt = '{date}'
        GROUP BY pipeline_type_name, event_type_code, event_type_name
    """)
    print("  生命周期事件统计完成")


def build_risk_assessment(spark, date):
    """DWD → DWS: 资产风险评估"""
    print(f"[步骤6] 资产风险评估 DWD → DWS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE dws.asset_risk_assessment PARTITION (dt = '{date}')
        SELECT
            asset_id,
            pipeline_type_code,
            pipeline_type_name,
            region_name,
            service_years,
            design_life,
            ROUND(CAST(service_years AS DOUBLE) / design_life, 4) AS aging_index,
            risk_score,
            CASE
                WHEN risk_score >= 80 THEN '极高'
                WHEN risk_score >= 60 THEN '高'
                WHEN risk_score >= 40 THEN '中'
                ELSE '低'
            END AS risk_level,
            remaining_life,
            material,
            ownership_unit,
            oam_unit
        FROM dwd.pipeline_asset_detail
        WHERE dt = '{date}'
    """)
    print("  资产风险评估完成")


def build_inventory_reconciliation(spark, date):
    """DWD → DWS: 盘点核对汇总"""
    print(f"[步骤7] 盘点核对汇总 DWD → DWS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE dws.inventory_reconciliation PARTITION (dt = '{date}')
        SELECT
            batch_id,
            pipeline_type_name,
            region_name,
            COUNT(*) AS total_checked,
            SUM(CASE WHEN diff_status = '一致' THEN 1 ELSE 0 END) AS match_count,
            SUM(CASE WHEN diff_status = '差异-缺失' THEN 1 ELSE 0 END) AS missing_count,
            SUM(CASE WHEN diff_status = '差异-多余' THEN 1 ELSE 0 END) AS extra_count,
            SUM(CASE WHEN diff_status = '差异-信息不符' THEN 1 ELSE 0 END) AS mismatch_count,
            ROUND(
                CAST(SUM(CASE WHEN diff_status != '一致' THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*),
                4
            ) AS diff_rate
        FROM dwd.inventory_check_detail
        WHERE dt = '{date}'
        GROUP BY batch_id, pipeline_type_name, region_name
    """)
    print("  盘点核对汇总完成")


def build_asset_overview(spark, date):
    """DWS → ADS: 资产全景总览"""
    print(f"[步骤8] 资产全景总览 DWS → ADS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE ads.asset_overview PARTITION (dt = '{date}')
        SELECT
            SUM(asset_count) AS total_assets,
            ROUND(SUM(total_length_m) / 1000, 2) AS total_length_km,
            SUM(total_original_value) AS total_original_value,
            SUM(total_net_value) AS total_net_value,
            SUM(in_service_count) AS in_service_count,
            SUM(retired_count) AS retired_count,
            0 AS pending_inspection_count,
            0 AS suspended_count,
            ROUND(AVG(avg_risk_score), 2) AS avg_risk_score,
            SUM(CASE WHEN avg_risk_score >= 60 THEN asset_count ELSE 0 END) AS high_risk_count,
            0.0 AS inventory_diff_rate,
            SUM(CASE WHEN pipeline_type_code = 'WSP' THEN asset_count ELSE 0 END) AS water_supply_count,
            SUM(CASE WHEN pipeline_type_code = 'HTP' THEN asset_count ELSE 0 END) AS heating_count,
            SUM(CASE WHEN pipeline_type_code = 'GSP' THEN asset_count ELSE 0 END) AS gas_count,
            SUM(CASE WHEN pipeline_type_code = 'SWP' THEN asset_count ELSE 0 END) AS sewage_count,
            SUM(CASE WHEN pipeline_type_code = 'HZP' THEN asset_count ELSE 0 END) AS hazardous_count
        FROM dws.asset_category_summary
        WHERE dt = '{date}'
    """)
    print("  资产全景总览完成")


def build_asset_distribution(spark, date):
    """DWS → ADS: 资产分布统计（多维度）"""
    print(f"[步骤9] 资产分布统计 DWS → ADS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE ads.asset_distribution PARTITION (dt = '{date}')
        SELECT 'type' AS dimension_type, pipeline_type_name AS dimension_value,
               pipeline_type_name, SUM(asset_count), SUM(total_length_m),
               SUM(total_original_value), AVG(avg_risk_score)
        FROM dws.asset_category_summary WHERE dt = '{date}'
        GROUP BY pipeline_type_name
        UNION ALL
        SELECT 'region' AS dimension_type, region_name AS dimension_value,
               pipeline_type_name, SUM(asset_count), SUM(total_length_m),
               SUM(total_original_value), AVG(avg_risk_score)
        FROM dws.asset_category_summary WHERE dt = '{date}'
        GROUP BY region_name, pipeline_type_name
        UNION ALL
        SELECT 'material' AS dimension_type, material AS dimension_value,
               pipeline_type_name, SUM(asset_count), SUM(total_length_m),
               SUM(total_original_value), AVG(avg_risk_score)
        FROM dws.asset_category_summary WHERE dt = '{date}'
        GROUP BY material, pipeline_type_name
        UNION ALL
        SELECT 'decade' AS dimension_type, decade AS dimension_value,
               pipeline_type_name, SUM(asset_count), SUM(total_length_m),
               SUM(total_original_value), AVG(avg_risk_score)
        FROM dws.asset_category_summary WHERE dt = '{date}'
        GROUP BY decade, pipeline_type_name
    """)
    print("  资产分布统计完成")


def build_ownership_summary(spark, date):
    """DWS → ADS: 权属责任汇总"""
    print(f"[步骤10] 权属责任汇总 DWS → ADS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE ads.ownership_summary PARTITION (dt = '{date}')
        SELECT
            ownership_unit,
            oam_unit,
            supervision_unit,
            pipeline_type_name,
            region_name,
            COUNT(*) AS asset_count,
            SUM(segment_length_m) AS total_length_m,
            SUM(original_value_yuan) AS total_value,
            AVG(risk_score) AS avg_risk_score
        FROM dwd.pipeline_asset_detail
        WHERE dt = '{date}'
        GROUP BY ownership_unit, oam_unit, supervision_unit, pipeline_type_name, region_name
    """)
    print("  权属责任汇总完成")


def build_inventory_diff_report(spark, date):
    """DWS → ADS: 盘点差异报告"""
    print(f"[步骤11] 盘点差异报告 DWS → ADS ({date})")

    spark.sql(f"""
        INSERT OVERWRITE TABLE ads.inventory_diff_report PARTITION (dt = '{date}')
        SELECT
            batch_id,
            pipeline_type_name,
            region_name,
            check_method,
            COUNT(*) AS total_checked,
            SUM(CASE WHEN diff_status != '一致' THEN 1 ELSE 0 END) AS diff_count,
            ROUND(CAST(SUM(CASE WHEN diff_status != '一致' THEN 1 ELSE 0 END) AS DOUBLE) / COUNT(*), 4) AS diff_rate,
            SUM(CASE WHEN diff_status = '差异-缺失' THEN 1 ELSE 0 END) AS missing_count,
            SUM(CASE WHEN diff_status = '差异-多余' THEN 1 ELSE 0 END) AS extra_count,
            SUM(CASE WHEN diff_status = '差异-信息不符' THEN 1 ELSE 0 END) AS mismatch_count
        FROM dwd.inventory_check_detail
        WHERE dt = '{date}'
        GROUP BY batch_id, pipeline_type_name, region_name, check_method
    """)
    print("  盘点差异报告完成")


def main():
    date = sys.argv[1] if len(sys.argv) > 1 else (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    print(f"=" * 60)
    print(f"城市管网资产台账分析 - 执行日期: {date}")
    print(f"=" * 60)

    spark = create_spark_session()

    try:
        clean_asset_data(spark, date)
        clean_lifecycle_events(spark, date)
        clean_inventory_checks(spark, date)

        build_asset_category_summary(spark, date)
        build_lifecycle_summary(spark, date)
        build_risk_assessment(spark, date)
        build_inventory_reconciliation(spark, date)

        build_asset_overview(spark, date)
        build_asset_distribution(spark, date)
        build_ownership_summary(spark, date)
        build_inventory_diff_report(spark, date)

        print(f"\n{'=' * 60}")
        print(f"全部分析步骤完成!")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"分析失败: {e}")
        raise
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
