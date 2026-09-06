#!/bin/bash

# ==============================================================================
# 工业设备数据ETL脚本 - ODS → DWD → DWS → ADS 全链路处理
# 使用纯Hive SQL实现，不依赖Spark
# ==============================================================================

DATE=$1
if [ -z "$DATE" ]; then
    DATE=$(date -d "-1 day" +%Y-%m-%d)
fi

echo "======================================"
echo "  工业设备数据ETL脚本"
echo "  处理日期: $DATE"
echo "======================================"

# ==============================================================================
# 步骤1：ODS → DWD 清洗转换（展开params MAP为独立字段）
# ==============================================================================
echo ""
echo "[1/4] 步骤1: ODS → DWD 数据清洗..."
sudo docker exec hive-server hive -e "
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.dynamic.partition=true;

USE dwd;

INSERT OVERWRITE TABLE dwd.device_sensor_detail PARTITION (dt='$DATE')
SELECT
    get_json_object(raw_json, '$.device_id') AS device_id,
    get_json_object(raw_json, '$.device_type') AS device_type,
    get_json_object(raw_json, '$.workshop') AS workshop,
    get_json_object(raw_json, '$.event_timestamp') AS event_time,
    hour(from_unixtime(CAST(get_json_object(raw_json, '$.ts') AS BIGINT) div 1000)) AS hour,
    CAST(get_json_object(raw_json, '$.status_code') AS INT) AS status_code,
    get_json_object(raw_json, '$.status_name') AS status_name,
    CAST(get_json_object(raw_json, '$.params.spindle_speed') AS DOUBLE) AS spindle_speed,
    CAST(get_json_object(raw_json, '$.params.feed_rate') AS DOUBLE) AS feed_rate,
    CAST(get_json_object(raw_json, '$.params.spindle_load') AS DOUBLE) AS spindle_load,
    CAST(get_json_object(raw_json, '$.params.spindle_temp') AS DOUBLE) AS spindle_temp,
    CAST(get_json_object(raw_json, '$.params.hydraulic_pressure') AS DOUBLE) AS hydraulic_pressure,
    CAST(get_json_object(raw_json, '$.params.coolant_temp') AS DOUBLE) AS coolant_temp,
    CAST(get_json_object(raw_json, '$.params.vibration') AS DOUBLE) AS vibration,
    CAST(get_json_object(raw_json, '$.params.joint_current') AS DOUBLE) AS joint_current,
    CAST(get_json_object(raw_json, '$.params.joint_torque') AS DOUBLE) AS joint_torque,
    CAST(get_json_object(raw_json, '$.params.servo_temp') AS DOUBLE) AS servo_temp,
    CAST(get_json_object(raw_json, '$.params.reducer_temp') AS DOUBLE) AS reducer_temp,
    CAST(get_json_object(raw_json, '$.params.joint_vibration') AS DOUBLE) AS joint_vibration,
    CAST(get_json_object(raw_json, '$.params.cycle_count') AS INT) AS cycle_count,
    CAST(get_json_object(raw_json, '$.params.injection_pressure') AS DOUBLE) AS injection_pressure,
    CAST(get_json_object(raw_json, '$.params.holding_pressure') AS DOUBLE) AS holding_pressure,
    CAST(get_json_object(raw_json, '$.params.system_oil_pressure') AS DOUBLE) AS system_oil_pressure,
    CAST(get_json_object(raw_json, '$.params.barrel_temp') AS DOUBLE) AS barrel_temp,
    CAST(get_json_object(raw_json, '$.params.mold_temp') AS DOUBLE) AS mold_temp,
    CAST(get_json_object(raw_json, '$.params.hydraulic_oil_temp') AS DOUBLE) AS hydraulic_oil_temp,
    CAST(get_json_object(raw_json, '$.params.cycle_time') AS DOUBLE) AS cycle_time,
    CAST(get_json_object(raw_json, '$.params.discharge_pressure') AS DOUBLE) AS discharge_pressure,
    CAST(get_json_object(raw_json, '$.params.discharge_temp') AS DOUBLE) AS discharge_temp,
    CAST(get_json_object(raw_json, '$.params.lubricant_temp') AS DOUBLE) AS lubricant_temp,
    CAST(get_json_object(raw_json, '$.params.motor_current') AS DOUBLE) AS motor_current,
    CAST(get_json_object(raw_json, '$.params.power') AS DOUBLE) AS power,
    CAST(get_json_object(raw_json, '$.alarm_code') AS INT) AS alarm_code,
    get_json_object(raw_json, '$.alarm_desc') AS alarm_desc,
    CAST(get_json_object(raw_json, '$.health_score') AS INT) AS health_score,
    CASE WHEN CAST(get_json_object(raw_json, '$.alarm_code') AS INT) != 0
          OR CAST(get_json_object(raw_json, '$.status_code') AS INT) IN (4, 5)
         THEN 1 ELSE 0 END AS is_abnormal
FROM ods.device_sensor_raw
WHERE dt = '$DATE';
"

# ==============================================================================
# 步骤2：DWD → DWS 设备状态汇总
# ==============================================================================
echo ""
echo "[2/4] 步骤2: DWD → DWS 设备状态统计..."
sudo docker exec hive-server hive -e "
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.dynamic.partition=true;

USE dws;

INSERT OVERWRITE TABLE dws.device_status_summary PARTITION (dt='$DATE')
SELECT
    device_id,
    device_type,
    workshop,
    COUNT(*) AS total_records,
    SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) AS run_count,
    SUM(CASE WHEN status_code = 1 THEN 1 ELSE 0 END) AS idle_count,
    SUM(CASE WHEN status_code = 3 THEN 1 ELSE 0 END) AS warning_count,
    SUM(CASE WHEN status_code = 4 THEN 1 ELSE 0 END) AS fault_count,
    SUM(CASE WHEN status_code = 0 THEN 1 ELSE 0 END) AS offline_count,
    CASE WHEN COUNT(*) > 0
         THEN SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) / COUNT(*)
         ELSE 0 END AS run_rate,
    AVG(health_score) AS avg_health_score,
    MIN(health_score) AS min_health_score,
    SUM(CASE WHEN alarm_code != 0 THEN 1 ELSE 0 END) AS alarm_count
FROM dwd.device_sensor_detail
WHERE dt = '$DATE'
GROUP BY device_id, device_type, workshop;
"

# ==============================================================================
# 步骤3：DWD → DWS 设备KPI汇总（OEE/MTBF/MTTR）
# ==============================================================================
echo ""
echo "[3/4] 步骤3: DWD → DWS 设备KPI计算..."
sudo docker exec hive-server hive -e "
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.dynamic.partition=true;

USE dws;

INSERT OVERWRITE TABLE dws.device_kpi_summary PARTITION (dt='$DATE')
SELECT
    device_id,
    device_type,
    workshop,
    -- 运行时长(分钟) = 运行次数 × 5秒 / 60
    SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) * 5.0 / 60 AS run_time_min,
    -- 计划生产时长(分钟) = 总记录数 × 5秒 / 60
    COUNT(*) * 5.0 / 60 AS planned_time_min,
    -- 故障时长(分钟) = 故障+急停次数 × 5秒 / 60
    SUM(CASE WHEN status_code IN (4, 5) THEN 1 ELSE 0 END) * 5.0 / 60 AS fault_time_min,
    SUM(CASE WHEN status_code IN (4, 5) THEN 1 ELSE 0 END) AS fault_count,
    -- OEE = 时间稼动率 × 性能 × 良品率
    CASE WHEN COUNT(*) > 0
         THEN (SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) / COUNT(*)) *
              0.85 *
              0.95
         ELSE 0 END AS oee,
    -- 时间稼动率 = 运行时长 / 计划时长
    CASE WHEN COUNT(*) > 0
         THEN SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) / COUNT(*)
         ELSE 0 END AS availability,
    0.85 AS performance,
    0.95 AS quality_rate,
    -- MTBF = 运行时长(小时) / 故障次数
    CASE WHEN SUM(CASE WHEN status_code IN (4, 5) THEN 1 ELSE 0 END) > 0
         THEN (SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) * 5.0 / 3600) /
              SUM(CASE WHEN status_code IN (4, 5) THEN 1 ELSE 0 END)
         ELSE 0 END AS mtbf,
    -- MTTR = 故障时长(小时) / 故障次数
    CASE WHEN SUM(CASE WHEN status_code IN (4, 5) THEN 1 ELSE 0 END) > 0
         THEN (SUM(CASE WHEN status_code IN (4, 5) THEN 1 ELSE 0 END) * 5.0 / 3600) /
              SUM(CASE WHEN status_code IN (4, 5) THEN 1 ELSE 0 END)
         ELSE 0 END AS mttr,
    -- 设备稼动率
    CASE WHEN COUNT(*) > 0
         THEN SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) / COUNT(*)
         ELSE 0 END AS utilization_rate
FROM dwd.device_sensor_detail
WHERE dt = '$DATE'
GROUP BY device_id, device_type, workshop;
"

# ==============================================================================
# 步骤4：DWS → ADS 应用层数据（大屏展示）
# ==============================================================================
echo ""
echo "[4/4] 步骤4: DWS → ADS 大屏应用数据..."
sudo docker exec hive-server hive -e "
SET hive.exec.dynamic.partition.mode=nonstrict;
SET hive.exec.dynamic.partition=true;

USE ads;

-- 设备实时状态总览（使用子查询避免JOIN重复）
INSERT OVERWRITE TABLE ads.device_overview PARTITION (dt='$DATE')
SELECT
    dws.total_devices,
    dws.online_devices,
    dws.running_devices,
    dws.warning_devices,
    dws.fault_devices,
    dws.offline_devices,
    dws.online_rate,
    dwd.avg_health_score,
    dwd.total_alarms,
    dwd.unhandled_alarms,
    kpi.factory_oee,
    dwd.total_output
FROM (
    SELECT
        COUNT(DISTINCT device_id) AS total_devices,
        COUNT(DISTINCT CASE WHEN run_count > 0 THEN device_id END) AS running_devices,
        COUNT(DISTINCT CASE WHEN warning_count > 0 THEN device_id END) AS warning_devices,
        COUNT(DISTINCT CASE WHEN fault_count > 0 THEN device_id END) AS fault_devices,
        COUNT(DISTINCT CASE WHEN offline_count > 0 THEN device_id END) AS offline_devices,
        COUNT(DISTINCT CASE WHEN offline_count = 0 THEN device_id END) AS online_devices,
        CASE WHEN COUNT(DISTINCT device_id) > 0
             THEN COUNT(DISTINCT CASE WHEN offline_count = 0 THEN device_id END) / COUNT(DISTINCT device_id)
             ELSE 0 END AS online_rate
    FROM dws.device_status_summary
    WHERE dt = '$DATE'
) dws
CROSS JOIN (
    SELECT
        AVG(oee) AS factory_oee
    FROM dws.device_kpi_summary
    WHERE dt = '$DATE'
) kpi
CROSS JOIN (
    SELECT
        AVG(health_score) AS avg_health_score,
        SUM(CASE WHEN alarm_code != 0 THEN 1 ELSE 0 END) AS total_alarms,
        SUM(CASE WHEN alarm_code != 0 AND status_code IN (3,4) THEN 1 ELSE 0 END) AS unhandled_alarms,
        SUM(CASE WHEN status_code = 2 THEN 1 ELSE 0 END) AS total_output
    FROM dwd.device_sensor_detail
    WHERE dt = '$DATE'
) dwd;

-- 告警类型统计
INSERT OVERWRITE TABLE ads.alarm_type_stats PARTITION (dt='$DATE')
SELECT
    alarm_code,
    MAX(alarm_desc) AS alarm_desc,
    COUNT(*) AS alarm_count,
    MAX(device_type) AS device_type,
    CASE
        WHEN alarm_code IN (1004, 2002, 2004, 3002, 3004, 4001, 4005) THEN '高'
        WHEN alarm_code IN (1001, 1005, 2001, 2005, 3001, 4002) THEN '中'
        ELSE '低'
    END AS severity
FROM dwd.device_sensor_detail
WHERE dt = '$DATE' AND alarm_code != 0
GROUP BY alarm_code;

-- 设备健康度排名
INSERT OVERWRITE TABLE ads.device_health_ranking PARTITION (dt='$DATE')
SELECT
    device_id,
    device_type,
    workshop,
    CAST(avg_health_score AS INT) AS health_score,
    run_rate,
    alarm_count,
    -- 简单RUL预测：基于健康度映射
    CASE
        WHEN avg_health_score >= 80 THEN CAST(80 + (avg_health_score - 80) * 0.5 AS INT)
        WHEN avg_health_score >= 60 THEN CAST(40 + (avg_health_score - 60) * 2 AS INT)
        WHEN avg_health_score >= 40 THEN CAST(20 + (avg_health_score - 40) AS INT)
        ELSE CAST(avg_health_score / 2 AS INT)
    END AS rul_prediction,
    ROW_NUMBER() OVER (ORDER BY avg_health_score DESC) AS ranking
FROM dws.device_status_summary
WHERE dt = '$DATE';
"

echo ""
echo "======================================"
echo "  ✅ ETL处理完成！"
echo "  处理日期: $DATE"
echo "======================================"
echo ""
echo "数据验证查询:"
sudo docker exec hive-server hive -e "
USE ads;
SELECT '设备总览' AS module, total_devices, online_devices, running_devices, fault_devices, factory_oee FROM device_overview WHERE dt='$DATE';
SELECT '告警类型' AS module, alarm_desc, alarm_count, severity FROM alarm_type_stats WHERE dt='$DATE' ORDER BY alarm_count DESC LIMIT 5;
SELECT '健康度TOP5' AS module, device_id, health_score, rul_prediction, ranking FROM device_health_ranking WHERE dt='$DATE' ORDER BY ranking LIMIT 5;
"
