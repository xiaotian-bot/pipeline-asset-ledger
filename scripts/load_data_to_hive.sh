#!/bin/bash

# ==============================================================================
# 工业设备数据加载脚本 - 加载JSON日志到Hive ODS层
# 由于ODS表使用JsonSerDe，可直接上传原始JSON文件，无需字段转换
# ==============================================================================

DATE=$1
if [ -z "$DATE" ]; then
    DATE=$(date +%Y-%m-%d)
fi

LOG_FILE="data/logs/device_sensor_${DATE//-/}.log"

echo "======================================"
echo "  工业设备数据加载脚本 - 加载到Hive ODS层"
echo "  日期: $DATE"
echo "  日志文件: $LOG_FILE"
echo "======================================"

# 检查日志文件是否存在
if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    echo "   请先执行: python3 src/python/generate_logs.py --count 50000 --output data/logs"
    exit 1
fi

# 统计日志行数
LINE_COUNT=$(wc -l < "$LOG_FILE")
echo "  日志记录数: $LINE_COUNT 条"
echo ""

echo "[1/4] 创建HDFS分区目录..."
cd ~/dataBase/docker
sudo docker exec hadoop-namenode hdfs dfs -mkdir -p /user/hive/warehouse/ods.db/device_sensor_raw/dt=$DATE

echo "[2/4] 上传原始JSON日志到HDFS..."
# 拷贝到namenode容器临时目录
sudo docker cp ~/dataBase/$LOG_FILE hadoop-namenode:/tmp/device_sensor_${DATE//-/}.log
# 上传到HDFS对应分区目录（-f 覆盖已存在文件）
sudo docker exec hadoop-namenode hdfs dfs -put -f /tmp/device_sensor_${DATE//-/}.log /user/hive/warehouse/ods.db/device_sensor_raw/dt=$DATE/

echo "[3/4] 修复Hive分区..."
sudo docker exec hive-server hive -e "USE ods; ALTER TABLE device_sensor_raw ADD IF NOT EXISTS PARTITION (dt='$DATE');"

echo "[4/4] 验证数据加载..."
echo ""
echo "======================================"
echo "  数据加载验证"
echo "======================================"
sudo docker exec hive-server hive -e "
USE ods;
SELECT COUNT(*) AS total_records FROM device_sensor_raw WHERE dt='$DATE';
SELECT get_json_object(raw_json, '$.device_type') AS device_type, COUNT(*) AS cnt FROM device_sensor_raw WHERE dt='$DATE' GROUP BY get_json_object(raw_json, '$.device_type') ORDER BY cnt DESC;
SELECT CAST(get_json_object(raw_json, '$.status_code') AS INT) AS status_code, COUNT(*) AS cnt FROM device_sensor_raw WHERE dt='$DATE' GROUP BY CAST(get_json_object(raw_json, '$.status_code') AS INT) ORDER BY status_code;
"

echo ""
echo "======================================"
echo "  ✅ 数据加载完成！"
echo "  下一步可执行ETL: bash scripts/etl_daily.sh $DATE"
echo "======================================"
