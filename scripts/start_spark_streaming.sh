#!/bin/bash
###############################################################################
# 城市管网资产数字化台账 - Spark Streaming 启动/停止/状态脚本
#
# 用法: bash scripts/start_spark_streaming.sh [start|stop|status]
#
# 自动识别运行环境：
#   - 检测到 spark-master 容器  → 在容器内提交（Kafka=kafka:9093, Hive=hive-metastore:9083）
#   - 未检测到容器               → 本机 spark-submit（Kafka=localhost:9092, Hive=localhost:9083）
# 依赖: spark-sql-kafka-0-10 连接器（--packages 自动下载）
###############################################################################

set -u

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGS_DIR="$PROJECT_ROOT/logs"
PID_FILE="$LOGS_DIR/spark_streaming.pid"
LOG_FILE="$LOGS_DIR/spark_streaming.log"
APP="spark_streaming.py"
PACKAGES="org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0"
EXTRA_ARGS="${SPARK_STREAM_ARGS:-}"

ACTION="${1:-status}"
case "$ACTION" in
  start|stop|status) ;;
  *) echo "用法: $0 [start|stop|status]"; exit 1 ;;
esac

mkdir -p "$LOGS_DIR"

# 检测 spark-master 容器
IN_DOCKER=0
if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'spark-master'; then
  IN_DOCKER=1
fi

do_start() {
  if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE" 2>/dev/null)" 2>/dev/null; then
    echo "Spark Streaming 已在运行 (PID $(cat "$PID_FILE"))"
    exit 0
  fi
  echo "启动 Spark Streaming ..."
  if [ "$IN_DOCKER" = "1" ]; then
    echo "[模式] spark-master 容器"
    REMOTE_PID=$(docker exec spark-master sh -c "nohup /opt/spark/bin/spark-submit \
      --packages $PACKAGES \
      --master spark://spark-master:7077 \
      --conf spark.sql.warehouse.dir=hdfs://hadoop-namenode:8020/user/hive/warehouse \
      --conf spark.hadoop.hive.metastore.uris=thrift://hive-metastore:9083 \
      --conf spark.hadoop.fs.defaultFS=hdfs://hadoop-namenode:8020 \
      /opt/spark/work-dir/src/python/$APP \
      --bootstrap kafka:9093 \
      --metastore-uris thrift://hive-metastore:9083 \
      --warehouse-dir hdfs://hadoop-namenode:8020/user/hive/warehouse \
      --fs-default hdfs://hadoop-namenode:8020 \
      $EXTRA_ARGS > /tmp/spark_streaming.log 2>&1 & echo \\\$!" 2>&1)
    echo "$REMOTE_PID" > "$PID_FILE"
  else
    echo "[模式] 宿主机 spark-submit"
    nohup spark-submit \
      --packages $PACKAGES \
      --conf spark.hadoop.hive.metastore.uris=thrift://localhost:9083 \
      "$PROJECT_ROOT/src/python/$APP" \
      --bootstrap localhost:9092 \
      --metastore-uris thrift://localhost:9083 \
      $EXTRA_ARGS >> "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
  fi
  echo "已提交，PID 记录于 $PID_FILE"
  echo "日志: $LOG_FILE"
  sleep 3
  do_status
}

do_stop() {
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if [ "$IN_DOCKER" = "1" ]; then
      docker exec spark-master kill "$PID" 2>/dev/null || true
      docker exec spark-master pkill -f "$APP" 2>/dev/null || true
    else
      kill "$PID" 2>/dev/null || true
      pkill -f "$APP" 2>/dev/null || true
    fi
    rm -f "$PID_FILE"
  else
    if [ "$IN_DOCKER" = "1" ]; then
      docker exec spark-master pkill -f "$APP" 2>/dev/null || true
    else
      pkill -f "$APP" 2>/dev/null || true
    fi
  fi
  echo "Spark Streaming 已停止"
}

do_status() {
  local alive=0
  if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$PID" ]; then
      if [ "$IN_DOCKER" = "1" ]; then
        docker exec spark-master sh -c "kill -0 $PID 2>/dev/null" >/dev/null 2>&1 && alive=1
      else
        kill -0 "$PID" 2>/dev/null && alive=1
      fi
    fi
  fi
  if [ "$alive" = "1" ]; then
    echo "Spark Streaming: 运行中 (PID $(cat "$PID_FILE"))"
  else
    if [ "$IN_DOCKER" = "1" ]; then
      docker exec spark-master sh -c "pgrep -f $APP" >/dev/null 2>&1 && echo "Spark Streaming: 运行中（容器内）" || echo "Spark Streaming: 已停止"
    else
      pgrep -f "$APP" >/dev/null 2>&1 && echo "Spark Streaming: 运行中" || echo "Spark Streaming: 已停止"
    fi
  fi
}

case "$ACTION" in
  start)  do_start ;;
  stop)   do_stop ;;
  status) do_status ;;
esac
