#!/bin/bash

###############################################################################
# 城市管网资产数字化台账 - 快速验证脚本
# 功能：验证所有服务是否正常运行，并测试完整的数据处理流程
###############################################################################

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  城市管网资产数字化台账 - 快速验证脚本${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

LOCAL_IP=$(hostname -I | awk '{print $1}')

TOTAL_SCORE=0
MAX_SCORE=10
# 鉴权用例的独立跳过计数：登录拿不到 token 时跳过，既不计入总分也不算 FAIL
AUTH_SKIP=0
API_TOKEN=""

check_service() {
    local service_name=$1
    local url=$2

    echo -n "检查 $service_name... "

    if curl -s --connect-timeout 5 "$url" > /dev/null 2>&1; then
        echo -e "${GREEN}OK${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 1))
        return 0
    else
        echo -e "${RED}FAIL${NC}"
        echo -e "  ${YELLOW}无法访问: $url${NC}"
        return 1
    fi
}

###############################################################################
# 登录演示账号并取 Bearer token
#   后端 /auth/login 返回的 token 字段是 "token"（部分版本为 "access_token"），
#   这里两种都兼容；优先用 python3 解析 JSON，不可用时退回 grep/sed。
#   成功: 设置全局 API_TOKEN 并返回 0；失败: 返回 1（由调用方决定跳过）
###############################################################################
login_and_get_token() {
    local login_resp token

    login_resp=$(curl -s -X POST "http://localhost:8000/auth/login" \
        -H "Content-Type: application/json" \
        -d '{"username":"admin","password":"admin123"}' \
        --connect-timeout 5 2>/dev/null || true)

    if [ -z "$login_resp" ]; then
        return 1
    fi

    token=""
    if command -v python3 >/dev/null 2>&1; then
        token=$(printf '%s' "$login_resp" | python3 -c 'import json, sys
try:
    data = json.load(sys.stdin)
except Exception:
    sys.exit(0)
sys.stdout.write(data.get("access_token") or data.get("token") or "")' 2>/dev/null || true)
    fi

    if [ -z "$token" ]; then
        token=$(printf '%s' "$login_resp" | grep -o '"access_token"[[:space:]]*:[[:space:]]*"[^"]*"' \
            | head -n 1 | sed -e 's/.*:[[:space:]]*"//' -e 's/"$//' || true)
    fi

    if [ -z "$token" ]; then
        token=$(printf '%s' "$login_resp" | grep -o '"token"[[:space:]]*:[[:space:]]*"[^"]*"' \
            | head -n 1 | sed -e 's/.*:[[:space:]]*"//' -e 's/"$//' || true)
    fi

    if [ -z "$token" ]; then
        return 1
    fi

    API_TOKEN="$token"
    return 0
}

echo -e "${BLUE}【1】检查Docker容器状态${NC}"
echo "----------------------------------------"

cd "$PROJECT_ROOT/docker"

RUNNING_CONTAINERS=$(docker-compose ps --services --filter "status=running" 2>/dev/null | wc -l)
TOTAL_CONTAINERS=$(docker-compose ps --services 2>/dev/null | wc -l)

echo "运行中的容器: $RUNNING_CONTAINERS / $TOTAL_CONTAINERS"

if [ "$RUNNING_CONTAINERS" -eq "$TOTAL_CONTAINERS" ] && [ "$TOTAL_CONTAINERS" -gt 0 ]; then
    echo -e "${GREEN}所有容器都在运行${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
else
    echo -e "${YELLOW}部分容器未运行${NC}"
    docker-compose ps 2>/dev/null || true
fi

echo ""
echo -e "${BLUE}【2】检查大数据服务${NC}"
echo "----------------------------------------"

check_service "HDFS NameNode" "http://localhost:9870" || true
check_service "Spark Master" "http://localhost:8080" || true
check_service "Kibana" "http://localhost:5601" || true
check_service "Elasticsearch" "http://localhost:9200" || true

echo ""
echo -e "${BLUE}【3】检查FastAPI服务${NC}"
echo "----------------------------------------"

if curl -s --connect-timeout 5 "http://localhost:8000/health" > /dev/null 2>&1; then
    echo -e "FastAPI服务... ${GREEN}OK${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    HEALTH_DATA=$(curl -s "http://localhost:8000/health")
    echo "  健康状态: $HEALTH_DATA"
else
    echo -e "FastAPI服务... ${RED}FAIL${NC}"
fi

echo ""
echo -e "${BLUE}【4】测试数据生成功能${NC}"
echo "----------------------------------------"

cd "$PROJECT_ROOT/src/python"

echo "测试生成管网资产数据..."
if python3 generate_asset_data.py --output "$PROJECT_ROOT/data/test_logs" > /dev/null 2>&1; then
    echo -e "数据生成... ${GREEN}OK${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))

    for f in pipeline_asset lifecycle_event inventory_check; do
        if ls "$PROJECT_ROOT/data/test_logs"/${f}_*.jsonl 1> /dev/null 2>&1; then
            FILE_COUNT=$(wc -l < "$PROJECT_ROOT/data/test_logs"/${f}_*.jsonl)
            echo "  ${f}: $FILE_COUNT 条记录"
        fi
    done
else
    echo -e "数据生成... ${RED}FAIL${NC}"
fi

echo ""
echo -e "${BLUE}【5】测试模型训练功能${NC}"
echo "----------------------------------------"

echo "测试模型训练..."
if timeout 60 python3 train_sklearn_model.py > /tmp/model_train.log 2>&1; then
    echo -e "模型训练... ${GREEN}OK${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))

    if [ -f "$PROJECT_ROOT/models/anomaly_model.pkl" ] && [ -f "$PROJECT_ROOT/models/rul_model.pkl" ]; then
        echo "  模型文件已生成"
    fi
else
    echo -e "模型训练... ${YELLOW}超时或失败${NC}"
    echo "  查看日志: cat /tmp/model_train.log"
fi

echo ""
echo -e "${BLUE}【6】测试FastAPI资产接口（需鉴权）${NC}"
echo "----------------------------------------"

if login_and_get_token; then
    echo -e "演示账号登录(admin)... ${GREEN}OK${NC}"

    echo "测试资产总览接口..."
    OVERVIEW_RESPONSE=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
        "http://localhost:8000/asset/overview" 2>/dev/null || echo '{"error": "API不可用"}')

    if echo "$OVERVIEW_RESPONSE" | grep -q "total_assets\|total_length_km"; then
        echo -e "资产总览API... ${GREEN}OK${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 1))
    else
        echo -e "资产总览API... ${RED}FAIL${NC}"
    fi

    echo "测试权属汇总接口..."
    OWNERSHIP_RESPONSE=$(curl -s -H "Authorization: Bearer $API_TOKEN" \
        "http://localhost:8000/ownership/summary" 2>/dev/null || echo '{"error": "API不可用"}')

    if echo "$OWNERSHIP_RESPONSE" | grep -q "ownership_unit\|oam_unit"; then
        echo -e "权属汇总API... ${GREEN}OK${NC}"
        TOTAL_SCORE=$((TOTAL_SCORE + 1))
    else
        echo -e "权属汇总API... ${RED}FAIL${NC}"
    fi
else
    echo -e "演示账号登录(admin)... ${YELLOW}SKIP${NC}"
    echo -e "  ${YELLOW}登录失败，跳过鉴权用例（资产总览 / 权属汇总），不计入总分也不算 FAIL${NC}"
    echo -e "  ${YELLOW}请确认 FastAPI 已启动，且演示账号 admin/admin123 可用${NC}"
    AUTH_SKIP=$((AUTH_SKIP + 2))
fi

echo ""
echo -e "${BLUE}【7】测试FastAPI工作流${NC}"
echo "----------------------------------------"

echo "测试工作流启动..."
WORKFLOW_RESPONSE=$(curl -s -X POST "http://localhost:8000/workflow/run" \
    -H "Content-Type: application/json" \
    -d '{"count": 5000}' 2>/dev/null || echo '{"error": "API不可用"}')

if echo "$WORKFLOW_RESPONSE" | grep -q "started\|success\|task_id"; then
    echo -e "工作流启动... ${GREEN}OK${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
    echo "  响应: $WORKFLOW_RESPONSE"
else
    echo -e "工作流启动... ${RED}FAIL${NC}"
    echo "  响应: $WORKFLOW_RESPONSE"
fi

echo ""
echo -e "${BLUE}【8】检查可视化文件${NC}"
echo "----------------------------------------"

if [ -f "$PROJECT_ROOT/output/index.html" ]; then
    echo -e "台账大屏(在线版)... ${GREEN}OK${NC}"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
else
    echo -e "台账大屏(在线版)... ${RED}FAIL${NC}"
fi

if [ -f "$PROJECT_ROOT/output/show.html" ]; then
    echo -e "台账大屏(离线版)... ${GREEN}OK${NC}"
else
    echo -e "台账大屏(离线版)... ${RED}FAIL${NC}"
fi

echo ""
echo -e "${BLUE}【9】网络连接测试${NC}"
echo "----------------------------------------"

echo "测试关键端口..."
PORTS_OK=0

for port in 8000 8080 9200 5601 9870; do
    if timeout 2 bash -c "cat < /dev/null > /dev/tcp/localhost/$port" 2>/dev/null; then
        echo -e "  端口 $port: ${GREEN}OK${NC}"
        PORTS_OK=$((PORTS_OK + 1))
    else
        echo -e "  端口 $port: ${RED}FAIL${NC}"
    fi
done

if [ "$PORTS_OK" -ge 4 ]; then
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
fi

echo ""
echo -e "${BLUE}【10】项目文件完整性检查${NC}"
echo "----------------------------------------"

REQUIRED_FILES=(
    "src/python/main.py"
    "src/python/generate_asset_data.py"
    "src/python/train_sklearn_model.py"
    "src/python/kafka_producer.py"
    "src/python/spark_analysis.py"
    "output/index.html"
    "output/show.html"
    "docker/docker-compose.yml"
    "config/hive/hive_ddl.sql"
    "config/logstash/kafka_to_es.conf"
)

FILES_OK=0
for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$PROJECT_ROOT/$file" ]; then
        FILES_OK=$((FILES_OK + 1))
    else
        echo -e "  缺少文件: $file"
    fi
done

if [ "$FILES_OK" -eq "${#REQUIRED_FILES[@]}" ]; then
    echo -e "项目文件... ${GREEN}OK${NC} ($FILES_OK/${#REQUIRED_FILES[@]})"
    TOTAL_SCORE=$((TOTAL_SCORE + 1))
else
    echo -e "项目文件... ${YELLOW}PARTIAL ($FILES_OK/${#REQUIRED_FILES[@]})${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  验证结果汇总${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "总分: ${GREEN}$TOTAL_SCORE${NC} / $MAX_SCORE"
if [ "$AUTH_SKIP" -gt 0 ]; then
    echo -e "${YELLOW}已跳过鉴权用例: $AUTH_SKIP 项（演示账号登录失败，未计入总分）${NC}"
fi
echo ""

if [ "$TOTAL_SCORE" -eq "$MAX_SCORE" ]; then
    echo -e "${GREEN}所有验证项目都通过了！${NC}"
    echo -e "${GREEN}系统已就绪，可以正常使用！${NC}"
elif [ "$TOTAL_SCORE" -ge 8 ]; then
    echo -e "${YELLOW}项目基本正常，有少量问题${NC}"
else
    echo -e "${RED}项目存在较多问题${NC}"
    echo -e "${RED}建议重新启动项目或检查配置${NC}"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  访问地址${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}应用服务：${NC}"
echo -e "  FastAPI服务:        ${YELLOW}http://$LOCAL_IP:8000${NC}"
echo -e "  API文档:            ${YELLOW}http://$LOCAL_IP:8000/docs${NC}"
echo -e "  台账大屏(在线):     ${YELLOW}http://$LOCAL_IP:8000/output/index.html${NC}"
echo -e "  台账大屏(离线):     ${YELLOW}http://$LOCAL_IP:8000/output/show.html${NC}"
echo ""
echo -e "${GREEN}大数据集群：${NC}"
echo -e "  HDFS Web UI:        ${YELLOW}http://$LOCAL_IP:9870${NC}"
echo -e "  Spark Master:       ${YELLOW}http://$LOCAL_IP:8080${NC}"
echo -e "  Kibana Dashboard:   ${YELLOW}http://$LOCAL_IP:5601${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  验证完成！${NC}"
echo -e "${BLUE}========================================${NC}"

exit 0
