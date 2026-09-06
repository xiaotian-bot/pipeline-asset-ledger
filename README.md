# 城市管网资产数字化台账系统

面向供水 / 燃气 / 供暖 / 污水 / 危废 5 类管网的资产数字化管理**演示原型**：5000 余段资产台账 + 100 个传感器实时监测 + 预测性维护 + AI 智能助手 + 多渠道预警推送，一条完整的数据链路：

```
传感器模拟(100点位) → Kafka → Spark Streaming(滑动窗口聚合)
    → Hive 数仓(ODS/DWD/DWS/ADS) → FastAPI → ECharts 大屏
    → XGBoost/IsolationForest 预测 + SHAP 归因
    → DeepSeek Agent + RAG 诊断 → 微信/邮件推送 → 工单闭环
```

## 功能模块

- **资产台账**：5000 段资产、生命周期档案、GIS 地图、盘点、工单、五级角色权限
- **传感器监控**：100 个模拟传感器（5 类管网 × 6 类指标 × 4 种场景），实时大屏 + 历史趋势
- **大数据链路**：Kafka → Spark Structured Streaming（60s/10s 窗口）→ Hive 四层数仓；组件故障自动降级本地数据源
- **预测性维护**：时序特征工程 + IsolationForest + LightGBM/XGBoost 风险预测与 RUL + SHAP 可解释；预测命中自动生成预警
- **AI 智能助手**：DeepSeek Function Calling（查询/预测/推送/建单）+ 运维规范知识库 RAG（带引用）
- **多渠道推送**：微信公众号测试号（客服消息）、SMTP 邮件（HTML 正文）、阿里云短信（预留）
- **资产二维码**：扫码直达公网详情页（免登录）

## 目录结构

```
├── src/python/            # FastAPI 后端 + Spark Streaming + 训练脚本
│   ├── main.py            # 主服务（含传感器模拟/大数据/预测/推送/AI）
│   ├── spark_streaming.py # Kafka→Spark→Hive 流计算
│   ├── predictive_models.py  # 特征工程/模型/在线预测
│   ├── train_predictive.py   # 训练入口
│   └── ...
├── output/                # 前端大屏（show.html）
├── config/                # Hive DDL / 大数据组件配置
├── docker/                # Docker Compose 大数据集群
└── scripts/               # 启停脚本
```

## 快速启动

```bash
# 1. 启动大数据集群（Kafka/Hive/Spark/ES）
cd docker && docker compose up -d

# 2. 配置密钥（复制模板并填写，详见下文"密钥配置"）
cp .env.example .env && vi .env

# 3. 启动后端
set -a; source .env; set +a
cd src/python && python3 -m uvicorn main:app --host 0.0.0.0 --port 8000

# 4. 浏览器打开大屏
# http://localhost:8000/output/show.html （默认账号 admin / admin123，仅演示用）
```

## 密钥配置（重要）

所有密钥均通过环境变量注入，仓库内不含任何真实凭据：

| 变量 | 用途 | 获取方式 |
| --- | --- | --- |
| `WX_TEST_APPID` / `WX_TEST_SECRET` / `WX_TEST_OPENID` | 微信测试号推送 | [微信公众平台测试号](https://mp.weixin.qq.com/debug/cgi-bin/sandbox?t=sandbox/login) |
| 邮箱推送 | 推送配置界面填写 | `data/push_config.json`（已被 .gitignore 忽略） |
| DeepSeek | AI 助手 | `data/agent_config.json`（已被 .gitignore 忽略） |

**资产二维码公网地址**：编辑 `output/show.html` 中 `amQrDataUrl()` 的 `base` 变量为你的 Serveo/内网穿透域名；留空则自动回退到当前访问地址。

## 预测模型训练

```bash
# 1. 启动传感器模拟积累历史（场景：正常运行为主 + 少量异常）
# 2. 导出数据并训练
curl -s http://localhost:8000/sensors/export -H "Authorization: Bearer <token>"
python3 src/python/train_predictive.py --input data/sensor_history.jsonl --model-dir models --window 15 --horizon 6
# 3. 重启后端加载模型，查看 models/predictive_report.json 评估指标
```

## 免责声明

本项目为**教学演示原型**，模拟数据仅用于展示链路与功能；Spark Streaming 状态支持模拟兜底，生产环境请接入真实设备与集群。
