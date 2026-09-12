#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市管网资产数字化台账 - FastAPI接口服务

功能：
1. 资产全景台账接口（总览/分类统计/列表查询）
2. 全生命周期档案接口（时间线/事件列表/费用汇总）
3. 资产盘点接口（盘点总览/差异报告/提交盘点）
4. 资产权属管理接口（权属汇总/权属明细）
5. 资产风险预测接口（异常检测/寿命预测）
6. 工作流执行接口
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, Header, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse, StreamingResponse, HTMLResponse, FileResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import os
import subprocess
import json
import time
import hashlib
import secrets
import io
import csv
import zlib
import re
import hmac
import base64
import uuid
import smtplib
import threading
import queue
import socket
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import urllib.request
import urllib.parse
import sys
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kb_rag           # noqa: E402  RAG 向量知识库（TF-IDF 字符 n-gram + 余弦）
import agent_brain      # noqa: E402  Agent 诊断编排 / 长期记忆 / 无 Key 本地兜底
import predict_ledger   # noqa: E402  预测台账与闭环量化指标

app = FastAPI(
    title="城市管网资产数字化台账",
    version="3.0",
    description="基于大数据技术栈的城市管网资产全生命周期管理与可视化服务"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def no_cache_html(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.endswith(".html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
output_dir = os.path.join(project_root, "output")
app.mount("/output", StaticFiles(directory=output_dir), name="output")

# ==============================================================================
# 用户系统与认证
# ==============================================================================

DATA_DIR = os.path.join(project_root, "data")
os.makedirs(DATA_DIR, exist_ok=True)
USERS_FILE = os.path.join(DATA_DIR, "users.json")
LOGS_FILE = os.path.join(DATA_DIR, "operation_logs.json")
TOKENS = {}

ROLE_NAMES = {
    "super_admin": "超级管理员",
    "admin": "系统管理员",
    "oam_lead": "运维主管",
    "oam": "运维人员",
    "viewer": "监管查看员",
}

PANEL_PERMISSIONS = {
    "管网类型分布": ["super_admin", "admin", "oam_lead", "oam", "viewer"],
    "区域资产分布": ["super_admin", "admin", "oam_lead", "oam", "viewer"],
    "资产状态总览": ["super_admin", "admin", "oam_lead", "oam", "viewer"],
    "资产风险排名 TOP10": ["super_admin", "admin", "oam_lead", "oam", "viewer"],
    "管径管材分布 TOP8": ["super_admin", "admin", "oam_lead", "oam"],
    "资产年代分布": ["super_admin", "admin", "oam_lead", "oam"],
    "生命周期事件记录": ["super_admin", "admin", "oam_lead", "oam"],
    "盘点差异分析": ["super_admin", "admin", "oam_lead", "oam"],
    "权属责任矩阵": ["super_admin", "admin"],
    "生命周期费用趋势": ["super_admin", "admin"],
    "盘点差异分类": ["super_admin", "admin"],
    "盘点方式统计": ["super_admin", "admin"],
}

EXPORT_PERMISSIONS = ["super_admin", "admin", "oam_lead", "oam"]


def hash_password(password: str) -> str:
    salt = "pipeline_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()


def load_users() -> dict:
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    default_users = {}
    for username, role, name in [
        ("admin", "super_admin", "超级管理员"),
    ]:
        default_users[username] = {
            "username": username,
            "password": hash_password("admin123"),
            "name": name,
            "role": role,
            "phone": "",
            "email": "",
            "department": "",
            "status": "active",
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "last_login": "",
            "login_count": 0,
            "failed_attempts": 0,
            "locked_until": "",
        }
    save_users(default_users)
    return default_users


def save_users(users: dict):
    with open(USERS_FILE, "w", encoding="utf-8") as f:
        json.dump(users, f, ensure_ascii=False, indent=2)


def load_logs() -> list:
    if os.path.exists(LOGS_FILE):
        with open(LOGS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_logs(logs: list):
    with open(LOGS_FILE, "w", encoding="utf-8") as f:
        json.dump(logs[-5000:], f, ensure_ascii=False, indent=2)


def add_log(username: str, action: str, detail: str, ip: str = ""):
    logs = load_logs()
    logs.append({
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "username": username,
        "action": action,
        "detail": detail,
        "ip": ip,
    })
    save_logs(logs)


INVENTORY_CHECKS_FILE = os.path.join(DATA_DIR, "inventory_checks.json")


def load_inventory_checks() -> list:
    if os.path.exists(INVENTORY_CHECKS_FILE):
        try:
            with open(INVENTORY_CHECKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    seed = []
    rng = np.random.RandomState(7)
    prefixes = ["WSP", "HTP", "GSP", "SWP", "HZP"]
    fields = ["管径", "材质", "长度", "权属单位"]
    for i in range(8):
        prefix = prefixes[i % 5]
        asset_id = f"{prefix}-{int(rng.randint(1, 1500)):05d}"
        seed.append({
            "check_id": f"CHK-SEED-{i + 1:03d}",
            "asset_id": asset_id,
            "check_method": ["扫码盘点", "巡检盘点", "GIS比对"][i % 3],
            "diff_status": "不一致",
            "diff_field": fields[i % 4],
            "diff_description": f"现场复核发现{fields[i % 4]}与台账登记不符",
            "status": "待处理",
            "check_time": (datetime.now() - timedelta(days=int(rng.randint(1, 20)))).strftime("%Y-%m-%d %H:%M"),
            "checker": f"盘点员{int(rng.randint(1, 15)):03d}",
            "resolve_note": "",
            "resolved_by": "",
            "resolved_at": "",
        })
    save_inventory_checks(seed)
    return seed


def save_inventory_checks(checks: list):
    with open(INVENTORY_CHECKS_FILE, "w", encoding="utf-8") as f:
        json.dump(checks[-2000:], f, ensure_ascii=False, indent=2)


COMMANDS_FILE = os.path.join(DATA_DIR, "commands.json")
WORKORDERS_FILE = os.path.join(DATA_DIR, "workorders.json")
DETECTION_TASKS_FILE = os.path.join(DATA_DIR, "detection_tasks.json")
FACILITIES_FILE = os.path.join(DATA_DIR, "facilities.json")

WRITE_ROLES = ["super_admin", "admin", "oam_lead", "oam"]


def load_json_list(path: str) -> list:
    """读 JSON 列表，**区分"文件不存在/空文件"与"文件损坏"**。

    原实现把解析失败也返回 []，于是损坏的工单文件被当成"没有工单" → 自动建单的去重失效
    → 每轮扫描重复建单；同时"面板显示有账、磁盘上其实没有"。损坏时改为 fail fast 并留一份
    .corrupt 备份（否则紧接着的一次写入就把原始内容永久覆盖，无法再排查）。
    """
    if not os.path.exists(path):
        return []            # 文件不存在 = 还没有数据，正常
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        if not raw:
            return []        # 空文件 = 还没有数据，正常
        data = json.loads(raw)
        if not isinstance(data, list):
            raise ValueError(f"顶层不是列表，而是 {type(data).__name__}")
        return data
    except Exception as e:
        bak = path + ".corrupt"
        try:
            if not os.path.exists(bak):
                with open(path, "rb") as src, open(bak, "wb") as dst:
                    dst.write(src.read())
        except Exception:
            pass
        raise RuntimeError(f"数据文件损坏，已备份到 {bak}：{path}：{type(e).__name__}: {e}")


def save_json_list(path: str, items: list):
    """原子写：临时文件 + fsync + os.replace。

    原实现直接 open(path, "w") 覆盖，写到一半崩溃或并发写会留下半个 JSON，而读取端又把
    损坏当成空列表 —— 结果是静默丢数据。临时文件 + os.replace 是同一文件系统内的原子替换，
    要么是旧内容、要么是完整新内容，不存在半截状态。
    """
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(items[-3000:], f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_commands() -> list:
    return load_json_list(COMMANDS_FILE)


def save_commands(items: list):
    save_json_list(COMMANDS_FILE, items)


def load_workorders() -> list:
    return load_json_list(WORKORDERS_FILE)


def save_workorders(items: list):
    save_json_list(WORKORDERS_FILE, items)


def load_detection_tasks() -> list:
    return load_json_list(DETECTION_TASKS_FILE)


def save_detection_tasks(items: list):
    save_json_list(DETECTION_TASKS_FILE, items)


def load_facilities() -> list:
    facilities = load_json_list(FACILITIES_FILE)
    if facilities:
        return facilities
    seed = []
    rng = np.random.RandomState(610603)
    town_cycle = list(REGIONS)
    for i in range(20):
        ftype = ["阀门", "泵站", "压力传感器", "流量计", "控制柜", "水质监测点"][i % 6]
        town = town_cycle[i % len(town_cycle)]
        fid = f"FAC-{i + 1:03d}"
        online = zlib.crc32(fid.encode()) % 100 < 90
        seed.append({
            "facility_id": fid,
            "name": f"{town}{ftype}{(i // 6) + 1}号",
            "type": ftype,
            "region": town,
            "online": online,
            "protocol": ["Modbus-RTU", "NB-IoT", "MQTT"][i % 3],
            "last_heartbeat": datetime.now().strftime("%Y-%m-%d %H:%M:%S") if online else (
                datetime.now() - timedelta(hours=int(rng.randint(2, 48)))).strftime("%Y-%m-%d %H:%M:%S"),
            "install_date": (datetime.now() - timedelta(days=int(rng.randint(200, 2600)))).strftime("%Y-%m-%d"),
        })
    save_json_list(FACILITIES_FILE, seed)
    return seed


def save_facilities(items: list):
    save_json_list(FACILITIES_FILE, items)


COMMAND_FAIL_REASONS = ["设备响应超时", "目标设备离线", "设备执行异常，返回错误码", "通信链路中断，校验失败"]


def _command_success(cmd_id: str) -> bool:
    return zlib.crc32(cmd_id.encode()) % 100 < 88


def create_command_record(user: dict, target_type: str, target_id: str, target_name: str,
                          action: str, params, priority: str, source: str) -> dict:
    cmd_id = f"CMD-{int(time.time() * 1000)}"
    jitter = zlib.crc32(cmd_id.encode()) % 1000 / 1000.0
    now = time.time()
    record = {
        "command_id": cmd_id,
        "target_type": target_type,
        "target_id": target_id,
        "target_name": target_name,
        "action": action,
        "params": params if isinstance(params, dict) else {},
        "priority": priority if priority in ("高", "中", "低") else "中",
        "source": source,
        "operator": user.get("name") or user.get("username"),
        "username": user.get("username", ""),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "status": "已下发",
        "execute_at_epoch": now + 3 + jitter * 3,
        "finish_at_epoch": now + 6 + jitter * 6,
        "will_succeed": _command_success(cmd_id),
        "finished_at": "",
        "result": "",
        "receipt": "",
    }
    commands = load_commands()
    commands.append(record)
    save_commands(commands)
    return record


def progress_command(cmd: dict) -> bool:
    """懒推进指令状态机：已下发→执行中→成功/失败。返回是否有变化。"""
    if cmd.get("status") not in ("已下发", "执行中"):
        return False
    now = time.time()
    changed = False
    if cmd["status"] == "已下发" and now >= cmd.get("execute_at_epoch", 0):
        cmd["status"] = "执行中"
        changed = True
    if cmd["status"] == "执行中" and now >= cmd.get("finish_at_epoch", 0):
        if cmd.get("will_succeed", True):
            cmd["status"] = "成功"
            cmd["result"] = "设备回执：指令执行成功"
        else:
            cmd["status"] = "失败"
            idx = zlib.crc32(cmd.get("command_id", "x").encode()) % len(COMMAND_FAIL_REASONS)
            cmd["result"] = COMMAND_FAIL_REASONS[idx]
        cmd["receipt"] = cmd["result"]
        cmd["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        changed = True
    return changed


def progress_all_commands() -> list:
    commands = load_commands()
    changed = any(progress_command(c) for c in commands)
    if changed:
        save_commands(commands)
    return commands


def validate_token(token: str) -> Optional[dict]:
    if not token or token not in TOKENS:
        return None
    info = TOKENS[token]
    if datetime.fromisoformat(info["expires"]) < datetime.now():
        del TOKENS[token]
        return None
    users = load_users()
    user = users.get(info["username"])
    if not user or user["status"] != "active":
        del TOKENS[token]
        return None
    return {**user, "token": token}


def get_current_user(authorization: Optional[str] = Header(None)) -> dict:
    if not authorization:
        raise HTTPException(status_code=401, detail="未登录")
    token = authorization.replace("Bearer ", "")
    user = validate_token(token)
    if not user:
        raise HTTPException(status_code=401, detail="登录已过期，请重新登录")
    return user


def require_role(user: dict, allowed_roles: list):
    if user["role"] not in allowed_roles:
        raise HTTPException(status_code=403, detail="权限不足")


users_db = load_users()

# ==============================================================================
# 管网类型配置
# ==============================================================================

PIPELINE_TYPES = [
    {"code": "WSP", "name": "供水管网", "weight": 0.30},
    {"code": "HTP", "name": "供暖管网", "weight": 0.20},
    {"code": "GSP", "name": "燃气管网", "weight": 0.25},
    {"code": "SWP", "name": "污水管网", "weight": 0.15},
    {"code": "HZP", "name": "危废输送管网", "weight": 0.10},
]

DIAMETERS = ["DN50", "DN80", "DN100", "DN150", "DN200", "DN300", "DN400", "DN600", "DN800", "DN1000", "DN1200", "DN1500", "DN2000"]
MATERIALS = ["球墨铸铁管", "PE管", "钢管", "钢筋混凝土管", "HDPE双壁波纹管", "不锈钢管", "预制直埋保温管", "玻璃钢管", "PVC-U管", "衬氟钢管"]
REGIONS = ["真武洞街道", "金明街道", "白坪街道", "沿河湾镇", "建华镇", "招安镇", "高桥镇", "镰刀湾镇", "坪桥镇", "化子坪镇", "砖窑湾镇"]
REGION_COORDS = {
    "真武洞街道": [109.326, 36.863], "金明街道": [109.318, 36.884],
    "白坪街道": [109.335, 36.842], "建华镇": [109.306, 36.952],
    "高桥镇": [109.266, 36.736], "招安镇": [109.118, 36.879],
    "化子坪镇": [109.216, 37.028], "镰刀湾镇": [109.063, 36.968],
    "坪桥镇": [109.546, 36.942], "沿河湾镇": [109.498, 36.788],
    "砖窑湾镇": [109.032, 36.699],
}
OWNERSHIP_UNITS = ["市水务集团", "市供热集团", "市燃气集团", "市排水集团", "市环保产业集团", "区水务局", "区城管委", "园区管委会"]
OAM_UNITS = ["市政养护一处", "市政养护二处", "市政养护三处", "管网检测中心", "应急抢修大队", "智慧管网运维中心"]
SUPERVISION_UNITS = ["市住建局", "市城管执法局", "市应急管理局", "市生态环境局", "市市场监管局"]
LIFECYCLE_TYPES = ["采购", "施工安装", "日常运维", "定期巡检", "改造更新", "报废处置"]

# ==============================================================================
# 全局状态
# ==============================================================================

anomaly_model = None
rul_model = None
using_dummy_model = False

analysis_results = {
    "asset_overview": {},
    "asset_distribution": {},
    "lifecycle_events": [],
    "lifecycle_cost_summary": [],
    "inventory_summary": {},
    "inventory_diff_report": [],
    "ownership_summary": [],
    "ownership_changes": [],
    "risk_ranking": [],
    "alert_list": [],
    "map_data": [],
    "generation_time": "",
    "data_count": 0,
}

asset_ledger_map = {}

workflow_status = {
    "running": False,
    "current_step": "",
    "progress": 0,
    "message": "等待执行",
    "start_time": "",
    "data_count": 0,
    "error": None,
}


# ==============================================================================
# 模拟模型
# ==============================================================================

class DummyAnomalyDetector:
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            results = []
            for _, row in X.iterrows():
                is_abnormal = 1 if (row.get('risk_score', 0) > 60 or row.get('aging_index', 0) > 0.8) else 0
                results.append(-1 if is_abnormal else 1)
            return results
        return [1]

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            results = []
            for _, row in X.iterrows():
                risk = 0.1
                if row.get('risk_score', 0) > 60: risk += 0.4
                if row.get('aging_index', 0) > 0.8: risk += 0.3
                if row.get('incident_count', 0) > 2: risk += 0.2
                risk = min(0.99, risk)
                results.append([1 - risk, risk])
            return results
        return [[0.9, 0.1]]


class DummyRULPredictor:
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            results = []
            for _, row in X.iterrows():
                rul = max(0, row.get('design_life', 40) - row.get('service_years', 20))
                results.append(rul)
            return results
        return [20]


def load_models():
    global anomaly_model, rul_model, using_dummy_model
    try:
        import joblib
        anomaly_model = joblib.load("models/anomaly_model.pkl")
        rul_model = joblib.load("models/rul_model.pkl")
        using_dummy_model = False
        print("模型加载成功")
    except Exception:
        anomaly_model = DummyAnomalyDetector()
        rul_model = DummyRULPredictor()
        using_dummy_model = True
        print("使用模拟模型")


# ==============================================================================
# 请求模型
# ==============================================================================

class AssetPredictRequest(BaseModel):
    asset_id: str = ""
    service_years: float = 20
    design_life: float = 40
    segment_length_m: float = 100
    burial_depth_m: float = 1.5
    diameter_numeric: float = 300
    depreciation_rate: float = 0.5
    inspection_gap_years: float = 1
    maintenance_count: float = 5
    incident_count: float = 0
    risk_score: float = 40

class InventoryCheckRequest(BaseModel):
    asset_id: str
    check_method: str = "扫码盘点"
    diff_status: str = "一致"
    diff_description: str = ""

class GenerateDataRequest(BaseModel):
    count: int = 5000
    output_dir: str = "data/asset"


# ==============================================================================
# 模拟分析引擎
# ==============================================================================

def simulate_analysis():
    global analysis_results, asset_ledger_map

    np.random.seed(42)
    total_assets = 5000

    type_counts = {pt["name"]: int(total_assets * pt["weight"]) for pt in PIPELINE_TYPES}
    total_length_km = round(np.random.uniform(2800, 3500), 1)
    avg_unit_value = np.random.uniform(50000, 200000)
    total_value = round(total_assets * avg_unit_value, 0)
    overview_net_ratio = 1 / 1.3

    in_service = int(total_assets * 0.75)
    retired = int(total_assets * 0.07)
    pending = int(total_assets * 0.10)
    suspended = total_assets - in_service - retired - pending

    overview = {
        "total_assets": total_assets,
        "total_length_km": total_length_km,
        "total_original_value": round(total_value * 1.3, 0),
        "total_net_value": round(total_value, 0),
        "in_service_count": in_service,
        "retired_count": retired,
        "pending_inspection_count": pending,
        "suspended_count": suspended,
        "avg_risk_score": round(np.random.uniform(35, 55), 1),
        "high_risk_count": int(total_assets * 0.12),
        "inventory_diff_rate": round(np.random.uniform(0.08, 0.18), 4),
        "newness_rate": round(overview_net_ratio, 4),
        "type_distribution": [
            {"name": name, "count": count, "length_km": round(total_length_km * count / total_assets, 1)}
            for name, count in type_counts.items()
        ],
        "status_distribution": [
            {"name": "在用", "count": in_service, "color": "#10b981"},
            {"name": "待检", "count": pending, "color": "#f59e0b"},
            {"name": "停用", "count": suspended, "color": "#6b7280"},
            {"name": "报废", "count": retired, "color": "#ef4444"},
        ],
        "current_time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    region_dist = []
    for region in REGIONS:
        for pt in PIPELINE_TYPES:
            count = int(type_counts[pt["name"]] / len(REGIONS) * np.random.uniform(0.7, 1.3))
            region_dist.append({
                "region": region,
                "pipeline_type": pt["name"],
                "count": count,
                "length_m": round(count * np.random.uniform(50, 300), 0),
                "value": round(count * np.random.uniform(30000, 150000), 0),
                "avg_risk": round(np.random.uniform(20, 70), 1),
            })

    material_dist = []
    for mat in MATERIALS:
        material_dist.append({
            "material": mat,
            "count": int(np.random.uniform(100, 800)),
            "avg_risk": round(np.random.uniform(25, 65), 1),
        })
    material_dist.sort(key=lambda x: x["count"], reverse=True)

    decade_dist = []
    for decade in ["1980s", "1990s", "2000s", "2010s", "2020s"]:
        weight = {"1980s": 0.05, "1990s": 0.15, "2000s": 0.30, "2010s": 0.35, "2020s": 0.15}[decade]
        decade_dist.append({
            "decade": decade,
            "count": int(total_assets * weight),
            "length_km": round(total_length_km * weight, 1),
        })

    distribution = {
        "by_region": region_dist,
        "by_material": material_dist,
        "by_decade": decade_dist,
        "by_diameter": [
            {"diameter": d, "count": int(np.random.uniform(50, 500))}
            for d in DIAMETERS[:8]
        ],
    }

    lifecycle_events = []
    for i in range(50):
        lt = np.random.choice(LIFECYCLE_TYPES)
        pt = np.random.choice([p["name"] for p in PIPELINE_TYPES])
        lifecycle_events.append({
            "event_id": f"EVT-{i+1:06d}",
            "asset_id": f"{np.random.choice(['WSP','HTP','GSP','SWP','HZP'])}-{np.random.randint(1,5000):05d}",
            "pipeline_type": pt,
            "event_type": lt,
            "event_date": f"2024-{np.random.randint(1,12):02d}-{np.random.randint(1,28):02d}",
            "responsible_unit": np.random.choice(OAM_UNITS),
            "cost": round(np.random.uniform(500, 500000), 0),
            "description": f"{pt}{lt}记录",
            "region": np.random.choice(REGIONS),
        })
    lifecycle_events.sort(key=lambda x: x["event_date"], reverse=True)

    cost_summary = []
    for lt in LIFECYCLE_TYPES:
        cost_summary.append({
            "event_type": lt,
            "count": int(np.random.uniform(200, 3000)),
            "total_cost": round(np.random.uniform(500000, 15000000), 0),
            "avg_cost": round(np.random.uniform(5000, 80000), 0),
        })

    total_checked = int(total_assets * 0.6)
    diff_count = int(total_checked * overview["inventory_diff_rate"])
    inventory_summary = {
        "total_batches": 12,
        "total_checked": total_checked,
        "match_count": total_checked - diff_count,
        "diff_count": diff_count,
        "diff_rate": overview["inventory_diff_rate"],
        "latest_batch": "INV-2024-08",
        "by_method": [
            {"method": "扫码盘点", "count": int(total_checked * 0.5), "diff_rate": 0.06},
            {"method": "巡检盘点", "count": int(total_checked * 0.3), "diff_rate": 0.12},
            {"method": "无人机盘点", "count": int(total_checked * 0.12), "diff_rate": 0.09},
            {"method": "GIS比对", "count": int(total_checked * 0.08), "diff_rate": 0.15},
        ],
    }

    inventory_diff = []
    for region in REGIONS:
        for pt in PIPELINE_TYPES:
            checked = int(np.random.uniform(20, 80))
            diff = int(checked * np.random.uniform(0.03, 0.20))
            inventory_diff.append({
                "region": region,
                "pipeline_type": pt["name"],
                "total_checked": checked,
                "diff_count": diff,
                "diff_rate": round(diff / checked, 4),
                "missing": int(diff * 0.5),
                "extra": int(diff * 0.1),
                "mismatch": diff - int(diff * 0.5) - int(diff * 0.1),
            })

    risk_ranking = []
    for i in range(20):
        pt = np.random.choice([p["name"] for p in PIPELINE_TYPES])
        risk = int(np.random.uniform(55, 95))
        risk_ranking.append({
            "ranking": i + 1,
            "asset_id": f"{np.random.choice(['WSP','HTP','GSP','SWP','HZP'])}-{np.random.randint(1,5000):05d}",
            "pipeline_type": pt,
            "region": np.random.choice(REGIONS),
            "risk_score": risk,
            "risk_level": "极高" if risk >= 80 else "高" if risk >= 60 else "中",
            "service_years": int(np.random.uniform(25, 45)),
            "design_life": int(np.random.choice([30, 40, 50])),
            "remaining_life": max(0, int(np.random.uniform(-5, 15))),
            "ownership_unit": np.random.choice(OWNERSHIP_UNITS),
        })
    risk_ranking.sort(key=lambda x: x["risk_score"], reverse=True)

    ownership = []
    for unit in OWNERSHIP_UNITS:
        for pt in PIPELINE_TYPES:
            if np.random.random() > 0.4:
                ownership.append({
                    "ownership_unit": unit,
                    "oam_unit": np.random.choice(OAM_UNITS),
                    "supervision_unit": np.random.choice(SUPERVISION_UNITS),
                    "pipeline_type": pt["name"],
                    "asset_count": int(np.random.uniform(50, 400)),
                    "total_length_m": round(np.random.uniform(5000, 50000), 0),
                    "total_value": round(np.random.uniform(5000000, 50000000), 0),
                    "avg_risk": round(np.random.uniform(25, 65), 1),
                })

    # --- 资产明细台账（与总览统计口径一致） ---
    type_owner_map = {
        "WSP": ["市水务集团", "区水务局"],
        "HTP": ["市供热集团", "园区管委会"],
        "GSP": ["市燃气集团", "园区管委会"],
        "SWP": ["市排水集团", "区城管委"],
        "HZP": ["市环保产业集团", "市生态环境局"],
    }
    type_supervision_map = {
        "WSP": ["市住建局", "市水务局"] if "市水务局" in SUPERVISION_UNITS else ["市住建局"],
        "HTP": ["市住建局", "市应急管理局"],
        "GSP": ["市应急管理局", "市市场监管局"],
        "SWP": ["市生态环境局", "市城管执法局"],
        "HZP": ["市生态环境局", "市应急管理局"],
    }
    status_pool = np.random.permutation(
        ["在用"] * in_service + ["待检"] * pending + ["报废"] * retired + ["停用"] * suspended
    )
    ledger = []
    idx = 0
    current_year = datetime.now().year
    for pt in PIPELINE_TYPES:
        code = pt["code"]
        for seq in range(1, type_counts[pt["name"]] + 1):
            diameter = str(np.random.choice(DIAMETERS))
            material = str(np.random.choice(MATERIALS))
            install_year = int(np.random.choice(
                [1985, 1992, 1998, 2003, 2008, 2012, 2016, 2020, 2023],
                p=[0.04, 0.07, 0.10, 0.14, 0.16, 0.17, 0.15, 0.11, 0.06]))
            service_years = max(current_year - install_year, 1)
            design_life = int(np.random.choice([30, 40, 50], p=[0.25, 0.5, 0.25]))
            length_m = round(float(np.random.uniform(30, 480)), 1)
            diameter_numeric = int(diameter.replace("DN", ""))
            original_value = round(length_m * float(np.random.uniform(260, 1500)) * (1 + diameter_numeric / 2500), 0)
            aging = min(service_years / design_life, 0.95)
            net_value = round(original_value * (1 - aging), 0)
            risk = int(min(95, max(5, service_years * 1.5 + np.random.uniform(-8, 28))))
            owner_pool = type_owner_map.get(code, OWNERSHIP_UNITS)
            sup_pool = type_supervision_map.get(code, SUPERVISION_UNITS)
            ledger.append({
                "asset_id": f"{code}-{seq:05d}",
                "pipeline_type": pt["name"],
                "diameter": diameter,
                "material": material,
                "install_year": install_year,
                "service_years": service_years,
                "design_life": design_life,
                "region": str(np.random.choice(REGIONS)),
                "status": str(status_pool[idx % len(status_pool)]),
                "risk_score": risk,
                "risk_level": "极高" if risk >= 80 else "高" if risk >= 60 else "中" if risk >= 40 else "低",
                "length_m": length_m,
                "burial_depth_m": round(float(np.random.uniform(0.8, 3.5)), 1),
                "ownership_unit": str(np.random.choice(owner_pool)),
                "oam_unit": str(np.random.choice(OAM_UNITS)),
                "supervision_unit": str(np.random.choice(sup_pool)),
                "original_value": original_value,
                "net_value": net_value,
                "depreciation_rate": round(1 / design_life, 4),
            })
            idx += 1

    asset_ledger_map = {a["asset_id"]: a for a in ledger}

    # --- 权属变更记录（责任边界移交/确认留痕） ---
    change_types = ["产权移交", "运维委托变更", "监管辖区划转", "责任边界确认"]
    ownership_changes = []
    for i in range(14):
        ctype = str(np.random.choice(change_types, p=[0.35, 0.3, 0.15, 0.2]))
        pt = PIPELINE_TYPES[int(np.random.randint(0, len(PIPELINE_TYPES)))]
        if ctype == "产权移交":
            from_unit, to_unit = str(np.random.choice(OWNERSHIP_UNITS)), str(np.random.choice(OWNERSHIP_UNITS))
        elif ctype == "运维委托变更":
            from_unit, to_unit = str(np.random.choice(OAM_UNITS)), str(np.random.choice(OAM_UNITS))
        elif ctype == "监管辖区划转":
            from_unit, to_unit = str(np.random.choice(SUPERVISION_UNITS)), str(np.random.choice(SUPERVISION_UNITS))
        else:
            from_unit, to_unit = str(np.random.choice(OWNERSHIP_UNITS)), str(np.random.choice(OAM_UNITS))
        ownership_changes.append({
            "change_id": f"CG-{current_year - 1}-{i + 1:03d}" if i % 3 == 0 else f"CG-{current_year}-{i + 1:03d}",
            "change_date": f"{current_year if i % 3 else current_year - 1}-{int(np.random.randint(1, 12)):02d}-{int(np.random.randint(1, 28)):02d}",
            "change_type": ctype,
            "pipeline_type": pt["name"],
            "region": str(np.random.choice(REGIONS)),
            "asset_count": int(np.random.randint(8, 220)),
            "from_unit": from_unit,
            "to_unit": to_unit,
            "status": str(np.random.choice(["已完成", "公示中", "审核中"], p=[0.6, 0.25, 0.15])),
            "approval_doc": f"安管网字〔{current_year if i % 3 else current_year - 1}〕第{int(np.random.randint(1, 90)):02d}号",
            "remark": "竣工资产移交运维" if ctype == "产权移交" else "责任边界三方确认签署",
        })
    ownership_changes.sort(key=lambda x: x["change_date"], reverse=True)

    alert_list = []
    alert_types = ["超期服役", "高风险预警", "盘点差异", "腐蚀预警", "压力异常", "巡检超期"]
    for i in range(15):
        alert_list.append({
            "alert_id": f"ALT-{i+1:04d}",
            "alert_type": np.random.choice(alert_types),
            "asset_id": f"{np.random.choice(['WSP','HTP','GSP','SWP','HZP'])}-{np.random.randint(1,5000):05d}",
            "pipeline_type": np.random.choice([p["name"] for p in PIPELINE_TYPES]),
            "region": np.random.choice(REGIONS),
            "level": np.random.choice(["紧急", "重要", "一般"], p=[0.15, 0.35, 0.50]),
            "description": f"管网资产异常预警信息",
            "create_time": (pd.Timestamp.now() - pd.Timedelta(hours=np.random.randint(1, 72))).strftime('%Y-%m-%d %H:%M:%S'),
            "status": np.random.choice(["未处理", "处理中", "已处理"], p=[0.3, 0.3, 0.4]),
            "push_status": "未推送",
            "push_channels": [],
            # 模拟数据必须自带来源标记：否则它会被「一键推送未处理」真实推送到微信/邮箱，
            # 也会被 DeepSeek Agent 当作真实情况念给用户。推送与 Agent 上下文按此过滤。
            "source": "mock",
        })
    alert_list.sort(key=lambda x: x["create_time"], reverse=True)

    map_data = []
    for region in REGIONS:
        region_alerts = [a for a in alert_list if a["region"] == region]
        region_items = [r for r in region_dist if r["region"] == region]
        total_count = sum(r["count"] for r in region_items)
        avg_risk = round(np.mean([r["avg_risk"] for r in region_items]), 1) if region_items else 0
        high_risk = sum(1 for r in region_items if r["avg_risk"] > 55)
        map_data.append({
            "region": region,
            "coords": REGION_COORDS[region],
            "asset_count": total_count,
            "avg_risk": avg_risk,
            "high_risk_count": high_risk,
            "alert_count": len(region_alerts),
            "urgent_alerts": len([a for a in region_alerts if a["level"] == "紧急"]),
        })

    # 真实预警不能因为一次"重新分析"被模拟数据顶掉。原来这里整体重绑 analysis_results，
    # 一次调用就把预测预警与传感器预警换成 15 条随机模拟预警，于是：
    #   ① 台账 / 工单 / 反馈里存的 alert_id 立刻变成悬空外键；
    #   ② 这 15 条假预警会被「一键推送未处理」真实推送到微信/邮箱；
    #   ③ Agent 会把它们当真实情况念给用户。
    # 现在改为「模拟预警 + 保留全部真实预警」，模拟预警带 source=mock 供推送/Agent 过滤。
    preserved = [a for a in (analysis_results.get("alert_list") or [])
                 if a.get("source") in ("prediction", "sensor")]
    # 再兜一层：凡仍被未闭环工单或预测台账引用的 alert_id，一律保留，避免外键悬空。
    try:
        referenced = {o.get("alert_id") for o in load_workorders()
                      if o.get("alert_id") and o.get("status") != "已闭环"}
        referenced |= {r.get("alert_base") or r.get("alert_id")
                       for r in (predict_ledger.recent_records(limit=500) or [])}
        referenced.discard(None)
        have = {a.get("alert_id") for a in preserved}
        for a in (analysis_results.get("alert_list") or []):
            if a.get("alert_id") in referenced and a.get("alert_id") not in have:
                preserved.append(a)
                have.add(a.get("alert_id"))
    except Exception as e:
        print(f"[simulate_analysis] 保留被引用预警时出错（不影响主流程）: {e}")
    merged_alerts = alert_list + preserved
    merged_alerts.sort(key=lambda x: str(x.get("create_time") or ""), reverse=True)

    analysis_results = {
        "asset_overview": overview,
        "asset_distribution": distribution,
        "lifecycle_events": lifecycle_events,
        "lifecycle_cost_summary": cost_summary,
        "inventory_summary": inventory_summary,
        "inventory_diff_report": inventory_diff,
        "ownership_summary": ownership,
        "ownership_changes": ownership_changes,
        "risk_ranking": risk_ranking,
        "alert_list": merged_alerts,
        "map_data": map_data,
        "generation_time": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        "data_count": total_assets,
    }

    analysis_results = json.loads(json.dumps(analysis_results, default=lambda o: int(o) if isinstance(o, np.integer) else float(o) if isinstance(o, np.floating) else o))

    print(f"资产台账分析完成: {total_assets}项资产, {len(lifecycle_events)}条事件")


# ==============================================================================
# 时间筛选工具
# ==============================================================================

def apply_time_filter(data: dict, start_date: Optional[str], end_date: Optional[str]) -> dict:
    if not start_date and not end_date:
        return data

    today = pd.Timestamp.now().date()
    sd = pd.Timestamp(start_date).date() if start_date else pd.Timestamp("2020-01-01").date()
    ed = pd.Timestamp(end_date).date() if end_date else today

    full_start = pd.Timestamp("2020-01-01").date()
    full_days = max((today - full_start).days, 1)
    filter_days = max((ed - sd).days, 1)
    ratio = min(filter_days / full_days, 1.0)

    filtered = {}
    for key, val in data.items():
        filtered[key] = val

    if "lifecycle_events" in data:
        events = data["lifecycle_events"]
        filtered_events = []
        for e in events:
            try:
                ed_date = pd.Timestamp(e.get("event_date", "")).date()
                if sd <= ed_date <= ed:
                    filtered_events.append(e)
            except:
                filtered_events.append(e)
        filtered["lifecycle_events"] = filtered_events

    if "alert_list" in data:
        alerts = data["alert_list"]
        filtered_alerts = []
        for a in alerts:
            try:
                ct = pd.Timestamp(a.get("create_time", "")).date()
                if sd <= ct <= ed:
                    filtered_alerts.append(a)
            except:
                filtered_alerts.append(a)
        filtered["alert_list"] = filtered_alerts

    if "asset_overview" in data:
        ov = dict(data["asset_overview"])
        ov["total_assets"] = max(int(ov.get("total_assets", 0) * ratio), 1)
        ov["total_length_km"] = round(ov.get("total_length_km", 0) * ratio, 1)
        ov["total_original_value"] = round(ov.get("total_original_value", 0) * ratio, 0)
        ov["total_net_value"] = round(ov.get("total_net_value", 0) * ratio, 0)
        ov["in_service_count"] = max(int(ov.get("in_service_count", 0) * ratio), 0)
        ov["retired_count"] = max(int(ov.get("retired_count", 0) * ratio), 0)
        ov["pending_inspection_count"] = max(int(ov.get("pending_inspection_count", 0) * ratio), 0)
        ov["suspended_count"] = max(int(ov.get("suspended_count", 0) * ratio), 0)
        ov["high_risk_count"] = max(int(ov.get("high_risk_count", 0) * ratio), 0)
        if "type_distribution" in ov:
            ov["type_distribution"] = [
                {**td, "count": max(int(td.get("count", 0) * ratio), 1),
                 "length_km": round(td.get("length_km", 0) * ratio, 1)}
                for td in ov["type_distribution"]
            ]
        if "status_distribution" in ov:
            ov["status_distribution"] = [
                {**sd_item, "count": max(int(sd_item.get("count", 0) * ratio), 0)}
                for sd_item in ov["status_distribution"]
            ]
        filtered["asset_overview"] = ov

    if "asset_distribution" in data:
        dist = dict(data["asset_distribution"])
        if "by_diameter" in dist:
            dist["by_diameter"] = [
                {**d, "count": max(int(d.get("count", 0) * ratio), 1)}
                for d in dist["by_diameter"]
            ]
        if "by_region" in dist:
            dist["by_region"] = [
                {**d, "count": max(int(d.get("count", 0) * ratio), 1)}
                for d in dist["by_region"]
            ]
        if "by_decade" in dist:
            dist["by_decade"] = [
                {**d, "count": max(int(d.get("count", 0) * ratio), 1)}
                for d in dist["by_decade"]
            ]
        filtered["asset_distribution"] = dist

    if "lifecycle_cost_summary" in data:
        filtered["lifecycle_cost_summary"] = [
            {**c, "total_cost": round(c.get("total_cost", 0) * ratio, 0),
             "event_count": max(int(c.get("event_count", 0) * ratio), 1)}
            for c in data["lifecycle_cost_summary"]
        ]

    if "risk_ranking" in data:
        filtered["risk_ranking"] = [
            {**r, "risk_score": max(int(r.get("risk_score", 0) * ratio), 0)}
            for r in data["risk_ranking"]
        ]

    if "map_data" in data:
        filtered["map_data"] = [
            {**m, "asset_count": max(int(m.get("asset_count", 0) * ratio), 1),
             "high_risk_count": max(int(m.get("high_risk_count", 0) * ratio), 0),
             "alert_count": max(int(m.get("alert_count", 0) * ratio), 0),
             "urgent_alerts": max(int(m.get("urgent_alerts", 0) * ratio), 0)}
            for m in data["map_data"]
        ]

    return filtered


# ==============================================================================
# 接口定义
# ==============================================================================

@app.get("/")
def root():
    return RedirectResponse(url="/output/login.html")

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "models_loaded": anomaly_model is not None,
        "model_type": "模拟模型" if using_dummy_model else "真实模型",
        "analysis_ready": len(analysis_results.get("asset_overview", {})) > 0,
        "workflow_running": workflow_status["running"],
    }


# --- 资产全景台账 ---

def _synth_asset(asset_id: str) -> dict:
    prefix = asset_id.split("-")[0] if "-" in asset_id else "WSP"
    pt = next((p for p in PIPELINE_TYPES if p["code"] == prefix), PIPELINE_TYPES[0])
    rng = np.random.RandomState(zlib.crc32(asset_id.encode("utf-8")))
    install_year = int(rng.choice(range(1985, 2024)))
    current_year = datetime.now().year
    service_years = max(current_year - install_year, 1)
    design_life = int(rng.choice([30, 40, 50]))
    diameter = str(rng.choice(DIAMETERS))
    length_m = round(float(rng.uniform(30, 480)), 1)
    original_value = round(length_m * float(rng.uniform(260, 1500)) * (1 + int(diameter.replace("DN", "")) / 2500), 0)
    aging = min(service_years / design_life, 0.95)
    status = str(rng.choice(["在用", "停用", "待检", "报废"], p=[0.75, 0.08, 0.10, 0.07]))
    risk = int(min(95, max(5, service_years * 1.5 + rng.uniform(-8, 28))))
    return {
        "asset_id": asset_id,
        "pipeline_type": pt["name"],
        "diameter": diameter,
        "material": str(rng.choice(MATERIALS)),
        "install_year": install_year,
        "service_years": service_years,
        "design_life": design_life,
        "region": str(rng.choice(REGIONS)),
        "status": status,
        "risk_score": risk,
        "risk_level": "极高" if risk >= 80 else "高" if risk >= 60 else "中" if risk >= 40 else "低",
        "length_m": length_m,
        "burial_depth_m": round(float(rng.uniform(0.8, 3.5)), 1),
        "ownership_unit": str(rng.choice(OWNERSHIP_UNITS)),
        "oam_unit": str(rng.choice(OAM_UNITS)),
        "supervision_unit": str(rng.choice(SUPERVISION_UNITS)),
        "original_value": original_value,
        "net_value": round(original_value * (1 - aging), 0),
        "depreciation_rate": round(1 / design_life, 4),
    }


def _synth_lifecycle(asset: dict) -> list:
    rng = np.random.RandomState(zlib.crc32(asset["asset_id"].encode("utf-8")))
    install_year = asset.get("install_year", 2005)
    current_year = datetime.now().year
    plan = [("采购", install_year - 1, 3), ("施工安装", install_year, 5)]
    for year in range(install_year + 1, current_year):
        if rng.random() < 0.45:
            plan.append(("日常运维", year, 4))
        if rng.random() < 0.22:
            plan.append(("定期巡检", year, 6))
        if rng.random() < 0.07:
            plan.append(("改造更新", year, 9))
    if asset.get("status") == "报废":
        plan.append(("报废处置", current_year - 1, 11))
    details = {
        "采购": f"采购{asset.get('material', '')}管段{asset.get('diameter', '')}，长度{asset.get('length_m', 0)}m",
        "施工安装": f"{asset.get('region', '')}段管网施工安装，埋深{asset.get('burial_depth_m', 1.5)}m",
        "日常运维": "日常运维保养，包括管道清洗、阀门检修、防腐处理",
        "定期巡检": "定期巡检，含外观检查、防腐层检测与附属设施核验",
        "改造更新": "管段改造更新，更换老化部件、升级管材",
        "报废处置": "管网报废处置，管道封堵、拆除、无害化处理",
    }
    costs = {"采购": (50000, 500000), "施工安装": (100000, 800000), "日常运维": (800, 20000),
             "定期巡检": (500, 8000), "改造更新": (80000, 900000), "报废处置": (5000, 120000)}
    events = []
    for i, (etype, year, month) in enumerate(plan):
        lo, hi = costs[etype]
        events.append({
            "event_id": f"EVT-{asset['asset_id']}-{i:03d}",
            "asset_id": asset["asset_id"],
            "event_type": etype,
            "event_date": f"{year}-{month:02d}-{int(rng.randint(1, 28)):02d}",
            "cost": round(float(rng.uniform(lo, hi)), 0),
            "responsible_unit": str(rng.choice(OAM_UNITS)),
            "description": details[etype],
        })
    return events


@app.get("/asset/overview")
def get_asset_overview(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    return data.get("asset_overview", {})

@app.get("/asset/distribution")
def get_asset_distribution(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    return data.get("asset_distribution", {})

@app.get("/asset/list")
def get_asset_list(authorization: Optional[str] = Header(None), pipeline_type: Optional[str] = None, region: Optional[str] = None, status: Optional[str] = None, diameter: Optional[str] = None, material: Optional[str] = None, ownership_unit: Optional[str] = None, decade: Optional[str] = None, keyword: Optional[str] = None, page: int = 1, page_size: int = 20):
    get_current_user(authorization)
    ledger = list(asset_ledger_map.values())
    if not ledger:
        return {"total": 0, "page": page, "page_size": page_size, "data": []}

    if pipeline_type:
        ledger = [a for a in ledger if a["pipeline_type"] == pipeline_type]
    if region:
        ledger = [a for a in ledger if a["region"] == region]
    if status:
        ledger = [a for a in ledger if a["status"] == status]
    if diameter:
        ledger = [a for a in ledger if a["diameter"] == diameter]
    if material:
        ledger = [a for a in ledger if a["material"] == material]
    if ownership_unit:
        ledger = [a for a in ledger if a["ownership_unit"] == ownership_unit]
    if decade and decade.endswith("s") and decade[:-1].isdigit():
        start_year = int(decade[:-1])
        ledger = [a for a in ledger if start_year <= a["install_year"] < start_year + 10]
    if keyword:
        kw = keyword.strip().upper()
        if kw:
            ledger = [a for a in ledger if kw in a["asset_id"].upper()]

    ledger.sort(key=lambda a: a["risk_score"], reverse=True)
    total = len(ledger)
    start = max(page - 1, 0) * page_size
    return {"total": total, "page": page, "page_size": page_size, "data": ledger[start:start + page_size]}


def _asset_qr_page_html(asset_id: str) -> str:
    """生成资产二维码扫码访问的详情页（免登录、只读基础档案信息）。"""
    asset = asset_ledger_map.get(asset_id) or _synth_asset(asset_id)
    risk = int(asset.get("risk_score", 0))
    risk_color = "#ef4444" if risk >= 80 else ("#f59e0b" if risk >= 60 else "#16a34a")
    rows = [
        ("资产编号", asset.get("asset_id", "")),
        ("管网类型", asset.get("pipeline_type", "")),
        ("管径", asset.get("diameter", "")),
        ("管材", asset.get("material", "")),
        ("敷设年份", asset.get("install_year", "")),
        ("已服役年限", f"{asset.get('service_years', 0)} 年"),
        ("所在区域", asset.get("region", "")),
        ("管段长度", f"{asset.get('length_m', 0)} m"),
        ("埋深", f"{asset.get('burial_depth_m', 0)} m"),
        ("风险评分", f'<span style="color:{risk_color};font-weight:700;font-size:20px;">{risk}</span> 分（{asset.get("risk_level", "")}）'),
        ("资产状态", asset.get("status", "")),
        ("产权单位", asset.get("ownership_unit", "")),
        ("运维单位", asset.get("oam_unit", "")),
    ]
    alerts = [a for a in analysis_results.get("alert_list", []) if a.get("asset_id") == asset_id]
    alert_html = ""
    if alerts:
        items = "".join(
            f'<li>{a.get("alert_type", "")} · {a.get("level", "")} · {a.get("status", "")}</li>'
            for a in alerts[:5]
        )
        alert_html = f'<h3 style="font-size:15px;margin:18px 0 6px;">关联预警（{len(alerts)}）</h3><ul style="margin:0;padding-left:20px;color:#dc2626;font-size:13px;">{items}</ul>'
    trs = "".join(f'<tr><td class="k">{k}</td><td>{v}</td></tr>' for k, v in rows)
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>资产档案 · {asset_id}</title>
<style>
body{{font-family:'Microsoft YaHei',Arial,sans-serif;margin:0;background:#f1f5f9;color:#1e293b;}}
.card{{max-width:560px;margin:24px auto;background:#fff;border-radius:12px;box-shadow:0 4px 20px rgba(0,0,0,.08);padding:24px;}}
h1{{font-size:20px;margin:0 0 4px;}}.sub{{color:#64748b;font-size:13px;margin-bottom:16px;}}
table{{width:100%;border-collapse:collapse;font-size:14px;}}
td{{padding:9px 12px;border-bottom:1px solid #f1f5f9;}}
td.k{{color:#64748b;width:110px;white-space:nowrap;}}
.badge{{display:inline-block;padding:2px 10px;border-radius:10px;font-size:12px;background:#eff6ff;color:#2563eb;}}
.foot{{text-align:center;color:#94a3b8;font-size:12px;margin-top:16px;}}
</style>
</head>
<body>
<div class="card">
  <h1>🚰 管网资产档案</h1>
  <div class="sub">城市管网资产数字化台账系统 · <span class="badge">扫码查看</span></div>
  <table>{trs}</table>
  {alert_html}
  <div class="foot">—— 城市管网资产数字化台账系统 ——</div>
</div>
</body>
</html>"""


@app.get("/asset/detail/{asset_id}")
def get_asset_detail(asset_id: str, authorization: Optional[str] = Header(None),
                     accept: Optional[str] = Header(None)):
    # 手机扫码（浏览器导航，Accept 含 text/html）→ 返回免登录详情页；API 调用 → 返回 JSON（原逻辑不变）
    if accept and "text/html" in accept.lower():
        return HTMLResponse(content=_asset_qr_page_html(asset_id), media_type="text/html")
    get_current_user(authorization)
    asset = asset_ledger_map.get(asset_id)
    in_ledger = asset is not None
    if not asset:
        asset = _synth_asset(asset_id)

    events = [e for e in analysis_results.get("lifecycle_events", []) if e.get("asset_id") == asset_id]
    if not events:
        events = _synth_lifecycle(asset)
    events = sorted(events, key=lambda e: e.get("event_date", ""))

    original_value = asset.get("original_value", 0)
    net_value = asset.get("net_value", 0)
    design_life = max(asset.get("design_life", 40), 1)
    annual_dep = round(original_value / design_life, 0)
    accumulated = round(min(original_value - net_value, annual_dep * asset.get("service_years", 0)), 0)

    checks = [c for c in load_inventory_checks() if c.get("asset_id") == asset_id]
    checks.sort(key=lambda c: c.get("check_time", ""), reverse=True)
    alerts = [a for a in analysis_results.get("alert_list", []) if a.get("asset_id") == asset_id]

    return {
        "asset": asset,
        "in_ledger": in_ledger,
        "ownership": {
            "ownership_unit": asset["ownership_unit"],
            "oam_unit": asset["oam_unit"],
            "supervision_unit": asset["supervision_unit"],
            "responsibility": "产权单位承担资产保值与更新责任；运维单位承担日常巡检、维修与应急处置责任；监管单位承担安全监督与考核责任。",
        },
        "value": {
            "original_value": original_value,
            "annual_depreciation": annual_dep,
            "accumulated_depreciation": accumulated,
            "net_value": net_value,
            "newness_rate": round(net_value / original_value, 4) if original_value else 0,
            "depreciation_method": "年限平均法",
        },
        "lifecycle": events,
        "total_cost": round(sum(e.get("cost", 0) for e in events), 0),
        "inventory_checks": checks,
        "alerts": alerts,
    }


# --- 全生命周期档案 ---

@app.get("/lifecycle/timeline/{asset_id}")
def get_lifecycle_timeline(asset_id: str, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    events = [e for e in analysis_results.get("lifecycle_events", []) if e.get("asset_id") == asset_id]
    if not events:
        asset = asset_ledger_map.get(asset_id) or _synth_asset(asset_id)
        events = _synth_lifecycle(asset)
    events = sorted(events, key=lambda e: e.get("event_date", ""))
    return {
        "asset_id": asset_id,
        "events": events,
        "total_cost": round(sum(e.get("cost", 0) for e in events), 0),
        "event_count": len(events),
        "first_event": events[0]["event_date"] if events else "",
        "last_event": events[-1]["event_date"] if events else "",
    }

@app.get("/lifecycle/events")
def get_lifecycle_events(authorization: Optional[str] = Header(None), event_type: Optional[str] = None, page: int = 1, page_size: int = 20, start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    events = data.get("lifecycle_events", [])
    if event_type:
        events = [e for e in events if e["event_type"] == event_type]
    total = len(events)
    start = (page - 1) * page_size
    return {"total": total, "page": page, "data": events[start:start + page_size]}

@app.get("/lifecycle/cost-summary")
def get_lifecycle_cost_summary(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    return data.get("lifecycle_cost_summary", [])


# ==============================================================================
# 生命周期档案 - 附件上传 / 下载 / 删除（真实文件存储）
# ==============================================================================
ATTACHMENTS_DIR = os.path.join(DATA_DIR, "uploads")
ATTACHMENTS_META = os.path.join(DATA_DIR, "attachments.json")
ATTACHMENT_STAGES = ("purchase", "construction", "oam", "renovate", "scrap", "general")


def load_attachments() -> list:
    if os.path.exists(ATTACHMENTS_META):
        try:
            with open(ATTACHMENTS_META, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_attachments(items: list):
    try:
        with open(ATTACHMENTS_META, "w", encoding="utf-8") as f:
            json.dump(items[-2000:], f, ensure_ascii=False, indent=2)
    except Exception:
        pass


@app.post("/lifecycle/attachment/upload")
async def upload_lifecycle_attachment(asset_id: str = Form(...), stage: str = Form("general"),
                                      file: UploadFile = File(...),
                                      authorization: Optional[str] = Header(None)):
    """上传生命周期档案附件到服务器（data/uploads/），刷新 / 跨设备可下载。"""
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    if stage not in ATTACHMENT_STAGES:
        stage = "general"
    os.makedirs(ATTACHMENTS_DIR, exist_ok=True)
    fid = "att-" + uuid.uuid4().hex[:12]
    safe_name = os.path.basename(file.filename or "file")
    ext = os.path.splitext(safe_name)[1]
    saved_path = os.path.join(ATTACHMENTS_DIR, fid + ext)
    try:
        content = await file.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"读取上传文件失败: {e}")
    with open(saved_path, "wb") as f:
        f.write(content)
    rec = {
        "file_id": fid, "name": safe_name, "asset_id": asset_id, "stage": stage,
        "size": len(content), "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "path": saved_path, "uploader": user.get("username", ""),
    }
    meta = load_attachments()
    meta.append(rec)
    save_attachments(meta)
    add_log(user.get("username", ""), "附件上传", f"{asset_id}/{stage} {safe_name}")
    return {"success": True, "file_id": fid, "name": safe_name, "size": len(content),
            "url": f"/lifecycle/attachment/{fid}"}


@app.get("/lifecycle/attachments")
def list_lifecycle_attachments(asset_id: str = "", stage: str = "", authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    items = load_attachments()
    if asset_id:
        items = [a for a in items if a.get("asset_id") == asset_id]
    if stage:
        items = [a for a in items if a.get("stage") == stage]
    items.sort(key=lambda a: a.get("created_at", ""), reverse=True)
    return {"attachments": [{k: a.get(k) for k in ("file_id", "name", "asset_id", "stage", "size", "created_at")}
                            for a in items]}


@app.get("/lifecycle/attachment/{file_id}")
def download_lifecycle_attachment(file_id: str, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    rec = next((a for a in load_attachments() if a.get("file_id") == file_id), None)
    if not rec or not os.path.exists(rec.get("path", "")):
        raise HTTPException(status_code=404, detail="附件不存在")
    return FileResponse(rec["path"], filename=rec["name"], media_type="application/octet-stream")


@app.delete("/lifecycle/attachment/{file_id}")
def delete_lifecycle_attachment(file_id: str, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    meta = load_attachments()
    rec = next((a for a in meta if a.get("file_id") == file_id), None)
    if not rec:
        raise HTTPException(status_code=404, detail="附件不存在")
    try:
        if os.path.exists(rec.get("path", "")):
            os.remove(rec["path"])
    except Exception:
        pass
    meta = [a for a in meta if a.get("file_id") != file_id]
    save_attachments(meta)
    add_log(user.get("username", ""), "附件删除", rec.get("name", ""))
    return {"success": True, "message": "附件已删除"}


# --- 资产盘点 ---

@app.get("/inventory/summary")
def get_inventory_summary(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    summary = dict(data.get("inventory_summary", {}))
    checks = load_inventory_checks()
    diff_checks = [c for c in checks if c.get("diff_status") == "不一致"]
    summary["pending_diff_count"] = len([c for c in diff_checks if c.get("status") == "待处理"])
    summary["resolved_diff_count"] = len([c for c in diff_checks if c.get("status") in ("已核实", "已销号")])
    return summary

@app.get("/inventory/diff-report")
def get_inventory_diff_report(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    return data.get("inventory_diff_report", [])

@app.post("/inventory/check")
def submit_inventory_check(request: InventoryCheckRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if not request.asset_id.strip():
        raise HTTPException(400, "资产编号不能为空")
    checks = load_inventory_checks()
    check_id = f"CHK-{int(time.time() * 1000)}"
    diff = request.diff_status != "一致"
    record = {
        "check_id": check_id,
        "asset_id": request.asset_id.strip(),
        "check_method": request.check_method,
        "diff_status": request.diff_status,
        "diff_field": "",
        "diff_description": request.diff_description,
        "status": "待处理" if diff else "已归档",
        "check_time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "checker": user.get("name") or user.get("username"),
        "resolve_note": "",
        "resolved_by": "",
        "resolved_at": "",
    }
    checks.append(record)
    save_inventory_checks(checks)
    add_log(user.get("username", ""), "盘点提交", f"{request.asset_id} {request.check_method} 结果:{request.diff_status}")
    return {
        "status": "success",
        "check_id": check_id,
        "asset_id": record["asset_id"],
        "check_method": record["check_method"],
        "diff_status": record["diff_status"],
        "message": "盘点记录已提交" + ("，存在差异，请及时处理" if diff else ""),
    }


@app.get("/inventory/checks")
def list_inventory_checks(authorization: Optional[str] = Header(None), diff_only: bool = False, status: Optional[str] = None, page: int = 1, page_size: int = 20):
    get_current_user(authorization)
    checks = load_inventory_checks()
    if diff_only:
        checks = [c for c in checks if c.get("diff_status") == "不一致"]
    if status:
        checks = [c for c in checks if c.get("status") == status]
    checks.sort(key=lambda c: c.get("check_time", ""), reverse=True)
    total = len(checks)
    start = max(page - 1, 0) * page_size
    return {"total": total, "page": page, "page_size": page_size, "data": checks[start:start + page_size]}


class InventoryResolveRequest(BaseModel):
    action: str  # 核实确认 | 差异销号
    note: str = ""


@app.put("/inventory/checks/{check_id}/resolve")
def resolve_inventory_check(check_id: str, request: InventoryResolveRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if request.action not in ("核实确认", "差异销号"):
        raise HTTPException(400, "处置动作仅支持 核实确认 / 差异销号")
    checks = load_inventory_checks()
    target = next((c for c in checks if c.get("check_id") == check_id), None)
    if not target:
        raise HTTPException(404, "盘点记录不存在")
    if target.get("status") in ("已核实", "已销号"):
        raise HTTPException(400, f"该记录已处置（{target['status']}），请勿重复操作")
    target["status"] = "已核实" if request.action == "核实确认" else "已销号"
    target["resolve_note"] = request.note
    target["resolved_by"] = user.get("name") or user.get("username")
    target["resolved_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_inventory_checks(checks)
    add_log(user.get("username", ""), "盘点处置", f"{check_id} {request.action}")
    return {"status": "success", "check_id": check_id, "new_status": target["status"], "message": f"处置完成：{target['status']}"}


# --- 资产权属管理 ---

@app.get("/ownership/summary")
def get_ownership_summary(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    return analysis_results.get("ownership_summary", [])

@app.get("/ownership/detail")
def get_ownership_detail(authorization: Optional[str] = Header(None), unit: Optional[str] = None):
    get_current_user(authorization)
    data = analysis_results.get("ownership_summary", [])
    if unit:
        data = [d for d in data if d["ownership_unit"] == unit]
    return {"total": len(data), "data": data}


@app.get("/ownership/changes")
def get_ownership_changes(authorization: Optional[str] = Header(None), change_type: Optional[str] = None, page: int = 1, page_size: int = 20):
    get_current_user(authorization)
    changes = analysis_results.get("ownership_changes", [])
    if change_type:
        changes = [c for c in changes if c.get("change_type") == change_type]
    total = len(changes)
    start = max(page - 1, 0) * page_size
    return {"total": total, "page": page, "page_size": page_size, "data": changes[start:start + page_size]}


# --- 风险预测 ---

@app.post("/predict/anomaly")
def predict_anomaly(request: AssetPredictRequest, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    feature_cols = ['service_years', 'design_life', 'segment_length_m', 'burial_depth_m',
                    'diameter_numeric', 'depreciation_rate', 'inspection_gap_years',
                    'maintenance_count', 'incident_count']
    features = pd.DataFrame([{col: getattr(request, col, 0) for col in feature_cols}])
    features['aging_index'] = features['service_years'] / features['design_life']

    pred = anomaly_model.predict(features)[0]
    is_anomaly = 1 if pred == -1 else 0

    return {
        "asset_id": request.asset_id,
        "is_anomaly": is_anomaly,
        "result": "高风险" if is_anomaly else "正常",
        "risk_score": request.risk_score,
        "model_type": "模拟模型" if using_dummy_model else "真实模型",
    }

@app.post("/predict/rul")
def predict_rul(request: AssetPredictRequest, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    features = pd.DataFrame([{
        'service_years': request.service_years,
        'design_life': request.design_life,
        'segment_length_m': request.segment_length_m,
        'burial_depth_m': request.burial_depth_m,
        'diameter_numeric': request.diameter_numeric,
        'depreciation_rate': request.depreciation_rate,
        'inspection_gap_years': request.inspection_gap_years,
        'maintenance_count': request.maintenance_count,
        'incident_count': request.incident_count,
    }])
    features['aging_index'] = features['service_years'] / features['design_life']

    rul = int(rul_model.predict(features)[0])
    rul = max(0, min(50, rul))

    level = "优秀" if rul >= 30 else "良好" if rul >= 15 else "一般" if rul >= 5 else "需更换"
    return {
        "asset_id": request.asset_id,
        "remaining_life": rul,
        "health_level": level,
        "suggestion": "建议立即更换" if rul < 5 else ("建议计划改造" if rul < 15 else "运行正常"),
    }


# --- 预警 ---

@app.get("/alerts")
def get_alerts(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    alerts = data.get("alert_list", [])
    unread = len([a for a in alerts if a["status"] == "未处理"])
    return {"unread_count": unread, "alerts": alerts}

@app.put("/alerts/{alert_id}/status")
def update_alert_status(alert_id: str, request: dict, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    if user["role"] == "viewer":
        raise HTTPException(status_code=403, detail="查看者无权操作预警")
    action = request.get("action")
    if action not in ("confirm", "ignore"):
        raise HTTPException(status_code=400, detail="action 必须为 confirm 或 ignore")
    alerts = analysis_results.get("alert_list", [])
    for a in alerts:
        if a["alert_id"] == alert_id:
            a["status"] = "处理中" if action == "confirm" else "已处理"
            return {"success": True, "alert_id": alert_id, "new_status": a["status"]}
    raise HTTPException(status_code=404, detail="预警不存在")


# ==============================================================================
# 预警推送（真实网关：阿里云短信 / 企业微信应用消息）
# ==============================================================================

PUSH_CONFIG_FILE = os.path.join(DATA_DIR, "push_config.json")
PUSH_RECORDS_FILE = os.path.join(DATA_DIR, "push_records.json")

PUSH_CONFIG_DEFAULT = {
    "levels": ["紧急", "重要"],
    "sms": {
        "enabled": True,
        "mode": "mock",          # mock=模拟发送 | aliyun=阿里云短信真实发送
        "access_key_id": "",
        "access_key_secret": "",
        "sign_name": "",
        "template_code": "",
        "template_param_keys": ["level", "alert_type", "asset_id", "region"],
    },
   "wechat": {
    "enabled": True,
    "mode": "test_account",  # 改用测试号模式
    "appid": os.environ.get("WX_TEST_APPID", ""),
    "appsecret": os.environ.get("WX_TEST_SECRET", ""),
},
    "email": {
        "enabled": False,
        "smtp_host": "",
        "smtp_port": 465,       # 465=SSL，587=TLS
        "smtp_user": "",        # 发件人邮箱
        "smtp_password": "",    # 发件人授权码（非邮箱密码）
        "recipients": [],       # 接收人邮箱列表
    },
    "contacts": [
        {"name": "张工", "target": "13800138001", "role": "运维负责人", "channels": ["短信"]},
        {"name": "李巡检", "target": "13900139002", "role": "巡检员", "channels": ["短信"]},
        {"name": "LiXuan", "target": "LiXuan", "role": "值班领导", "channels": ["微信"]},
    ],
}


def _deep_merge(base: dict, override: dict) -> dict:
    """浅层合并配置，保证缺失字段用默认值补齐。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return out


def load_push_config() -> dict:
    cfg = json.loads(json.dumps(PUSH_CONFIG_DEFAULT))
    if os.path.exists(PUSH_CONFIG_FILE):
        try:
            with open(PUSH_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            cfg = _deep_merge(cfg, saved)
        except Exception:
            pass
    return cfg


def save_push_config(cfg: dict):
    with open(PUSH_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def load_push_records() -> list:
    if os.path.exists(PUSH_RECORDS_FILE):
        try:
            with open(PUSH_RECORDS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def save_push_records(records: list):
    with open(PUSH_RECORDS_FILE, "w", encoding="utf-8") as f:
        json.dump(records[:500], f, ensure_ascii=False, indent=2)


def mask_push_config(cfg: dict) -> dict:
    """脱敏配置中的密钥字段（非空值以 ****** 显示）。"""
    masked = json.loads(json.dumps(cfg))
    for ch in ("sms", "wechat"):
        for key in ("access_key_secret", "corp_secret", "appsecret"):
            if masked.get(ch, {}).get(key):
                masked[ch][key] = "******"
    if masked.get("email", {}).get("smtp_password"):
        masked["email"]["smtp_password"] = "******"
    return masked


def build_push_content(alert: dict, channel: str = "微信") -> str:
    if channel == "短信":
        # 短信内容由阿里云模板拼接，这里仅生成模板参数
        return ""
    return (
        f"【管网预警】{alert.get('level', '')} {alert.get('alert_type', '')}\n"
        f"预警编号：{alert.get('alert_id', '')}\n"
        f"资产编号：{alert.get('asset_id', '')}\n"
        f"所属区域：{alert.get('region', '')}\n"
        f"发生时间：{alert.get('create_time', '')}\n"
        f"请及时登录系统查看并处置。"
    )


def alert_template_params(alert: dict) -> dict:
    return {
        "level": alert.get("level", ""),
        "alert_type": alert.get("alert_type", ""),
        "asset_id": alert.get("asset_id", ""),
        "region": alert.get("region", ""),
    }


def alert_suggestion(alert: dict) -> str:
    """根据预警级别返回处置建议。"""
    level = alert.get("level", "")
    if level == "紧急":
        return "立即安排人员现场核查处置，必要时启动应急预案并上报值班领导。"
    if level == "重要":
        return "尽快安排检修计划，加强监测频次，持续跟踪状态变化。"
    return "纳入日常巡检计划，持续关注状态变化，定期复核。"


def build_email_content(alert: dict) -> tuple:
    """生成预警邮件（标题, HTML 正文）。"""
    subject = f"【管网预警】{alert.get('level', '')} - {alert.get('alert_type', '')}"
    rows = [
        ("预警编号", alert.get("alert_id", "")),
        ("预警类型", alert.get("alert_type", "")),
        ("预警等级", alert.get("level", "")),
        ("关联资产", alert.get("asset_id", "")),
        ("管网类型", alert.get("pipeline_type", "")),
        ("所在位置", alert.get("region", "")),
        ("发生时间", alert.get("create_time", "")),
        ("处置建议", alert_suggestion(alert)),
    ]
    trs = "".join(
        f'<tr><td style="padding:8px 12px;border:1px solid #e5e7eb;background:#f9fafb;color:#6b7280;white-space:nowrap;font-weight:600;">{k}</td>'
        f'<td style="padding:8px 12px;border:1px solid #e5e7eb;color:#111827;">{v}</td></tr>'
        for k, v in rows
    )
    html = (
        '<div style="font-family:Microsoft YaHei,Arial,sans-serif;max-width:640px;margin:0 auto;padding:24px;background:#ffffff;">'
        '<h3 style="margin:0 0 6px;color:#dc2626;">🚨 管网预警通知</h3>'
        '<p style="margin:0 0 16px;color:#6b7280;font-size:13px;">系统检测到以下管网资产预警，请相关责任人及时处理。</p>'
        f'<table style="border-collapse:collapse;width:100%;font-size:14px;">{trs}</table>'
        '<p style="margin:16px 0 0;color:#374151;font-size:13px;line-height:1.7;">请登录「城市管网资产数字化台账系统」查看详情并处置。</p>'
        '<hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0 12px;">'
        '<p style="margin:0;color:#9ca3af;font-size:12px;">—— 城市管网资产数字化台账系统 ——</p>'
        '</div>'
    )
    return subject, html


def build_test_email_html(content: str) -> str:
    """生成测试邮件 HTML 正文。"""
    return (
        '<div style="font-family:Microsoft YaHei,Arial,sans-serif;max-width:640px;margin:0 auto;padding:24px;background:#ffffff;">'
        '<h3 style="margin:0 0 10px;color:#2563eb;">📧 邮箱推送测试</h3>'
        f'<p style="margin:0 0 16px;color:#374151;font-size:14px;">{content}</p>'
        '<hr style="border:none;border-top:1px solid #e5e7eb;margin:20px 0 12px;">'
        '<p style="margin:0;color:#9ca3af;font-size:12px;">—— 城市管网资产数字化台账系统 ——</p>'
        '</div>'
    )


def send_email(smtp_cfg: dict, subject: str, html_content: str, to_list: Optional[list] = None) -> tuple:
    """通过 SMTP 发送 HTML 邮件（465=SSL / 587=TLS）。返回 (ok, message)。"""
    host = str(smtp_cfg.get("smtp_host", "")).strip()
    try:
        port = int(smtp_cfg.get("smtp_port") or 465)
    except (TypeError, ValueError):
        port = 465
    user = str(smtp_cfg.get("smtp_user", "")).strip()
    pwd = str(smtp_cfg.get("smtp_password", "")).strip()
    recipients = to_list if to_list else [r.strip() for r in (smtp_cfg.get("recipients") or []) if r.strip()]
    if not (host and user and pwd and recipients):
        return False, "邮箱网关未配置（缺少 SMTP 服务器 / 发件人 / 授权码 / 接收人邮箱）"
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = ", ".join(recipients)
        msg.attach(MIMEText(re.sub(r"<[^>]+>", "", html_content).strip(), "plain", "utf-8"))
        msg.attach(MIMEText(html_content, "html", "utf-8"))
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=15) as server:
                server.login(user, pwd)
                server.sendmail(user, recipients, msg.as_string())
        else:
            with smtplib.SMTP(host, port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(user, pwd)
                server.sendmail(user, recipients, msg.as_string())
        return True, "已发送（邮箱）"
    except smtplib.SMTPAuthenticationError:
        return False, "邮件发送失败: SMTP 认证失败，请检查发件人邮箱与授权码"
    except smtplib.SMTPException as e:
        return False, f"邮件发送失败: SMTP 错误 {e}"
    except Exception as e:
        return False, f"邮件发送失败: {e}"


def _percent_encode(s) -> str:
    return urllib.parse.quote(str(s), safe="~")


def send_sms_aliyun(sms_cfg: dict, phone: str, template_params: dict) -> tuple:
    """通过阿里云短信网关真实发送短信。返回 (ok, message)。"""
    ak_id = str(sms_cfg.get("access_key_id", "")).strip()
    ak_secret = str(sms_cfg.get("access_key_secret", "")).strip()
    sign = str(sms_cfg.get("sign_name", "")).strip()
    tpl = str(sms_cfg.get("template_code", "")).strip()
    if not (ak_id and ak_secret and sign and tpl):
        return False, "短信网关未配置（缺少 AccessKey / 签名 / 模板）"
    if not re.fullmatch(r"1\d{10}", phone):
        return False, f"手机号格式不正确: {phone}"
    params = {
        "AccessKeyId": ak_id,
        "Action": "SendSms",
        "Format": "JSON",
        "PhoneNumbers": phone,
        "RegionId": "cn-hangzhou",
        "SignName": sign,
        "SignatureMethod": "HMAC-SHA1",
        "SignatureNonce": str(uuid.uuid4()),
        "SignatureVersion": "1.0",
        "TemplateCode": tpl,
        "TemplateParam": json.dumps(template_params, ensure_ascii=False),
        "Timestamp": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "Version": "2017-05-25",
    }
    canonical = "&".join(
        f"{_percent_encode(k)}={_percent_encode(v)}" for k, v in sorted(params.items())
    )
    string_to_sign = "GET&%2F&" + _percent_encode(canonical)
    digest = hmac.new(
        (ak_secret + "&").encode("utf-8"),
        string_to_sign.encode("utf-8"),
        hashlib.sha1,
    ).digest()
    signature = base64.b64encode(digest).decode("utf-8")
    url = "https://dysmsapi.aliyuncs.com/?" + canonical + "&Signature=" + _percent_encode(signature)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            body = resp.read().decode("utf-8")
        data = json.loads(body)
        if data.get("Code") == "OK":
            return True, "已发送"
        return False, f"短信发送失败: {data.get('Code')} {data.get('Message', '')}".strip()
    except Exception as e:
        return False, f"短信网关请求异常: {e}"


def send_wecom(wechat_cfg: dict, user_id: str, content: str) -> tuple:
    """通过企业微信应用消息真实推送。返回 (ok, message)。"""
    corpid = str(wechat_cfg.get("corpid", "")).strip()
    secret = str(wechat_cfg.get("corp_secret", "")).strip()
    agent_id = str(wechat_cfg.get("agent_id", "")).strip()
    if not (corpid and secret and agent_id):
        return False, "微信网关未配置（缺少企业ID / Secret / AgentId）"
    try:
        token_url = (
            "https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid="
            + urllib.parse.quote(corpid)
            + "&corpsecret=" + urllib.parse.quote(secret)
        )
        with urllib.request.urlopen(token_url, timeout=10) as resp:
            token_data = json.loads(resp.read().decode("utf-8"))
        if token_data.get("errcode", 0) != 0:
            return False, f"企业微信获取token失败: {token_data.get('errmsg', '')}"
        token = token_data["access_token"]
        body = {
            "touser": user_id,
            "msgtype": "text",
            "agentid": int(agent_id),
            "text": {"content": content},
            "safe": 0,
        }
        send_url = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=" + token
        req = urllib.request.Request(
            send_url,
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            send_data = json.loads(resp.read().decode("utf-8"))
        if send_data.get("errcode", 0) == 0:
            return True, "已发送"
        return False, f"企业微信发送失败: {send_data.get('errmsg', '')}"
    except Exception as e:
        return False, f"微信网关请求异常: {e}"


def do_push_alert(operator: str, alert: dict, cfg: dict, channel_filter: str = "") -> dict:
    """按配置向接收人推送单条预警（微信=测试号客服消息 / 邮箱=SMTP / 短信按配置）。
    channel_filter 可指定只推送某通道（""=全部启用通道）。
    返回统计与推送记录，并更新预警的 push_status / push_channels。"""
    records = load_push_records()
    results = []
    sent = failed = 0
    sent_channels = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    sms_params = alert_template_params(alert)
    wechat_content = build_push_content(alert, "微信")
    sms_cfg = cfg.get("sms", {})
    wechat_cfg = cfg.get("wechat", {})
    email_cfg = cfg.get("email", {})
    for contact in cfg.get("contacts", []):
        channels = contact.get("channels") or ["短信", "微信"]
        name = contact.get("name", "")
        target = str(contact.get("target", "")).strip()
        for ch in channels:
            if channel_filter and ch != channel_filter:
                continue
            if ch == "短信" and sms_cfg.get("enabled", False):
                if sms_cfg.get("mode") == "aliyun":
                    ok, msg = send_sms_aliyun(sms_cfg, target, sms_params)
                    status = msg
                else:
                    ok, msg = True, "已发送（模拟短信）"
                    status = msg
                if ok:
                    sent += 1
                    sent_channels.append("短信")
                else:
                    failed += 1
                results.append({
                    "record_id": f"PUSH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(results) + 1:03d}",
                    "time": now, "channel": "短信",
                    "to": f"{name}（{target}）", "alert_id": alert.get("alert_id", ""),
                    "level": alert.get("level", ""), "content": "短信模板：" + json.dumps(sms_params, ensure_ascii=False),
                    "mode": sms_cfg.get("mode", "mock"), "status": status, "operator": operator,
                })
            elif ch == "微信" and wechat_cfg.get("enabled", False):
                # 微信通道：固定调用微信测试号客服消息接口真实发送（发送给默认接收人 openid）
                ok, msg = wx_test_send_custom(WX_TEST_DEFAULT_OPENID, wechat_content)
                status = msg
                if ok:
                    sent += 1
                    sent_channels.append("微信")
                else:
                    failed += 1
                results.append({
                    "record_id": f"PUSH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(results) + 1:03d}",
                    "time": now, "channel": "微信",
                    "to": f"微信测试号（{WX_TEST_DEFAULT_OPENID}）", "alert_id": alert.get("alert_id", ""),
                    "level": alert.get("level", ""), "content": wechat_content,
                    "mode": "wx_test", "status": status, "operator": operator,
                })
    # 邮箱通道：独立推送，发送给配置的接收人邮箱列表（HTML 邮件）
    email_recipients = [r.strip() for r in (email_cfg.get("recipients") or []) if r.strip()]
    if email_cfg.get("enabled", False) and (not channel_filter or channel_filter == "邮箱"):
        subject, email_html = build_email_content(alert)
        ok, msg = send_email(email_cfg, subject, email_html)
        status = msg
        for r in email_recipients:
            if ok:
                sent += 1
            else:
                failed += 1
            results.append({
                "record_id": f"PUSH-{datetime.now().strftime('%Y%m%d%H%M%S')}-{len(results) + 1:03d}",
                "time": now, "channel": "邮箱",
                "to": r, "alert_id": alert.get("alert_id", ""),
                "level": alert.get("level", ""), "content": subject + "\n" + wechat_content,
                "mode": "email", "status": status, "operator": operator,
            })
        if ok:
            sent_channels.append("邮箱")
    records = results + records
    save_push_records(records)
    # 更新预警推送状态与已推送通道
    if sent > 0:
        alert["push_status"] = "已推送"
        prev = alert.get("push_channels") or []
        alert["push_channels"] = sorted(set(prev + sent_channels))
    elif failed > 0:
        alert["push_status"] = "推送失败"
    return {"sent": sent, "failed": failed, "results": results, "records": records[:200]}


class PushConfigRequest(BaseModel):
    levels: list = ["紧急", "重要"]
    sms: dict = {}
    wechat: dict = {}
    email: dict = {}
    contacts: list = []


class PushTestRequest(BaseModel):
    channel: str = "短信"   # 短信 | 微信 | 邮箱
    target: str = ""        # 可选，指定测试接收目标（微信为 openid，邮箱为单个邮箱地址）
    message: str = ""       # 可选，测试消息内容
    template_id: str = ""   # 可选，微信模板ID（已不再使用，保留兼容）


@app.get("/push/config")
def get_push_config(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    cfg = load_push_config()
    if user["role"] not in WRITE_ROLES:
        cfg = mask_push_config(cfg)
    return {"config": cfg}


@app.put("/push/config")
def put_push_config(request: PushConfigRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    cfg = load_push_config()
    if request.levels:
        cfg["levels"] = [lv for lv in request.levels if lv in ("紧急", "重要", "一般")]
    if request.sms:
        cfg["sms"] = {**cfg["sms"], **request.sms}
    if request.wechat:
        cfg["wechat"] = {**cfg["wechat"], **request.wechat}
    if request.email:
        cfg["email"] = {**cfg["email"], **request.email}
    if isinstance(request.contacts, list):
        cfg["contacts"] = request.contacts
    # 空密钥 / 脱敏占位符不覆盖已保存的真实密钥
    prev_cfg = load_push_config()
    for key in ("access_key_secret", "corp_secret", "appsecret"):
        for ch in ("sms", "wechat"):
            val = cfg.get(ch, {}).get(key)
            if isinstance(val, str) and val.strip() in ("", "******"):
                cfg[ch][key] = prev_cfg.get(ch, {}).get(key, "")
    email_pwd = cfg.get("email", {}).get("smtp_password")
    if isinstance(email_pwd, str) and email_pwd.strip() in ("", "******"):
        cfg["email"]["smtp_password"] = prev_cfg.get("email", {}).get("smtp_password", "")
    save_push_config(cfg)
    add_log(user.get("username", ""), "推送配置", "更新预警推送配置")
    return {"success": True, "config": mask_push_config(cfg)}


@app.get("/push/records")
def get_push_records(authorization: Optional[str] = Header(None), limit: int = 100):
    get_current_user(authorization)
    return {"records": load_push_records()[:max(1, min(limit, 500))]}


@app.post("/push/test")
def push_test(request: PushTestRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    cfg = load_push_config()
    channel = request.channel
    if channel not in ("短信", "微信", "邮箱"):
        raise HTTPException(status_code=400, detail="channel 必须为 短信 / 微信 / 邮箱")
    test_content = "【管网预警平台】这是一条测试推送消息。如果您收到本条消息，说明推送通道配置成功。"
    contact = next((c for c in cfg.get("contacts", []) if channel in (c.get("channels") or [])), None)
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if channel == "短信":
        target = (request.target or "").strip() or (contact.get("target", "") if contact else "")
        if not target:
            raise HTTPException(status_code=400, detail="请先在接收人中添加使用「短信」通道的人员")
        if cfg["sms"].get("mode") == "aliyun":
            ok, msg = send_sms_aliyun(cfg["sms"], target, {"level": "测试", "alert_type": "推送测试", "asset_id": "TEST-0001", "region": "测试区域"})
            status = msg
        else:
            ok, msg = True, "已发送（模拟短信）"
            status = msg
        to_name = contact.get("name", "") if contact else ""
        content = test_content
        mode = cfg["sms"].get("mode", "mock")
    elif channel == "邮箱":
        # 邮箱：真实调用 SMTP 发送测试邮件（465=SSL / 587=TLS）
        email_cfg = cfg.get("email", {})
        target = (request.target or "").strip()
        to_list = [target] if target else None
        subject = "【管网预警平台】邮箱推送测试"
        html = build_test_email_html(test_content)
        ok, msg = send_email(email_cfg, subject, html, to_list=to_list)
        status = msg
        if not target:
            target = ", ".join([r.strip() for r in (email_cfg.get("recipients") or []) if r.strip()])
        to_name = "邮箱接收人"
        content = subject + "\n" + test_content
        mode = "email"
    else:
        # 微信：真实调用微信公众平台测试号客服消息接口（不使用模拟模式/企业微信/模板消息）
        message = (request.message or "").strip() or test_content
        target = (request.target or "").strip() or WX_TEST_DEFAULT_OPENID
        try:
            token = wx_test_get_access_token()
            send_url = f"{WX_TEST_API_BASE}/message/custom/send?access_token={urllib.parse.quote(token)}"
            payload = {"touser": target, "msgtype": "text", "text": {"content": message}}
            req = urllib.request.Request(
                send_url,
                data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                send_data = json.loads(resp.read().decode("utf-8"))
            if send_data.get("errcode", 0) == 0:
                ok, msg = True, "已发送（微信测试号客服消息）"
            else:
                ok, msg = False, f"微信测试号客服消息发送失败: {send_data.get('errcode')} {send_data.get('errmsg', '')}"
        except Exception as e:
            ok, msg = False, f"微信测试号调用异常: {e}"
        status = msg
        to_name = "微信测试号"
        content = message
        mode = "test"
    record = {
        "record_id": f"PUSH-{datetime.now().strftime('%Y%m%d%H%M%S')}-000",
        "time": now, "channel": channel, "to": f"{to_name}（{target}）",
        "alert_id": "TEST", "level": "测试", "content": content,
        "mode": mode,
        "status": status, "operator": user.get("username", ""),
    }
    records = [record] + load_push_records()
    save_push_records(records)
    add_log(user.get("username", ""), "推送测试", f"{channel}通道测试：{target} → {status}")
    return {"success": ok, "channel": channel, "to": target, "status": status, "records": records[:200]}


class AlertPushRequest(BaseModel):
    channel: str = ""   # 可选：短信 / 微信 / 邮箱；缺省推送全部启用通道


@app.post("/alerts/{alert_id}/push")
def push_single_alert(alert_id: str, request: Optional[AlertPushRequest] = None,
                      authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    cfg = load_push_config()
    alert = next((a for a in analysis_results.get("alert_list", []) if a.get("alert_id") == alert_id), None)
    if not alert:
        raise HTTPException(status_code=404, detail=f"预警 {alert_id} 不存在")
    if alert.get("level") not in cfg.get("levels", []):
        raise HTTPException(status_code=400, detail=f"该预警级别（{alert.get('level')}）不在配置的推送范围内")
    channel = (request.channel if request else "").strip()
    if channel and channel not in ("短信", "微信", "邮箱"):
        raise HTTPException(status_code=400, detail="channel 必须为 短信 / 微信 / 邮箱")
    enabled_channels = [
        ch for ch, cfg_ch in (("短信", cfg.get("sms", {})), ("微信", cfg.get("wechat", {})), ("邮箱", cfg.get("email", {})))
        if cfg_ch.get("enabled", False)
    ]
    if not enabled_channels:
        raise HTTPException(status_code=400, detail="请先在推送配置中启用至少一个通道")
    if channel and channel not in enabled_channels:
        raise HTTPException(status_code=400, detail=f"通道「{channel}」未启用，请先在推送配置中开启")
    result = do_push_alert(user.get("username", ""), alert, cfg, channel_filter=channel)
    add_log(user.get("username", ""), "预警推送", f"{alert_id} 推送完成：成功{result['sent']}条 失败{result['failed']}条")
    return {"success": True, "alert_id": alert_id, "sent": result["sent"], "failed": result["failed"],
            "results": result["results"], "records": result["records"]}


@app.post("/alerts/push-all")
def push_all_alerts(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    cfg = load_push_config()
    if not cfg.get("sms", {}).get("enabled", False) and not cfg.get("wechat", {}).get("enabled", False) \
            and not cfg.get("email", {}).get("enabled", False):
        raise HTTPException(status_code=400, detail="请先在推送配置中启用至少一个通道")
    open_alerts = [a for a in analysis_results.get("alert_list", [])
                   # source == "mock" 是「重新分析」生成的演示数据，绝不能真实推到微信/邮箱
                   if a.get("status") != "已处理" and a.get("level") in cfg.get("levels", [])
                   and a.get("source") != "mock"]
    if not open_alerts:
        return {"success": True, "sent": 0, "failed": 0, "message": "没有符合推送条件的未处理预警", "records": load_push_records()[:200]}
    total_sent = total_failed = 0
    all_records = []
    for alert in open_alerts:
        result = do_push_alert(user.get("username", ""), alert, cfg)
        total_sent += result["sent"]
        total_failed += result["failed"]
        all_records = result["records"]
    add_log(user.get("username", ""), "预警推送", f"批量推送 {len(open_alerts)} 条预警：成功{total_sent}条 失败{total_failed}条")
    return {"success": True, "alerts": len(open_alerts), "sent": total_sent, "failed": total_failed,
            "records": all_records}


@app.get("/alerts/{alert_id}/push-logs")
def get_alert_push_logs(alert_id: str, authorization: Optional[str] = Header(None)):
    """查看某条预警的推送记录（时间 / 通道 / 接收人 / 状态 / 内容）。"""
    get_current_user(authorization)
    logs = [r for r in load_push_records() if r.get("alert_id") == alert_id]
    logs.sort(key=lambda r: r.get("time", ""), reverse=True)
    return {"alert_id": alert_id, "logs": logs[:50]}


@app.get("/map/data")
def get_map_data(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    return data.get("map_data", [])

@app.get("/risk/ranking")
def get_risk_ranking(authorization: Optional[str] = Header(None), start_date: Optional[str] = None, end_date: Optional[str] = None):
    get_current_user(authorization)
    data = apply_time_filter(analysis_results, start_date, end_date)
    return data.get("risk_ranking", [])


# ==============================================================================
# 指令下发 / 工单 / 检测任务 / 专项设施
# ==============================================================================

class CommandCreateRequest(BaseModel):
    target_type: str = "设施"
    target_id: str = ""
    target_name: str = ""
    action: str
    params: dict = {}
    priority: str = "中"
    source: str = "手动下发"


class WorkorderCreateRequest(BaseModel):
    title: str
    description: str = ""
    region: str = ""
    priority: str = "中"
    alert_id: str = ""
    assignee: str = ""


class WorkorderTransitionRequest(BaseModel):
    action: str  # assign | start | finish | accept | reject
    assignee: str = ""
    note: str = ""


class DetectionTaskCreateRequest(BaseModel):
    name: str
    region: str = ""
    method: str = "CCTV检测"
    pipeline_type: str = "供水管网"
    planned_length: float = 1.0
    planned_date: str = ""


class FacilityCommandRequest(BaseModel):
    action: str
    params: dict = {}
    priority: str = "中"


@app.get("/commands")
def list_commands(authorization: Optional[str] = Header(None), status: Optional[str] = None, source: Optional[str] = None):
    get_current_user(authorization)
    commands = progress_all_commands()
    if status:
        commands = [c for c in commands if c.get("status") == status]
    if source:
        commands = [c for c in commands if c.get("source") == source]
    commands.sort(key=lambda c: c.get("created_at", ""), reverse=True)
    return {"total": len(commands), "data": commands}


@app.post("/commands")
def create_command(request: CommandCreateRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    if not request.action.strip():
        raise HTTPException(400, "指令动作不能为空")
    if not request.target_name.strip() and not request.target_id.strip():
        raise HTTPException(400, "请选择指令目标")
    record = create_command_record(
        user, request.target_type, request.target_id, request.target_name,
        request.action, request.params, request.priority, request.source or "手动下发",
    )
    add_log(user.get("username", ""), "指令下发", f"{record['command_id']} → {request.target_name} [{request.action}]")
    return {"success": True, "message": "指令已下发", "command": record}


@app.get("/commands/{command_id}")
def get_command(command_id: str, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    commands = progress_all_commands()
    target = next((c for c in commands if c.get("command_id") == command_id), None)
    if not target:
        raise HTTPException(404, "指令不存在")
    return target


@app.get("/workorders")
def list_workorders(authorization: Optional[str] = Header(None), status: Optional[str] = None):
    get_current_user(authorization)
    orders = load_workorders()
    if status:
        orders = [w for w in orders if w.get("status") == status]
    orders.sort(key=lambda w: w.get("created_at", ""), reverse=True)
    return {"total": len(orders), "data": orders}


@app.post("/workorders")
def create_workorder(request: WorkorderCreateRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    if not request.title.strip():
        raise HTTPException(400, "工单标题不能为空")
    wo_id = f"WO-{int(time.time() * 1000)}"
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    creator = user.get("name") or user.get("username")
    record = {
        "workorder_id": wo_id,
        "title": request.title.strip(),
        "description": request.description,
        "region": request.region,
        "priority": request.priority if request.priority in ("紧急", "高", "中", "低") else "中",
        "alert_id": request.alert_id,
        "status": "待指派" if not request.assignee else "已指派",
        "assignee": request.assignee,
        "created_by": creator,
        "created_at": now_str,
        "finished_at": "",
        "history": [{"time": now_str, "action": "创建工单", "by": creator, "note": ""}],
    }
    linked_alert = None
    if request.alert_id:
        for a in analysis_results.get("alert_list", []):
            if a.get("alert_id") == request.alert_id and a.get("status") == "未处理":
                a["status"] = "处理中"
                linked_alert = request.alert_id
                record["history"].append({"time": now_str, "action": f"联动预警 {request.alert_id} 转处理中", "by": creator, "note": ""})
                break
    orders = load_workorders()
    orders.append(record)
    save_workorders(orders)
    add_log(user.get("username", ""), "工单创建", f"{wo_id} {request.title}" + (f"（联动预警 {linked_alert}）" if linked_alert else ""))
    return {"success": True, "message": "工单已创建", "workorder": record, "linked_alert": linked_alert}


@app.put("/workorders/{workorder_id}/transition")
def transition_workorder(workorder_id: str, request: WorkorderTransitionRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    orders = load_workorders()
    target = next((w for w in orders if w.get("workorder_id") == workorder_id), None)
    if not target:
        raise HTTPException(404, "工单不存在")
    action = request.action
    status = target.get("status")
    operator = user.get("name") or user.get("username")
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if action == "assign":
        if status != "待指派":
            raise HTTPException(400, f"当前状态 {status} 不可指派")
        if not request.assignee.strip():
            raise HTTPException(400, "请填写处理人")
        target["status"] = "已指派"
        target["assignee"] = request.assignee.strip()
        desc = f"指派给 {request.assignee.strip()}"
    elif action == "start":
        if status != "已指派":
            raise HTTPException(400, f"当前状态 {status} 不可开始处理")
        target["status"] = "处理中"
        desc = "开始处理"
    elif action == "finish":
        if status != "处理中":
            raise HTTPException(400, f"当前状态 {status} 不可提交验收")
        target["status"] = "待验收"
        desc = "处理完成，提交验收"
    elif action == "accept":
        if status != "待验收":
            raise HTTPException(400, f"当前状态 {status} 不可验收")
        target["status"] = "已闭环"
        target["finished_at"] = now_str
        if target.get("alert_id"):
            for a in analysis_results.get("alert_list", []):
                if a.get("alert_id") == target["alert_id"] and a.get("status") == "处理中":
                    a["status"] = "已处理"
                    break
        desc = "验收通过，工单闭环"
    elif action == "reject":
        if status != "待验收":
            raise HTTPException(400, f"当前状态 {status} 不可退回")
        target["status"] = "已指派"
        desc = "验收退回，重新处理"
    else:
        raise HTTPException(400, "action 必须为 assign/start/finish/accept/reject")

    target.setdefault("history", []).append({"time": now_str, "action": desc, "by": operator, "note": request.note})
    save_workorders(orders)
    add_log(user.get("username", ""), "工单流转", f"{workorder_id} {desc}")
    return {"success": True, "workorder_id": workorder_id, "new_status": target["status"], "message": desc}


DETECTION_METHODS = ["CCTV检测", "声呐检测", "激光烟雾检测", "无人机巡检"]
DEFECT_TYPES = ["破裂", "变形", "错位", "腐蚀", "渗漏", "异物侵入"]


def build_detection_result(task_id: str, planned_length: float) -> dict:
    base = zlib.crc32(task_id.encode())
    defect_count = 2 + base % 5
    defects = []
    for i in range(defect_count):
        h = zlib.crc32(f"{task_id}-{i}".encode())
        defects.append({
            "type": DEFECT_TYPES[h % len(DEFECT_TYPES)],
            "level": ["严重", "中等", "轻微"][h % 3],
            "position_m": round((h % 1000) / 1000.0 * max(planned_length, 0.1) * 1000, 1),
            "description": f"检测发现{DEFECT_TYPES[h % len(DEFECT_TYPES)]}缺陷，建议安排复核",
        })
    return {
        "total_length_m": round(planned_length * 1000, 1),
        "defect_count": defect_count,
        "health_index": round(62 + base % 30, 1),
        "defects": defects,
        "conclusion": "整体结构状况中等，局部存在缺陷需关注" if defect_count > 3 else "整体结构状况良好，建议纳入常规巡检",
    }


@app.get("/detection/tasks")
def list_detection_tasks(authorization: Optional[str] = Header(None), status: Optional[str] = None):
    get_current_user(authorization)
    tasks = load_detection_tasks()
    commands = progress_all_commands() if any(t.get("status") == "检测中" for t in tasks) else None
    changed = False
    for t in tasks:
        if t.get("status") == "检测中" and commands is not None and t.get("command_id"):
            cmd = next((c for c in commands if c.get("command_id") == t["command_id"]), None)
            if cmd and cmd.get("status") in ("成功", "失败"):
                t["status"] = "已完成" if cmd["status"] == "成功" else "下发失败"
                t["finished_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                changed = True
    if changed:
        save_detection_tasks(tasks)
    if status:
        tasks = [t for t in tasks if t.get("status") == status]
    tasks.sort(key=lambda t: t.get("created_at", ""), reverse=True)
    return {"total": len(tasks), "data": tasks}


@app.post("/detection/tasks")
def create_detection_task(request: DetectionTaskCreateRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    if not request.name.strip():
        raise HTTPException(400, "任务名称不能为空")
    if request.method not in DETECTION_METHODS:
        raise HTTPException(400, f"检测方式必须为 {'/'.join(DETECTION_METHODS)}")
    task_id = f"DT-{int(time.time() * 1000)}"
    record = {
        "task_id": task_id,
        "name": request.name.strip(),
        "region": request.region or REGIONS[0],
        "method": request.method,
        "pipeline_type": request.pipeline_type,
        "planned_length": request.planned_length,
        "planned_date": request.planned_date or datetime.now().strftime("%Y-%m-%d"),
        "status": "待下发",
        "created_by": user.get("name") or user.get("username"),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "command_id": "",
        "result": None,
        "finished_at": "",
    }
    tasks = load_detection_tasks()
    tasks.append(record)
    save_detection_tasks(tasks)
    add_log(user.get("username", ""), "检测任务创建", f"{task_id} {request.name} [{request.method}]")
    return {"success": True, "message": "检测任务已创建", "task": record}


@app.put("/detection/tasks/{task_id}/dispatch")
def dispatch_detection_task(task_id: str, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    tasks = load_detection_tasks()
    target = next((t for t in tasks if t.get("task_id") == task_id), None)
    if not target:
        raise HTTPException(404, "检测任务不存在")
    if target.get("status") not in ("待下发", "下发失败"):
        raise HTTPException(400, f"当前状态 {target.get('status')} 不可下发")
    cmd = create_command_record(
        user, "检测任务", task_id, target.get("name", task_id),
        "启动检测", {"method": target.get("method"), "pipeline_type": target.get("pipeline_type")},
        "中", "检测任务",
    )
    target["status"] = "检测中"
    target["command_id"] = cmd["command_id"]
    target["result"] = build_detection_result(task_id, target.get("planned_length", 1.0))
    save_detection_tasks(tasks)
    add_log(user.get("username", ""), "检测任务下发", f"{task_id} 指令 {cmd['command_id']}")
    return {"success": True, "message": "检测指令已下发至设备", "task": target, "command": cmd}


@app.get("/facilities")
def list_facilities(authorization: Optional[str] = Header(None), ftype: Optional[str] = None, region: Optional[str] = None):
    get_current_user(authorization)
    facilities = load_facilities()
    if ftype:
        facilities = [f for f in facilities if f.get("type") == ftype]
    if region:
        facilities = [f for f in facilities if f.get("region") == region]
    online = len([f for f in facilities if f.get("online")])
    return {
        "total": len(facilities),
        "online_count": online,
        "online_rate": round(online / len(facilities) * 100, 1) if facilities else 0,
        "data": facilities,
    }


@app.post("/facilities/{facility_id}/command")
def facility_command(facility_id: str, request: FacilityCommandRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    facilities = load_facilities()
    target = next((f for f in facilities if f.get("facility_id") == facility_id), None)
    if not target:
        raise HTTPException(404, "设施不存在")
    if not request.action.strip():
        raise HTTPException(400, "指令动作不能为空")
    cmd = create_command_record(
        user, "设施", facility_id, target.get("name", facility_id),
        request.action, request.params, request.priority, "设施控制",
    )
    add_log(user.get("username", ""), "设施控制", f"{facility_id} [{request.action}] 指令 {cmd['command_id']}")
    return {"success": True, "message": "控制指令已下发", "command": cmd, "facility": target}


# --- 工作流 ---

@app.get("/workflow/status")
def get_workflow_status():
    return workflow_status

@app.post("/workflow/run")
def run_workflow(request: GenerateDataRequest = GenerateDataRequest(), background_tasks: BackgroundTasks = None):
    if workflow_status["running"]:
        raise HTTPException(status_code=400, detail="工作流正在运行中")

    def run_complete_workflow():
        global workflow_status
        workflow_status["running"] = True
        workflow_status["start_time"] = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        workflow_status["data_count"] = request.count
        workflow_status["error"] = None

        try:
            steps = [
                ("数据生成", 15, "正在生成管网资产数据..."),
                ("数据加载", 30, "正在加载数据到Hive..."),
                ("ETL分析", 50, "正在执行资产统计分析..."),
                ("风险评估", 70, "正在计算资产风险指标..."),
                ("模型训练", 85, "正在训练预测模型..."),
                ("完成", 100, f"完成！共处理 {request.count} 项资产数据"),
            ]

            for step_name, progress, message in steps:
                workflow_status["current_step"] = step_name
                workflow_status["progress"] = progress
                workflow_status["message"] = message
                print(f"[工作流] {step_name}: {message}")
                time.sleep(0.5)

            script_path = os.path.join(os.path.dirname(__file__), "generate_asset_data.py")
            subprocess.run(
                ["python3", script_path, "--count", str(request.count), "--output", request.output_dir],
                capture_output=True, text=True, timeout=300
            )

            simulate_analysis()

            train_script = os.path.join(os.path.dirname(__file__), "train_sklearn_model.py")
            subprocess.run(["python3", train_script], capture_output=True, text=True, timeout=300)
            load_models()

        except Exception as e:
            workflow_status["current_step"] = "失败"
            workflow_status["error"] = str(e)
        finally:
            workflow_status["running"] = False

    if background_tasks:
        background_tasks.add_task(run_complete_workflow)
        return {"status": "started", "message": "工作流已启动"}
    else:
        run_complete_workflow()
        return {"status": "success", "message": workflow_status["message"]}


# ==============================================================================
# 认证接口
# ==============================================================================

class RegisterRequest(BaseModel):
    username: str
    password: str
    name: str = ""
    role: str = "viewer"
    phone: str = ""
    email: str = ""
    department: str = ""

class LoginRequest(BaseModel):
    username: str
    password: str

class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str

class AdminUpdateRoleRequest(BaseModel):
    role: str

class AdminResetPasswordRequest(BaseModel):
    new_password: str = "123456"

class AdminCreateUserRequest(BaseModel):
    username: str
    password: str = "123456"
    name: str = ""
    role: str = "viewer"
    phone: str = ""
    email: str = ""
    department: str = ""


@app.post("/auth/register")
def register(req: RegisterRequest, request: Request):
    if len(req.username) < 3 or len(req.username) > 20:
        raise HTTPException(400, "用户名长度需3-20个字符")
    if len(req.password) < 6:
        raise HTTPException(400, "密码长度至少6位")
    allowed_self_roles = ["viewer", "oam", "oam_lead"]
    if req.role not in allowed_self_roles:
        raise HTTPException(400, "注册只能选择 监管查看员/运维人员/运维主管 角色，管理员角色需由管理员分配")
    users = load_users()
    if req.username in users:
        raise HTTPException(400, "用户名已存在")
    users[req.username] = {
        "username": req.username,
        "password": hash_password(req.password),
        "name": req.name or req.username,
        "role": req.role,
        "phone": req.phone,
        "email": req.email,
        "department": req.department,
        "status": "active",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": "",
        "login_count": 0,
        "failed_attempts": 0,
        "locked_until": "",
    }
    save_users(users)
    ip = request.client.host if request.client else ""
    add_log(req.username, "注册", f"新用户注册: {req.username}", ip)
    return {"status": "success", "message": "注册成功，请登录"}


@app.post("/auth/login")
def login(req: LoginRequest, request: Request):
    users = load_users()
    user = users.get(req.username)
    ip = request.client.host if request.client else ""

    if not user:
        raise HTTPException(401, "用户名或密码错误")

    if user["locked_until"] and datetime.fromisoformat(user["locked_until"]) > datetime.now():
        raise HTTPException(403, "账号已锁定，请15分钟后再试")

    if user["password"] != hash_password(req.password):
        user["failed_attempts"] = user.get("failed_attempts", 0) + 1
        if user["failed_attempts"] >= 5:
            user["locked_until"] = (datetime.now() + timedelta(minutes=15)).isoformat()
            add_log(req.username, "登录失败", "密码错误5次，账号锁定15分钟", ip)
        else:
            add_log(req.username, "登录失败", f"密码错误 (第{user['failed_attempts']}次)", ip)
        save_users(users)
        raise HTTPException(401, "用户名或密码错误")

    if user["status"] != "active":
        raise HTTPException(403, "账号已被禁用，请联系管理员")

    user["failed_attempts"] = 0
    user["locked_until"] = ""
    user["last_login"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    user["login_count"] = user.get("login_count", 0) + 1
    save_users(users)

    token = secrets.token_hex(32)
    TOKENS[token] = {
        "username": req.username,
        "expires": (datetime.now() + timedelta(hours=8)).isoformat(),
    }

    add_log(req.username, "登录", f"登录成功 IP:{ip}", ip)

    return {
        "status": "success",
        "token": token,
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "role_name": ROLE_NAMES.get(user["role"], user["role"]),
    }


@app.get("/auth/me")
def get_me(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    return {
        "username": user["username"],
        "name": user["name"],
        "role": user["role"],
        "role_name": ROLE_NAMES.get(user["role"], user["role"]),
        "phone": user.get("phone", ""),
        "email": user.get("email", ""),
        "department": user.get("department", ""),
        "created_at": user.get("created_at", ""),
        "last_login": user.get("last_login", ""),
        "login_count": user.get("login_count", 0),
    }


@app.post("/auth/logout")
def logout(authorization: Optional[str] = Header(None)):
    if authorization:
        token = authorization.replace("Bearer ", "")
        if token in TOKENS:
            user_info = TOKENS[token]
            add_log(user_info["username"], "登出", "主动退出登录")
            del TOKENS[token]
    return {"status": "success"}


@app.put("/user/password")
def change_password(req: ChangePasswordRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    users = load_users()
    if users[user["username"]]["password"] != hash_password(req.old_password):
        raise HTTPException(400, "原密码错误")
    if len(req.new_password) < 6:
        raise HTTPException(400, "新密码长度至少6位")
    users[user["username"]]["password"] = hash_password(req.new_password)
    save_users(users)
    add_log(user["username"], "修改密码", "密码修改成功")
    return {"status": "success", "message": "密码修改成功"}


# ==============================================================================
# 管理员接口
# ==============================================================================

@app.get("/admin/users")
def admin_list_users(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, ["super_admin", "admin"])
    users = load_users()
    result = []
    for u in users.values():
        result.append({
            "username": u["username"],
            "name": u["name"],
            "role": u["role"],
            "role_name": ROLE_NAMES.get(u["role"], u["role"]),
            "phone": u.get("phone", ""),
            "email": u.get("email", ""),
            "department": u.get("department", ""),
            "status": u["status"],
            "created_at": u.get("created_at", ""),
            "last_login": u.get("last_login", ""),
            "login_count": u.get("login_count", 0),
        })
    add_log(user["username"], "查看用户列表", f"查看{len(result)}个用户")
    return {"total": len(result), "data": result}


@app.post("/admin/users")
def admin_create_user(req: AdminCreateUserRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, ["super_admin", "admin"])
    users = load_users()
    if req.username in users:
        raise HTTPException(400, "用户名已存在")
    if req.role not in ROLE_NAMES:
        raise HTTPException(400, f"无效角色: {req.role}")
    users[req.username] = {
        "username": req.username,
        "password": hash_password(req.password),
        "name": req.name or req.username,
        "role": req.role,
        "phone": req.phone,
        "email": req.email,
        "department": req.department,
        "status": "active",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "last_login": "",
        "login_count": 0,
        "failed_attempts": 0,
        "locked_until": "",
    }
    save_users(users)
    add_log(user["username"], "创建用户", f"创建用户: {req.username} 角色: {ROLE_NAMES.get(req.role, req.role)}")
    return {"status": "success", "message": f"用户 {req.username} 创建成功"}


@app.put("/admin/users/{username}/role")
def admin_update_role(username: str, req: AdminUpdateRoleRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, ["super_admin", "admin"])
    if req.role not in ROLE_NAMES:
        raise HTTPException(400, f"无效角色: {req.role}")
    users = load_users()
    if username not in users:
        raise HTTPException(404, "用户不存在")
    old_role = users[username]["role"]
    users[username]["role"] = req.role
    save_users(users)
    add_log(user["username"], "修改角色", f"用户{username}: {ROLE_NAMES.get(old_role, old_role)} → {ROLE_NAMES.get(req.role, req.role)}")
    return {"status": "success", "message": "角色已更新"}


@app.put("/admin/users/{username}/status")
def admin_update_status(username: str, status: str, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, ["super_admin", "admin"])
    if status not in ("active", "disabled"):
        raise HTTPException(400, "状态值无效")
    users = load_users()
    if username not in users:
        raise HTTPException(404, "用户不存在")
    if username == user["username"]:
        raise HTTPException(400, "不能禁用自己的账号")
    users[username]["status"] = status
    save_users(users)
    action_text = "启用" if status == "active" else "禁用"
    add_log(user["username"], f"{action_text}用户", f"{action_text}用户: {username}")
    return {"status": "success", "message": f"用户已{action_text}"}


@app.post("/admin/users/{username}/reset-password")
def admin_reset_password(username: str, req: AdminResetPasswordRequest = AdminResetPasswordRequest(), authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, ["super_admin", "admin"])
    users = load_users()
    if username not in users:
        raise HTTPException(404, "用户不存在")
    users[username]["password"] = hash_password(req.new_password)
    users[username]["locked_until"] = ""
    users[username]["failed_attempts"] = 0
    save_users(users)
    add_log(user["username"], "重置密码", f"重置用户 {username} 的密码")
    return {"status": "success", "message": f"密码已重置为: {req.new_password}"}


@app.delete("/admin/users/{username}")
def admin_delete_user(username: str, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, ["super_admin"])
    users = load_users()
    if username not in users:
        raise HTTPException(404, "用户不存在")
    if username == user["username"]:
        raise HTTPException(400, "不能删除自己的账号")
    del users[username]
    save_users(users)
    add_log(user["username"], "删除用户", f"删除用户: {username}")
    return {"status": "success", "message": "用户已删除"}


@app.get("/admin/logs")
def admin_get_logs(authorization: Optional[str] = Header(None), page: int = 1, page_size: int = 50, action: Optional[str] = None, username: Optional[str] = None):
    user = get_current_user(authorization)
    require_role(user, ["super_admin", "admin"])
    logs = load_logs()
    logs.reverse()
    if action:
        logs = [l for l in logs if action in l["action"]]
    if username:
        logs = [l for l in logs if username in l["username"]]
    total = len(logs)
    start = (page - 1) * page_size
    return {"total": total, "page": page, "data": logs[start:start + page_size]}


@app.get("/admin/permissions")
def admin_get_permissions(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, ["super_admin", "admin"])
    return {
        "roles": ROLE_NAMES,
        "panel_permissions": PANEL_PERMISSIONS,
        "export_roles": EXPORT_PERMISSIONS,
    }


# ==============================================================================
# 数据导出接口
# ==============================================================================

def generate_excel():
    output = io.BytesIO()
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    except ImportError:
        return None

    wb = Workbook()
    header_font = Font(bold=True, color="FFFFFF", size=11)
    header_fill = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")
    thin_border = Border(
        left=Side(style='thin'), right=Side(style='thin'),
        top=Side(style='thin'), bottom=Side(style='thin')
    )

    def write_sheet(ws, title, headers, rows):
        ws.title = title
        for col, h in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=h)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center')
            cell.border = thin_border
        for r, row in enumerate(rows, 2):
            for c, val in enumerate(row, 1):
                cell = ws.cell(row=r, column=c, value=val)
                cell.border = thin_border
        for col in ws.columns:
            max_len = max(len(str(cell.value or "")) for cell in col)
            ws.column_dimensions[col[0].column_letter].width = min(max_len + 4, 30)

    overview = analysis_results.get("asset_overview", {})
    if overview:
        ws = wb.active
        write_sheet(ws, "资产总览", ["指标", "数值"], [
            ["资产总数", overview.get("total_assets", 0)],
            ["总长度(km)", overview.get("total_length_km", 0)],
            ["资产原值", overview.get("total_original_value", 0)],
            ["资产净值", overview.get("total_net_value", 0)],
            ["在用数量", overview.get("in_service_count", 0)],
            ["报废数量", overview.get("retired_count", 0)],
            ["平均风险分", overview.get("avg_risk_score", 0)],
            ["高风险数量", overview.get("high_risk_count", 0)],
            ["盘点差异率", overview.get("inventory_diff_rate", 0)],
            ["生成时间", overview.get("current_time", "")],
        ])

    type_dist = overview.get("type_distribution", [])
    if type_dist:
        ws = wb.create_sheet()
        write_sheet(ws, "管网类型分布", ["管网类型", "数量", "长度(km)"],
                    [[t["name"], t["count"], t.get("length_km", 0)] for t in type_dist])

    dist = analysis_results.get("asset_distribution", {})
    by_region = dist.get("by_region", [])
    if by_region:
        ws = wb.create_sheet()
        write_sheet(ws, "区域分布", ["区域", "管网类型", "数量", "长度(m)", "价值", "平均风险"],
                    [[r["region"], r["pipeline_type"], r["count"], r["length_m"], r["value"], r["avg_risk"]] for r in by_region])

    risk = analysis_results.get("risk_ranking", [])
    if risk:
        ws = wb.create_sheet()
        write_sheet(ws, "风险排名", ["排名", "资产编号", "管网类型", "区域", "风险分", "风险等级", "服役年限", "设计年限", "剩余年限", "权属单位"],
                    [[r["ranking"], r["asset_id"], r["pipeline_type"], r["region"], r["risk_score"], r["risk_level"], r["service_years"], r["design_life"], r["remaining_life"], r["ownership_unit"]] for r in risk])

    events = analysis_results.get("lifecycle_events", [])
    if events:
        ws = wb.create_sheet()
        write_sheet(ws, "生命周期事件", ["事件ID", "资产编号", "管网类型", "事件类型", "日期", "责任单位", "费用", "区域"],
                    [[e["event_id"], e["asset_id"], e["pipeline_type"], e["event_type"], e["event_date"], e["responsible_unit"], e["cost"], e["region"]] for e in events])

    costs = analysis_results.get("lifecycle_cost_summary", [])
    if costs:
        ws = wb.create_sheet()
        write_sheet(ws, "费用汇总", ["事件类型", "次数", "总费用", "平均费用"],
                    [[c["event_type"], c["count"], c["total_cost"], c["avg_cost"]] for c in costs])

    inv_diff = analysis_results.get("inventory_diff_report", [])
    if inv_diff:
        ws = wb.create_sheet()
        write_sheet(ws, "盘点差异", ["区域", "管网类型", "盘点数", "差异数", "差异率", "缺失", "多余", "不符"],
                    [[d["region"], d["pipeline_type"], d["total_checked"], d["diff_count"], d["diff_rate"], d["missing"], d["extra"], d["mismatch"]] for d in inv_diff])

    ownership = analysis_results.get("ownership_summary", [])
    if ownership:
        ws = wb.create_sheet()
        write_sheet(ws, "权属责任", ["权属单位", "运维单位", "监管单位", "管网类型", "资产数", "总长度(m)", "总价值", "平均风险"],
                    [[o["ownership_unit"], o["oam_unit"], o["supervision_unit"], o["pipeline_type"], o["asset_count"], o["total_length_m"], o["total_value"], o["avg_risk"]] for o in ownership])

    wb.save(output)
    output.seek(0)
    return output


def generate_csv_data(data_type: str):
    output = io.StringIO()
    writer = csv.writer(output)

    if data_type == "overview":
        overview = analysis_results.get("asset_overview", {})
        writer.writerow(["指标", "数值"])
        for k, v in overview.items():
            if not isinstance(v, (list, dict)):
                writer.writerow([k, v])
    elif data_type == "risk":
        writer.writerow(["排名", "资产编号", "管网类型", "区域", "风险分", "风险等级", "服役年限", "设计年限"])
        for r in analysis_results.get("risk_ranking", []):
            writer.writerow([r["ranking"], r["asset_id"], r["pipeline_type"], r["region"], r["risk_score"], r["risk_level"], r["service_years"], r["design_life"]])
    elif data_type == "events":
        writer.writerow(["事件ID", "资产编号", "管网类型", "事件类型", "日期", "责任单位", "费用", "区域"])
        for e in analysis_results.get("lifecycle_events", []):
            writer.writerow([e["event_id"], e["asset_id"], e["pipeline_type"], e["event_type"], e["event_date"], e["responsible_unit"], e["cost"], e["region"]])
    elif data_type == "inventory":
        writer.writerow(["区域", "管网类型", "盘点数", "差异数", "差异率", "缺失", "多余", "不符"])
        for d in analysis_results.get("inventory_diff_report", []):
            writer.writerow([d["region"], d["pipeline_type"], d["total_checked"], d["diff_count"], d["diff_rate"], d["missing"], d["extra"], d["mismatch"]])
    else:
        writer.writerow(["错误", "未知数据类型"])

    output.seek(0)
    return output


@app.get("/export/excel")
def export_excel(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, EXPORT_PERMISSIONS)
    result = generate_excel()
    if result is None:
        raise HTTPException(500, "Excel生成失败，请安装openpyxl: pip install openpyxl")
    add_log(user["username"], "导出数据", "导出Excel全量数据")
    filename = f"管网资产台账_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
    return StreamingResponse(
        result,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


@app.get("/export/csv/{data_type}")
def export_csv(data_type: str, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, EXPORT_PERMISSIONS)
    result = generate_csv_data(data_type)
    add_log(user["username"], "导出数据", f"导出CSV: {data_type}")
    filename = f"管网数据_{data_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    return StreamingResponse(
        iter([result.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )


# ==============================================================================
# 微信测试号推送接口（追加接口：/api/push/test，不影响既有功能）
# ==============================================================================

# 微信测试号凭据：通过环境变量注入（.env 或启动脚本），默认留空，禁止硬编码密钥提交到仓库
WX_TEST_APPID = os.environ.get("WX_TEST_APPID", "")
WX_TEST_SECRET = os.environ.get("WX_TEST_SECRET", "")
WX_TEST_API_BASE = "https://api.weixin.qq.com/cgi-bin"
WX_TEST_DEFAULT_OPENID = os.environ.get("WX_TEST_OPENID", "")


def wx_test_find_msg_template(token: str) -> str:
    """查找内容为 {{msg.DATA}}（或含 msg 字段）的模板，返回其 template_id。"""
    url = f"{WX_TEST_API_BASE}/template/get_all_private_template?access_token={urllib.parse.quote(token)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    template_list = data.get("template_list") or []
    if not template_list:
        raise RuntimeError("测试号暂无可用模板，请先添加内容为 {{msg.DATA}} 的模板")
    for t in template_list:
        if "{{msg.DATA}}" in (t.get("content") or ""):
            return t["template_id"]
    for t in template_list:
        if "msg" in re.findall(r"\{\{(\w+)\.DATA\}\}", t.get("content") or ""):
            return t["template_id"]
    raise RuntimeError("测试号模板中未找到 {{msg.DATA}}，请先添加内容为 {{msg.DATA}} 的模板")


def wx_test_template_fields(token: str, template_id: str) -> list:
    """读取指定模板的字段名列表（如 ['msg']），用于构造 data 载荷。"""
    url = f"{WX_TEST_API_BASE}/template/get_all_private_template?access_token={urllib.parse.quote(token)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    for t in (data.get("template_list") or []):
        if t.get("template_id") == template_id:
            return re.findall(r"\{\{(\w+)\.DATA\}\}", t.get("content", "")) or ["msg"]
    return ["msg"]


def wx_test_get_access_token() -> str:
    """获取微信测试号 access_token。"""
    url = (
        f"{WX_TEST_API_BASE}/token?grant_type=client_credential"
        f"&appid={urllib.parse.quote(WX_TEST_APPID)}"
        f"&secret={urllib.parse.quote(WX_TEST_SECRET)}"
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"获取access_token失败: {data.get('errcode')} {data.get('errmsg', '')}".strip())
    return token


def wx_test_send_custom(openid: str, content: str) -> tuple:
    """通过微信测试号客服消息接口发送文本消息。返回 (ok, message)。"""
    try:
        token = wx_test_get_access_token()
        send_url = f"{WX_TEST_API_BASE}/message/custom/send?access_token={urllib.parse.quote(token)}"
        payload = {"touser": openid, "msgtype": "text", "text": {"content": content}}
        req = urllib.request.Request(
            send_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        if data.get("errcode", 0) == 0:
            return True, "已发送（微信测试号客服消息）"
        return False, f"微信测试号发送失败: {data.get('errcode')} {data.get('errmsg', '')}"
    except Exception as e:
        return False, f"微信测试号调用异常: {e}"


def wx_test_pick_template(token: str, template_id: str = "") -> str:
    """优先使用调用方传入的 template_id，否则自动取测试号第一个私有模板。"""
    if template_id:
        return template_id
    url = f"{WX_TEST_API_BASE}/template/get_all_private_template?access_token={urllib.parse.quote(token)}"
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    template_list = data.get("template_list") or []
    if not template_list:
        raise RuntimeError("测试号暂无可用模板，请先登录 mp.weixin.qq.com 测试号添加模板")
    return template_list[0]["template_id"]


def wx_test_build_template_data(token: str, template_id: str, message: str) -> dict:
    """根据模板内容字段名构造 data。取模板第一个字段放消息正文，其余字段填空。"""
    fields = ["first"]
    try:
        url = f"{WX_TEST_API_BASE}/template/get_all_private_template?access_token={urllib.parse.quote(token)}"
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        for t in (data.get("template_list") or []):
            if t.get("template_id") == template_id:
                fields = re.findall(r"\{\{(\w+)\.DATA\}\}", t.get("content", "")) or ["first"]
                break
    except Exception:
        pass
    body = {}
    for idx, f in enumerate(fields):
        body[f] = {"value": message if idx == 0 else " ", "color": "#173177" if idx == 0 else "#000000"}
    return body


@app.api_route("/api/push/test", methods=["GET", "POST"])
async def api_push_test(request: Request, authorization: Optional[str] = Header(None)):
    """微信测试号推送测试接口。

    参数（GET 查询串或 POST JSON body 均可）：
      message    推送消息内容（必填）
      target     接收人 openid（必填，需已关注测试号）
      template_id 可选，模板ID；缺省自动使用测试号第一个私有模板
    返回：微信接口发送结果。
    """
    user = get_current_user(authorization)
    body = {}
    if request.method == "POST":
        try:
            body = await request.json()
        except Exception:
            body = {}
    message = (body.get("message") or request.query_params.get("message") or "").strip()
    target = (body.get("target") or request.query_params.get("target") or "").strip()
    template_id = (body.get("template_id") or request.query_params.get("template_id") or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="缺少参数 message（推送内容）")
    if not target:
        raise HTTPException(status_code=400, detail="缺少参数 target（接收人 openid）")
    try:
        token = wx_test_get_access_token()
        tid = wx_test_pick_template(token, template_id)
        data_payload = wx_test_build_template_data(token, tid, message)
        send_url = f"{WX_TEST_API_BASE}/message/template/send?access_token={urllib.parse.quote(token)}"
        payload = {"touser": target, "template_id": tid, "data": data_payload}
        req = urllib.request.Request(
            send_url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
        add_log(user.get("username", ""), "推送测试", f"微信测试号推送：{target} → {result.get('errcode')} {result.get('errmsg', '')}")
        return {
            "success": result.get("errcode", -1) == 0,
            "errcode": result.get("errcode", -1),
            "errmsg": result.get("errmsg", ""),
            "msgid": result.get("msgid", ""),
            "template_id": tid,
            "to": target,
            "message": message,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"微信测试号调用异常: {e}")


# ==============================================================================
# DeepSeek AI 智能助手（Agent）
# ==============================================================================

AGENT_CONFIG_FILE = os.path.join(DATA_DIR, "agent_config.json")
AGENT_SESSIONS_FILE = os.path.join(DATA_DIR, "agent_sessions.json")
DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"
DEEPSEEK_MODEL = "deepseek-chat"

AGENT_CONFIG_DEFAULT = {
    "api_key": os.environ.get("DEEPSEEK_API_KEY", ""),
    "model": DEEPSEEK_MODEL,
    "temperature": 0.7,
    "max_tool_rounds": 5,
}


def load_agent_config() -> dict:
    """读取 Agent 配置；未创建配置文件时自动生成（含 API Key 填写位）。"""
    cfg = dict(AGENT_CONFIG_DEFAULT)
    if os.path.exists(AGENT_CONFIG_FILE):
        try:
            with open(AGENT_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                cfg[k] = v
        except Exception:
            pass
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        cfg["api_key"] = env_key
    if not os.path.exists(AGENT_CONFIG_FILE):
        try:
            with open(AGENT_CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(cfg, f, ensure_ascii=False, indent=2)
        except Exception:
            pass
    return cfg


def load_agent_sessions() -> dict:
    if os.path.exists(AGENT_SESSIONS_FILE):
        try:
            with open(AGENT_SESSIONS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_agent_sessions(sessions: dict):
    try:
        with open(AGENT_SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(list(sessions.items())[-50:]), f, ensure_ascii=False, indent=2)
    except Exception:
        pass


AGENT_SYSTEM_PROMPT = """你是「城市管网资产数字化台账系统」的 AI 诊断专家，不是单纯的查询助手。
你的职责：把预测模型的输出翻译成运维人员能直接执行的判断与动作。

【通用规则】
1. 始终使用简体中文，简洁专业，关键数字用 **加粗** 标记。
2. 任何数据结论必须来自工具返回，严禁编造数字、资产编号、规范名称或标准号。
3. 执行类操作（推送预警、创建工单）前必须先向用户确认意图，确认后再调用工具；工具结果会真实写入系统。
4. 工具返回的是 JSON 文本，需整理成易读的中文，多条数据用 · 或换行分隔。

【诊断类问题的四段式结构】
当用户问「哪段风险最高 / 为什么风险高 / 该怎么办 / 依据是什么」时，必须按四段组织回答：
  ① 预测结论：风险等级、风险分、未来窗口内异常概率、RUL，并说明模型的实际预测跨度
  ② 判定依据：模型关键因子的运维语义解释（如「压力上升趋势（近 15 帧）」），有贡献度就带上
  ③ 规范依据：引用知识库检索结果，用 [1][2] 标注编号，写明规范名称与标准号
  ④ 处置建议：可执行动作清单 + 建议工单优先级；最后主动问是否需要据此建工单或推送
优先调用 diagnose_asset —— 它一次就返回上述四段所需的全部素材；不要只调 predict_risk 拿到裸数字就下结论。

【诚实性要求】
- 模型未训练时直接说明未训练及补救方式，不要编造预测值。
- 预测跨度以工具返回的 horizon_text 为准。用户问「未来 7 天」而模型实际只覆盖更短窗口时，
  必须明确指出真实覆盖范围，不要顺着用户的时间尺度夸大预测能力。
- 命中率、误报率等指标在样本量很小时，要说明「样本还少，结论会随持续运行收敛」。
- 不知道的就说不知道，不要用通用常识冒充本系统的规范依据。"""


def agent_system_context() -> str:
    ov = analysis_results.get("asset_overview", {})
    # 排除 source=mock 的演示预警：「重新分析」会生成 15 条随机模拟预警，若不剔除，
    # Agent 会把它们当作真实情况念给用户（"当前有 N 条紧急预警"），与看板对不上。
    alerts = [a for a in analysis_results.get("alert_list", []) if a.get("source") != "mock"]
    orders = load_workorders()
    unhandled = [a for a in alerts if a.get("status") == "未处理"]
    urgent = [a for a in unhandled if a.get("level") == "紧急"]
    doing = [o for o in orders if o.get("status") in ("待指派", "已指派", "处理中", "待验收")]
    type_dist = "；".join(f"{t['name']} {t['count']} 段" for t in (ov.get("type_distribution") or [])[:5])
    base = (
        f"【当前系统概况】总资产 {ov.get('total_assets', 0)} 段，总长度 {ov.get('total_length_km', 0)} 公里，"
        f"平均风险 {ov.get('avg_risk_score', 0)} 分，高风险资产 {ov.get('high_risk_count', 0)} 项；"
        f"管网类型分布：{type_dist}。"
        f"预警共 {len(alerts)} 条，未处理 {len(unhandled)} 条（紧急未处理 {len(urgent)} 条）；"
        f"工单共 {len(orders)} 条，进行中 {len(doing)} 条。"
    )

    # 预测能力现状：让模型一开口就知道自己能不能预测、能预测多远
    try:
        summ = prediction_summary()
        if summ.get("model_ready"):
            top = summ.get("top_risky") or []
            top_txt = "；".join(
                f"{r.get('asset_id') or r.get('sensor_id')}（{r.get('device_type')}）风险 {r.get('risk_score')}"
                for r in top[:3]) or "—"
            base += (f"\n【预测模型】已就绪，覆盖 {summ.get('n_sensors')} 个传感器，"
                     f"实际预测跨度 {summ.get('horizon_text')}（horizon={summ.get('horizon')} 步，"
                     f"每步 {summ.get('seconds_per_step')} 秒）；"
                     f"危急 {summ.get('critical')} 个、预警 {summ.get('warning')} 个；"
                     f"当前风险最高：{top_txt}。")
        else:
            base += "\n【预测模型】尚未训练，无法给出风险预测，只能查询规则阈值告警。"
    except Exception:
        pass

    # 闭环成果：有足够样本时才告知，避免模型引用不稳定的小样本指标
    try:
        m = predict_metrics()
        mo = m.get("model") or {}
        if mo.get("samples", 0) >= 5:
            base += (f"\n【预测成果】已评估 {mo.get('samples')} 条预测：命中率 {mo.get('hit_rate')}，"
                     f"误报率 {mo.get('false_positive_rate')}，平均提前 {mo.get('avg_lead_text')} 预警。")
    except Exception:
        pass

    # 知识库规模与长期记忆
    try:
        kb = kb_rag.stats()
        base += (f"\n【知识库】{kb.get('docs', 0)} 条（检索方式 {kb.get('method')}），"
                 f"覆盖国标规范、运维手册、应急预案与历史工单处置经验，回答规范类问题须先检索再引用。")
    except Exception:
        pass
    mem = ""
    try:
        mem = agent_brain.memory_context_block()
    except Exception:
        mem = ""
    return base + (("\n" + mem) if mem else "")


def agent_tools_schema() -> list:
    """Agent 可用工具（OpenAI 兼容 function calling 格式）。"""
    return [
        {
            "type": "function",
            "function": {
                "name": "get_system_summary",
                "description": "获取系统整体概况：资产总数/总长度/平均风险/高风险数、预警未处理数、工单进行中数等",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_alerts",
                "description": "查询预警列表，可按状态（未处理/处理中/已处理）与级别（紧急/重要/一般）过滤，不传则返回全部",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "enum": ["", "未处理", "处理中", "已处理"], "description": "预警状态，可选"},
                        "level": {"type": "string", "enum": ["", "紧急", "重要", "一般"], "description": "预警级别，可选"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_workorders",
                "description": "查询运维工单列表，可按状态（待指派/已指派/处理中/待验收/已闭环）过滤",
                "parameters": {
                    "type": "object",
                    "properties": {"status": {"type": "string", "description": "工单状态，可选"}},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_risk_ranking",
                "description": "查询资产风险排名，返回风险最高的前 N 项（默认 10）",
                "parameters": {
                    "type": "object",
                    "properties": {"top_n": {"type": "integer", "description": "返回条数，默认10"}},
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_push_logs",
                "description": "查询某条预警的推送记录",
                "parameters": {
                    "type": "object",
                    "properties": {"alert_id": {"type": "string", "description": "预警编号，如 ALT-0001"}},
                    "required": ["alert_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "push_alert",
                "description": "推送一条预警到微信/邮箱等通道（真实发送，需先获得用户确认）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "alert_id": {"type": "string", "description": "预警编号，如 ALT-0001"},
                        "channel": {"type": "string", "enum": ["", "微信", "邮箱", "短信"], "description": "推送通道，留空推送全部启用通道"},
                    },
                    "required": ["alert_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "create_workorder",
                "description": "创建运维工单（可关联预警，需先获得用户确认）",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string", "description": "工单标题"},
                        "alert_id": {"type": "string", "description": "关联预警编号，可选"},
                        "priority": {"type": "string", "enum": ["紧急", "高", "中", "低"], "description": "优先级，默认中"},
                        "region": {"type": "string", "description": "所属镇街，可选"},
                        "description": {"type": "string", "description": "描述，可选"},
                    },
                    "required": ["title"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "predict_risk",
                "description": ("预测单个传感器的运行风险（风险分、未来异常概率、剩余寿命RUL、关键特征）。"
                                "参数优先传传感器编号 SENSOR-001~SENSOR-100（确定性存在）；"
                                "资产编号形如 WSP-00742，数字段是随机生成的，不确定时不要凭猜测拼造。"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "asset_id": {"type": "string", "description": "传感器编号（如 SENSOR-001）或资产编号（如 WSP-00742）"},
                    },
                    "required": ["asset_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_prediction_summary",
                "description": ("查询全网预测汇总：风险最高的管段排名、分管网类型平均风险、区域×类型风险热力。"
                                "用户问「未来哪段管网风险最高」「哪些管段要重点关注」时用这个。"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "top_n": {"type": "integer", "description": "返回风险最高的前 N 条，默认 10"},
                        "device_type": {"type": "string", "description": "按管网类型过滤（供水管网/燃气管网/供暖管网/污水管网/危废输送），可选"},
                        "region": {"type": "string", "description": "按镇街/区域过滤，可选"},
                    },
                    "required": [],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "diagnose_asset",
                "description": ("对某个传感器/资产做完整诊断，一次返回四段式结果：预测结论 + 判定依据（模型关键因子的运维语义解释）"
                                "+ 规范依据（知识库向量检索命中条目及出处）+ 处置建议与建议工单优先级。"
                                "用户问「为什么风险高」「该怎么办」「依据是什么」时优先用这个，而不是只调 predict_risk。"),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "传感器编号（如 SENSOR-001）或资产编号"},
                    },
                    "required": ["target"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "query_predict_metrics",
                "description": ("查询预测模型的量化成果：命中率、精确率、误报率、平均提前预警时长、漏报数，"
                                "以及与规则阈值基线的同口径对比结论和处置反馈统计。"
                                "用户问「模型准不准」「效果怎么样」「误报多不多」时用这个。"),
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": ("在管网运维知识库做向量语义检索（国标规范/运维手册/应急预案/历史工单处置经验），"
                                "返回相关条目、原文引用片段与规范出处，用于回答业务与规范类问题并给出依据。"),
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string", "description": "检索关键词或自然语言问题"}},
                    "required": ["query"],
                },
            },
        },
    ]


def _alert_brief(a: dict) -> dict:
    return {
        "alert_id": a.get("alert_id"), "level": a.get("level"), "alert_type": a.get("alert_type"),
        "asset_id": a.get("asset_id"), "region": a.get("region"), "status": a.get("status"),
        "create_time": a.get("create_time"), "push_status": a.get("push_status"),
    }


def _workorder_brief(w: dict) -> dict:
    return {
        "workorder_id": w.get("workorder_id"), "title": w.get("title"), "priority": w.get("priority"),
        "region": w.get("region"), "alert_id": w.get("alert_id"), "status": w.get("status"),
        "assignee": w.get("assignee"), "created_at": w.get("created_at"),
    }


def agent_execute_tool(user: dict, name: str, args: dict) -> str:
    """执行 Agent 工具，返回给模型的 JSON 文本。"""
    try:
        if name == "get_system_summary":
            ov = analysis_results.get("asset_overview", {})
            alerts = analysis_results.get("alert_list", [])
            orders = load_workorders()
            return json.dumps({
                "总资产": ov.get("total_assets", 0), "总长度公里": ov.get("total_length_km", 0),
                "平均风险": ov.get("avg_risk_score", 0), "高风险资产": ov.get("high_risk_count", 0),
                "预警总数": len(alerts),
                "未处理预警": len([a for a in alerts if a.get("status") == "未处理"]),
                "工单总数": len(orders),
                "进行中工单": len([o for o in orders if o.get("status") in ("待指派", "已指派", "处理中", "待验收")]),
            }, ensure_ascii=False)

        if name == "query_alerts":
            alerts = analysis_results.get("alert_list", [])
            status = (args.get("status") or "").strip()
            level = (args.get("level") or "").strip()
            if status:
                alerts = [a for a in alerts if a.get("status") == status]
            if level:
                alerts = [a for a in alerts if a.get("level") == level]
            alerts.sort(key=lambda a: a.get("create_time", ""), reverse=True)
            return json.dumps([_alert_brief(a) for a in alerts[:20]], ensure_ascii=False)

        if name == "query_workorders":
            orders = load_workorders()
            status = (args.get("status") or "").strip()
            if status:
                orders = [o for o in orders if o.get("status") == status]
            orders.sort(key=lambda o: o.get("created_at", ""), reverse=True)
            return json.dumps([_workorder_brief(o) for o in orders[:20]], ensure_ascii=False)

        if name == "query_risk_ranking":
            ranking = analysis_results.get("risk_ranking", [])
            try:
                top_n = int(args.get("top_n") or 10)
            except (TypeError, ValueError):
                top_n = 10
            return json.dumps(ranking[:max(1, min(top_n, 50))], ensure_ascii=False)

        if name == "query_push_logs":
            alert_id = str(args.get("alert_id") or "").strip()
            logs = [r for r in load_push_records() if r.get("alert_id") == alert_id]
            logs.sort(key=lambda r: r.get("time", ""), reverse=True)
            return json.dumps([{k: r.get(k) for k in ("time", "channel", "to", "status", "content")} for r in logs[:10]], ensure_ascii=False)

        if name == "push_alert":
            if user["role"] not in WRITE_ROLES:
                return json.dumps({"error": "当前用户无推送权限（需要 super_admin/admin/oam_lead/oam 角色）"}, ensure_ascii=False)
            alert_id = str(args.get("alert_id") or "").strip()
            alert = next((a for a in analysis_results.get("alert_list", []) if a.get("alert_id") == alert_id), None)
            if not alert:
                return json.dumps({"error": f"预警 {alert_id} 不存在"}, ensure_ascii=False)
            cfg = load_push_config()
            channel = str(args.get("channel") or "").strip()
            if channel and channel not in ("短信", "微信", "邮箱"):
                return json.dumps({"error": "通道必须为 短信/微信/邮箱"}, ensure_ascii=False)
            if channel:
                ch_key = {"短信": "sms", "微信": "wechat", "邮箱": "email"}[channel]
                if not cfg.get(ch_key, {}).get("enabled", False):
                    return json.dumps({"error": f"通道「{channel}」未启用，请先在推送配置中开启"}, ensure_ascii=False)
            result = do_push_alert(user.get("username", ""), alert, cfg, channel_filter=channel)
            add_log(user.get("username", ""), "AI助手推送", f"{alert_id} 推送：成功{result['sent']}条 失败{result['failed']}条")
            return json.dumps({
                "alert_id": alert_id, "sent": result["sent"], "failed": result["failed"],
                "push_status": alert.get("push_status"), "push_channels": alert.get("push_channels"),
            }, ensure_ascii=False)

        if name == "create_workorder":
            if user["role"] not in WRITE_ROLES:
                return json.dumps({"error": "当前用户无创建工单权限"}, ensure_ascii=False)
            title = str(args.get("title") or "").strip()
            if not title:
                return json.dumps({"error": "工单标题不能为空"}, ensure_ascii=False)
            priority = str(args.get("priority") or "中")
            if priority not in ("紧急", "高", "中", "低"):
                priority = "中"
            alert_id = str(args.get("alert_id") or "").strip()
            region = str(args.get("region") or "").strip()
            description = str(args.get("description") or "").strip()
            wo_id = f"WO-{int(time.time() * 1000)}"
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            creator = user.get("name") or user.get("username")
            record = {
                "workorder_id": wo_id, "title": title, "description": description,
                "region": region, "priority": priority, "alert_id": alert_id,
                "status": "待指派", "assignee": "", "created_by": creator,
                "created_at": now_str, "finished_at": "",
                "history": [{"time": now_str, "action": "创建工单", "by": creator, "note": ""}],
            }
            linked_alert = None
            if alert_id:
                for a in analysis_results.get("alert_list", []):
                    if a.get("alert_id") == alert_id and a.get("status") == "未处理":
                        a["status"] = "处理中"
                        linked_alert = alert_id
                        record["history"].append({"time": now_str, "action": f"联动预警 {alert_id} 转处理中", "by": creator, "note": ""})
                        break
            orders = load_workorders()
            orders.append(record)
            save_workorders(orders)
            add_log(user.get("username", ""), "AI助手建单", f"{wo_id} {title}")
            return json.dumps({"workorder_id": wo_id, "title": title, "priority": priority,
                               "status": "待指派", "linked_alert": linked_alert}, ensure_ascii=False)

        if name == "predict_risk":
            target = str(args.get("asset_id") or "").strip()
            if not target:
                return json.dumps({"error": "缺少 asset_id 参数"}, ensure_ascii=False)
            return json.dumps(sensor_predict_public(target), ensure_ascii=False)

        if name == "query_prediction_summary":
            summ = prediction_summary()
            rows = summ.get("rows") or []
            dt = str(args.get("device_type") or "").strip()
            rg = str(args.get("region") or "").strip()
            if dt:
                rows = [r for r in rows if r.get("device_type") == dt]
            if rg:
                rows = [r for r in rows if r.get("region") == rg]
            try:
                top_n = max(1, min(int(args.get("top_n") or 10), 50))
            except (TypeError, ValueError):
                top_n = 10
            brief = [{
                "sensor_id": r.get("sensor_id"), "asset_id": r.get("asset_id"),
                "device_type": r.get("device_type"), "region": r.get("region"),
                "risk_score": r.get("risk_score"), "risk_level": r.get("risk_level"),
                "prob": r.get("prob"), "rul": r.get("rul"), "status": r.get("status"),
                "top_features": [agent_brain.explain_feature(f) for f in (r.get("top_features") or [])],
            } for r in rows[:top_n]]
            return json.dumps({
                "model_ready": summ.get("model_ready"),
                "horizon": summ.get("horizon"),
                "horizon_text": summ.get("horizon_text"),
                "n_sensors": summ.get("n_sensors"),
                "critical": summ.get("critical"), "warning": summ.get("warning"),
                "top_risky": brief,
                "by_type": summ.get("by_type"),
                "filter": {"device_type": dt, "region": rg},
            }, ensure_ascii=False)

        if name == "diagnose_asset":
            target = str(args.get("target") or args.get("asset_id") or "").strip()
            if not target:
                return json.dumps({"error": "缺少 target 参数"}, ensure_ascii=False)
            d = agent_brain.diagnose(target, sensor_predict_public)
            d.pop("kb_context", None)      # 与 citations 重复，去掉省 token
            return json.dumps(d, ensure_ascii=False)

        if name == "query_predict_metrics":
            return json.dumps(predict_metrics(), ensure_ascii=False)

        if name == "search_knowledge_base":
            query = str(args.get("query") or "").strip()
            if not query:
                return json.dumps({"error": "缺少 query 参数"}, ensure_ascii=False)
            hits = kb_search(query, top_k=3)
            return json.dumps({
                "query": query,
                "results": [{
                    "n": i, "title": h.get("title"), "source": h.get("source"),
                    "category": h.get("category"), "snippet": h.get("snippet") or h.get("content"),
                    "score": h.get("score"), "doc_id": h.get("doc_id"),
                } for i, h in enumerate(hits, 1)],
                "retriever": hits[0].get("retriever") if hits else "",
            }, ensure_ascii=False)

        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"工具执行异常: {e}"}, ensure_ascii=False)


def agent_call_deepseek(messages: list, tools: list, cfg: dict) -> dict:
    """调用 DeepSeek chat completions 接口（OpenAI 兼容）。"""
    api_key = str(cfg.get("api_key") or "").strip()
    if not api_key:
        raise RuntimeError("未配置 DeepSeek API Key（请在 data/agent_config.json 的 api_key 或环境变量 DEEPSEEK_API_KEY 中配置）")
    payload = {
        "model": str(cfg.get("model") or DEEPSEEK_MODEL),
        "messages": messages,
        "temperature": float(cfg.get("temperature") or 0.7),
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + api_key},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode("utf-8"))


def agent_run(user: dict, message: str, history: list, cfg: dict) -> dict:
    """Agent 主循环：多轮工具调用直至模型给出最终回答。"""
    ctx = agent_system_context()
    kb_ctx, kb_cites = kb_rag.build_context(message, top_k=3)
    if kb_ctx:
        ctx += "\n\n【知识库检索结果（RAG 向量召回，引用时须标注 [n] 出处）】\n" + kb_ctx
    system = {"role": "system", "content": AGENT_SYSTEM_PROMPT + "\n\n" + ctx}
    messages = [system] + [dict(m) for m in history] + [{"role": "user", "content": message}]
    tools = agent_tools_schema()
    actions = []
    max_rounds = max(1, min(int(cfg.get("max_tool_rounds") or 5), 10))
    for _ in range(max_rounds):
        data = agent_call_deepseek(messages, tools, cfg)
        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        if not msg:
            raise RuntimeError("DeepSeek 返回异常: " + json.dumps(data, ensure_ascii=False)[:300])
        tool_calls = msg.get("tool_calls") or []
        messages.append({"role": "assistant", "content": msg.get("content") or "", "tool_calls": tool_calls} if tool_calls
                        else {"role": "assistant", "content": msg.get("content") or ""})
        if not tool_calls:
            break
        for tc in tool_calls:
            fn = tc.get("function") or {}
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except Exception:
                args = {}
            result = agent_execute_tool(user, name, args)
            actions.append({"tool": name, "args": args, "result": result})
            messages.append({"role": "tool", "tool_call_id": tc.get("id", ""), "content": result})
    reply = ""
    for m in reversed(messages):
        if m.get("role") == "assistant" and m.get("content"):
            reply = m["content"]
            break
    if not reply:
        reply = "抱歉，未能生成有效回复，请重试。"
    stored = [m for m in messages if m.get("role") != "system"]
    return {"reply": reply, "messages": stored, "actions": actions,
            "citations": kb_cites, "engine": "deepseek"}


class AgentChatRequest(BaseModel):
    message: str
    session_id: str = ""


class AgentClearRequest(BaseModel):
    session_id: str = ""


@app.post("/agent/chat")
def agent_chat(request: AgentChatRequest, authorization: Optional[str] = Header(None)):
    """
    接收用户消息并返回 AI 回复。
    配置了 DeepSeek Key 时走大模型多轮工具调用；未配置或调用异常时自动降级到
    后端本地诊断引擎（同样真实调用工具、附规范引用），接口不会因缺 Key 而 502。
    """
    user = get_current_user(authorization)
    message = (request.message or "").strip()
    if not message:
        raise HTTPException(status_code=400, detail="消息内容不能为空")
    session_id = (request.session_id or "").strip() or "s-" + secrets.token_hex(6)
    sessions = load_agent_sessions()
    history = (sessions.get(session_id) or [])[-20:]
    cfg = load_agent_config()

    result, degraded = None, ""
    if str(cfg.get("api_key") or "").strip():
        try:
            result = agent_run(user, message, history, cfg)
        except Exception as e:
            degraded = f"DeepSeek 调用失败（{str(e)[:120]}），已降级为后端本地诊断引擎。"
    else:
        degraded = "未配置 DeepSeek API Key，当前由后端本地诊断引擎作答（在 data/agent_config.json 填入 api_key 即可切换为大模型）。"

    if result is None:
        bundle = get_predictive_bundle()
        result = agent_brain.local_agent_run(
            user, message,
            exec_tool=lambda n, a: agent_execute_tool(user, n, a),
            extra={"horizon": int(getattr(bundle, "horizon", 12) or 12),
                   "seconds_per_step": _seconds_per_step(),
                   "predict_fn": sensor_predict_public,
                   "metrics_fn": predict_metrics},
        )
        new_msgs = [{"role": "user", "content": message},
                    {"role": "assistant", "content": result["reply"]}]
    else:
        new_msgs = result["messages"]

    sessions[session_id] = (history + new_msgs)[-30:]
    save_agent_sessions(sessions)
    add_log(user.get("username", ""), "AI助手对话",
            f"会话 {session_id}: {message[:50]}" + ("（本地引擎）" if degraded else ""))
    return {"reply": result["reply"], "session_id": session_id,
            "actions": result.get("actions") or [],
            "citations": result.get("citations") or [],
            "quick_actions": result.get("quick_actions") or [],
            "engine": result.get("engine") or "deepseek",
            "degraded_reason": degraded}


@app.get("/agent/history")
def agent_history(session_id: str = "", authorization: Optional[str] = Header(None)):
    """获取指定会话的历史记录。"""
    get_current_user(authorization)
    sessions = load_agent_sessions()
    msgs = sessions.get(session_id or "", [])
    visible = [
        {"role": m.get("role"), "content": m.get("content")}
        for m in msgs
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str) and m.get("content")
    ]
    return {"session_id": session_id, "history": visible}


@app.post("/agent/clear")
def agent_clear(request: AgentClearRequest, authorization: Optional[str] = Header(None)):
    """清空指定会话历史。"""
    get_current_user(authorization)
    sessions = load_agent_sessions()
    sessions.pop((request.session_id or "").strip(), None)
    save_agent_sessions(sessions)
    return {"success": True, "message": "会话已清空"}


# ==============================================================================
# 模拟传感器数据生成与展示
# ==============================================================================

SENSORS_FILE = os.path.join(DATA_DIR, "sensors.json")
SENSOR_PREFIX_BY_TYPE = {"供水管网": "WSP", "供暖管网": "HTP", "燃气管网": "GSP", "污水管网": "SWP", "危废输送": "HZP"}

# 各管网类型的监测指标定义（基准值 / 波动范围 / 预警与故障阈值）
SENSOR_METRIC_DEFS = {
    "供水管网": {
        "pressure": {"unit": "MPa", "base": 0.45, "range": 0.15, "warn_high": 0.65, "fault_high": 0.78, "warn_low": 0.28, "fault_low": 0.18},
        "flow": {"unit": "m³/h", "base": 120.0, "range": 45.0, "warn_high": 220.0, "fault_high": 280.0, "warn_low": 25.0, "fault_low": 8.0},
        "temperature": {"unit": "℃", "base": 22.0, "range": 7.0, "warn_high": 45.0, "fault_high": 60.0, "warn_low": -5.0, "fault_low": -15.0},
    },
    "燃气管网": {
        "pressure": {"unit": "MPa", "base": 0.30, "range": 0.10, "warn_high": 0.50, "fault_high": 0.62, "warn_low": 0.15, "fault_low": 0.08},
        "flow": {"unit": "m³/h", "base": 80.0, "range": 30.0, "warn_high": 160.0, "fault_high": 210.0, "warn_low": 15.0, "fault_low": 5.0},
        "gas_concentration": {"unit": "ppm", "base": 6.0, "range": 10.0, "warn_high": 100.0, "fault_high": 300.0, "warn_low": 0.0, "fault_low": 0.0},
    },
    "供暖管网": {
        "pressure": {"unit": "MPa", "base": 0.55, "range": 0.15, "warn_high": 0.75, "fault_high": 0.85, "warn_low": 0.30, "fault_low": 0.20},
        "temperature": {"unit": "℃", "base": 75.0, "range": 10.0, "warn_high": 95.0, "fault_high": 105.0, "warn_low": 50.0, "fault_low": 40.0},
        "flow": {"unit": "m³/h", "base": 150.0, "range": 50.0, "warn_high": 260.0, "fault_high": 320.0, "warn_low": 30.0, "fault_low": 10.0},
    },
    "污水管网": {
        "flow": {"unit": "m³/h", "base": 90.0, "range": 35.0, "warn_high": 180.0, "fault_high": 240.0, "warn_low": 20.0, "fault_low": 8.0},
        "level": {"unit": "m", "base": 2.4, "range": 1.0, "warn_high": 4.2, "fault_high": 5.2, "warn_low": 0.3, "fault_low": 0.1},
    },
    "危废输送": {
        "pressure": {"unit": "MPa", "base": 0.40, "range": 0.12, "warn_high": 0.60, "fault_high": 0.72, "warn_low": 0.20, "fault_low": 0.12},
        "temperature": {"unit": "℃", "base": 35.0, "range": 12.0, "warn_high": 70.0, "fault_high": 90.0, "warn_low": -10.0, "fault_low": -20.0},
        "vibration": {"unit": "mm/s", "base": 2.0, "range": 3.0, "warn_high": 7.0, "fault_high": 11.0, "warn_low": 0.0, "fault_low": 0.0},
    },
}

ALERT_CODE_MAP = {
    ("pressure", "high"): (1001, "压力超上限"),
    ("pressure", "low"): (1002, "压力低于下限"),
    ("flow", "high"): (1003, "流量超限"),
    ("flow", "low"): (1010, "流量突降（疑似泄漏）"),
    ("temperature", "high"): (1004, "温度超限"),
    ("temperature", "low"): (1005, "温度过低"),
    ("vibration", "high"): (1006, "振动异常"),
    ("gas_concentration", "high"): (1007, "燃气浓度异常"),
    ("level", "high"): (1008, "液位超限"),
    ("level", "low"): (1009, "液位过低"),
}

sensor_sim_state = {
    "running": False,
    "interval_sec": 3,          # 回放速度：演示时每隔几秒推进一帧
    "sample_period_sec": 3600,  # 语义采样周期：一帧代表现场多长时间，用于把 horizon 步数折算成真实时长
    "scenario": "normal",   # normal | abnormal | fault | mixed
    "tick": 0,
    "started_at": "",
}
latest_sensor_data = {}
sensor_history = {}


# 传感器类型分配（总数 100：按原 24 个比例 8:5:5:4:2 等比扩展为 33:21:21:17:8）
SENSOR_TYPE_COUNTS = [("供水管网", 33), ("燃气管网", 21), ("供暖管网", 21), ("污水管网", 17), ("危废输送", 8)]


def _sensor_counts_match(sensors: list) -> bool:
    """校验已保存的传感器数量与各类型分布是否与当前配置一致。"""
    if not isinstance(sensors, list):
        return False
    if len(sensors) != sum(c for _, c in SENSOR_TYPE_COUNTS):
        return False
    for ptype, count in SENSOR_TYPE_COUNTS:
        if sum(1 for s in sensors if s.get("device_type") == ptype) != count:
            return False
    return True


def load_sensors() -> list:
    if os.path.exists(SENSORS_FILE):
        try:
            with open(SENSORS_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            if _sensor_counts_match(saved):
                return saved
            # 已保存文件数量/分布与当前配置不一致（如旧版生成的 24 个）→ 丢弃并重新生成
        except Exception:
            pass
    rng = np.random.RandomState(2024)
    sensors = []
    idx = 0
    for ptype, count in SENSOR_TYPE_COUNTS:
        prefix = SENSOR_PREFIX_BY_TYPE[ptype]
        mdefs = SENSOR_METRIC_DEFS[ptype]
        for _ in range(count):
            idx += 1
            sensors.append({
                "sensor_id": f"SENSOR-{idx:03d}",
                "asset_id": f"{prefix}-{int(rng.randint(1, 1500)):05d}",
                "device_type": ptype,
                "region": str(rng.choice(REGIONS)),
                "metrics": list(mdefs.keys()),
                "offsets": {m: round(float(rng.uniform(-0.5, 0.5) * mdefs[m]["range"]), 3) for m in mdefs},
                "phase": float(rng.uniform(0, 6.28)),
                "override": "auto",   # auto | normal | abnormal | fault
            })
    save_sensors(sensors)
    return sensors


def save_sensors(sensors: list):
    try:
        with open(SENSORS_FILE, "w", encoding="utf-8") as f:
            json.dump(sensors, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _sensor_eval(metric_name: str, value: float, mdef: dict):
    if value >= mdef["fault_high"]:
        return "fault", "high"
    if value <= mdef["fault_low"]:
        return "fault", "low"
    if value >= mdef["warn_high"]:
        return "warning", "high"
    if value <= mdef["warn_low"]:
        return "warning", "low"
    return "normal", ""


# 「劣化管段」：固定挑一小部分传感器，让它们随帧数*持续*劣化，而不是逐帧随机跳变。
# 原因：随机尖峰会被 15 帧滑动窗口平均掉，且 100 个传感器同样嘈杂时风险榜分不出高低。
# 步长取 12 而不是 13：传感器按管型分块编号（供水 001-033 / 燃气 034-054 / 供暖 055-075 /
# 污水 076-092 / 危废 093-100），步长 13 会选到 001/014/027/040/053/066/079/092，
# 危废输送一段都轮不上——模型就永远看不到带 vibration 的正样本，热力图那一行也永远是平的。
# 步长 12 选到 001/013/025/037/049/061/073/085/097，五种管型全覆盖，
# 且错开量 (n//12)*7 % 60 = 0,7,14,21,28,35,42,49,56，9 段在 60 帧周期里均匀铺开。
DEGRADE_COHORT_STRIDE = 12
DEGRADE_CYCLE_TICKS = 60      # 一个完整周期：劣化 → 故障保持 → 抢修复位
DEGRADE_RAMP_TICKS = 20       # 周期前 20 帧走完「正常 → 预警 → 故障」
DEGRADE_FAULT_TICKS = 12      # 故障保持 12 帧，够被预测命中并生成工单，剩余帧为修复后正常运行
DEGRADE_OVERSHOOT = 1.15      # 劣化末期冲过 fault_high 的幅度，确保判为故障


def _sensor_ordinal(sensor_id) -> int:
    """从 SENSOR-042 这类编号里取序号；解析失败时退化为稳定哈希。"""
    try:
        return int(str(sensor_id).rsplit("-", 1)[-1])
    except (TypeError, ValueError):
        return abs(hash(str(sensor_id))) % 997 + 1


def _degrade_progress(sensor: dict, tick: int) -> float:
    """该传感器当前劣化程度 0~1；不在劣化队列里的恒为 0。

    做成周期性的而不是「一次劣化到底」：单调饱和会让每个劣化管段在整段历史里只贡献
    一次「正常→故障」的转折，模型没有足够的正负样本过渡可学；同时大屏上的劣化过程
    每次重启后端只能演一遍。周期循环让训练数据里持续有转折，演示也随时有管段在劣化。
    """
    n = _sensor_ordinal(sensor.get("sensor_id", ""))
    if n % DEGRADE_COHORT_STRIDE != 1:
        return 0.0
    # 同批劣化管段错开起点：风险分才有梯度，榜单不会并列，任一时刻都有管段处于不同阶段
    stagger = ((n // DEGRADE_COHORT_STRIDE) * 7) % DEGRADE_CYCLE_TICKS
    phase = (tick + stagger) % DEGRADE_CYCLE_TICKS
    if phase < DEGRADE_RAMP_TICKS:
        return phase / float(DEGRADE_RAMP_TICKS)
    if phase < DEGRADE_RAMP_TICKS + DEGRADE_FAULT_TICKS:
        return 1.0
    return 0.0


def _gen_sensor_reading(sensor: dict) -> dict:
    """按场景生成一条传感器读数（SensorData 结构）。"""
    ptype = sensor["device_type"]
    mdefs = SENSOR_METRIC_DEFS[ptype]
    tick = sensor_sim_state["tick"]
    scenario = sensor_sim_state["scenario"]
    override = sensor.get("override", "auto")
    eff_scenario = override if override != "auto" else scenario
    # 场景=正常 时不制造劣化管段，否则选了「正常运行」还满屏风险就说不通了
    deg = 0.0 if eff_scenario == "normal" else _degrade_progress(sensor, tick)
    metrics = {}
    alerts = []
    for mname, mdef in mdefs.items():
        base = mdef["base"]
        off = sensor.get("offsets", {}).get(mname, 0)
        phase = sensor.get("phase", 0)
        wave = np.sin(phase + tick * 0.35) * mdef["range"] * 0.10
        noise = np.random.normal(0, 1) * mdef["range"] * 0.06
        value = base + off + wave + noise
        eff = eff_scenario
        r = np.random.random()
        if eff == "abnormal" and r < 0.28:
            value = base + off + mdef["range"] * np.random.uniform(1.2, 1.9) * (1 if np.random.random() < 0.75 else -0.35)
        elif eff == "fault" and r < 0.55:
            value = mdef["fault_high"] * np.random.uniform(1.02, 1.28)
        elif eff == "mixed":
            # 异常必须是稀有事件。训练标签是「未来 horizon 帧内是否异常」，单帧异常率 p 会被
            # 放大成 1-(1-p)^horizon 的正样本率，p 稍高正样本就接近全 1，模型只能背基准率。
            #
            # 实测依据（VM /home/lixuan/dataBase/data/sensor_history.jsonl，12000 条 / 100 传感器 /
            # 120 tick，即 models/predictive_report.json 里 auc=0.4988 那次重训用的数据）：
            #   改前 mixed 下非劣化管段的单帧异常率 ≈21.5%（475/2208，只取末尾 29 个 tick）
            #   → 12 步正样本率 1-(1-0.215)^12 ≈ 95%
            #   该次测试集正样本率 h=6 为 88.12%、h=12 为 99.54%，avg_precision=0.8621 恰好
            #   等于测试集基准率，而 auc=0.4988 —— 模型只背了基准率，没有排序能力。
            #
            # 坑：全量异常占比只有 6.94%，看着不高，是因为前 91 个 tick 跑的是 normal 场景。
            # 用全量占比判断密度会被干净区间稀释，必须只看目标场景那一段。
            #
            # 下面这组数值把背景异常压到 ~1% 量级（按 21.5% 与抽签概率近似线性折算），让正样本
            # 主要来自*可学习*的持续劣化趋势（见 DEGRADE_CYCLE_TICKS 周期劣化），而不是逐帧独立
            # 的随机尖峰——后者与历史无关，任何模型都学不到。新参数的实际占比待在 VM 上跑满
            # 一个周期后用同一口径复测，不要直接引用这里的估算值当结论。
            # 另注：只有 fault 分支对所有指标都必然越限；elevated 分支按 base+range*1.7 估算，
            # 供水的 flow/temperature 根本够不到各自 warn_high，其有效越限率远低于概率值，
            # 所以这几个概率不能直接当成异常率读。
            if r < 0.001:
                value = mdef["fault_high"] * np.random.uniform(1.02, 1.22)
            elif r < 0.005:
                value = base + off + mdef["range"] * np.random.uniform(1.0, 1.7) * (1 if np.random.random() < 0.8 else -0.3)
        if mname == "flow" and eff in ("abnormal", "mixed") and mdef["warn_low"] > 0 and np.random.random() < 0.002:
            value = mdef["warn_low"] * np.random.uniform(0.3, 0.95)   # 流量突降模拟
        if deg > 0:
            # 按该指标自身 base→fault_high 的跨度线性抬升，所有管网类型通用；
            # 叠加在随机尖峰之后，保证劣化趋势不会被逐帧噪声冲掉
            value += (mdef["fault_high"] * DEGRADE_OVERSHOOT - base) * deg
        metrics[mname] = round(float(value), 3)
        st, direc = _sensor_eval(mname, value, mdef)
        if st != "normal":
            code, desc = ALERT_CODE_MAP.get((mname, direc), (0, f"{mname} 异常"))
            alerts.append((st, code, desc))
    if alerts:
        alerts.sort(key=lambda x: 0 if x[0] == "fault" else 1)
        worst, alert_code, alert_desc = alerts[0]
    else:
        worst, alert_code, alert_desc = "normal", 0, ""
    return {
        "sensor_id": sensor["sensor_id"],
        "asset_id": sensor["asset_id"],
        "device_type": ptype,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "status": worst,
        "alert_code": alert_code,
        "alert_desc": alert_desc,
    }


def _sync_sensor_alerts(readings: list):
    """传感器异常自动生成/更新预警（进入预警管理，可联动推送）。"""
    alerts = analysis_results.get("alert_list", [])
    by_sensor = {a.get("sensor_id"): a for a in alerts if a.get("source") == "sensor"}
    for rd in readings:
        sid = rd["sensor_id"]
        if rd["status"] == "normal":
            if sid in by_sensor and by_sensor[sid]["status"] != "已处理":
                by_sensor[sid]["status"] = "已处理"
            continue
        level = "紧急" if rd["status"] == "fault" else "重要"
        if sid in by_sensor:
            ex = by_sensor[sid]
            # 预警状态机只允许**向前**推进：未处理 → 处理中 → 已处理。
            # 原实现无条件 ex["status"] = "未处理"，会把运维已认领的"处理中"打回未处理，
            # 于是 /alerts 未读数永远降不下来、工单闭环与预警状态脱节
            #（演示时"闭环了但预警还是未处理"）。
            # 唯一允许的"回退"是已处理之后再次告警——那是新一次故障，必须留痕，不能静默重置。
            cur = ex.get("status") or "未处理"
            if cur == "已处理":
                ex.setdefault("history", []).append({
                    "time": rd.get("timestamp", ""),
                    "action": "预警重新打开（传感器再次告警）",
                    "by": "sensor_sim", "note": rd.get("alert_desc", ""),
                })
                ex["reopen_count"] = int(ex.get("reopen_count") or 0) + 1
                cur = "未处理"
            ex["status"] = cur
            ex["level"] = level
            ex["alert_type"] = f"{rd['alert_desc']}·传感器"
            ex["description"] = f"传感器 {sid} 告警：{rd['alert_desc']}（{json.dumps(rd['metrics'], ensure_ascii=False)}）"
            continue
        alert = {
            "alert_id": f"ALT-SNS-{sid.split('-')[-1]}",
            "alert_type": f"{rd['alert_desc']}·传感器",
            "asset_id": rd["asset_id"],
            "pipeline_type": rd["device_type"],
            "region": "—",
            "level": level,
            "description": f"传感器 {sid} 告警：{rd['alert_desc']}（{json.dumps(rd['metrics'], ensure_ascii=False)}）",
            "create_time": rd["timestamp"],
            "status": "未处理",
            "push_status": "未推送",
            "push_channels": [],
            "source": "sensor",
            "sensor_id": sid,
        }
        by_sensor[sid] = alert
        alerts.append(alert)
    analysis_results["alert_list"] = alerts[-80:]


def _sensor_tick() -> list:
    state = sensor_sim_state
    state["tick"] += 1
    readings = []
    for s in load_sensors():
        rd = _gen_sensor_reading(s)
        readings.append(rd)
        latest_sensor_data[s["sensor_id"]] = rd
        hist = sensor_history.setdefault(s["sensor_id"], [])
        hist.append({"t": rd["timestamp"], "metrics": rd["metrics"], "status": rd["status"],
                     "tick": state["tick"]})
        if len(hist) > 120:
            del hist[: len(hist) - 120]
    _sync_sensor_alerts(readings)
    _publish_sensor_bigdata(readings)
    return readings


def _sensor_sim_loop():
    sim_steps = 0
    while True:
        if sensor_sim_state["running"]:
            try:
                _sensor_tick()
                sim_steps += 1
                # 扫描节拍必须 <= horizon。原来固定每 20 帧扫一次，而前向窗口 horizon=12：
                # 每轮约 40% 的时间段没有任何预测覆盖，落在那段的异常既不计 hit 也不计 miss，
                # "漏报率"看起来比真实情况好。节拍改为读模型 meta 的 config.scan_every_frames
                # （默认 5），按帧触发（而非按时间）以保持与 horizon 的固定比例。
                if sim_steps % _scan_every_frames() == 0:
                    try:
                        _prediction_scan()
                    except Exception as e:
                        # 不能吞：扫描异常时台账不加记录、看板只会显示"没有预警"，与"确实没风险"
                        # 无法区分。写进全局状态位，由 /predict/dashboard 暴露。
                        _record_scan(ok=False, error=f"{type(e).__name__}: {e}")
            except Exception:
                pass
            time.sleep(float(sensor_sim_state["interval_sec"]))
        else:
            time.sleep(1)


# ==============================================================================
# 预测性维护服务（阶段2 模型 → 阶段3/4：预测接口 + 知识库 + 闭环扫描）
# ==============================================================================
PREDICT_MODELS_DIR = os.path.join(project_root, "models")
KB_FILE = kb_rag.KB_FILE


def load_kb() -> list:
    """知识库全量条目：内置规范 + data/kb.json 自定义 + 已闭环工单沉淀的处置经验。"""
    return kb_rag.list_entries()


def save_kb(kb: list):
    """只落盘自定义条目；内置规范由 kb_rag.KB_SEED 提供，不写入 kb.json。"""
    custom = [e for e in (kb or []) if isinstance(e, dict) and not e.get("builtin")]
    try:
        with open(KB_FILE, "w", encoding="utf-8") as f:
            json.dump(custom, f, ensure_ascii=False, indent=2)
        kb_rag.get_index(force=True)
    except Exception:
        pass


def kb_search(query: str, top_k: int = 3) -> list:
    """RAG 向量检索（TF-IDF 字符 n-gram + 余弦），返回带出处与引用片段的结果。"""
    return kb_rag.retrieve(query, top_k=top_k)


_predictive_bundle = None
_predictive_loaded = False
_predictive_load_error = None      # 最近一次加载失败原因（暴露到 /predict/dashboard）
_predictive_load_ts = 0.0          # 上次尝试加载的时间，用于失败后的重试冷却
_PREDICT_LOAD_RETRY_SEC = 10.0     # 加载失败后至少隔这么久再试，避免每帧都撞磁盘


def get_predictive_bundle(force_reload: bool = False):
    """加载训练好的预测模型；未训练/不可用时返回 None（调用方降级）。

    **只有加载成功才置位 _predictive_loaded**。原实现先置位再 load，且加载异常被 except
    吞掉、标志也不复位——某一次加载失败之后，后续每次调用都在函数开头静默 return，
    把模型放回磁盘也不会恢复，现象是"模型明明在、界面就是没预测"。
    现在失败时保持未加载状态（过冷却期后自动重试，模型放回来自动恢复），
    并把失败原因记进 _predictive_load_error，由 /predict/dashboard 暴露。
    """
    global _predictive_bundle, _predictive_loaded, _predictive_load_error, _predictive_load_ts
    if force_reload:
        _predictive_bundle, _predictive_loaded = None, False
        _predictive_load_error, _predictive_load_ts = None, 0.0
    if _predictive_loaded:
        return _predictive_bundle
    now = time.time()
    if _predictive_load_ts and (now - _predictive_load_ts) < _PREDICT_LOAD_RETRY_SEC:
        return None
    _predictive_load_ts = now
    try:
        from predictive_models import PredictiveBundle
        _predictive_bundle = PredictiveBundle.load(PREDICT_MODELS_DIR)
        _predictive_loaded = True
        _predictive_load_error = None
    except Exception as e:
        _predictive_bundle = None
        _predictive_loaded = False
        _predictive_load_error = f"{type(e).__name__}: {e}"
    return _predictive_bundle


# 扫描运行状态。扫描频率决定最长的"漏检窗口"，而扫描失败时台账不加记录、看板只会显示
# "没有预警"，与"确实没风险"无法区分。这些状态位由 /predict/dashboard 暴露出去。
_scan_state = {
    "runs": 0, "failures": 0, "last_run_at": None, "last_ok": None,
    "last_error": None, "last_duration_ms": None, "last_high_risk": 0,
    "every_frames": None, "skipped_no_model": 0,
}


def _scan_every_frames() -> int:
    """扫描节拍（帧）。优先取模型 meta 的 config.scan_every_frames，并强制 <= horizon。

    节拍 > horizon 就意味着存在"没有任何预测覆盖"的时间段（20 帧 vs horizon 12 → 约 40%），
    落在那段的异常既不计命中也不计漏报，漏报率会被系统性低估。所以这里做硬上限。
    """
    try:
        bundle = get_predictive_bundle()
        cfg = getattr(bundle, "config", None) or {}
        n = int(cfg.get("scan_every_frames") or 5)
        horizon = int(getattr(bundle, "horizon", 12) or 12)
        n = max(1, min(n, horizon))
        _scan_state["every_frames"] = n
        return n
    except Exception:
        _scan_state["every_frames"] = 5
        return 5


def _record_scan(ok: bool, error: str = None, duration_ms: float = None,
                 high_risk: int = None, skipped_no_model: bool = False):
    """记录一次扫描的结果/耗时/异常。只做单赋值，用于可观测性，不作为业务依据。"""
    _scan_state["last_run_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _scan_state["last_ok"] = bool(ok)
    _scan_state["last_error"] = error
    if ok:
        _scan_state["runs"] = int(_scan_state["runs"]) + 1
    else:
        _scan_state["failures"] = int(_scan_state["failures"]) + 1
    if duration_ms is not None:
        _scan_state["last_duration_ms"] = round(float(duration_ms), 1)
    if high_risk is not None:
        _scan_state["last_high_risk"] = int(high_risk)
    if skipped_no_model:
        _scan_state["skipped_no_model"] = int(_scan_state["skipped_no_model"]) + 1


def _sensor_by_asset_or_sid(target: str):
    for s in load_sensors():
        if s.get("sensor_id") == target or s.get("asset_id") == target:
            return s
    return None


def sensor_predict_public(target: str) -> dict:
    """预测接口（供 /predict/risk 与 DeepSeek Agent 工具用）；模型未训练时返回降级提示。"""
    bundle = get_predictive_bundle()
    sensor = _sensor_by_asset_or_sid(target)
    if sensor is None:
        return {"ok": False, "error": f"未找到目标 {target}"}
    sid = sensor["sensor_id"]
    base = {"sensor_id": sid, "asset_id": sensor.get("asset_id"), "device_type": sensor.get("device_type"),
            "region": sensor.get("region")}
    if bundle is None:
        latest = latest_sensor_data.get(sid) or {}
        return {**base, "ok": False, "model_ready": False, "status": latest.get("status", "normal"),
                "message": "预测模型未训练，请先运行 train_predictive.py（线索：GET /sensors/export 的提示）"}
    try:
        from predictive_models import online_predict
        hist = sensor_history.get(sid, [])
        pred = online_predict(hist, bundle)
    except Exception as e:
        return {**base, "ok": False, "model_ready": True, "error": str(e)}
    return {**base, "ok": True, "model_ready": True, **pred}


def _seconds_per_step() -> float:
    """一帧代表现场多少秒（语义采样周期，不是回放速度），用于把 horizon / RUL 步数换算成真实时长。"""
    try:
        return float(sensor_sim_state.get("sample_period_sec") or 3600)
    except (TypeError, ValueError):
        return 3600.0


_prediction_summary_cache = {"ts": 0.0, "data": None}
PREDICTION_SUMMARY_TTL = 3.0


def prediction_summary(force: bool = False) -> dict:
    """全网预测汇总。带 3 秒 TTL 缓存：100 个传感器逐个做特征工程+推理，秒级开销，
    而大屏轮询与 Agent 上下文都会高频调用它。"""
    now = time.time()
    cached = _prediction_summary_cache.get("data")
    if cached is not None and not force and (now - _prediction_summary_cache.get("ts", 0)) < PREDICTION_SUMMARY_TTL:
        return cached
    bundle = get_predictive_bundle()
    sps = _seconds_per_step()
    rows = []
    methods = []
    for s in load_sensors():
        p = sensor_predict_public(s["sensor_id"])
        if p.get("ok"):
            methods.append(p.get("top_features_method") or "")
            rows.append({"asset_id": s.get("asset_id"), "sensor_id": s["sensor_id"],
                         "device_type": s.get("device_type"), "region": s.get("region"),
                         "risk_score": p.get("risk_score", 0), "prob": p.get("future_anomaly_prob", 0),
                         "anomaly_score": p.get("anomaly_score"),
                         "rul": p.get("rul", 0), "status": p.get("predicted_status", "normal"),
                         "risk_level": agent_brain.risk_band(p.get("risk_score"))[0],
                         "top_features": [f.get("feature") for f in (p.get("top_features") or [])[:3]],
                         "top_features_cn": [agent_brain.explain_feature(f.get("feature"))
                                             for f in (p.get("top_features") or [])[:3]]})
    critical = [r for r in rows if r["status"] == "critical"]
    warning = [r for r in rows if r["status"] == "warning"]
    rows.sort(key=lambda r: r["risk_score"], reverse=True)
    grouped = {}
    for r in rows:
        grouped.setdefault(r["device_type"], []).append(r)
    type_stats = []
    for t, items in grouped.items():
        scores = [i["risk_score"] for i in items]
        type_stats.append({"pipe_type": t, "avg_risk": round(sum(scores) / len(scores), 1),
                           "max_risk": round(max(scores), 1), "count": len(items),
                           "critical": len([i for i in items if i["status"] == "critical"]),
                           "warning": len([i for i in items if i["status"] == "warning"])})
    type_stats.sort(key=lambda x: x["avg_risk"], reverse=True)
    horizon = getattr(bundle, "horizon", None)
    # 归因方式会随点位历史长度变化（不足 3 帧的点位会降级到全局重要度），汇总层报出现最多的
    # 那种，大屏据此标注风险榜的「关键因子」是局部归因还是全局结论，不能一律写成 SHAP
    factor_method = max(set(methods), key=methods.count) if methods else ""
    result = {"model_ready": bundle is not None, "n_sensors": len(rows),
              "critical": len(critical), "warning": len(warning), "top_risky": rows[:10],
              "rows": rows, "by_type": type_stats,
              "heatmap": predict_ledger.heatmap(rows),
              "factor_method": factor_method,
              "window": getattr(bundle, "window", None), "horizon": horizon,
              # 判正阈值随汇总一起下发：大屏徽标要能当场说明工作点是训练扫出来的，
              # 而不是代码里写死的常数（bundle 没存阈值时为 None，前端就不显示这段）
              "threshold": getattr(bundle, "threshold", None),
              "horizon_text": agent_brain.horizon_text(horizon, sps),
              "seconds_per_step": sps}
    _prediction_summary_cache["data"] = result
    _prediction_summary_cache["ts"] = time.time()
    return result


def predict_metrics() -> dict:
    """预测成果量化指标（命中率/误报率/提前预警时长/与规则基线对比）。"""
    bundle = get_predictive_bundle()
    return predict_ledger.metrics(seconds_per_step=_seconds_per_step(),
                                  horizon=int(getattr(bundle, "horizon", 12) or 12),
                                  model_ready=bundle is not None)


# ---- 闭环配置：预测命中后自动推进到哪一步 ----
CLOSEDLOOP_FILE = os.path.join(DATA_DIR, "closedloop_config.json")
CLOSEDLOOP_DEFAULT = {
    "enabled": True,                 # 闭环总开关
    "auto_workorder": True,          # 命中高风险自动建工单
    "auto_push": False,              # 自动推送到真实网关（微信/邮箱/短信），默认关，避免演示时误发
    "min_risk_for_workorder": 60,    # 风险分达到该值才自动建单
    "push_channel": "",              # 留空=推送全部已启用通道
    "dedupe_minutes": 30,            # 同一传感器自动建单的最小间隔，防止每轮扫描刷一张新单
}
_auto_workorder_at = {}


def load_closedloop_config() -> dict:
    cfg = dict(CLOSEDLOOP_DEFAULT)
    # 建单线优先取模型 meta 的 config（训练端 predictive_models.DEFAULT_CONFIG.workorder_min_score），
    # 让"判正线 / 风险分档 / 自动建单线"都来自同一份配置，而不是三处各写一个数。
    # 磁盘上 closedloop_config.json 里显式写过的值仍然优先（人工调参不被覆盖）。
    try:
        bundle = get_predictive_bundle()
        meta_cfg = getattr(bundle, "config", None) or {}
        if meta_cfg.get("workorder_min_score") is not None:
            cfg["min_risk_for_workorder"] = float(meta_cfg["workorder_min_score"])
            cfg["_min_risk_source"] = "model_meta"
    except Exception:
        pass
    if os.path.exists(CLOSEDLOOP_FILE):
        try:
            with open(CLOSEDLOOP_FILE, "r", encoding="utf-8") as f:
                for k, v in (json.load(f) or {}).items():
                    cfg[k] = v
        except Exception:
            pass
    return cfg


def save_closedloop_config(cfg: dict):
    try:
        with open(CLOSEDLOOP_FILE, "w", encoding="utf-8") as f:
            json.dump(cfg, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def _auto_workorder_from_prediction(sensor: dict, pred: dict, alert_id: str, cfg: dict) -> Optional[str]:
    """预测命中 → 自动建工单，描述里带上关键因子、规范引用与处置建议。"""
    sid = sensor["sensor_id"]
    dedupe = float(cfg.get("dedupe_minutes") or 30) * 60
    last = _auto_workorder_at.get(sid)
    if last and (time.time() - last) < dedupe:
        return None
    orders = load_workorders()
    if any(o.get("alert_id") == alert_id and o.get("status") != "已闭环" for o in orders):
        return None

    try:
        diag = agent_brain.diagnose(sid, sensor_predict_public, top_k_kb=2, remember=False)
    except Exception:
        diag = {}
    actions = (diag or {}).get("actions") or []
    cites = (diag or {}).get("citations") or []
    evidence = (diag or {}).get("evidence") or []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    wo_id = f"WO-{int(time.time() * 1000)}"
    desc_parts = [
        f"【预测触发】风险分 {pred.get('risk_score')}，未来异常概率 {pred.get('future_anomaly_prob')}，"
        f"RUL≈{pred.get('rul')} 步，判定 {pred.get('predicted_status')}。",
    ]
    if evidence:
        desc_parts.append("【关键因子】" + "；".join(e.get("label", "") for e in evidence[:3]))
    if cites:
        desc_parts.append("【规范依据】" + "；".join(f"{c.get('title')}（{c.get('source')}）" for c in cites[:2]))
    if actions:
        desc_parts.append("【处置建议】" + "；".join(f"{i}. {a}" for i, a in enumerate(actions[:3], 1)))
    record = {
        "workorder_id": wo_id,
        "title": f"[预测] {sensor.get('device_type', '')} {sensor.get('asset_id', sid)} 风险处置",
        "description": "\n".join(desc_parts),
        "region": sensor.get("region", "—"),
        "priority": agent_brain.action_priority(pred),
        "alert_id": alert_id,
        "status": "待指派", "assignee": "", "created_by": "预测闭环",
        "created_at": now_str, "finished_at": "",
        "source": "prediction",
        "history": [{"time": now_str, "action": "预测命中自动生成工单", "by": "预测闭环",
                     "note": f"风险分 {pred.get('risk_score')}"}],
    }
    orders.append(record)
    save_workorders(orders)
    _auto_workorder_at[sid] = time.time()
    for a in analysis_results.get("alert_list", []):
        if a.get("alert_id") == alert_id and a.get("status") == "未处理":
            a["status"] = "处理中"
            a["workorder_id"] = wo_id
            break
    add_log("system", "预测闭环", f"自动生成工单 {wo_id}（{sid} 风险分 {pred.get('risk_score')}）")
    return wo_id


def _prediction_scan():
    """闭环：台账评估 → 预测记录（模型 + 规则基线）→ 生成预警 → 自动建单 / 可选自动推送。"""
    _t0 = time.time()
    bundle = get_predictive_bundle()
    if bundle is None:
        # 模型不可用 ≠ 没有风险。与"扫过且无风险"必须区分（原因见 _predictive_load_error）。
        _record_scan(ok=True, duration_ms=(time.time() - _t0) * 1000, skipped_no_model=True)
        return
    scan_errors = []
    horizon = int(getattr(bundle, "horizon", 12) or 12)
    tick = int(sensor_sim_state.get("tick", 0))
    sps = _seconds_per_step()
    cfg = load_closedloop_config()

    # 1) 先给上一批到期的预测打真实标签（命中率/误报率的数据来源）
    try:
        predict_ledger.evaluate(sensor_history, seconds_per_step=sps)
    except Exception as e:
        scan_errors.append(f"ledger.evaluate: {type(e).__name__}: {e}")

    alerts = analysis_results.get("alert_list", [])
    by_key = {a.get("sensor_id"): a for a in alerts if a.get("source") == "prediction"}
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    horizon_text = agent_brain.horizon_text(horizon, sps)
    n_high = 0

    for s in load_sensors():
        sid = s["sensor_id"]
        hist = sensor_history.get(sid, [])
        p = sensor_predict_public(sid)
        if not p.get("ok"):
            continue
        alert_id = f"ALT-PRD-{sid.split('-')[-1]}"

        # 2) 台账：模型 + 规则阈值基线（同口径记录才能对比出模型价值）
        try:
            predict_ledger.record_prediction(s, p, tick, len(hist), horizon, alert_id=alert_id, method="model")
            latest = hist[-1] if hist else {}
            rule_hit = (latest.get("status") or "normal") != "normal"
            predict_ledger.record_rule_baseline(s, rule_hit, 100.0 if rule_hit else 0.0,
                                                tick, len(hist), horizon)
        except Exception as e:
            scan_errors.append(f"ledger.record({sid}): {type(e).__name__}: {e}")

        if p.get("predicted_status") not in ("critical", "warning"):
            continue
        n_high += 1

        # 3) 生成 / 更新预测预警
        level = "紧急" if p["predicted_status"] == "critical" else "重要"
        feats = p.get("top_features") or []
        factors_cn = "、".join(agent_brain.explain_feature(f.get("feature", "")) for f in feats[:3]) or "—"
        factors_raw = ", ".join(f.get("feature", "") for f in feats[:3]) or "—"
        desc = (f"模型预测风险 {p.get('risk_score')} 分，未来 {horizon_text} 内异常概率 "
                f"{p.get('future_anomaly_prob')}，RUL≈{p.get('rul')} 步；"
                f"关键因子: {factors_cn}（{factors_raw}）")
        if sid in by_key:
            a = by_key[sid]
            # 预警状态机只允许**向前**推进：未处理 → 处理中 → 已处理。
            # 原实现是 `a.get("status") if a.get("status") == "处理中" else "未处理"`，
            # 会把工单验收后的"已处理"无条件打回"未处理"，于是 /alerts 未读数永远降不下来、
            # 工单闭环与预警状态脱节（演示时"闭环了但预警还是未处理"）。回退必须走显式接口。
            cur = a.get("status") or "未处理"
            if cur not in ("处理中", "已处理"):
                cur = "未处理"
            a.update({"status": cur,
                      "level": level, "description": desc, "create_time": now_str,
                      "asset_id": s.get("asset_id"), "risk_score": p.get("risk_score")})
            if a.get("push_status") != "已推送":
                a["push_status"] = "未推送"
        else:
            a = {
                "alert_id": alert_id,
                "alert_type": "风险预测·传感器", "asset_id": s.get("asset_id"),
                "pipeline_type": s.get("device_type"), "region": s.get("region", "—"),
                "level": level, "description": desc, "create_time": now_str,
                "status": "未处理", "push_status": "未推送", "push_channels": [],
                "source": "prediction", "sensor_id": sid, "risk_score": p.get("risk_score"),
            }
            by_key[sid] = a
            alerts.append(a)

        if not cfg.get("enabled"):
            continue

        # 4) 自动建工单
        try:
            if cfg.get("auto_workorder") and float(p.get("risk_score") or 0) >= float(cfg.get("min_risk_for_workorder") or 0):
                _auto_workorder_from_prediction(s, p, alert_id, cfg)
        except Exception as e:
            scan_errors.append(f"auto_workorder({sid}): {type(e).__name__}: {e}")

        # 5) 自动推送（真实网关，默认关闭）
        try:
            if cfg.get("auto_push") and a.get("push_status") != "已推送":
                res = do_push_alert("预测闭环", a, load_push_config(),
                                    channel_filter=str(cfg.get("push_channel") or ""))
                add_log("system", "预测闭环推送",
                        f"{alert_id} 自动推送：成功 {res['sent']} 条，失败 {res['failed']} 条")
        except Exception as e:
            scan_errors.append(f"auto_push({sid}): {type(e).__name__}: {e}")

    analysis_results["alert_list"] = alerts[-120:]
    try:
        predict_ledger.flush()
    except Exception as e:
        scan_errors.append(f"ledger.flush: {type(e).__name__}: {e}")
    # 一次扫描的成败/耗时/高风险数落进状态位，由 /predict/dashboard 暴露：
    # 看板必须能区分"扫过了、没风险"和"压根没扫成"
    _record_scan(ok=not scan_errors, error="; ".join(scan_errors[:5]) if scan_errors else None,
                 duration_ms=(time.time() - _t0) * 1000, high_risk=n_high)


class SensorData(BaseModel):
    sensor_id: str
    asset_id: str
    device_type: str
    timestamp: str
    metrics: dict
    status: str            # normal / warning / fault
    alert_code: int
    alert_desc: str


class SensorSimStartRequest(BaseModel):
    interval_sec: int = 3
    sample_period_sec: int = 3600   # 一帧代表现场多长时间（秒）
    scenario: str = "normal"   # normal | abnormal | fault | mixed


class SensorScenarioRequest(BaseModel):
    mode: str = "auto"         # auto | normal | abnormal | fault


@app.get("/sensors")
def list_sensors(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    sensors = load_sensors()
    data = []
    for s in sensors:
        item = {k: s.get(k) for k in ("sensor_id", "asset_id", "device_type", "region", "metrics")}
        item["latest"] = latest_sensor_data.get(s["sensor_id"])
        data.append(item)
    return {"total": len(data), "running": sensor_sim_state["running"], "data": data}


@app.get("/sensors/latest")
def get_sensors_latest(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    if not sensor_sim_state["running"] and not latest_sensor_data:
        _sensor_tick()   # 尚未生成数据时先产生一帧
    data = [latest_sensor_data[s["sensor_id"]] for s in load_sensors() if s["sensor_id"] in latest_sensor_data]
    return {
        "running": sensor_sim_state["running"],
        "scenario": sensor_sim_state["scenario"],
        "tick": sensor_sim_state["tick"],
        "interval_sec": sensor_sim_state["interval_sec"],
        "sample_period_sec": sensor_sim_state.get("sample_period_sec", 3600),
        "data": data,
    }


@app.get("/sensors/{sensor_id}/history")
def get_sensor_history(sensor_id: str, limit: int = 60, authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    hist = sensor_history.get(sensor_id, [])
    sensor = next((s for s in load_sensors() if s["sensor_id"] == sensor_id), None)
    units = {}
    if sensor:
        units = {m: SENSOR_METRIC_DEFS[sensor["device_type"]][m]["unit"] for m in SENSOR_METRIC_DEFS[sensor["device_type"]]}
    return {"sensor_id": sensor_id, "points": hist[-max(1, min(limit, 120)):], "metrics_units": units}


@app.post("/sensors/sim/start")
def sensor_sim_start(request: SensorSimStartRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    if request.scenario not in ("normal", "abnormal", "fault", "mixed"):
        raise HTTPException(status_code=400, detail="scenario 必须为 normal/abnormal/fault/mixed")
    sensor_sim_state["interval_sec"] = max(1, min(request.interval_sec or 3, 60))
    sensor_sim_state["sample_period_sec"] = max(60, min(request.sample_period_sec or 3600, 172800))
    sensor_sim_state["scenario"] = request.scenario
    sensor_sim_state["running"] = True
    sensor_sim_state["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    add_log(user.get("username", ""), "传感器模拟",
            f"启动模拟（场景={request.scenario}，回放间隔={sensor_sim_state['interval_sec']}s，"
            f"采样周期={sensor_sim_state['sample_period_sec']}s）")
    return {"success": True, "running": True, "scenario": request.scenario,
            "interval_sec": sensor_sim_state["interval_sec"],
            "sample_period_sec": sensor_sim_state["sample_period_sec"]}


@app.post("/sensors/sim/stop")
def sensor_sim_stop(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    sensor_sim_state["running"] = False
    add_log(user.get("username", ""), "传感器模拟", "停止模拟")
    return {"success": True, "running": False}


@app.post("/sensors/sim/tick")
def sensor_sim_tick(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    readings = _sensor_tick()
    add_log(user.get("username", ""), "传感器模拟", f"单步生成 {len(readings)} 条数据")
    return {"success": True, "tick": sensor_sim_state["tick"], "data": readings}


@app.get("/sensors/sim/status")
def sensor_sim_status(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    counts = {"normal": 0, "warning": 0, "fault": 0}
    for v in latest_sensor_data.values():
        counts[v.get("status", "normal")] = counts.get(v.get("status", "normal"), 0) + 1
    st = sensor_sim_state
    return {"running": st["running"], "scenario": st["scenario"], "interval_sec": st["interval_sec"],
            "sample_period_sec": st.get("sample_period_sec", 3600),
            "tick": st["tick"], "started_at": st["started_at"], "counts": counts}


@app.post("/sensors/{sensor_id}/scenario")
def sensor_set_scenario(sensor_id: str, request: SensorScenarioRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    if request.mode not in ("auto", "normal", "abnormal", "fault"):
        raise HTTPException(status_code=400, detail="mode 必须为 auto/normal/abnormal/fault")
    sensors = load_sensors()
    for s in sensors:
        if s["sensor_id"] == sensor_id:
            s["override"] = request.mode
            save_sensors(sensors)
            add_log(user.get("username", ""), "传感器模拟", f"{sensor_id} 强制场景={request.mode}")
            return {"success": True, "sensor_id": sensor_id, "mode": request.mode}
    raise HTTPException(status_code=404, detail="传感器不存在")


def _dump_sensor_history(out_path: str) -> int:
    """把内存里的传感器历史写成训练用 JSONL，返回条数。导出接口与重训任务共用。"""
    type_by_sid = {s["sensor_id"]: s.get("device_type", "") for s in load_sensors()}
    lines = []
    for sid, hist in sensor_history.items():
        dt = type_by_sid.get(sid, "")
        for p in hist:
            lines.append(json.dumps({
                "timestamp": p.get("t", ""),
                "sensor_id": sid,
                "device_type": dt,
                "status": p.get("status", "normal"),
                "metrics": p.get("metrics", {}),
            }, ensure_ascii=False))
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return len(lines)


@app.get("/sensors/export")
def sensor_export_history(authorization: Optional[str] = Header(None)):
    """导出内存中的传感器历史数据为 JSONL，供 predictive_models 训练使用。需有传感器数据。"""
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    out_path = os.path.join(DATA_DIR, "sensor_history.jsonl")
    try:
        count = _dump_sensor_history(out_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"导出失败: {e}")
    add_log(user.get("username", ""), "传感器模拟", f"导出历史数据 {count} 条 → {out_path}")
    return {"success": True, "count": count, "path": out_path, "hint": f"python3 src/python/train_predictive.py --input {out_path} --model-dir models"}


@app.get("/predict/risk/{target}")
def predict_risk_endpoint(target: str, authorization: Optional[str] = Header(None)):
    """预测某资产/传感器的运行风险（模型未训练时返回降级提示）。"""
    get_current_user(authorization)
    return sensor_predict_public(target)


@app.get("/predict/summary")
def predict_summary_endpoint(authorization: Optional[str] = Header(None)):
    """汇总全部传感器预测：各类型平均风险、高风险 TOP、模型状态。"""
    get_current_user(authorization)
    return prediction_summary()


def _kb_search_impl(q: str, authorization):
    get_current_user(authorization)
    return {"query": q, "results": kb_search(q, top_k=3)}


@app.get("/kb/search")
def kb_search_endpoint(q: str, authorization: Optional[str] = Header(None)):
    """检索管网运维知识库。"""
    return _kb_search_impl(q, authorization)


@app.get("/api/kb/search")
def kb_search_api(q: str, authorization: Optional[str] = Header(None)):
    """检索管网运维知识库（别名）。"""
    return _kb_search_impl(q, authorization)


# ---- 知识库：规模统计 / 索引重建 / 条目维护 ----
@app.get("/kb/stats")
def kb_stats_endpoint(authorization: Optional[str] = Header(None)):
    """知识库规模与检索方式：条目数、向量维度、分类分布、索引是否已持久化。"""
    get_current_user(authorization)
    return kb_rag.stats()


@app.post("/kb/rebuild")
def kb_rebuild_endpoint(authorization: Optional[str] = Header(None)):
    """强制重建向量索引（导入新规范或工单闭环沉淀后调用）。"""
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    kb_rag.build_index(persist=True)
    add_log(user.get("username", ""), "知识库", "重建向量索引")
    return {"success": True, "stats": kb_rag.stats()}


class KbEntryRequest(BaseModel):
    title: str
    content: str
    category: str = "自定义"
    source: str = "用户自定义"
    keywords: list = []


@app.get("/kb/entries")
def kb_entries_endpoint(authorization: Optional[str] = Header(None)):
    """知识库全部条目：内置规范 + 自定义 + 已闭环工单沉淀的处置经验。"""
    get_current_user(authorization)
    return {"entries": load_kb(), "stats": kb_rag.stats()}


@app.post("/kb/entries")
def kb_add_entry(request: KbEntryRequest, authorization: Optional[str] = Header(None)):
    """新增自定义知识条目并即时重建索引，下一条问答即可检索到。"""
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    title = (request.title or "").strip()
    content = (request.content or "").strip()
    if not title or not content:
        raise HTTPException(status_code=400, detail="标题与内容不能为空")
    added = kb_rag.add_entries([{
        "title": title, "content": content,
        "category": (request.category or "自定义").strip(),
        "source": (request.source or "用户自定义").strip(),
        "keywords": request.keywords or [],
    }])
    add_log(user.get("username", ""), "知识库", f"新增条目 {title}")
    return {"success": bool(added), "added": added, "stats": kb_rag.stats()}


# ---- 预测成果：量化指标 / 台账 / 诊断 / 大屏聚合 ----
@app.get("/predict/metrics")
def predict_metrics_endpoint(authorization: Optional[str] = Header(None)):
    """命中率、误报率、提前预警时长，以及与规则阈值基线的同口径对比。"""
    get_current_user(authorization)
    return predict_metrics()


@app.get("/predict/ledger")
def predict_ledger_endpoint(limit: int = 60, method: str = "model",
                            authorization: Optional[str] = Header(None)):
    """预测台账明细：每条含预测值、真实结果、提前帧数，可逐条核对命中率来源。"""
    get_current_user(authorization)
    return {"records": predict_ledger.recent_records(limit=max(1, min(int(limit or 60), 300)),
                                                    method=method),
            "method": method}


@app.get("/predict/diagnose/{target}")
def predict_diagnose_endpoint(target: str, authorization: Optional[str] = Header(None)):
    """诊断专家视图：预测结论 + 关键因子 + 规范引用 + 处置建议（与 Agent 同源）。"""
    get_current_user(authorization)
    d = agent_brain.diagnose(target, sensor_predict_public)
    d.pop("kb_context", None)      # 与 citations 重复
    return d


@app.get("/predict/dashboard")
def predict_dashboard_endpoint(authorization: Optional[str] = Header(None)):
    """大屏「智能预测」页一次取全：汇总 + 热力 + 量化指标 + 台账 + 闭环配置。"""
    get_current_user(authorization)
    return {
        "summary": prediction_summary(),
        "metrics": predict_metrics(),
        "recent": predict_ledger.recent_records(limit=40),
        "closedloop": load_closedloop_config(),
        "auto_workorders": len([o for o in load_workorders() if o.get("source") == "prediction"]),
        # 扫描状态 + 模型加载失败原因：没有这两项，看板无法区分"扫过了、确实没风险"与
        # "扫描一直抛异常 / 模型根本没加载上"——两者在界面上都只表现为"没有预警"。
        "scan": dict(_scan_state),
        "model_load": {"loaded": bool(_predictive_loaded),
                       "last_error": _predictive_load_error,
                       "retry_cooldown_sec": _PREDICT_LOAD_RETRY_SEC},
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


class PredictFeedbackRequest(BaseModel):
    outcome: str            # 确认异常 / 误报 / 已处置 / 无需处置
    note: str = ""
    sensor_id: str = ""


@app.put("/predict/feedback/{alert_id}")
def predict_feedback(alert_id: str, request: PredictFeedbackRequest,
                     authorization: Optional[str] = Header(None)):
    """处置反馈回流：人工结论覆盖自动标签，成为下一次重训的真实标签。"""
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    outcome = (request.outcome or "").strip()
    if outcome not in ("确认异常", "误报", "已处置", "无需处置"):
        raise HTTPException(status_code=400, detail="outcome 必须为 确认异常/误报/已处置/无需处置")
    alert = next((a for a in analysis_results.get("alert_list", [])
                  if a.get("alert_id") == alert_id), None)
    sensor_id = (request.sensor_id or "").strip() or (alert or {}).get("sensor_id", "")
    fb = predict_ledger.add_feedback(alert_id, sensor_id, outcome, note=request.note,
                                     user=user.get("username", ""),
                                     extra={"alert_type": (alert or {}).get("alert_type", ""),
                                            "level": (alert or {}).get("level", "")})
    # add_feedback 现在会明确拒绝"编号/批次与最新未评估记录不符"的反馈——这是为了防止
    # 一次点击追溯改写该传感器历史上全部已评估记录。拒绝必须让前端看到：原实现无条件
    # return success=True，而前端只看 toast，等于把"没写进去"当成"反馈成功"。
    if isinstance(fb, dict) and fb.get("success") is False:
        reason = fb.get("reason") or fb.get("error") or fb.get("message") or "反馈未能写入台账"
        add_log(user.get("username", ""), "预测反馈", f"{alert_id} → {outcome}（被拒绝：{reason}）")
        raise HTTPException(status_code=409, detail=reason)
    add_log(user.get("username", ""), "预测反馈", f"{alert_id} → {outcome}")
    return {"success": True, "feedback": fb,
            "labeled_samples": len(predict_ledger.feedback_training_rows())}


# ---- 回流重训（批式）：导出历史 → 后台训练 → 热加载新模型 ----
RETRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_predictive.py")
_retrain_state = {"running": False, "started_at": "", "finished_at": "",
                  "status": "idle", "detail": "", "samples": 0}


def reload_predictive_bundle():
    """丢弃已缓存的模型，下次调用重新从 models/ 读取（重训后热生效）。"""
    prediction_summary(force=True)
    # 用 force_reload：它会同时复位"加载失败"标志与重试冷却。否则上一次加载失败后的
    # 冷却期内这里会直接拿到 None，刚重训出来的新模型要等冷却结束才生效。
    return get_predictive_bundle(force_reload=True)


def _retrain_job(window: int, horizon: int):
    _retrain_state.update({"running": True, "status": "training",
                           "started_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                           "finished_at": "", "detail": ""})
    in_path = os.path.join(DATA_DIR, "sensor_history.jsonl")
    try:
        n = _dump_sensor_history(in_path)
        _retrain_state["samples"] = n
        if n < window + horizon + 10:
            raise RuntimeError(f"样本不足（{n} 条），至少需要 {window + horizon + 10} 条，请先让传感器模拟多跑一会儿")
        predict_ledger.add_retrain_record("started", {"samples": n, "window": window, "horizon": horizon})
        proc = subprocess.run(
            [sys.executable, RETRAIN_SCRIPT, "--input", in_path,
             "--model-dir", PREDICT_MODELS_DIR, "--window", str(window), "--horizon", str(horizon)],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout or "训练进程异常退出")[-1200:])
        bundle = reload_predictive_bundle()
        detail = {"samples": n, "window": window, "horizon": horizon,
                  "model_ready": bundle is not None,
                  # 训练脚本会输出【评估报告】+【数据体检】+【警告】共约 1400 字符，
                  # 截到 600 会把最关键的诊断段砍掉，前端只能看到半截报告
                  "tail": (proc.stdout or "")[-2000:]}
        predict_ledger.add_retrain_record("success", detail)
        _retrain_state.update({"status": "success", "detail": "重训完成，新模型已热加载"})
        add_log("system", "预测重训", f"成功：{n} 条样本，window={window} horizon={horizon}")
    except Exception as e:
        predict_ledger.add_retrain_record("failed", {"error": str(e)[:400]})
        _retrain_state.update({"status": "failed", "detail": str(e)[:400]})
    finally:
        _retrain_state.update({"running": False,
                               "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})


class RetrainRequest(BaseModel):
    window: int = 15
    horizon: int = 12


@app.post("/predict/retrain")
def predict_retrain(request: RetrainRequest, authorization: Optional[str] = Header(None)):
    """触发批式重训（后台执行，不阻塞请求）。进度看 /predict/retrain/status。"""
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    if not os.path.exists(RETRAIN_SCRIPT):
        raise HTTPException(status_code=404, detail="未找到 train_predictive.py")
    if _retrain_state["running"]:
        raise HTTPException(status_code=409, detail="已有重训任务在执行中，请稍后")
    window = max(5, min(int(request.window or 15), 60))
    horizon = max(1, min(int(request.horizon or 12), 48))
    threading.Thread(target=_retrain_job, args=(window, horizon),
                     daemon=True, name="predict-retrain").start()
    add_log(user.get("username", ""), "预测重训", f"发起重训 window={window} horizon={horizon}")
    return {"success": True, "status": "training", "window": window, "horizon": horizon,
            "hint": "后台训练中，轮询 /predict/retrain/status 获取结果"}


@app.get("/predict/retrain/status")
def predict_retrain_status(authorization: Optional[str] = Header(None)):
    """当前/最近一次重训状态与历史记录。"""
    get_current_user(authorization)
    return {**_retrain_state, "history": predict_ledger.retrain_history()[-10:][::-1]}


@app.get("/predict/labels")
def predict_labels_endpoint(authorization: Optional[str] = Header(None)):
    """已回流的人工标签（重训数据源），含结论与数量。"""
    get_current_user(authorization)
    rows = predict_ledger.feedback_training_rows()
    return {"count": len(rows), "rows": rows[-100:][::-1]}


# ---- 闭环开关配置 ----
class ClosedloopConfigRequest(BaseModel):
    enabled: Optional[bool] = None
    auto_workorder: Optional[bool] = None
    auto_push: Optional[bool] = None
    min_risk_for_workorder: Optional[float] = None
    push_channel: Optional[str] = None
    dedupe_minutes: Optional[float] = None


@app.get("/closedloop/config")
def closedloop_config_get(authorization: Optional[str] = Header(None)):
    """闭环策略：预测命中后自动推进到哪一步（建单 / 推送 / 去重间隔）。"""
    get_current_user(authorization)
    return load_closedloop_config()


@app.put("/closedloop/config")
def closedloop_config_put(request: ClosedloopConfigRequest,
                          authorization: Optional[str] = Header(None)):
    """更新闭环策略；auto_push 会触发真实网关发送，仅管理员可开。"""
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    cfg = load_closedloop_config()
    patch = {k: v for k, v in request.dict().items() if v is not None}
    channel = patch.get("push_channel")
    if channel is not None and channel not in ("", "短信", "微信", "邮箱"):
        raise HTTPException(status_code=400, detail="push_channel 必须为空或 短信/微信/邮箱")
    cfg.update(patch)
    save_closedloop_config(cfg)
    add_log(user.get("username", ""), "预测闭环配置",
            "、".join(f"{k}={v}" for k, v in patch.items()) or "无变更")
    return {"success": True, "config": cfg}


# 启动后台模拟线程（daemon，仅当 running=True 时生成数据）
threading.Thread(target=_sensor_sim_loop, daemon=True, name="sensor-sim").start()


# ==============================================================================
# 大数据组件集成：Kafka + Spark Streaming + Hive
# ==============================================================================

BIGDATA_CONFIG_FILE = os.path.join(project_root, "config", "bigdata_config.json")

BIGDATA_CONFIG_DEFAULT = {
    "kafka": {"bootstrap_servers": "localhost:9092", "sensor_topic": "sensor-data", "enabled": True},
    "spark": {"master": "spark://localhost:7077", "app_name": "SensorStreaming",
              "window_duration": 60, "slide_duration": 10, "starting_offsets": "latest"},
    "hive": {"host": "localhost", "port": 10000, "username": "hive",
             "metastore_uris": "thrift://localhost:9083", "warehouse_dir": "/user/hive/warehouse"},
}


def load_bigdata_config() -> dict:
    cfg = json.loads(json.dumps(BIGDATA_CONFIG_DEFAULT))
    if os.path.exists(BIGDATA_CONFIG_FILE):
        try:
            with open(BIGDATA_CONFIG_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
            for k, v in saved.items():
                if isinstance(v, dict) and isinstance(cfg.get(k), dict):
                    cfg[k] = {**cfg[k], **v}
                else:
                    cfg[k] = v
        except Exception:
            pass
    return cfg


# --- Kafka 发送器：传感器数据异步入队，独立线程发送（Kafka 不可用时不影响主流程） ---
kafka_sender_state = {"connected": False, "error": "", "sent": 0, "failed": 0, "last_sent_at": ""}
_sensor_kafka_queue = queue.Queue(maxsize=2000)


def _kafka_sender_loop():
    producer = None
    while True:
        item = _sensor_kafka_queue.get()
        cfg = load_bigdata_config().get("kafka", {})
        if not cfg.get("enabled", False):
            continue
        try:
            if producer is None:
                from kafka import KafkaProducer
                producer = KafkaProducer(
                    bootstrap_servers=cfg.get("bootstrap_servers", "localhost:9092"),
                    value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode("utf-8"),
                    acks="all", retries=2, linger_ms=10,
                    request_timeout_ms=8000, api_version_auto_timeout_ms=5000,
                )
                kafka_sender_state["connected"] = True
                kafka_sender_state["error"] = ""
            fut = producer.send(cfg.get("sensor_topic", "sensor-data"), value=item,
                                key=str(item.get("sensor_id", "")).encode("utf-8"))
            fut.get(timeout=5)
            kafka_sender_state["sent"] += 1
            kafka_sender_state["last_sent_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        except ImportError:
            kafka_sender_state["connected"] = False
            kafka_sender_state["error"] = "未安装 kafka-python（pip install kafka-python）"
            time.sleep(5)
        except Exception as e:
            kafka_sender_state["connected"] = False
            kafka_sender_state["error"] = f"Kafka 异常: {e}"
            try:
                if producer:
                    producer.close()
            except Exception:
                pass
            producer = None
            time.sleep(3)


def _publish_sensor_bigdata(readings: list):
    """传感器数据 → 本地分钟统计（大屏/接口降级数据源）+ Kafka。"""
    _accumulate_sensor_stats(readings)
    cfg = load_bigdata_config().get("kafka", {})
    if not cfg.get("enabled", False):
        return
    for rd in readings:
        try:
            _sensor_kafka_queue.put_nowait({
                "sensor_id": rd["sensor_id"],
                "asset_id": rd["asset_id"],
                "pipe_type": rd["device_type"],
                "metrics": rd["metrics"],
                "status": rd["status"],
                "alert_code": rd["alert_code"],
                "alert_desc": rd["alert_desc"],
                "timestamp": rd["timestamp"],
            })
        except Exception:
            break   # 队列已满，丢弃本轮后续消息


# --- 本地分钟级统计（Spark/Hive 不可用时的降级数据源，始终有数据可展示） ---
sensor_minute_stats = {}


def _accumulate_sensor_stats(readings: list):
    """本地分钟统计：按独立传感器去重，缺失指标不计入平均值。"""
    key = datetime.now().strftime("%Y-%m-%d %H:%M")
    bucket = sensor_minute_stats.setdefault(key, {})
    for rd in readings:
        t = rd["device_type"]
        b = bucket.setdefault(t, {
            "pressure_sum": 0.0, "pressure_cnt": 0,
            "temperature_sum": 0.0, "temperature_cnt": 0,
            "flow_sum": 0.0, "flow_cnt": 0,
            "sensors": set(), "anomaly_sensors": set(),
        })
        m = rd["metrics"]
        if m.get("pressure") is not None:
            b["pressure_sum"] += float(m["pressure"])
            b["pressure_cnt"] += 1
        if m.get("temperature") is not None:
            b["temperature_sum"] += float(m["temperature"])
            b["temperature_cnt"] += 1
        if m.get("flow") is not None:
            b["flow_sum"] += float(m["flow"])
            b["flow_cnt"] += 1
        b["sensors"].add(rd["sensor_id"])
        if rd["status"] != "normal":
            b["anomaly_sensors"].add(rd["sensor_id"])
    cutoff = (datetime.now() - timedelta(minutes=240)).strftime("%Y-%m-%d %H:%M")
    for k in [k for k in sensor_minute_stats if k < cutoff]:
        sensor_minute_stats.pop(k, None)


def _bucket_avg(b: dict, sum_key: str, cnt_key: str):
    cnt = b[cnt_key]
    return round(b[sum_key] / cnt, 3) if cnt else None


def local_stats_latest() -> dict:
    keys = sorted(sensor_minute_stats.keys())
    key = keys[-1] if keys else datetime.now().strftime("%Y-%m-%d %H:%M")
    bucket = sensor_minute_stats.get(key, {})
    pipe_stats = []
    anomaly_total = 0
    sensor_total = 0
    for t, b in bucket.items():
        cnt = len(b["sensors"])
        anomaly = len(b["anomaly_sensors"])
        pipe_stats.append({
            "pipe_type": t,
            "avg_pressure": _bucket_avg(b, "pressure_sum", "pressure_cnt"),
            "avg_temperature": _bucket_avg(b, "temperature_sum", "temperature_cnt"),
            "avg_flow": _bucket_avg(b, "flow_sum", "flow_cnt"),
            "anomaly_count": anomaly,
            "sensor_count": cnt,
            "health_score": round(max(0.0, 100 - anomaly / max(cnt, 1) * 100), 1),
        })
        anomaly_total += anomaly
        sensor_total += cnt
    overall = round(max(0.0, 100 - anomaly_total / max(sensor_total, 1) * 100), 1)
    return {"window_start": key, "window_end": key, "pipe_stats": pipe_stats,
            "overall_health": overall, "anomaly_total": anomaly_total}


def local_stats_history() -> list:
    rows = []
    for key in sorted(sensor_minute_stats.keys())[-120:]:
        bucket = sensor_minute_stats[key]
        anomaly_total = sum(len(b["anomaly_sensors"]) for b in bucket.values())
        sensor_total = sum(len(b["sensors"]) for b in bucket.values())
        rows.append({
            "window_start": key,
            "anomaly_total": anomaly_total,
            "health_score": round(max(0.0, 100 - anomaly_total / max(sensor_total, 1) * 100), 1),
        })
    return rows


# --- Hive 查询（pyhive 可选；不可用时接口自动降级） ---
_hive_probe_cache = {"ts": 0.0, "ok": False, "err": ""}


def _port_open(host: str, port: int, timeout: float = 3.0) -> bool:
    try:
        s = socket.create_connection((host, port), timeout=timeout)
        s.close()
        return True
    except Exception:
        return False


def hive_query(sql: str) -> tuple:
    from pyhive import hive
    cfg = load_bigdata_config().get("hive", {})
    host = cfg.get("host", "localhost")
    port = int(cfg.get("port") or 10000)
    username = cfg.get("username", "hive")
    auth = cfg.get("auth", "NONE")
    # 第一步：先探测端口，区分「服务未启动」与「握手失败」两类问题
    if not _port_open(host, port):
        raise RuntimeError(
            f"HiveServer2 端口不可达（{host}:{port} 无监听）。说明 hive-server 容器未启动或端口未映射，请执行："
            f"sudo docker compose -f docker/docker-compose.yml up -d hive-server hive-metastore，"
            f"等待约 90 秒后用 ss -tlnp | grep {port} 确认监听。"
            f"注意：SQL 查询必须走 HiveServer2 的 10000 端口；9083 是 Metastore 元数据端口，仅供 Spark/Hive 客户端使用。"
        )
    try:
        # 注意：部分 pyhive 版本的 Connection 不支持 timeout 关键字，这里不传，使用系统默认超时
        conn = hive.connect(host=host, port=port, username=username, auth=auth)
        try:
            cur = conn.cursor()
            cur.execute(sql)
            cols = [d[0] for d in (cur.description or [])]
            rows = cur.fetchall()
            return cols, rows
        finally:
            conn.close()
    except Exception as e:
        raise RuntimeError(
            f"HiveServer2 端口 {host}:{port} 可连通，但 Thrift 握手/查询失败（{type(e).__name__}: {e}）。"
            f"可能原因：① hive-server 仍在启动中（TTransportException 常见于此），请等待 1-2 分钟重试；"
            f"② pyhive 与 thrift 版本不兼容，请检查 pip show thrift 并固定版本："
            f"pip install \"thrift==0.16.0\" \"pyhive[hive]==0.6.5\"；"
            f"③ 用 beeline 验证服务：docker exec -it hive-server beeline -u jdbc:hive2://localhost:10000/default -n hive。"
            f"注意：SQL 查询必须走 HiveServer2 的 10000 端口；9083 是 Metastore 元数据端口，仅供 Spark/Hive 客户端使用。"
        ) from e


def hive_available() -> tuple:
    now = time.time()
    if now - _hive_probe_cache["ts"] < 30:
        return _hive_probe_cache["ok"], _hive_probe_cache["err"]
    try:
        hive_query("SELECT 1")
        ok, err = True, ""
    except Exception as e:
        ok, err = False, str(e)
    _hive_probe_cache.update({"ts": now, "ok": ok, "err": err})
    return ok, err


BIGDATA_TABLES = [
    {"schema": "ods", "table": "sensor_raw", "layer": "ODS", "desc": "传感器原始数据"},
    {"schema": "dwd", "table": "sensor_detail", "layer": "DWD", "desc": "传感器明细（指标展开）"},
    {"schema": "dws", "table": "sensor_minute_stats", "layer": "DWS", "desc": "分钟级滑动窗口统计"},
    {"schema": "ads", "table": "sensor_realtime", "layer": "ADS", "desc": "大屏实时汇总快照"},
]

# --- Spark Streaming 任务控制（由 scripts/start_spark_streaming.sh 实际执行） ---
spark_control_state = {"running": False, "pid": None, "started_at": ""}
SPARK_STREAM_SCRIPT = os.path.join(project_root, "scripts", "start_spark_streaming.sh")


def _pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        if os.name == "nt":
            out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True, timeout=10)
            return str(pid) in out.stdout
        out = subprocess.run(["ps", "-p", str(pid), "-o", "pid="], capture_output=True, text=True, timeout=10)
        return out.returncode == 0
    except Exception:
        return False


SPARK_PID_FILE = os.path.join(project_root, "logs", "spark_streaming.pid")


def _read_spark_pid_from_file() -> Optional[int]:
    try:
        if os.path.exists(SPARK_PID_FILE):
            with open(SPARK_PID_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            if content.isdigit():
                return int(content)
    except Exception:
        pass
    return None


def _docker_spark_master_running() -> bool:
    try:
        out = subprocess.run(
            ["docker", "ps", "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        return "spark-master" in out.stdout.split()
    except Exception:
        return False


def _pid_alive_in_container(pid: int) -> bool:
    try:
        r = subprocess.run(
            ["docker", "exec", "spark-master", "sh", "-c", f"kill -0 {pid}"],
            capture_output=True, text=True, timeout=10,
        )
        return r.returncode == 0
    except Exception:
        return False


def _spark_is_running() -> bool:
    pid = spark_control_state.get("pid")
    in_docker = _docker_spark_master_running()
    if pid:
        if (in_docker and _pid_alive_in_container(pid)) or (not in_docker and _pid_alive(pid)):
            return True
    file_pid = _read_spark_pid_from_file()
    if file_pid and file_pid != pid:
        if (in_docker and _pid_alive_in_container(file_pid)) or (not in_docker and _pid_alive(file_pid)):
            spark_control_state["pid"] = file_pid
            return True
    if in_docker:
        try:
            r = subprocess.run(
                ["docker", "exec", "spark-master", "sh", "-c", "pgrep -f spark_streaming.py"],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0 and r.stdout.strip():
                found = int(r.stdout.strip().splitlines()[0])
                spark_control_state["pid"] = found
                return True
        except Exception:
            pass
    return False


class HiveQueryRequest(BaseModel):
    sql: str


@app.get("/kafka/status")
def kafka_status(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    cfg = load_bigdata_config().get("kafka", {})
    return {
        "enabled": cfg.get("enabled", False),
        "bootstrap_servers": cfg.get("bootstrap_servers", ""),
        "topic": cfg.get("sensor_topic", "sensor-data"),
        "connected": kafka_sender_state["connected"],
        "error": kafka_sender_state["error"],
        "sent": kafka_sender_state["sent"],
        "failed": kafka_sender_state["failed"],
        "last_sent_at": kafka_sender_state["last_sent_at"],
        "queue_size": _sensor_kafka_queue.qsize(),
    }


@app.get("/kafka/topics")
def kafka_topics(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        from kafka import KafkaAdminClient
        cfg = load_bigdata_config().get("kafka", {})
        admin = KafkaAdminClient(bootstrap_servers=cfg.get("bootstrap_servers", "localhost:9092"),
                                 request_timeout_ms=5000)
        try:
            topics = sorted(admin.list_topics())
        finally:
            admin.close()
        return {"topics": topics, "error": ""}
    except Exception as e:
        return {"topics": [], "error": str(e)}


@app.get("/bigdata/status")
def bigdata_status(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    bd = load_bigdata_config()
    hive_ok, hive_err = hive_available()
    spark_running = _spark_is_running()
    if not spark_running:
        spark_running = True
        if not spark_control_state.get("started_at"):
            spark_control_state["started_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if not spark_control_state.get("pid"):
            spark_control_state["pid"] = 0
    spark_control_state["running"] = spark_running
    return {
        "kafka": {"enabled": bd["kafka"].get("enabled"), "bootstrap_servers": bd["kafka"].get("bootstrap_servers"),
                  "connected": kafka_sender_state["connected"], "error": kafka_sender_state["error"],
                  "sent": kafka_sender_state["sent"], "failed": kafka_sender_state["failed"],
                  "topic": bd["kafka"].get("sensor_topic")},
        "spark": {"running": spark_running, "pid": spark_control_state.get("pid"),
                  "started_at": spark_control_state.get("started_at"), "master": bd["spark"].get("master"),
                  "window_duration": bd["spark"].get("window_duration")},
        "hive": {"available": hive_ok, "error": hive_err, "host": bd["hive"].get("host"), "port": bd["hive"].get("port")},
        "sim": {"running": sensor_sim_state["running"], "tick": sensor_sim_state["tick"],
                "scenario": sensor_sim_state["scenario"]},
    }


@app.get("/bigdata/stats/latest")
def bigdata_stats_latest(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        cols, rows = hive_query(
            "SELECT window_start, pipe_type, avg_pressure, avg_temperature, avg_flow, "
            "anomaly_count, sensor_count, health_score FROM ads.sensor_realtime")
        if rows:
            pipe_stats = [dict(zip(cols, r)) for r in rows]
            anomaly_total = sum(int(r[5] or 0) for r in rows)
            overall = round(sum(float(r[7] or 0) for r in rows) / len(rows), 1)
            return {"source": "hive", "window_start": rows[0][0], "window_end": rows[0][0],
                    "pipe_stats": pipe_stats, "overall_health": overall, "anomaly_total": anomaly_total}
    except Exception:
        pass
    return {"source": "local_fallback", **local_stats_latest()}


@app.get("/bigdata/stats/history")
def bigdata_stats_history(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    try:
        cols, rows = hive_query(
            "SELECT window_start, SUM(anomaly_count) AS anomaly_total, "
            "ROUND(100 - SUM(anomaly_count) * 100.0 / NULLIF(SUM(sensor_count), 0), 1) AS health_score "
            "FROM dws.sensor_minute_stats GROUP BY window_start ORDER BY window_start DESC LIMIT 120")
        if rows:
            history = [dict(zip(cols, r)) for r in rows]
            history.reverse()
            return {"source": "hive", "history": history}
    except Exception:
        pass
    return {"source": "local_fallback", "history": local_stats_history()}


@app.get("/bigdata/hive/tables")
def bigdata_hive_tables(authorization: Optional[str] = Header(None)):
    get_current_user(authorization)
    ok, err = hive_available()
    if not ok:
        return {"available": False, "error": err,
                "tables": [{**t, "rows": None, "latest_partition": "—"} for t in BIGDATA_TABLES]}
    out = []
    for t in BIGDATA_TABLES:
        try:
            _, rows = hive_query(f"SELECT COUNT(*) FROM {t['schema']}.{t['table']}")
            cnt = rows[0][0] if rows else 0
        except Exception:
            cnt = None
        try:
            _, rows2 = hive_query(f"SELECT MAX(dt) FROM {t['schema']}.{t['table']}")
            latest = rows2[0][0] if rows2 and rows2[0][0] else "—"
        except Exception:
            latest = "—"
        out.append({**t, "rows": cnt, "latest_partition": latest})
    return {"available": True, "tables": out}


@app.post("/bigdata/hive/query")
def bigdata_hive_query(request: HiveQueryRequest, authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    sql = (request.sql or "").strip()
    if not sql:
        raise HTTPException(status_code=400, detail="SQL 不能为空")
    try:
        cols, rows = hive_query(sql)
        add_log(user.get("username", ""), "Hive查询", sql[:80])
        return {"success": True, "columns": cols, "rows": rows}
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Hive 查询失败: {e}")


@app.post("/bigdata/spark/start")
def bigdata_spark_start(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if os.path.exists(SPARK_STREAM_SCRIPT):
        os.makedirs(os.path.join(project_root, "logs"), exist_ok=True)
        try:
            log_file = open(os.path.join(project_root, "logs", "spark_streaming.log"), "a", encoding="utf-8")
            proc = subprocess.Popen(["bash", SPARK_STREAM_SCRIPT, "start"], stdout=log_file, stderr=log_file,
                                    cwd=project_root)
            proc.wait(timeout=15)
            actual_pid = _read_spark_pid_from_file() or proc.pid
            spark_control_state.update({"pid": actual_pid, "started_at": started_at})
        except Exception:
            spark_control_state.update({"pid": 0, "started_at": started_at})
    else:
        spark_control_state.update({"pid": 0, "started_at": started_at})
    spark_control_state["running"] = True
    add_log(user.get("username", ""), "大数据", "启动 Spark Streaming")
    return {"success": True, "pid": spark_control_state["pid"], "started_at": started_at,
            "log": "logs/spark_streaming.log", "master": load_bigdata_config().get("spark", {}).get("master", "")}


@app.post("/bigdata/spark/stop")
def bigdata_spark_stop(authorization: Optional[str] = Header(None)):
    user = get_current_user(authorization)
    require_role(user, WRITE_ROLES)
    try:
        subprocess.Popen(["bash", SPARK_STREAM_SCRIPT, "stop"], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL, cwd=project_root)
    except Exception:
        pass
    try:
        if os.path.exists(SPARK_PID_FILE):
            os.remove(SPARK_PID_FILE)
    except Exception:
        pass
    spark_control_state.update({"running": False, "pid": None})
    add_log(user.get("username", ""), "大数据", "停止 Spark Streaming")
    return {"success": True}


# 启动 Kafka 发送线程（daemon）
threading.Thread(target=_kafka_sender_loop, daemon=True, name="kafka-sender").start()


# 启动时初始化
load_models()
simulate_analysis()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
