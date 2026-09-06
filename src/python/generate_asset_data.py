#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市管网资产数字化台账 - 模拟数据生成器

生成3类数据：
1. 管网资产数据（5类管网，含完整属性）
2. 生命周期事件（采购/施工/运维/巡检/改造/报废）
3. 盘点记录（扫码盘点/巡检盘点，含差异标记）
"""

import json
import os
import random
import argparse
from datetime import datetime, timedelta

# ==============================================================================
# 管网类型配置
# ==============================================================================

PIPELINE_TYPES = [
    {
        "type_code": "WSP",
        "type_name": "供水管网",
        "prefix": "WSP",
        "diameters": ["DN100", "DN150", "DN200", "DN300", "DN400", "DN600", "DN800", "DN1000", "DN1200"],
        "materials": ["球墨铸铁管", "PE管", "钢管", "预应力混凝土管", "玻璃钢管"],
        "pressure_levels": ["低压(≤0.6MPa)", "中压(0.6~1.6MPa)", "高压(>1.6MPa)"],
        "year_range": (1990, 2024),
        "unit_cost_range": (200, 2000),
        "design_life": 50,
        "weight": 0.30,
    },
    {
        "type_code": "HTP",
        "type_name": "供暖管网",
        "prefix": "HTP",
        "diameters": ["DN50", "DN80", "DN100", "DN150", "DN200", "DN300", "DN500", "DN800"],
        "materials": ["无缝钢管", "螺旋焊缝钢管", "预制直埋保温管", "PE-RT管"],
        "pressure_levels": ["低压(≤1.0MPa)", "中压(1.0~2.5MPa)", "高压(>2.5MPa)"],
        "year_range": (2000, 2024),
        "unit_cost_range": (300, 2500),
        "design_life": 40,
        "weight": 0.20,
    },
    {
        "type_code": "GSP",
        "type_name": "燃气管网",
        "prefix": "GSP",
        "diameters": ["DN50", "DN80", "DN100", "DN150", "DN200", "DN300", "DN400", "DN600"],
        "materials": ["PE管", "钢管", "球墨铸铁管", "不锈钢管"],
        "pressure_levels": ["低压(≤0.01MPa)", "中压A(0.01~0.4MPa)", "中压B(0.4~0.8MPa)", "高压(>0.8MPa)"],
        "year_range": (1995, 2024),
        "unit_cost_range": (150, 3000),
        "design_life": 40,
        "weight": 0.25,
    },
    {
        "type_code": "SWP",
        "type_name": "污水管网",
        "prefix": "SWP",
        "diameters": ["DN200", "DN300", "DN400", "DN600", "DN800", "DN1000", "DN1200", "DN1500", "DN2000"],
        "materials": ["钢筋混凝土管", "HDPE双壁波纹管", "球墨铸铁管", "玻璃钢夹砂管", "PVC-U管"],
        "pressure_levels": ["重力流", "压力流(≤0.6MPa)", "压力流(>0.6MPa)"],
        "year_range": (1985, 2024),
        "unit_cost_range": (100, 1500),
        "design_life": 50,
        "weight": 0.15,
    },
    {
        "type_code": "HZP",
        "type_name": "危废输送管网",
        "prefix": "HZP",
        "diameters": ["DN50", "DN80", "DN100", "DN150", "DN200", "DN300"],
        "materials": ["不锈钢管", "衬氟钢管", "衬塑钢管", "钛合金管", "哈氏合金管"],
        "pressure_levels": ["低压(≤1.0MPa)", "中压(1.0~4.0MPa)", "高压(>4.0MPa)"],
        "year_range": (2010, 2024),
        "unit_cost_range": (1000, 8000),
        "design_life": 30,
        "weight": 0.10,
    },
]

REGIONS = [
    {"code": "DC", "name": "东城区"},
    {"code": "XC", "name": "西城区"},
    {"code": "CY", "name": "朝阳区"},
    {"code": "HD", "name": "海淀区"},
    {"code": "FT", "name": "丰台区"},
    {"code": "SJS", "name": "石景山区"},
    {"code": "TZ", "name": "通州区"},
    {"code": "DX", "name": "大兴区"},
]

OWNERSHIP_UNITS = [
    "市水务集团", "市供热集团", "市燃气集团", "市排水集团",
    "市环保产业集团", "区水务局", "区城管委", "开发区管委会",
    "高新水务公司", "城北供热公司", "蓝天燃气公司", "绿源环保公司",
]

OAM_UNITS = [
    "市政养护一处", "市政养护二处", "市政养护三处",
    "管网检测中心", "应急抢修大队", "智慧管网运维中心",
    "第一运维服务站", "第二运维服务站", "第三运维服务站",
]

SUPERVISION_UNITS = [
    "市住建局", "市城管执法局", "市应急管理局",
    "市生态环境局", "市市场监管局", "区住建局",
]

ASSET_STATUS = ["在用", "停用", "待检", "报废"]
ASSET_STATUS_WEIGHTS = [0.75, 0.08, 0.10, 0.07]

LIFECYCLE_EVENT_TYPES = [
    {"code": "PURCHASE", "name": "采购", "cost_range": (5000, 500000)},
    {"code": "CONSTRUCTION", "name": "施工安装", "cost_range": (10000, 800000)},
    {"code": "MAINTENANCE", "name": "日常运维", "cost_range": (500, 50000)},
    {"code": "INSPECTION", "name": "定期巡检", "cost_range": (200, 10000)},
    {"code": "RENOVATION", "name": "改造更新", "cost_range": (20000, 1000000)},
    {"code": "DECOMMISSION", "name": "报废处置", "cost_range": (5000, 200000)},
]

INVENTORY_METHODS = ["扫码盘点", "巡检盘点", "无人机盘点", "GIS比对"]


def generate_asset_id(prefix, index):
    return f"{prefix}-{index:05d}"


def generate_coordinate(region_name):
    base_lon = 116.3 + random.uniform(-0.15, 0.15)
    base_lat = 39.9 + random.uniform(-0.15, 0.15)
    return round(base_lon, 6), round(base_lat, 6)


def generate_pipeline_asset(global_index, pipeline_type):
    asset_id = generate_asset_id(pipeline_type["prefix"], global_index)
    diameter = random.choice(pipeline_type["diameters"])
    material = random.choice(pipeline_type["materials"])
    install_year = random.randint(*pipeline_type["year_range"])
    current_year = 2024
    age = current_year - install_year
    design_life = pipeline_type["design_life"]
    region = random.choice(REGIONS)
    lon, lat = generate_coordinate(region["name"])

    segment_length = round(random.uniform(20, 500), 1)
    burial_depth = round(random.uniform(0.5, 4.0), 2)
    pressure_level = random.choice(pipeline_type["pressure_levels"])
    unit_cost = random.uniform(*pipeline_type["unit_cost_range"])
    original_value = round(segment_length * unit_cost, 2)
    depreciation_rate = min(0.95, age / design_life)
    net_value = round(original_value * (1 - depreciation_rate), 2)

    status = random.choices(ASSET_STATUS, weights=ASSET_STATUS_WEIGHTS, k=1)[0]
    if age > design_life:
        status = random.choices(["待检", "报废", "在用"], weights=[0.4, 0.4, 0.2], k=1)[0]

    risk_score = min(100, max(0, int(
        age / design_life * 50 +
        (10 if material in ["钢筋混凝土管", "钢管"] else 0) +
        random.uniform(-10, 10)
    )))

    ownership_unit = random.choice(OWNERSHIP_UNITS)
    oam_unit = random.choice(OAM_UNITS)
    supervision_unit = random.choice(SUPERVISION_UNITS)

    return {
        "asset_id": asset_id,
        "pipeline_type_code": pipeline_type["type_code"],
        "pipeline_type_name": pipeline_type["type_name"],
        "diameter": diameter,
        "material": material,
        "install_year": install_year,
        "service_years": age,
        "design_life": design_life,
        "remaining_life": max(0, design_life - age),
        "region_code": region["code"],
        "region_name": region["name"],
        "longitude": lon,
        "latitude": lat,
        "segment_length_m": segment_length,
        "burial_depth_m": burial_depth,
        "pressure_level": pressure_level,
        "asset_status": status,
        "original_value_yuan": original_value,
        "net_value_yuan": net_value,
        "depreciation_rate": round(depreciation_rate, 4),
        "risk_score": risk_score,
        "ownership_unit": ownership_unit,
        "oam_unit": oam_unit,
        "supervision_unit": supervision_unit,
        "last_inspection_date": (datetime(2024, 1, 1) - timedelta(days=random.randint(0, 365))).strftime("%Y-%m-%d"),
        "qr_code": f"QR-{asset_id}",
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_lifecycle_event(asset, event_index):
    event_type = random.choice(LIFECYCLE_EVENT_TYPES)
    install_year = asset["install_year"]
    current_year = 2024

    if event_type["code"] == "PURCHASE":
        event_year = install_year - random.randint(0, 1)
        event_month = random.randint(1, 12)
    elif event_type["code"] == "CONSTRUCTION":
        event_year = install_year
        event_month = random.randint(1, 12)
    elif event_type["code"] == "DECOMMISSION":
        if asset["asset_status"] != "报废":
            return None
        event_year = random.randint(install_year + 20, current_year)
        event_month = random.randint(1, 12)
    else:
        event_year = random.randint(install_year, current_year)
        event_month = random.randint(1, 12)

    event_day = random.randint(1, 28)
    event_date = f"{event_year}-{event_month:02d}-{event_day:02d}"

    cost = round(random.uniform(*event_type["cost_range"]), 2)
    responsible = random.choice(OAM_UNITS)

    descriptions = {
        "PURCHASE": f"采购{asset['material']}管段{asset['diameter']}，长度{asset['segment_length_m']}m",
        "CONSTRUCTION": f"{asset['region_name']}段管网施工安装，埋深{asset['burial_depth_m']}m",
        "MAINTENANCE": f"日常运维保养，包括管道清洗、阀门检修、防腐处理",
        "INSPECTION": f"定期巡检检测，管道内窥检测、压力测试",
        "RENOVATION": f"管段改造更新，更换老化部件、升级管材",
        "DECOMMISSION": f"管网报废处置，管道封堵、拆除、无害化处理",
    }

    return {
        "event_id": f"EVT-{event_index:06d}",
        "asset_id": asset["asset_id"],
        "pipeline_type_name": asset["pipeline_type_name"],
        "event_type_code": event_type["code"],
        "event_type_name": event_type["name"],
        "event_date": event_date,
        "responsible_unit": responsible,
        "cost_yuan": cost,
        "description": descriptions[event_type["code"]],
        "operator": f"操作员{random.randint(1, 20):03d}",
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_inventory_check(assets, check_index):
    batch_id = f"INV-2024-{random.randint(1, 12):02d}"
    method = random.choice(INVENTORY_METHODS)
    check_date = (datetime(2024, 1, 1) + timedelta(days=random.randint(0, 270))).strftime("%Y-%m-%d")
    asset = random.choice(assets)

    diff_status = random.choices(
        ["一致", "差异-缺失", "差异-多余", "差异-信息不符"],
        weights=[0.82, 0.08, 0.03, 0.07],
        k=1
    )[0]

    diff_desc = ""
    if diff_status == "差异-缺失":
        diff_desc = f"台账记录资产{asset['asset_id']}，现场未找到对应资产"
    elif diff_status == "差异-多余":
        diff_desc = f"现场发现未入账管段，暂估{asset['diameter']} {asset['material']}"
    elif diff_status == "差异-信息不符":
        field = random.choice(["管径", "材质", "长度", "权属单位"])
        diff_desc = f"台账{field}与现场实际不符"

    return {
        "check_id": f"CHK-{check_index:06d}",
        "batch_id": batch_id,
        "asset_id": asset["asset_id"],
        "pipeline_type_name": asset["pipeline_type_name"],
        "region_name": asset["region_name"],
        "check_method": method,
        "check_date": check_date,
        "diff_status": diff_status,
        "diff_description": diff_desc,
        "checker": f"盘点员{random.randint(1, 15):03d}",
        "checker_unit": random.choice(OAM_UNITS),
        "create_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }


def generate_all_data(asset_count=5000, output_dir="data/asset"):
    os.makedirs(output_dir, exist_ok=True)
    today = datetime.now().strftime("%Y%m%d")

    assets = []
    global_idx = 1
    for pt in PIPELINE_TYPES:
        count = int(asset_count * pt["weight"])
        for _ in range(count):
            asset = generate_pipeline_asset(global_idx, pt)
            assets.append(asset)
            global_idx += 1

    asset_file = os.path.join(output_dir, f"pipeline_asset_{today}.jsonl")
    with open(asset_file, "w", encoding="utf-8") as f:
        for asset in assets:
            f.write(json.dumps(asset, ensure_ascii=False) + "\n")
    print(f"[数据生成] 管网资产数据: {len(assets)} 条 -> {asset_file}")

    events = []
    event_idx = 1
    for asset in assets:
        num_events = random.randint(2, 8)
        for _ in range(num_events):
            event = generate_lifecycle_event(asset, event_idx)
            if event:
                events.append(event)
                event_idx += 1

    event_file = os.path.join(output_dir, f"lifecycle_event_{today}.jsonl")
    with open(event_file, "w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    print(f"[数据生成] 生命周期事件: {len(events)} 条 -> {event_file}")

    checks = []
    for i in range(int(asset_count * 0.6)):
        check = generate_inventory_check(assets, i + 1)
        checks.append(check)

    check_file = os.path.join(output_dir, f"inventory_check_{today}.jsonl")
    with open(check_file, "w", encoding="utf-8") as f:
        for check in checks:
            f.write(json.dumps(check, ensure_ascii=False) + "\n")
    print(f"[数据生成] 盘点记录: {len(checks)} 条 -> {check_file}")

    return asset_file, event_file, check_file


def main():
    parser = argparse.ArgumentParser(description="城市管网资产数据生成器")
    parser.add_argument("--count", type=int, default=5000, help="资产记录总数（默认5000）")
    parser.add_argument("--output", type=str, default="data/asset", help="输出目录")
    args = parser.parse_args()
    generate_all_data(args.count, args.output)


if __name__ == "__main__":
    main()
