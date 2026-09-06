#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
工业车间设备状态监测数据生成脚本

功能：
1. 生成模拟的工业设备时序传感数据（CNC、工业机器人、注塑机、空压机）
2. 支持设备状态编码、工艺参数、温度压力、运维健康四大维度
3. 随机注入3%~5%异常数据用于机器学习异常检测
4. 生成的数据可直接用于Kafka实时传输或HDFS离线分析

使用示例：
    python generate_logs.py --count 100000 --output data/logs

依赖：
    无第三方依赖（纯标准库）
"""

import argparse
import json
import os
import random
from datetime import datetime, timedelta


# ==============================================================================
# 设备配置定义
# ==============================================================================

# 4类核心工业设备配置
DEVICE_CONFIGS = [
    # CNC数控加工中心
    {
        "device_type": "CNC加工中心",
        "device_prefix": "CNC",
        "count": 3,
        "workshop": "机加工一车间",
        "params": {
            "spindle_speed": (2000, 12000, "rpm"),        # 主轴转速
            "feed_rate": (500, 15000, "mm/min"),          # 进给速度
            "spindle_load": (20, 90, "%"),                # 主轴负载率
            "spindle_temp": (35, 75, "℃"),                # 主轴轴承温度
            "hydraulic_pressure": (4.5, 5.8, "MPa"),      # 液压系统压力
            "coolant_temp": (22, 32, "℃"),                # 冷却系统温度
            "vibration": (0.5, 3.0, "mm/s"),              # 主轴振动烈度
        },
        "alarm_codes": {
            1001: "主轴过载",
            1002: "超行程",
            1003: "刀具磨损",
            1004: "主轴过热",
            1005: "液压低压",
        }
    },
    # 六轴工业机器人
    {
        "device_type": "六轴工业机器人",
        "device_prefix": "RBT",
        "count": 3,
        "workshop": "装配车间",
        "params": {
            "joint_current": (2.0, 6.5, "A"),             # 关节电机电流
            "joint_torque": (30, 250, "N·m"),             # 关节力矩
            "servo_temp": (30, 65, "℃"),                  # 伺服电机温度
            "reducer_temp": (35, 60, "℃"),                # 减速器壳体温度
            "joint_vibration": (0.3, 3.5, "mm/s"),        # 关节振动值
            "cycle_count": (1, 500, "次"),                # 循环计数
        },
        "alarm_codes": {
            2001: "关节过载",
            2002: "碰撞检测",
            2003: "编码器异常",
            2004: "伺服过热",
            2005: "减速器润滑失效",
        }
    },
    # 伺服注塑机
    {
        "device_type": "伺服注塑机",
        "device_prefix": "INJ",
        "count": 2,
        "workshop": "注塑车间",
        "params": {
            "injection_pressure": (90, 145, "MPa"),       # 注射压力
            "holding_pressure": (45, 95, "MPa"),          # 保压压力
            "system_oil_pressure": (11, 15, "MPa"),       # 系统油压
            "barrel_temp": (190, 270, "℃"),               # 料筒温度
            "mold_temp": (45, 80, "℃"),                   # 模具温度
            "hydraulic_oil_temp": (35, 55, "℃"),          # 液压油温度
            "cycle_time": (25, 110, "s"),                 # 成型周期
        },
        "alarm_codes": {
            3001: "注射压力异常",
            3002: "料筒温度超限",
            3003: "模具温度异常",
            3004: "液压油高温",
            3005: "螺杆背压异常",
        }
    },
    # 永磁变频螺杆空压机
    {
        "device_type": "螺杆空压机",
        "device_prefix": "AIR",
        "count": 2,
        "workshop": "公用工程车间",
        "params": {
            "discharge_pressure": (0.65, 0.78, "MPa"),    # 排气压力
            "oil_separator_diff": (0.02, 0.08, "MPa"),    # 油气分离器压差
            "air_filter_diff": (0.01, 0.04, "MPa"),       # 空气过滤器压差
            "discharge_temp": (78, 92, "℃"),              # 排气温度
            "lubricant_temp": (62, 80, "℃"),             # 润滑油温度
            "motor_current": (45, 85, "A"),               # 主机电机电流
            "power": (15, 45, "kW"),                      # 设备总功率
        },
        "alarm_codes": {
            4001: "排气高温",
            4002: "排气压力超限",
            4003: "油气分离器堵塞",
            4004: "空气滤芯堵塞",
            4005: "电机过载",
        }
    },
]

# 设备状态编码（遵循工业三色灯标准）
STATUS_CODES = {
    0: "关机",
    1: "待机",
    2: "运行",
    3: "预警",
    4: "故障",
    5: "急停",
    6: "调试",
}

# 车间环境监测指标基准值
ENVIRONMENT_BASELINE = {
    "workshop_temp": (20, 26, "℃"),
    "workshop_humidity": (45, 65, "%RH"),
    "workshop_pressure": (95, 102, "kPa"),
    "pm25": (10, 50, "μg/m³"),
    "noise": (70, 82, "dB"),
    "cabinet_temp": (25, 38, "℃"),
}


def generate_device_list():
    """
    根据设备配置生成完整的设备清单

    Returns:
        list: 所有设备的基础信息列表
    """
    devices = []
    for config in DEVICE_CONFIGS:
        for i in range(config["count"]):
            device_id = f"{config['device_prefix']}-{i + 1:03d}"
            # 为每台设备设置不同的老化系数（0.8~1.2）
            aging_factor = round(random.uniform(0.8, 1.2), 2)
            devices.append({
                "device_id": device_id,
                "device_type": config["device_type"],
                "workshop": config["workshop"],
                "params_config": config["params"],
                "alarm_codes": config["alarm_codes"],
                "aging_factor": aging_factor,
            })
    return devices


def generate_params(params_config, status_code, aging_factor, inject_anomaly=False):
    """
    根据设备状态和老化系数生成工艺参数

    Args:
        params_config (dict): 参数配置
        status_code (int): 设备状态编码
        aging_factor (float): 设备老化系数（>1表示老化严重）
        inject_anomaly (bool): 是否注入异常数据

    Returns:
        dict: 生成的参数字典
    """
    params = {}
    for param_name, (min_val, max_val, unit) in params_config.items():
        # 基准值取正常范围中值
        base_val = (min_val + max_val) / 2
        # 波动幅度为范围的20%
        fluctuation = (max_val - min_val) * 0.2

        if status_code == 2:
            # 运行状态：参数在正常区间波动，老化设备波动更大
            value = base_val + random.uniform(-fluctuation, fluctuation) * aging_factor
        elif status_code in (1, 6):
            # 待机/调试：参数接近下限
            value = min_val + random.uniform(0, fluctuation)
        elif status_code == 3:
            # 预警状态：参数接近上限
            value = max_val - random.uniform(0, fluctuation * 0.5)
        elif status_code in (4, 5):
            # 故障/急停：温度类参数升高，其他参数骤降
            if "temp" in param_name:
                value = max_val + random.uniform(2, 10)
            else:
                value = min_val * random.uniform(0.1, 0.5)
        else:
            # 关机：参数为零或极低
            value = 0 if "temp" not in param_name else min_val * 0.5

        # 异常注入：参数超出正常范围
        if inject_anomaly:
            if "temp" in param_name:
                value = max_val + random.uniform(5, 20)
            elif "pressure" in param_name:
                value = max_val * random.uniform(1.1, 1.3)
            elif "vibration" in param_name:
                value = max_val * random.uniform(1.5, 2.0)

        # 保留2位小数
        params[param_name] = round(value, 2)

    return params


def generate_status_code(hour):
    """
    根据时间段生成设备状态编码

    Args:
        hour (int): 当前小时（0-23）

    Returns:
        int: 状态编码
    """
    # 白班（8:00-20:00）高运行率
    if 8 <= hour < 20:
        status_weights = [0, 5, 80, 8, 5, 1, 1]
    # 夜班（20:00-次日8:00）低运行率
    else:
        status_weights = [10, 40, 30, 10, 8, 1, 1]

    return random.choices(list(STATUS_CODES.keys()), weights=status_weights)[0]


def generate_health_score(status_code, params, params_config, aging_factor):
    """
    根据设备状态和参数计算健康度评分（0-100）

    Args:
        status_code (int): 设备状态编码
        params (dict): 当前参数值
        params_config (dict): 参数配置
        aging_factor (float): 老化系数

    Returns:
        int: 健康度评分
    """
    if status_code in (4, 5):
        return random.randint(20, 40)
    if status_code == 0:
        return random.randint(30, 50)

    # 基于参数偏离度计算健康度
    base_score = 95
    for param_name, value in params.items():
        if param_name in params_config:
            min_val, max_val, _ = params_config[param_name]
            # 参数越接近上限，扣分越多
            if max_val > 0:
                ratio = value / max_val
                if ratio > 0.85:
                    base_score -= (ratio - 0.85) * 100

    # 老化系数影响
    base_score -= (aging_factor - 1.0) * 20

    # 预警状态额外扣分
    if status_code == 3:
        base_score -= 10

    return max(10, min(100, int(base_score)))


def generate_log_entry(device, base_time, index):
    """
    生成单条设备监测数据记录

    Args:
        device (dict): 设备信息
        base_time (datetime): 基准时间
        index (int): 记录索引

    Returns:
        dict: 单条日志记录
    """
    # 时间戳：基于索引递增，模拟5秒采集间隔
    timestamp = base_time + timedelta(seconds=index * 5)
    hour = timestamp.hour

    # 生成状态编码
    status_code = generate_status_code(hour)

    # 随机注入3%~5%异常数据
    inject_anomaly = random.random() < 0.04

    # 生成工艺参数
    params = generate_params(
        device["params_config"],
        status_code,
        device["aging_factor"],
        inject_anomaly
    )

    # 生成告警代码
    alarm_code = 0
    if status_code == 4:
        alarm_code = random.choice(list(device["alarm_codes"].keys()))
    elif status_code == 3:
        # 预警状态有30%概率产生告警
        if random.random() < 0.3:
            alarm_code = random.choice(list(device["alarm_codes"].keys()))

    # 计算健康度评分
    health_score = generate_health_score(
        status_code, params, device["params_config"], device["aging_factor"]
    )

    return {
        "device_id": device["device_id"],
        "device_type": device["device_type"],
        "workshop": device["workshop"],
        "event_timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
        "status_code": status_code,
        "status_name": STATUS_CODES[status_code],
        "params": params,
        "alarm_code": alarm_code,
        "alarm_desc": device["alarm_codes"].get(alarm_code, "正常") if alarm_code else "正常",
        "health_score": health_score,
        "ts": int(timestamp.timestamp() * 1000),
    }


def generate_logs(count, output_dir):
    """
    生成指定数量的工业设备监测数据

    Args:
        count (int): 记录数量
        output_dir (str): 输出目录

    Returns:
        str: 生成的日志文件路径
    """
    os.makedirs(output_dir, exist_ok=True)

    # 生成设备清单
    devices = generate_device_list()
    device_count = len(devices)
    print(f"设备清单：共 {device_count} 台设备")
    for dev in devices:
        print(f"  - {dev['device_id']} ({dev['device_type']} @ {dev['workshop']})")

    base_time = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

    filename = f"device_sensor_{base_time.strftime('%Y%m%d')}.log"
    filepath = os.path.join(output_dir, filename)

    print(f"\n开始生成 {count:,} 条设备监测数据...")
    print(f"输出文件: {filepath}")

    batch_size = 1000
    batches = (count + batch_size - 1) // batch_size

    with open(filepath, 'w', encoding='utf-8') as f:
        for batch in range(batches):
            start_idx = batch * batch_size
            end_idx = min((batch + 1) * batch_size, count)
            batch_count = end_idx - start_idx

            batch_logs = []
            for i in range(batch_count):
                # 轮询分配设备
                global_idx = start_idx + i
                device = devices[global_idx % device_count]
                log_entry = generate_log_entry(device, base_time, global_idx // device_count)
                batch_logs.append(json.dumps(log_entry, ensure_ascii=False))

            f.write('\n'.join(batch_logs) + '\n')

            if (batch + 1) % 10 == 0 or batch == batches - 1:
                progress = ((batch + 1) / batches) * 100
                print(f"进度: {progress:.1f}% ({end_idx:,}/{count:,} 条记录)")

    file_size = os.path.getsize(filepath)
    print(f"✓ 数据生成完成！")
    print(f"  文件路径: {filepath}")
    print(f"  文件大小: {file_size / 1024 / 1024:.2f} MB")
    print(f"  记录数量: {count:,}")
    print(f"  设备数量: {device_count} 台")

    return filepath


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="工业车间设备状态监测数据生成器")
    parser.add_argument("--count", type=int, default=100000,
                        help="生成记录数量，默认100000")
    parser.add_argument("--output", type=str, default="data/logs",
                        help="输出目录，默认为 data/logs")

    args = parser.parse_args()
    generate_logs(args.count, args.output)


if __name__ == "__main__":
    main()
