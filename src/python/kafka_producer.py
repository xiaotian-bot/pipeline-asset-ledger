#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
城市管网资产数字化台账 - Kafka数据生产者

将管网资产数据、生命周期事件、盘点记录发送至Kafka
"""

import json
import os
import time
import argparse
from datetime import datetime

try:
    from kafka import KafkaProducer
except ImportError:
    print("请安装 kafka-python: pip install kafka-python")
    exit(1)

ASSET_TOPIC = "pipeline-asset-topic"
LIFECYCLE_TOPIC = "pipeline-lifecycle-topic"
INVENTORY_TOPIC = "pipeline-inventory-topic"


def create_producer(bootstrap_servers):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v, ensure_ascii=False).encode('utf-8'),
        acks='all',
        retries=3,
        linger_ms=10,
    )


def send_to_kafka(producer, topic, message, key=None):
    try:
        future = producer.send(topic, value=message, key=key)
        future.get(timeout=10)
        return True
    except Exception as e:
        print(f"  发送失败 [{topic}]: {e}")
        return False


def load_and_send(file_path, producer, topic, label):
    if not os.path.exists(file_path):
        print(f"文件不存在: {file_path}")
        return 0

    count = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = record.get('asset_id', '').encode('utf-8')
            if send_to_kafka(producer, topic, record, key):
                count += 1
            if count % 1000 == 0 and count > 0:
                print(f"  [{label}] 已发送 {count} 条...")

    return count


def main():
    parser = argparse.ArgumentParser(description="管网资产数据Kafka生产者")
    parser.add_argument("--input", type=str, default="data/asset", help="数据目录")
    parser.add_argument("--bootstrap", type=str, default="localhost:9092", help="Kafka地址")
    parser.add_argument("--speed", type=float, default=10, help="发送速度倍率")
    args = parser.parse_args()

    today = datetime.now().strftime("%Y%m%d")
    asset_file = os.path.join(args.input, f"pipeline_asset_{today}.jsonl")
    event_file = os.path.join(args.input, f"lifecycle_event_{today}.jsonl")
    check_file = os.path.join(args.input, f"inventory_check_{today}.jsonl")

    print(f"连接 Kafka: {args.bootstrap}")
    producer = create_producer([args.bootstrap])

    print(f"\n[1/3] 发送管网资产数据...")
    asset_count = load_and_send(asset_file, producer, ASSET_TOPIC, "资产")
    print(f"  资产数据发送完成: {asset_count} 条")

    print(f"\n[2/3] 发送生命周期事件...")
    event_count = load_and_send(event_file, producer, LIFECYCLE_TOPIC, "事件")
    print(f"  事件数据发送完成: {event_count} 条")

    print(f"\n[3/3] 发送盘点记录...")
    check_count = load_and_send(check_file, producer, INVENTORY_TOPIC, "盘点")
    print(f"  盘点数据发送完成: {check_count} 条")

    producer.flush()
    producer.close()

    print(f"\n{'=' * 50}")
    print(f"全部发送完成!")
    print(f"  管网资产: {asset_count} 条 -> {ASSET_TOPIC}")
    print(f"  生命周期: {event_count} 条 -> {LIFECYCLE_TOPIC}")
    print(f"  盘点记录: {check_count} 条 -> {INVENTORY_TOPIC}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
