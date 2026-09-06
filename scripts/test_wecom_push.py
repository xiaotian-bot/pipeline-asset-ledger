#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
企业微信应用消息推送 - 独立连通性测试脚本
用法: python3 scripts/test_wecom_push.py [要发送的测试内容]
不依赖 FastAPI，直接读取 data/push_config.json 中的微信配置并调用企业微信 API。
"""
import json
import os
import sys
import urllib.request
import urllib.parse

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_FILE = os.path.join(BASE_DIR, "data", "push_config.json")


def load_wechat_cfg() -> dict:
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("wechat", {})
    return {}


def get_access_token(cfg: dict) -> str:
    url = (
        "https://qyapi.weixin.qq.com/cgi-bin/gettoken?corpid="
        + urllib.parse.quote(str(cfg["corpid"]))
        + "&corpsecret=" + urllib.parse.quote(str(cfg["corp_secret"]))
    )
    with urllib.request.urlopen(url, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    if data.get("errcode", 0) != 0:
        raise RuntimeError(f"获取access_token失败: {data.get('errmsg')}")
    return data["access_token"]


def send_message(cfg: dict, token: str, touser: str, content: str) -> dict:
    body = {
        "touser": touser,
        "msgtype": "text",
        "agentid": int(cfg["agent_id"]),
        "text": {"content": content},
        "safe": 0,
    }
    url = "https://qyapi.weixin.qq.com/cgi-bin/message/send?access_token=" + token
    req = urllib.request.Request(
        url,
        data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main():
    cfg = load_wechat_cfg()
    if not (cfg.get("corpid") and cfg.get("corp_secret") and cfg.get("agent_id")):
        print("错误: data/push_config.json 中缺少企业微信配置（corpid / corp_secret / agent_id）")
        sys.exit(1)
    touser = "LiXuan"
    content = "【管网预警平台】这是一条来自后端的测试推送。如果您收到本条消息，说明企业微信推送通道已连通。"
    if len(sys.argv) > 1:
        touser = sys.argv[1]
    if len(sys.argv) > 2:
        content = "【管网预警平台】测试推送：" + " ".join(sys.argv[2:])
    print(f"企业ID: {cfg['corpid']}  AgentId: {cfg['agent_id']}")
    print(f"接收人: {touser}")
    try:
        token = get_access_token(cfg)
        print("✓ 获取 access_token 成功")
        result = send_message(cfg, token, touser, content)
        if result.get("errcode", 0) == 0:
            print(f"✓ 消息发送成功 → {touser}")
            print(f"  内容: {content}")
            sys.exit(0)
        else:
            print(f"✗ 消息发送失败: {result.get('errcode')} {result.get('errmsg')}")
            sys.exit(1)
    except Exception as e:
        print(f"✗ 调用企业微信API异常: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
