#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent 诊断大脑
==============
在既有 DeepSeek function calling 之上叠三层能力：

  1) 诊断编排  预测结果 + 关键因子解释 + 规范引用 + 处置建议，一次成型
  2) 长期记忆  data/agent_memory.json，跨会话记住近期诊断结论与用户偏好，注入上下文
  3) 本地兜底  api_key 未配置或 DeepSeek 调用失败时，用确定性意图规划器真实调用工具，
              组织出同样带引用的四段式回答 —— 保证 /agent/chat 永不 502，演示不中断

设计约束：不引入 langchain / faiss / chromadb，只用标准库 + 已有的 kb_rag。
"""
import json
import os
import re
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import kb_rag  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MEMORY_FILE = os.path.join(DATA_DIR, "agent_memory.json")

MAX_RECENT_DIAGNOSES = 20
MAX_PREFERENCES = 30


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==============================================================================
# 长期记忆
# ==============================================================================
def load_memory() -> dict:
    try:
        if os.path.exists(MEMORY_FILE):
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                m = json.load(f)
            if isinstance(m, dict):
                m.setdefault("recent_diagnoses", [])
                m.setdefault("preferences", [])
                m.setdefault("facts", {})
                return m
    except Exception:
        pass
    return {"recent_diagnoses": [], "preferences": [], "facts": {}, "created_at": _now()}


def save_memory(mem: dict):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(mem, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def remember_diagnosis(target: str, conclusion: str, risk_score=None, status: str = ""):
    """记下一次诊断结论，后续追问可直接引用（多步推理 + 记忆）。"""
    mem = load_memory()
    items = [d for d in mem["recent_diagnoses"] if d.get("target") != target]
    items.append({"target": target, "conclusion": conclusion, "risk_score": risk_score,
                  "status": status, "time": _now()})
    mem["recent_diagnoses"] = items[-MAX_RECENT_DIAGNOSES:]
    save_memory(mem)


def remember_preference(text: str):
    text = (text or "").strip()
    if not text:
        return
    mem = load_memory()
    prefs = [p for p in mem["preferences"] if p.get("text") != text]
    prefs.append({"text": text, "time": _now()})
    mem["preferences"] = prefs[-MAX_PREFERENCES:]
    save_memory(mem)


def memory_context_block() -> str:
    """注入 system prompt 的记忆片段。"""
    mem = load_memory()
    lines = []
    recent = mem.get("recent_diagnoses") or []
    if recent:
        lines.append("【近期诊断记忆（用户可能基于这些结论追问）】")
        for d in recent[-6:][::-1]:
            lines.append(f"· {d.get('time', '')} {d.get('target', '')}："
                         f"{d.get('conclusion', '')}")
    prefs = mem.get("preferences") or []
    if prefs:
        lines.append("【用户偏好】")
        for p in prefs[-5:][::-1]:
            lines.append(f"· {p.get('text', '')}")
    return "\n".join(lines)


# ==============================================================================
# 特征可解释性：把模型特征名翻成运维语言
# ==============================================================================
_METRIC_CN = {
    "pressure": "压力", "flow": "流量", "temperature": "温度",
    "gas_concentration": "燃气浓度", "level": "液位", "vibration": "振动",
}

_STAT_CN = {
    "val": "当前值", "mean": "均值", "std": "波动幅度（标准差）",
    "min": "最低值", "max": "最高值", "diff": "相邻帧变化量", "slope": "趋势斜率",
    "z": "偏离自身常态的程度（z 分数）", "zslope": "归一化劣化趋势",
}

# zslope 必须排在 slope 和 z 前面：正则的分支是从左到右试的，先列长的才能整体匹配
_FEATURE_RE = re.compile(r"^([a-z_]+?)_(val|mean|std|min|max|diff|zslope|slope|z)(\d+)?$")


def explain_feature(name: str, importance=None) -> str:
    """把 pressure_slope15 / window_anomaly_cnt 之类的特征名解释成中文运维语义。"""
    if not name:
        return "—"
    if name == "window_anomaly_cnt":
        return "滑动窗口内历史告警次数（近期异常频次）"
    if name == "hour":
        return "小时段特征（昼夜工况差异）"
    if name == "dow":
        return "星期特征（工作日/周末用水用气差异）"
    m = _FEATURE_RE.match(name)
    if not m:
        return name
    metric, stat, win = m.group(1), m.group(2), m.group(3)
    cn_metric = _METRIC_CN.get(metric, metric)
    cn_stat = _STAT_CN.get(stat, stat)
    tail = f"（近 {win} 帧）" if win else ""
    if stat == "slope":
        return f"{cn_metric}上升/下降趋势{tail}"
    if stat == "zslope":
        return f"{cn_metric}劣化速率（已按自身常态归一，跨管型可比）{tail}"
    if stat == "std":
        return f"{cn_metric}抖动剧烈程度{tail}"
    if stat == "diff":
        return f"{cn_metric}瞬时突变量"
    return f"{cn_metric}{cn_stat}{tail}"


def _feature_direction(name: str) -> str:
    # "_zslope" 里 slope 前面是 z 不是下划线，所以 "_slope" in name 匹配不到，必须单列
    if "_zslope" in name or name.endswith("_slope15") or "_slope" in name:
        return "趋势项"
    if "_std" in name:
        return "波动项"
    if "_diff" in name:
        return "突变项"
    if name == "window_anomaly_cnt":
        return "历史告警项"
    return "水平项"


# ==============================================================================
# 处置建议：按风险档 + 管网类型 + 命中的关键因子给出可执行动作
# ==============================================================================
_ACTION_BY_BAND = [
    (80, "危急", [
        "24 小时内派员到场核查，同步推送属地运维负责人与值班领导",
        "立即创建紧急工单并关联本条预测预警，处置结论回填以校准模型",
        "核查上游压力/流量调度，必要时隔离区段防止事故扩大",
    ]),
    (60, "高风险", [
        "7 天内完成现场核查与专项检测（测厚 / CCTV / 检漏按管型选择）",
        "加密该点位监测频次，把该传感器纳入重点关注清单",
        "创建高优先级工单，明确责任人与完成时限",
    ]),
    (40, "预警", [
        "30 天内安排专项检测，纳入下次巡检路线",
        "核对传感器是否漂移或断线（数据长期恒定时优先排查设备本体）",
        "观察后续 3 轮扫描风险分是否持续上行",
    ]),
    (0, "正常", [
        "维持常规巡检频次，无需专项处置",
        "如出现单帧越限，等待连续 3 帧确认后再判定，避免抖动误报",
    ]),
]

_ACTION_BY_TYPE = {
    "供水管网": "供水管段优先做 DMA 夜间最小流量分析与音听/相关仪检漏，重点排查接口渗漏与第三方开挖破坏。",
    "燃气管网": "燃气管段严格执行禁火与防爆要求，浓度超限先警戒疏散、切断气源、强制通风，检测须用防爆仪器。",
    "供暖管网": "供暖管段核查供回水温差与循环流量，判断水力失衡或短路，检查除污器堵塞与平衡阀开度。",
    "污水管网": "污水管段下井前必须执行「先通风、再检测、后作业」并办理有限空间审批，防范硫化氢中毒。",
    "危废输送": "危废输送管段检查双套管检漏通道与紧急切断阀有效性，泄漏须导流至应急池，严禁冲入市政管网。",
}


def _risk_band(risk_score, levels=None):
    """风险分 → (档位名, 处置动作清单, 风险分)。

    分档线**优先取模型 meta 下发的 risk_levels**（main.py 把 bundle.config 透传进 pred dict）。
    原来 80/60/40 在 predictive_models、main.py、agent_brain 三处各写一遍，改一处就对不上：
    demo 时表现为"同一张预警卡片上，风险档与阈值描述互相矛盾、紧急工单永远升不上去"。
    """
    try:
        rs = float(risk_score or 0)
    except (TypeError, ValueError):
        rs = 0.0
    bands = _ACTION_BY_BAND
    if isinstance(levels, dict) and levels:
        try:
            # 只替换阈值，保留原有的档位名与动作清单
            bands = [
                (float(levels.get("critical", _ACTION_BY_BAND[0][0])), _ACTION_BY_BAND[0][1], _ACTION_BY_BAND[0][2]),
                (float(levels.get("warning", _ACTION_BY_BAND[1][0])), _ACTION_BY_BAND[1][1], _ACTION_BY_BAND[1][2]),
                (float(levels.get("attention", _ACTION_BY_BAND[2][0])), _ACTION_BY_BAND[2][1], _ACTION_BY_BAND[2][2]),
                _ACTION_BY_BAND[3],
            ]
        except Exception:
            bands = _ACTION_BY_BAND
    for threshold, label, actions in bands:
        if rs >= threshold:
            return label, actions, rs
    return "正常", _ACTION_BY_BAND[-1][2], rs


def _kb_query(pred: dict, device_type: str) -> str:
    """由预测结果反推检索式，让 RAG 命中对应的规范/预案条目。"""
    feats = " ".join(explain_feature(f.get("feature", "")) for f in (pred.get("top_features") or [])[:3])
    band_label = _risk_band(pred.get("risk_score"), pred.get("risk_levels"))[0]
    parts = [device_type or "", band_label, feats]
    st = pred.get("predicted_status")
    if st == "critical":
        parts.append("应急处置 处置流程")
    elif st == "warning":
        parts.append("检测周期 风险评估")
    else:
        parts.append("巡检 维护要点")
    return " ".join(p for p in parts if p).strip()


def _rul_span(rul, horizon=None) -> str:
    """把 RUL 步数转成能念出口的话术，_conclude 与 _suggested_actions 共用。

    说「N 步内」而不是「还剩 N 步」：RUL 头在未删失子集上 R² 为负（依据见
    predictive_models.consistent_rul），只有「快出事 / 还早」的判别力，报绝对步数会被
    当成精确倒计时念。rul<1 时 :.0f 会印成「0 步」，大屏上看着像故障，单独走一档。

    两侧的含义不对称，必须分开措辞。consistent_rul 在 prob 低于**判正线**（bundle.threshold，
    实测 0.80；不再是硬编码的 0.5）时把 rul 抬到 >= horizon，那是个*下界*——模型只敢说
    「horizon 步内不会出事」，再往后具体几步它给不出。一律念成「约 N 步内」会变成反方向的
    过度承诺（实测抓到过：SENSOR-001 判定「正常」、概率 10.3%，却输出「RUL 约 14 步内」，
    等于告诉运维 14 步后必然出事）。
    """
    r = float(rul)
    h = int(horizon or 12)
    if r >= h:
        return f"超出预测窗口（≥ {h} 步）"
    return "不足 1 步内" if r < 1 else f"约 {r:.0f} 步内"


def _suggested_actions(pred: dict, device_type: str) -> list:
    label, actions, rs = _risk_band(pred.get("risk_score"), pred.get("risk_levels"))
    out = list(actions)
    extra = _ACTION_BY_TYPE.get(device_type or "")
    if extra:
        out.append(extra)
    # 必须**同时**满足「RUL 落进窗口」与「状态被判为预警及以上」才给紧急话术。
    # 原来只看 rul < horizon：判正线（bundle.threshold，实测 0.80）高于 0.5 时，
    # prob ∈ [0.5, 0.8) 的点位 predicted_status = "normal" 但仍输出 rul < horizon，
    # 于是同一张卡片上写着"正常"、下面又催"已进入预警窗口、时限提前"——自相矛盾。
    st = pred.get("predicted_status") or "normal"
    if pred.get("rul") is not None and st in ("warning", "critical"):
        try:
            rul = float(pred["rul"])
            # 窗口取自模型而非写死 12：horizon 是重训时可调的（1-48）。
            horizon = int(pred.get("horizon") or 12)
            # 用 < 不用 <=：consistent_rul 保证 prob < 判正线时 rul >= horizon，
            # 写成 <= 会把恰好等于 horizon 的正常点位也套上紧急话术。
            if rul < horizon:
                out.insert(0, f"RUL {_rul_span(rul, horizon)}可能出现异常，已进入预警窗口，处置时限按上一档提前执行。")
        except (TypeError, ValueError):
            pass
    return out


def _action_for_alert(pred: dict) -> str:
    label = _risk_band(pred.get("risk_score"), pred.get("risk_levels"))[0]
    return {"危急": "紧急", "高风险": "高", "预警": "中", "正常": "低"}.get(label, "中")


# ==============================================================================
# 诊断编排
# ==============================================================================
def risk_band(risk_score, levels=None):
    """风险分 → (档位标签, 处置动作清单, 数值)。main.py 与本模块共用同一套分档口径。

    levels 传模型 meta 下发的 risk_levels 时以它为准（见 _risk_band）。
    """
    return _risk_band(risk_score, levels)


def action_priority(pred: dict) -> str:
    """由预测结果推导建议工单优先级。"""
    return _action_for_alert(pred)


def diagnose(target: str, predict_fn, top_k_kb: int = 3, remember: bool = True) -> dict:
    """
    组合诊断：预测 + 依据 + 规范引用 + 处置建议。
    predict_fn(target) -> sensor_predict_public 的返回结构。
    remember=False 用于后台自动扫描，避免机器产生的结论刷掉用户的交互记忆。
    """
    pred = predict_fn(target) or {}
    if not pred.get("ok") and not pred.get("model_ready"):
        return {
            "ok": False,
            "target": target,
            "stage": "prediction",
            "message": pred.get("error") or pred.get("message") or "目标不存在或模型不可用",
            "raw": pred,
        }

    device_type = pred.get("device_type") or ""
    top_features = pred.get("top_features") or []
    label, _, rs = _risk_band(pred.get("risk_score"), pred.get("risk_levels"))
    actions = _suggested_actions(pred, device_type)

    evidence = []
    for f in top_features[:5]:
        evidence.append({
            "feature": f.get("feature", ""),
            "label": explain_feature(f.get("feature", "")),
            "importance": f.get("importance"),
            "kind": _feature_direction(f.get("feature", "")),
        })

    kb_ctx, citations = kb_rag.build_context(_kb_query(pred, device_type), top_k=top_k_kb)

    conclusion = _conclude(pred, device_type, label, rs, evidence)

    result = {
        "ok": True,
        "target": target,
        "sensor_id": pred.get("sensor_id"),
        "asset_id": pred.get("asset_id"),
        "device_type": device_type,
        "region": pred.get("region"),
        "model_ready": bool(pred.get("model_ready")),
        "prediction": {
            "risk_score": pred.get("risk_score"),
            "future_anomaly_prob": pred.get("future_anomaly_prob"),
            "anomaly_score": pred.get("anomaly_score"),
            "rul": pred.get("rul"),
            "rul_unit": pred.get("rul_unit", "steps"),
            # RUL 的语义完全取决于窗口：>= horizon 是下界（「窗口内不会出事」），
            # < horizon 才是上界。不透传窗口，读响应的人没法判断该往哪个方向理解
            "horizon": pred.get("horizon"),
            "predicted_status": pred.get("predicted_status"),
            "risk_level": label,
        },
        "evidence": evidence,
        # 归因方法必须透传：遮挡归因/SHAP 解释的是「这个点位为什么被判高风险」，
        # 全局特征重要度解释的是「训练集里哪些特征普遍有用」，前端要照实标注
        "factor_method": pred.get("top_features_method") or "",
        "citations": citations,
        "kb_context": kb_ctx,
        "actions": actions,
        "suggested_priority": _action_for_alert(pred),
        "conclusion": conclusion,
    }
    if remember:
        remember_diagnosis(target, conclusion, pred.get("risk_score"), pred.get("predicted_status") or "")
    return result


def _conclude(pred, device_type, label, rs, evidence) -> str:
    st = pred.get("predicted_status") or "normal"
    st_cn = {"critical": "危急", "warning": "预警", "normal": "正常"}.get(st, st)
    prob = pred.get("future_anomaly_prob")
    rul = pred.get("rul")
    head = f"{device_type or '该管段'} {pred.get('asset_id') or pred.get('sensor_id') or ''}".strip()
    bits = [f"{head} 当前判定为「{st_cn}」，风险分 {rs:.1f}"]
    if prob is not None:
        bits.append(f"未来窗口内异常概率 {float(prob) * 100:.1f}%")
    if rul is not None:
        bits.append(f"RUL {_rul_span(rul, pred.get('horizon'))}")
    method = pred.get("top_features_method") or ""
    if evidence:
        top = evidence[0]
        imp = top.get("importance")
        # importance 的量纲取决于归因方式，不能一律念成「推高多少个百分点」：
        #   遮挡归因 = P(异常|实际) − P(异常|遮挡)，概率差值，可以说成百分点
        #   SHAP     = LightGBM/XGBoost 的贡献落在*对数几率*尺度，直接乘 100 会念出
        #              「推高 257.7 个百分点」这种超过概率上限的数（VM 实跑抓到过）
        #   全局重要度 = 树的分裂增益，与单个点位无关，只能原样报数并说明
        if imp is None:
            imp_txt = ""
        elif "全局" in method or "排列" in method:
            imp_txt = f"（全局重要度 {float(imp):.3f}，非本点位局部归因）"
        elif "SHAP" in method:
            imp_txt = f"（SHAP 贡献 {float(imp):+.2f}，对数几率尺度）"
        else:
            imp_txt = f"（单独推高异常概率 {float(imp) * 100:.1f} 个百分点）"
        bits.append(f"主因是{top['label']}{imp_txt}")
    elif "遮挡" in method:
        # 空榜单在局部归因下是有效结论，不是「算不出来」：把每个特征都换回该点位常态后
        # 概率不变，说明它本来就贴着常态运行
        bits.append("各指标均接近该点位自身常态，无显著推高因子")
    elif "SHAP" in method:
        bits.append("各特征 SHAP 贡献均不推高风险，该点位贴近自身常态")
    return "，".join(bits) + "。"


def horizon_text(horizon, seconds_per_step=3.0) -> str:
    """把「horizon 步」翻成人话，避免夸大预测跨度。"""
    try:
        h = float(horizon or 0)
        s = float(seconds_per_step or 3.0)
    except (TypeError, ValueError):
        return "—"
    sec = h * s
    if sec >= 86400:
        return f"{sec / 86400:.1f} 天"
    if sec >= 3600:
        return f"{sec / 3600:.1f} 小时"
    if sec >= 60:
        return f"{sec / 60:.1f} 分钟"
    return f"{sec:.0f} 秒"


# ==============================================================================
# 本地兜底 Agent：无 API Key 时的确定性意图规划器
# ==============================================================================
_ASSET_RE = re.compile(r"(?<![A-Za-z0-9\-])(SENSOR-\d{1,4}|[A-Z]{2,4}-\d{3,6})(?![A-Za-z0-9])", re.I)
_ALERT_RE = re.compile(r"(?<![A-Za-z0-9\-])(ALT-[A-Za-z0-9\-]+)", re.I)

_P_METRICS = [r"命中率", r"误报", r"提前预警", r"模型效果", r"预测成果", r"量化指标", r"基线对比", r"准不准", r"漏报"]
_P_PUSH = [r"推送", r"发(微信|邮件|短信)", r"通知.*(微信|邮件|短信|负责人)"]
_P_CREATE_WO = [r"建.*工单", r"创建.*工单", r"派单", r"生成工单", r"确认建单"]
_P_DIAG = [r"诊断", r"为什么", r"什么原因", r"怎么办", r"如何处置", r"处置建议", r"建议动作", r"分析一下", r"什么情况", r"怎么回事"]
_P_KB = [r"规范", r"国标", r"标准", r"检测周期", r"规程", r"怎么做", r"如何检测", r"阈值", r"规定", r"依据", r"要求是"]
_P_TOP = [r"风险最高", r"最危险", r"哪段", r"哪个", r"哪些", r"排名", r"top", r"未来.*(天|小时|风险)", r"重点关注"]
_P_ALERTS = [r"预警", r"告警", r"报警"]
_P_WO = [r"工单"]
_P_SUMMARY = [r"概况", r"总览", r"多少", r"统计", r"汇总", r"整体情况"]
_P_CONFIRM = [r"^确认", r"^执行", r"^同意", r"^好的", r"^可以", r"确认(推送|建单|执行|创建)", r"就(这么|这样)(办|做|推送|建单)"]

_ANAPHORA_RE = re.compile(r"(它|这个|那个|这段|那段|该管段|上述|上面|刚才|前面|继续|同样|还是)")
_ORDINAL_RE = re.compile(r"第\s*([一二两三四五六七八九十\d]+)\s*(段|个|条|名|位)")
_CN_NUM = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
           "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}


def _last_diagnosed_target() -> str:
    recent = load_memory().get("recent_diagnoses") or []
    return (recent[-1].get("target") or "") if recent else ""


def resolve_target(message: str, call):
    """
    连续追问的指代消解，返回 (target, note)：
      「第三段呢」         → 当前风险榜第 3 名
      「它的规范依据」      → 长期记忆里上一次诊断过的资产
    消息里已带编号时直接返回，不做推断。
    """
    msg = (message or "").strip()
    m = _ASSET_RE.search(msg)
    if m:
        return m.group(1).upper(), ""

    mo = _ORDINAL_RE.search(msg)
    if mo:
        raw = mo.group(1)
        n = int(raw) if raw.isdigit() else _CN_NUM.get(raw, 0)
        if n:
            summ = call("query_prediction_summary", {"top_n": n})
            rows = (summ.get("top_risky") or []) if isinstance(summ, dict) else []
            if len(rows) >= n:
                r = rows[n - 1]
                return (r.get("sensor_id") or r.get("asset_id") or ""), f"风险榜第 {n} 名"

    if _ANAPHORA_RE.search(msg) or len(msg) <= 10:
        last = _last_diagnosed_target()
        if last:
            return last, "沿用上次诊断对象"
    return "", ""


def detect_intent(message: str):
    """返回 (intent, asset_target)。按优先级判定，避免多意图互相抢答。"""
    msg = message or ""

    def has(pats):
        return any(re.search(p, msg, re.I) for p in pats)

    m = _ASSET_RE.search(msg)
    target = m.group(1).upper() if m else None

    if has(_P_PUSH):
        return "push", target
    if has(_P_CREATE_WO):
        return "create_workorder", target
    if has(_P_METRICS):
        return "predict_metrics", target
    if target and has(_P_DIAG):
        return "diagnose", target
    if has(_P_KB):
        return "kb", target
    if has(_P_TOP):
        return "top_risk", target
    if has(_P_DIAG) or target:
        return "diagnose", target
    if has(_P_ALERTS):
        return "alerts", target
    if has(_P_WO):
        return "workorders", target
    if has(_P_SUMMARY):
        return "summary", target
    return "summary", target


def _fmt_pct(v):
    return f"{float(v) * 100:.1f}%" if v is not None else "—"


def _add_cite(cites: list, c: dict):
    """收集完整引用条目（标题/出处/相关度），按 doc_id 去重，前端据此渲染引用卡片。"""
    if not isinstance(c, dict) or not c.get("title"):
        return
    key = c.get("doc_id") or c.get("title")
    if any((x.get("doc_id") or x.get("title")) == key for x in cites):
        return
    cites.append(c)


def _render_diagnosis(d: dict, lines: list, cites: list):
    if not d.get("ok"):
        lines.append(f"· 未能完成诊断：{d.get('message') or d.get('error') or '目标不存在'}")
        return
    p = d["prediction"]
    lines.append(f"**{d.get('device_type') or ''} {d.get('asset_id') or d.get('sensor_id')}**（{d.get('region') or '—'}）")
    lines.append("")
    lines.append("【预测结论】")
    lines.append(f"· 风险等级 **{p['risk_level']}**，风险分 **{p['risk_score']}**，"
                 f"未来窗口异常概率 **{_fmt_pct(p['future_anomaly_prob'])}**，RUL ≈ **{p['rul']}** 步")
    if d.get("evidence"):
        lines.append("")
        lines.append("【判定依据（模型关键因子）】")
        for e in d["evidence"][:4]:
            imp = e.get("importance")
            lines.append(f"· {e['label']}{'，贡献度 ' + format(float(imp), '.3f') if imp is not None else ''}")
    if d.get("citations"):
        lines.append("")
        lines.append("【规范依据】")
        for c in d["citations"]:
            _add_cite(cites, c)
            lines.append(f"· [{c['n']}] {c['title']} —— {c['source']}"
                         f"（相关度 {float(c.get('score') or 0):.2f}）")
    if d.get("actions"):
        lines.append("")
        lines.append("【处置建议】")
        for i, a in enumerate(d["actions"][:5], 1):
            lines.append(f"{i}. {a}")
        lines.append("")
        lines.append(f"建议工单优先级：**{d.get('suggested_priority')}**")


def _render_citation_footer(cites: list, lines: list):
    if cites:
        lines.append("")
        lines.append(f"（本回答引用知识库条目 {len(cites)} 条，可在「智能预测 → 知识库」查看原文）")


def local_agent_run(user: dict, message: str, exec_tool, extra: dict = None) -> dict:
    """
    无 DeepSeek Key 时的兜底：真实调用工具 → 组织四段式回答。
    exec_tool(name, args) -> JSON 字符串（复用 main.agent_execute_tool）
    extra: {"horizon":, "seconds_per_step":, "predict_fn":, "metrics_fn":}
    """
    extra = extra or {}
    intent, target = detect_intent(message)
    confirmed = any(re.search(p, (message or "").strip(), re.I) for p in _P_CONFIRM)
    lines, cites, actions = [], [], []

    def call(name, args=None):
        raw = exec_tool(name, args or {})
        actions.append({"tool": name, "args": args or {}})
        try:
            return json.loads(raw)
        except Exception:
            return {"raw": raw}

    if intent == "diagnose" or (target and intent not in ("push", "create_workorder")):
        note = ""
        if not target:
            target, note = resolve_target(message, call)
        if not target:
            summ = call("query_prediction_summary", {"top_n": 1})
            rows = (summ.get("top_risky") or []) if isinstance(summ, dict) else []
            target = rows[0].get("sensor_id") if rows else None
            note = note or "当前风险最高管段"
        if target:
            if note:
                lines.append(f"（{note}：{target}）")
            d = diagnose(target, extra.get("predict_fn") or (lambda t: call("predict_risk", {"asset_id": t})))
            _render_diagnosis(d, lines, cites)
        else:
            lines.append("· 请指定要诊断的资产编号（如 SENSOR-001）或资产号（如 WSP-00742）。")

    elif intent == "top_risk":
        summ = call("query_prediction_summary", {"top_n": 8})
        if not isinstance(summ, dict) or not summ.get("model_ready"):
            lines.append("· 预测模型尚未训练，暂无法给出风险排名。请先运行 train_predictive.py。")
        else:
            win = horizon_text(extra.get("horizon") or summ.get("horizon"),
                               extra.get("seconds_per_step") or 3.0)
            lines.append(f"**未来 {win} 内风险最高的管段**（模型 horizon={summ.get('horizon')} 步）：")
            lines.append("")
            rows = summ.get("top_risky") or []
            if not rows:
                lines.append("· 暂无预测样本，请先启动传感器模拟积累数据。")
            for i, r in enumerate(rows[:8], 1):
                st_cn = {"critical": "危急", "warning": "预警", "normal": "正常"}.get(r.get("status"), r.get("status"))
                lines.append(f"{i}. {r.get('asset_id') or r.get('sensor_id')}（{r.get('device_type')}·{r.get('region')}）"
                             f" 风险分 **{r.get('risk_score')}**，异常概率 {_fmt_pct(r.get('prob'))}，"
                             f"RUL≈{r.get('rul')} 步，判定 {st_cn}")
            by_type = summ.get("by_type") or []
            if by_type:
                lines.append("")
                lines.append("【分类型平均风险】")
                for t in by_type:
                    lines.append(f"· {t.get('pipe_type')}：{t.get('avg_risk')}（{t.get('count')} 个传感器，"
                                 f"危急 {t.get('critical', 0)}）")
            if rows:
                worst = rows[0]
                lines.append("")
                lines.append("【为什么是它】")
                d = diagnose(worst.get("sensor_id") or worst.get("asset_id"),
                             extra.get("predict_fn") or (lambda t: call("predict_risk", {"asset_id": t})))
                if d.get("ok"):
                    for e in (d.get("evidence") or [])[:3]:
                        lines.append(f"· {e['label']}")
                    for c in (d.get("citations") or [])[:2]:
                        _add_cite(cites, c)
                        lines.append(f"· [{c['n']}] {c['title']} —— {c['source']}")
                    lines.append("")
                    lines.append("【建议动作】")
                    for i, a in enumerate((d.get("actions") or [])[:3], 1):
                        lines.append(f"{i}. {a}")
                    lines.append(f"（建议工单优先级：{d.get('suggested_priority')}）")

    elif intent == "kb":
        res = call("search_knowledge_base", {"query": message})
        hits = (res.get("results") or []) if isinstance(res, dict) else []
        if not hits:
            lines.append("· 知识库未命中相关条目，可换个说法（如「球墨铸铁管检测周期」「燃气泄漏处置」）。")
        else:
            lines.append("**知识库检索结果**（向量语义检索，附出处）：")
            lines.append("")
            for i, h in enumerate(hits, 1):
                lines.append(f"[{i}] **{h.get('title')}** —— {h.get('source') or '内部资料'}"
                             f"（相关度 {float(h.get('score') or 0):.2f}）")
                lines.append(f"    {h.get('snippet') or h.get('content') or ''}")
                _add_cite(cites, {"n": i, "title": h.get("title"),
                                  "source": h.get("source") or "内部资料",
                                  "score": h.get("score"), "doc_id": h.get("doc_id"),
                                  "category": h.get("category"),
                                  "snippet": h.get("snippet") or ""})
                lines.append("")

    elif intent == "predict_metrics":
        m = (extra.get("metrics_fn") or (lambda: call("query_predict_metrics", {})))()
        if not isinstance(m, dict):
            lines.append("· 指标暂不可用。")
        else:
            mo, ru, dl = m.get("model") or {}, m.get("rule_baseline") or {}, m.get("delta") or {}
            lines.append("**预测成果量化**（模型 vs 规则阈值基线）")
            lines.append("")
            lines.append(f"· 台账样本 {m.get('ledger_size', 0)} 条，已评估 {mo.get('samples', 0)} 条，"
                         f"待评估 {m.get('pending_evaluation', 0)} 条")
            lines.append(f"· 命中率（召回）：**{_fmt_pct(mo.get('hit_rate'))}** ｜ 基线 {_fmt_pct(ru.get('hit_rate'))}")
            lines.append(f"· 精确率：**{_fmt_pct(mo.get('precision'))}** ｜ 基线 {_fmt_pct(ru.get('precision'))}")
            lines.append(f"· 误报率：**{_fmt_pct(mo.get('false_positive_rate'))}** ｜ 基线 {_fmt_pct(ru.get('false_positive_rate'))}")
            lines.append(f"· 平均提前预警：**{mo.get('avg_lead_text') or '—'}**（{mo.get('avg_lead_steps', 0)} 步）")
            if dl and dl.get("conclusion"):
                lines.append("")
                lines.append(f"【对比结论】{dl['conclusion']}")
            if mo.get("samples", 0) < 5:
                lines.append("")
                lines.append("（样本量还很小，指标会随传感器持续运行逐步收敛）")

    elif intent == "alerts":
        rows = call("query_alerts", {"status": "未处理"})
        rows = rows if isinstance(rows, list) else []
        lines.append(f"**未处理预警 {len(rows)} 条**")
        for a in rows[:10]:
            lines.append(f"· {a.get('alert_id')} [{a.get('level')}] {a.get('alert_type')} "
                         f"{a.get('asset_id')}（{a.get('region')}）{a.get('create_time')} 推送:{a.get('push_status')}")
        if not rows:
            lines.append("· 当前没有未处理预警。")

    elif intent == "workorders":
        rows = call("query_workorders", {})
        rows = rows if isinstance(rows, list) else []
        lines.append(f"**工单 {len(rows)} 条**")
        for w in rows[:10]:
            lines.append(f"· {w.get('workorder_id')} [{w.get('status')}] {w.get('title')} "
                         f"优先级{w.get('priority')} {w.get('region') or ''}")
        if not rows:
            lines.append("· 暂无工单。")

    elif intent == "push":
        alert_id = None
        mm = _ALERT_RE.search(message or "")
        if mm:
            alert_id = mm.group(1).upper()
        if not alert_id:
            rows = call("query_alerts", {"status": "未处理"})
            rows = rows if isinstance(rows, list) else []
            alert_id = rows[0].get("alert_id") if rows else None
        if not alert_id:
            lines.append("· 没有可推送的未处理预警。")
        elif confirmed:
            res = call("push_alert", {"alert_id": alert_id})
            if isinstance(res, dict) and res.get("error"):
                lines.append(f"· 推送失败：{res['error']}")
            else:
                lines.append(f"· 已推送预警 **{alert_id}**：成功 {res.get('sent', 0)} 条，失败 {res.get('failed', 0)} 条。")
                chs = res.get("push_channels") or []
                if chs:
                    lines.append(f"· 实际送达通道：{'、'.join(chs)}")
                lines.append("· 推送明细可在「预警管理 → 推送记录」查看。")
        else:
            lines.append(f"· 待确认：把预警 **{alert_id}** 推送到已启用通道（微信/邮箱/短信）。")
            lines.append("· 这是真实发送操作，请回复「确认推送 {alert_id}」我再执行。".format(alert_id=alert_id))

    elif intent == "create_workorder":
        if not confirmed:
            lines.append("· 待确认：需要我基于当前最高风险管段创建工单吗？")
            lines.append("· 请回复「确认建单」，我会带上风险分、关键因子与规范依据写入工单描述。")
        else:
            sid = target
            if not sid:
                summ = call("query_prediction_summary", {"top_n": 1})
                rows = (summ.get("top_risky") or []) if isinstance(summ, dict) else []
                sid = rows[0].get("sensor_id") if rows else None
            d = diagnose(sid, extra.get("predict_fn") or (lambda t: call("predict_risk", {"asset_id": t}))) if sid else {}
            if not d.get("ok"):
                lines.append(f"· 建单失败：{d.get('message') or '未找到风险最高的管段，请先启动传感器模拟或指定资产编号'}")
            else:
                p = d["prediction"]
                aid = d.get("asset_id") or d.get("sensor_id")
                desc_lines = [
                    f"【预测触发】风险分 {p['risk_score']}（{p['risk_level']}），"
                    f"未来窗口异常概率 {_fmt_pct(p['future_anomaly_prob'])}，RUL ≈ {p['rul']} 步。",
                    "【关键因子】" + ("；".join(e["label"] for e in (d.get("evidence") or [])[:3]) or "—"),
                    "【规范依据】" + ("；".join(f"{c['title']}（{c['source']}）" for c in (d.get("citations") or [])[:2]) or "—"),
                    "【处置建议】" + ("；".join((d.get("actions") or [])[:3]) or "—"),
                    "（本工单由 AI 助手依据预测诊断自动生成）",
                ]
                res = call("create_workorder", {
                    "title": f"{d.get('device_type') or ''}{aid} 预测性检修",
                    "priority": d.get("suggested_priority") or "中",
                    "region": d.get("region") or "",
                    "description": "\n".join(desc_lines),
                })
                if isinstance(res, dict) and res.get("error"):
                    lines.append(f"· 建单失败：{res['error']}")
                else:
                    lines.append(f"· 已创建工单 **{res.get('workorder_id')}**：{res.get('title')}")
                    lines.append(f"· 优先级 {res.get('priority')}，状态 {res.get('status')}，创建人已记入操作日志。")
                    lines.append("· 工单描述已写入预测结论、关键因子、规范依据与处置建议。")

    else:
        s = call("get_system_summary", {})
        if isinstance(s, dict) and not s.get("error"):
            lines.append("**系统概况**")
            for k, v in s.items():
                lines.append(f"· {k}：{v}")
        else:
            lines.append("· " + str(s.get("error") if isinstance(s, dict) else s))
        summ = call("query_prediction_summary", {"top_n": 3})
        if isinstance(summ, dict) and summ.get("model_ready"):
            rows = summ.get("top_risky") or []
            if rows:
                lines.append("")
                lines.append("**当前风险 TOP3**")
                for r in rows[:3]:
                    lines.append(f"· {r.get('asset_id') or r.get('sensor_id')}（{r.get('device_type')}）"
                                 f" 风险分 {r.get('risk_score')}")

    quick_actions = []
    if intent in ("diagnose", "top_risk"):
        quick_actions = [{"label": "据此建工单", "message": f"确认建单 {target}" if target else "确认建单"},
                         {"label": "推送最新预警", "message": "确认推送最新未处理预警"},
                         {"label": "模型命中率如何", "message": "模型命中率怎么样"}]
    elif intent == "kb":
        quick_actions = [{"label": "哪段风险最高", "message": "未来风险最高的管段是哪段"},
                         {"label": "据此建工单", "message": "确认建单"}]
    elif intent == "predict_metrics":
        quick_actions = [{"label": "哪段风险最高", "message": "未来风险最高的管段是哪段"}]

    if intent in ("diagnose", "top_risk"):
        mem = memory_context_block()
        prior = [l[2:] for l in mem.split("\n") if l.startswith("· ")][1:3] if mem else []
        if prior:
            lines.append("")
            lines.append("【记忆】此前诊断：" + "；".join(prior))

    _render_citation_footer(cites, lines)

    if not lines:
        lines.append("· 我没太理解这个问题。可以试试：「未来风险最高的管段是哪段」「诊断 SENSOR-001」"
                     "「球墨铸铁管检测周期依据」「模型命中率怎么样」。")

    reply = "\n".join(lines)
    return {"reply": reply, "actions": actions, "citations": cites, "engine": "local_rule",
            "intent": intent, "target": target, "quick_actions": quick_actions}
