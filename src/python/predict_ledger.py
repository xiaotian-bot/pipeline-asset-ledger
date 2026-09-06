#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
预测台账与闭环量化
==================
把「预测 → 预警 → 推送 → 建单 → 处置 → 回流」串成可度量的闭环：

  1) 台账  每轮预测扫描落一条记录（模型 + 规则基线各一条，便于同口径对比）
  2) 评估  horizon 步后回看该区段真实是否出现异常 → 命中 / 误报 / 漏报
  3) 指标  命中率、精确率、误报率、平均提前预警时长、与规则基线对比
  4) 反馈  人工处置结论回填为标签（确认异常 / 误报 / 已处置），供批式重训

存储：data/prediction_ledger.json（记录）、data/feedback_labels.json（反馈标签）
"""
import json
import os
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LEDGER_FILE = os.path.join(DATA_DIR, "prediction_ledger.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_labels.json")
RETRAIN_FILE = os.path.join(DATA_DIR, "retrain_history.json")

MAX_RECORDS = 2000
MAX_FEEDBACK = 2000


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                v = json.load(f)
            return v if v is not None else default
    except Exception:
        pass
    return default


def _save(path, obj, compact=False):
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
        tmp = path + ".tmp"
        kw = {"ensure_ascii": False}
        if compact:
            kw["separators"] = (",", ":")
        else:
            kw["indent"] = 2
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, **kw)
        os.replace(tmp, path)
        return True
    except Exception:
        return False


# ==============================================================================
# 台账读写
# ==============================================================================
_ledger_cache = None


def load_ledger() -> list:
    global _ledger_cache
    if _ledger_cache is None:
        data = _load(LEDGER_FILE, {})
        _ledger_cache = data.get("records", []) if isinstance(data, dict) else []
    return _ledger_cache


def _flush_ledger():
    global _ledger_cache
    recs = load_ledger()[-MAX_RECORDS:]
    # 内存这份也必须一起裁。只裁落盘副本的话，_ledger_cache 会无上界地涨，而 metrics()
    # 算的正是这份内存列表——于是重训前旧模型写的记录会一直赖在分母里（实测涨到 5000 条，
    # 精确率成了新旧模型的混合值），磁盘上反而是干净的最近 2000 条，
    # 变成「重启后看到的比一直跑着的更准」这种不一致。
    _ledger_cache = recs
    _save(LEDGER_FILE, {"records": recs, "updated_at": _now()}, compact=True)


def flush():
    """把内存台账落盘（供每轮预测扫描结束时调用）。"""
    _flush_ledger()


def load_feedback() -> list:
    return _load(FEEDBACK_FILE, []) or []


def save_feedback(items: list):
    _save(FEEDBACK_FILE, items[-MAX_FEEDBACK:])


# ==============================================================================
# 记录预测
# ==============================================================================
def record_prediction(sensor: dict, pred: dict, tick: int, hist_len: int,
                      horizon: int, alert_id: str = "", method: str = "model") -> dict:
    """落一条预测记录。hist_len 用于事后在 sensor_history 上按索引回看真实标签。"""
    rec = {
        "id": f"PL-{int(time.time() * 1000)}-{sensor.get('sensor_id', '')}",
        "time": _now(),
        "tick": tick,
        "sensor_id": sensor.get("sensor_id", ""),
        "asset_id": sensor.get("asset_id", ""),
        "device_type": sensor.get("device_type", ""),
        "region": sensor.get("region", ""),
        "method": method,
        "horizon": horizon,
        "hist_len": hist_len,
        "risk_score": pred.get("risk_score"),
        "future_anomaly_prob": pred.get("future_anomaly_prob"),
        "anomaly_score": pred.get("anomaly_score"),
        "rul": pred.get("rul"),
        "predicted_status": pred.get("predicted_status"),
        "top_features": [f.get("feature") for f in (pred.get("top_features") or [])[:3]],
        "alert_id": alert_id,
        "evaluated": False,
        "outcome": None,
        "actual_anomaly": None,
        "lead_steps": None,
        "feedback": None,
    }
    load_ledger().append(rec)
    return rec


def record_rule_baseline(sensor: dict, triggered: bool, risk_score, tick: int,
                         hist_len: int, horizon: int) -> dict:
    """规则阈值基线：当前帧越限即外推「未来 horizon 内会异常」。"""
    return record_prediction(
        sensor,
        {"risk_score": risk_score, "future_anomaly_prob": 1.0 if triggered else 0.0,
         "anomaly_score": None, "rul": None,
         "predicted_status": "critical" if triggered else "normal", "top_features": []},
        tick, hist_len, horizon, alert_id="", method="rule_baseline")


# ==============================================================================
# 事后评估
# ==============================================================================
def _forward_anomaly(history: list, rec: dict):
    """
    回看预测点之后 horizon 帧内是否出现异常，返回 (是否异常, 首次异常偏移帧数)。
    用 tick（仿真帧号）定位而非列表索引：main.py 里每个传感器的历史只留最近 120 条
    并会裁掉头部，索引会整体前移导致回看错位。
    """
    if not history:
        return None, None
    horizon = max(1, int(rec.get("horizon") or 12))
    tick = rec.get("tick")
    entries = [r for r in history if isinstance(r, dict)]
    has_ticks = any(r.get("tick") is not None for r in entries)

    if has_ticks and tick is not None:
        newest = max((r.get("tick") or 0) for r in entries)
        if newest - tick > horizon * 4:
            return "expired", None          # 窗口已被裁剪掉，永远凑不满，标记过期
        window = [r for r in entries if (r.get("tick") or 0) > tick][:horizon]
    else:
        start = max(0, min(int(rec.get("hist_len") or 0), len(history)))
        window = history[start:start + horizon]

    if len(window) < horizon:
        return None, None                   # 还没走够 horizon 帧，暂不评估
    for i, r in enumerate(window):
        st = (r.get("status") or "normal") if isinstance(r, dict) else "normal"
        if st != "normal":
            return True, i
    return False, None


def evaluate(history_by_sensor: dict, seconds_per_step: float = 3.0) -> dict:
    """对未评估记录做标签回填。history_by_sensor: {sensor_id: [readings...]}"""
    recs = load_ledger()
    n_eval = n_expired = 0
    for r in recs:
        if r.get("evaluated"):
            continue
        hist = history_by_sensor.get(r.get("sensor_id")) or []
        actual, offset = _forward_anomaly(hist, r)
        if actual is None:
            continue
        if actual == "expired":
            r.update({"evaluated": True, "outcome": "expired", "actual_anomaly": None,
                      "evaluated_at": _now()})
            n_expired += 1
            continue
        r["evaluated"] = True
        r["actual_anomaly"] = bool(actual)
        predicted = r.get("predicted_status") in ("critical", "warning")
        if predicted and actual:
            r["outcome"] = "hit"
            r["lead_steps"] = int(offset or 0)
        elif predicted and not actual:
            r["outcome"] = "false_positive"
            r["lead_steps"] = None
        elif actual:
            r["outcome"] = "miss"           # 漏报：真出异常了，模型没提前预警
            r["lead_steps"] = None
        else:
            r["outcome"] = "correct_normal"
            r["lead_steps"] = None
        r["evaluated_at"] = _now()
        r["seconds_per_step"] = seconds_per_step
        n_eval += 1
    if n_eval or n_expired:
        _flush_ledger()
    return {"evaluated_now": n_eval, "expired_now": n_expired}


# ==============================================================================
# 指标
# ==============================================================================
def _confusion(recs: list):
    hit = fp = cn = miss = 0
    leads = []
    for r in recs:
        if not r.get("evaluated"):
            continue
        o = r.get("outcome")
        if o == "hit":
            hit += 1
            if r.get("lead_steps") is not None:
                leads.append(float(r["lead_steps"]))
        elif o == "false_positive":
            fp += 1
        elif o == "correct_normal":
            cn += 1
        elif o == "miss":
            miss += 1
    return {"hit": hit, "false_positive": fp, "correct_normal": cn, "miss": miss, "leads": leads}


def _rate_block(c: dict, seconds_per_step: float):
    hit, fp, miss, cn = c["hit"], c["false_positive"], c["miss"], c["correct_normal"]
    pred_pos = hit + fp
    actual_pos = hit + miss
    total = hit + fp + miss + cn
    leads = c["leads"]
    avg_lead = (sum(leads) / len(leads)) if leads else 0.0
    return {
        "samples": total,
        "evaluated": total,
        "hit": hit,
        "false_positive": fp,
        "miss": miss,
        "correct_normal": cn,
        "predicted_positive": pred_pos,
        "actual_positive": actual_pos,
        "hit_rate": round(hit / actual_pos, 4) if actual_pos else None,        # 召回率：真实异常中被提前预警的比例
        "precision": round(hit / pred_pos, 4) if pred_pos else None,           # 精确率：预警中确实发生异常的比例
        "false_positive_rate": round(fp / pred_pos, 4) if pred_pos else None,  # 误报率
        "avg_lead_steps": round(avg_lead, 2),
        "avg_lead_seconds": round(avg_lead * seconds_per_step, 1),
        "avg_lead_text": _lead_text(avg_lead, seconds_per_step),
        "max_lead_steps": int(max(leads)) if leads else 0,
    }


def _lead_text(steps: float, seconds_per_step: float):
    if not steps:
        return "—"
    sec = steps * seconds_per_step
    if sec >= 3600:
        return f"{sec / 3600:.1f} 小时"
    if sec >= 60:
        return f"{sec / 60:.1f} 分钟"
    return f"{sec:.0f} 秒"


def metrics(seconds_per_step: float = 3.0, horizon: int = 12, model_ready: bool = False) -> dict:
    recs = load_ledger()
    model = _rate_block(_confusion([r for r in recs if r.get("method") == "model"]), seconds_per_step)
    rule = _rate_block(_confusion([r for r in recs if r.get("method") == "rule_baseline"]), seconds_per_step)

    delta = None
    if model["samples"] and rule["samples"]:
        def _d(a, b):
            return round(a - b, 4) if (a is not None and b is not None) else None
        delta = {
            "hit_rate": _d(model["hit_rate"], rule["hit_rate"]),
            "precision": _d(model["precision"], rule["precision"]),
            "false_positive_rate": _d(model["false_positive_rate"], rule["false_positive_rate"]),
            "avg_lead_steps": _d(model["avg_lead_steps"], rule["avg_lead_steps"]),
            "conclusion": _compare_text(model, rule),
        }

    fb = load_feedback()
    fb_stat = {"total": len(fb)}
    for k in ("确认异常", "误报", "已处置", "无需处置"):
        fb_stat[k] = len([x for x in fb if x.get("outcome") == k])

    # 趋势：按小时聚合命中率与风险分，供大屏折线图
    trend = _trend(recs)
    retrains = retrain_history()

    return {
        "model_ready": model_ready,
        "horizon": horizon,
        "seconds_per_step": seconds_per_step,
        "ledger_size": len(recs),
        "pending_evaluation": len([r for r in recs if not r.get("evaluated")]),
        "model": model,
        "rule_baseline": rule,
        "delta": delta,
        "feedback": fb_stat,
        "trend": trend,
        "last_retrain": retrains[-1] if retrains else None,
        "generated_at": _now(),
    }


def _compare_text(m: dict, r: dict):
    parts = []
    if m["hit_rate"] is not None and r["hit_rate"] is not None:
        d = (m["hit_rate"] - r["hit_rate"]) * 100
        parts.append(f"命中率 {'高' if d >= 0 else '低'} {abs(d):.1f} 个百分点")
    if m["false_positive_rate"] is not None and r["false_positive_rate"] is not None:
        d = (r["false_positive_rate"] - m["false_positive_rate"]) * 100
        parts.append(f"误报率 {'低' if d >= 0 else '高'} {abs(d):.1f} 个百分点")
    if m["avg_lead_steps"]:
        parts.append(f"平均提前 {m['avg_lead_text']} 预警")
    return "；".join(parts) if parts else "样本不足，暂无可比结论"


def _trend(recs: list):
    """按扫描轮次聚合：模型预警数、命中数、平均风险分。

    一次 _prediction_scan 写入的记录共享同一个 tick，所以 tick 就是天然的轮次编号。
    原先按 time 的小时聚合，但台账只留最近 MAX_RECORDS 条，而一轮扫描写 200 条
    （100 传感器 × 模型/规则基线），可见窗口只有 10 轮 ≈ 十几分钟，几乎必然整段落在
    同一个小时桶里——趋势线永远只剩一个点，看不出任何走势。
    """
    buckets = {}
    for r in recs:
        if r.get("method") != "model":
            continue
        tick = r.get("tick")
        if tick is None:
            continue
        b = buckets.setdefault(int(tick), {"predictions": 0, "warnings": 0, "hits": 0,
                                           "risk_sum": 0.0, "risk_n": 0, "time": ""})
        b["predictions"] += 1
        if r.get("predicted_status") in ("critical", "warning"):
            b["warnings"] += 1
        if r.get("outcome") == "hit":
            b["hits"] += 1
        if r.get("risk_score") is not None:
            b["risk_sum"] += float(r["risk_score"])
            b["risk_n"] += 1
        if not b["time"]:
            b["time"] = r.get("time") or ""
    out = []
    for tick in sorted(buckets.keys()):
        b = buckets[tick]
        out.append({
            "tick": tick, "time": b["time"],
            "predictions": b["predictions"], "warnings": b["warnings"], "hits": b["hits"],
            "avg_risk": round(b["risk_sum"] / b["risk_n"], 1) if b["risk_n"] else None,
        })
    return out[-48:]


# ==============================================================================
# 处置反馈（回流）
# ==============================================================================
def add_feedback(alert_id: str, sensor_id: str, outcome: str, note: str = "",
                 user: str = "", extra: dict = None) -> dict:
    """outcome: 确认异常 / 误报 / 已处置 / 无需处置"""
    fb = {
        "id": f"FB-{int(time.time() * 1000)}",
        "time": _now(),
        "alert_id": alert_id or "",
        "sensor_id": sensor_id or "",
        "outcome": outcome,
        "note": note or "",
        "by": user or "",
    }
    if extra:
        fb["extra"] = extra
    items = load_feedback()
    items.append(fb)
    save_feedback(items)

    # 回填台账：人工结论覆盖自动标签，作为重训的真实标签
    changed = 0
    label = {"确认异常": "hit", "误报": "false_positive", "已处置": "hit", "无需处置": "correct_normal"}.get(outcome)
    for r in load_ledger():
        if (alert_id and r.get("alert_id") == alert_id) or (sensor_id and r.get("sensor_id") == sensor_id and not r.get("evaluated")):
            r["feedback"] = {"outcome": outcome, "note": note, "by": user, "time": fb["time"]}
            if label and alert_id and r.get("alert_id") == alert_id:
                r["evaluated"] = True
                r["outcome"] = label
                r["actual_anomaly"] = label in ("hit",)
                r["feedback_label"] = True
                changed += 1
    if changed:
        _flush_ledger()
    return {"success": True, "feedback": fb, "ledger_updated": changed}


def feedback_training_rows() -> list:
    """导出可直接喂给重训的标签行（sensor_id + 结论 + 时间）。"""
    rows = []
    for fb in load_feedback():
        rows.append({
            "sensor_id": fb.get("sensor_id", ""),
            "alert_id": fb.get("alert_id", ""),
            "label": {"确认异常": 1, "已处置": 1, "误报": 0, "无需处置": 0}.get(fb.get("outcome"), None),
            "outcome": fb.get("outcome", ""),
            "time": fb.get("time", ""),
            "note": fb.get("note", ""),
        })
    return [r for r in rows if r["label"] is not None]


# ==============================================================================
# 重训记录
# ==============================================================================
def add_retrain_record(status: str, detail: dict):
    items = _load(RETRAIN_FILE, []) or []
    items.append({"time": _now(), "status": status, **detail})
    _save(RETRAIN_FILE, items[-50:])
    return items[-1]


def retrain_history() -> list:
    return _load(RETRAIN_FILE, []) or []


# ==============================================================================
# 大屏用聚合视图
# ==============================================================================
def heatmap(rows: list):
    """区域 × 管网类型 的风险热力矩阵。rows 来自 prediction_summary 的逐传感器结果。"""
    regions, types = [], []
    cell = {}
    for r in rows:
        reg = r.get("region") or "—"
        typ = r.get("device_type") or "—"
        if reg not in regions:
            regions.append(reg)
        if typ not in types:
            types.append(typ)
        cell.setdefault((reg, typ), []).append(float(r.get("risk_score") or 0))
    data = []
    for i, reg in enumerate(regions):
        for j, typ in enumerate(types):
            vals = cell.get((reg, typ))
            data.append([j, i, round(sum(vals) / len(vals), 1) if vals else 0, len(vals) if vals else 0])
    return {"regions": regions, "types": types, "data": data}


def recent_records(limit: int = 60, method: str = "model") -> list:
    recs = [r for r in load_ledger() if r.get("method") == method]
    recs.sort(key=lambda r: r.get("time", ""), reverse=True)
    return recs[:limit]
