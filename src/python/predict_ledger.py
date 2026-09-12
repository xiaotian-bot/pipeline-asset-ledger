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

可靠性约定（审计后补的，别再退回旧的"静默"写法）：
  · 落盘一律原子写（同目录 .tmp → flush + fsync → os.replace）；失败抛 LedgerPersistError 并记日志
  · 读盘区分「文件不存在 / 空文件」（= 空台账）与「文件损坏」（抛 LedgerCorruptError）；
    损坏文件绝不当成空台账继续，否则按"没有记录"去重会重复建单
  · 记录主键 id 与预警号 alert_id 都带 tick 维度保证唯一（同一毫秒 / 同一传感器的记录不再撞号）
"""
import json
import logging
import os
import time
import uuid
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
LEDGER_FILE = os.path.join(DATA_DIR, "prediction_ledger.json")
FEEDBACK_FILE = os.path.join(DATA_DIR, "feedback_labels.json")
RETRAIN_FILE = os.path.join(DATA_DIR, "retrain_history.json")

MAX_RECORDS = 2000
MAX_FEEDBACK = 2000

logger = logging.getLogger(__name__)


class LedgerPersistError(RuntimeError):
    """落盘失败（磁盘满 / 权限 / 只读挂载）。调用方必须能察觉，不能当成功。"""


class LedgerCorruptError(ValueError):
    """文件存在但已损坏。必须显式暴露：当成空台账会让指标统计与建单去重一起失效。"""


# 落盘健康度：main.py 里 flush() / evaluate() 都被 try/except 包着，异常到那一层会被吞掉，
# 所以这里额外留一份状态，由 metrics() 暴露给指标页——避免"面板有账、磁盘没账"的假象。
_persist_state = {"last_ok_at": "", "last_error": "", "last_error_at": "", "last_error_path": ""}


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _load(path, default, strict=True):
    """读 JSON，区分「文件不存在 / 空文件」与「文件损坏」。

    文件不存在或内容为空 → 返回 default（首次启动的正常语义，保留不变）。
    内容损坏 → strict=True 时抛 LedgerCorruptError。
    旧实现把任何异常都 pass 掉再返回 default，于是损坏的台账/工单文件被当成"空的"，
    下游"按 alert_id 去重建单"直接失效 → 每轮扫描重复建单；这条静默路径必须堵死。
    """
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except OSError as e:
        if strict:
            raise LedgerCorruptError(f"读取失败 {path}: {e}") from e
        logger.warning("读取 %s 失败：%s", path, e)
        return default
    if not raw.strip():
        return default                       # 空文件按"还没有账"处理
    try:
        v = json.loads(raw)
    except Exception as e:
        if strict:
            raise LedgerCorruptError(f"文件损坏 {path}: {e}") from e
        logger.warning("文件损坏 %s：%s", path, e)
        return default
    return default if v is None else v


def atomic_write_json(path, obj, compact=False) -> bool:
    """原子写 JSON：同目录临时文件 → flush + fsync → os.replace。

    为什么这么写：os.replace 本身是原子的，但替换前数据若还在页缓存里没落盘，
    断电/被 kill 后磁盘上的"新文件"可能是半截内容；先 fsync 才能保证替换过去的是完整 JSON。
    写失败一律抛 LedgerPersistError（不再 return False 让调用方悄悄忽略），并留痕供 metrics() 暴露。
    公开出来是为了让同样写 data/*.json 的相邻模块（如 workorders.json 的写盘）能复用同一套原子写。
    """
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path) or DATA_DIR, exist_ok=True)
        kw = {"ensure_ascii": False}
        if compact:
            kw["separators"] = (",", ":")
        else:
            kw["indent"] = 2
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, **kw)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError as e:
                # 少数文件系统（部分网络盘）不支持 fsync：降级为 flush + 原子替换，但不静默
                logger.warning("fsync 不可用（%s），仅保证 flush + 原子替换：%s", path, e)
        os.replace(tmp, path)
        _persist_state["last_ok_at"] = _now()
        return True
    except Exception as e:
        _persist_state.update({"last_error": f"{type(e).__name__}: {e}",
                               "last_error_at": _now(), "last_error_path": path})
        logger.error("落盘失败 %s：%s", path, e)
        try:
            if os.path.exists(tmp):
                os.remove(tmp)               # 清掉半截临时文件，别让它污染下一次读取
        except OSError:
            pass
        raise LedgerPersistError(f"落盘失败 {path}: {e}") from e


def _save(path, obj, compact=False) -> bool:
    """保留旧函数名/签名：内部已换成原子写，失败抛错而不是返回 False。"""
    return atomic_write_json(path, obj, compact=compact)


def _persist_status() -> dict:
    """落盘健康度快照，供指标接口 / 大屏判断磁盘上到底有没有账。"""
    st = _persist_state
    err = st["last_error"] or ""
    # 失败之后又成功落盘过，说明当前磁盘状态是好的（报错不 sticky，否则一次抖动会一直吓唬面板）
    healed = bool(err) and st["last_ok_at"] >= st["last_error_at"]
    return {"ok": not err or healed, "last_error": err,
            "last_error_at": st["last_error_at"], "last_ok_at": st["last_ok_at"],
            "last_error_path": st["last_error_path"]}


# ==============================================================================
# 台账读写
# ==============================================================================
_ledger_cache = None


def _normalize_record(r: dict) -> dict:
    """兼容旧台账：只补新字段，不改写旧字段、不改旧结论。

    locked 是本次新增的（人工标签写一次就加锁）。旧 JSON 没有这个字段，就按 evaluated 推断
    ——已评估即已锁定；否则升级后第一次点反馈，会把历史上已被自动评估成
    hit / miss / false_positive 的记录再批量改写一遍。字段缺失也不会 KeyError。
    """
    if not isinstance(r, dict):
        return r
    if "locked" not in r:
        r["locked"] = bool(r.get("evaluated"))
    return r


def load_ledger() -> list:
    """读台账（带内存缓存，返回的就是缓存列表本身，调用方可直接 append）。

    文件不存在 / 空文件 → 空台账；结构损坏 → 抛 LedgerCorruptError，绝不返回空台账
    （旧实现会安静地给出 []，于是指标分母清零、建单去重失效，问题被藏起来）。
    """
    global _ledger_cache
    if _ledger_cache is None:
        data = _load(LEDGER_FILE, {})
        recs = data.get("records") if isinstance(data, dict) else data   # 容忍早期直接存数组的写法
        if recs is None:
            recs = []
        if not isinstance(recs, list) or any(not isinstance(r, dict) for r in recs):
            raise LedgerCorruptError(f"台账结构异常 {LEDGER_FILE}: records 不是记录数组")
        # 旧记录主键/预警号保持原样（不重写历史），只补默认字段，保证仍能被正常读取与统计
        _ledger_cache = [_normalize_record(r) for r in recs]
    return _ledger_cache


def _flush_ledger() -> bool:
    global _ledger_cache
    recs = load_ledger()[-MAX_RECORDS:]
    # 内存这份也必须一起裁。只裁落盘副本的话，_ledger_cache 会无上界地涨，而 metrics()
    # 算的正是这份内存列表——于是重训前旧模型写的记录会一直赖在分母里（实测涨到 5000 条，
    # 精确率成了新旧模型的混合值），磁盘上反而是干净的最近 2000 条，
    # 变成「重启后看到的比一直跑着的更准」这种不一致。
    _ledger_cache = recs
    _save(LEDGER_FILE, {"records": recs, "updated_at": _now()}, compact=True)
    return True


def flush():
    """把内存台账落盘（供每轮预测扫描结束时调用）。失败抛 LedgerPersistError 并留痕到
    _persist_state（main.py 的 try/except 会吞异常，指标页靠这份状态兜底）。"""
    return _flush_ledger()


def load_feedback() -> list:
    """读反馈标签。文件不存在 / 空 → []；有内容但不是数组 = 损坏，抛错而不是当空标签继续。"""
    data = _load(FEEDBACK_FILE, [])
    if not data:
        return []
    if not isinstance(data, list):
        raise LedgerCorruptError(f"反馈标签文件结构异常 {FEEDBACK_FILE}: 顶层不是数组")
    return data


def save_feedback(items: list):
    _save(FEEDBACK_FILE, items[-MAX_FEEDBACK:])


# ==============================================================================
# 记录预测
# ==============================================================================
def _unique_alert_id(alert_id: str, tick) -> str:
    """把「每传感器恒定」的预警号改成「传感器 × tick」唯一，同时保留可读前缀。

    为什么必须唯一：人工反馈是按 alert_id 回填台账的。编号恒定 ⇒ 一次点击会命中该传感器的
    全部历史记录（含已被自动评估的），前向评估被不可逆污染。

    编号形如 ALT-PRD-0007-T1234-a1b2c3d4：前缀（含传感器号与 T<tick>）供人工排查对上是哪一轮，
    后缀 8 位 uuid 兜住"仿真重启后 tick 归零"造成的跨批次重号。alert_id 为空（规则基线）时保持空。
    """
    base = (alert_id or "").strip()
    if not base:
        return ""
    return f"{base}-T{int(tick or 0)}-{uuid.uuid4().hex[:8]}"


def record_prediction(sensor: dict, pred: dict, tick: int, hist_len: int,
                      horizon: int, alert_id: str = "", method: str = "model") -> dict:
    """落一条预测记录。hist_len 用于事后在 sensor_history 上按索引回看真实标签。"""
    sensor_id = sensor.get("sensor_id", "")
    rec = {
        # 主键加 tick + uuid：旧的 PL-<毫秒>-<sensor> 在同一毫秒写"模型 / 规则基线"两条时会撞号
        # （实测 2000/4000 重复）。旧记录的主键原样保留，仍能正常读出来。
        "id": f"PL-{int(time.time() * 1000)}-{sensor_id}-T{int(tick or 0)}-{method}-{uuid.uuid4().hex[:8]}",
        "time": _now(),
        "tick": tick,
        "sensor_id": sensor_id,
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
        "alert_id": _unique_alert_id(alert_id, tick),
        # 大屏上那个每传感器恒定的编号单独留一份：人工只拿到它时仍能反查到台账记录
        "alert_base": (alert_id or "").strip(),
        "evaluated": False,
        "locked": False,
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
    """对未评估记录做标签回填。history_by_sensor: {sensor_id: [readings...]}

    评估完立刻 locked：自动结论和人工结论一样是"已定稿"的标签，人工反馈不得追溯改写。
    """
    recs = load_ledger()
    n_eval = n_expired = 0
    for r in recs:
        # locked 兜底判一次：锁定但漏置 evaluated 的异常态也不该被自动评估覆盖
        if r.get("evaluated") or r.get("locked"):
            continue
        hist = history_by_sensor.get(r.get("sensor_id")) or []
        actual, offset = _forward_anomaly(hist, r)
        if actual is None:
            continue
        if actual == "expired":
            r.update({"evaluated": True, "locked": True, "outcome": "expired", "actual_anomaly": None,
                      "evaluated_at": _now(), "seconds_per_step": seconds_per_step})
            n_expired += 1
            continue
        r["evaluated"] = True
        r["locked"] = True
        r["actual_anomaly"] = bool(actual)
        predicted = r.get("predicted_status") in ("critical", "warning")
        if predicted and actual:
            r["outcome"] = "hit"
            r["lead_steps"] = int(offset or 0)
            # 提前秒数在评估当时就按"这一次的采样周期"固定下来：采样周期是可改配置，
            # 留到统计时才换算的话，改一次周期会把整本历史台账的提前时长一起伸缩（一本账两种口径）。
            r["lead_seconds"] = round(r["lead_steps"] * float(seconds_per_step or 0), 1)
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
def _record_sps(r: dict, default: float):
    """取这条记录自己的采样周期，返回 (秒/步, 是否用了兜底默认值)。

    缺失 / 非法 / ≤0 一律退回默认值并标记为兜底：兜底必须被计数，不能静默——否则
    "提前 3 分钟"这类结论里混了多少条按当前周期硬算的旧记录，面板上根本看不出来。
    """
    try:
        sps = float(r.get("seconds_per_step"))
    except (TypeError, ValueError):
        return float(default or 0), True
    if sps <= 0:
        return float(default or 0), True
    return sps, False


def _confusion(recs: list, default_seconds_per_step: float = 3.0):
    """混淆矩阵 + 提前量。

    提前量按每条记录自己的 seconds_per_step 换算后再聚合（缺字段的旧记录退回默认值并计数），
    不再统一乘"当前采样周期"——那样改一次周期，整本历史台账的提前时长会一起伸缩。
    """
    hit = fp = cn = miss = 0
    leads = []              # 提前帧数（步）
    lead_seconds = []       # 每条记录按自身采样周期换算出的提前秒数
    fallback = 0            # 缺 seconds_per_step、只能按默认周期兜底的记录条数
    for r in recs:
        if not r.get("evaluated"):
            continue
        o = r.get("outcome")
        if o == "hit":
            hit += 1
            if r.get("lead_steps") is not None:
                steps = float(r["lead_steps"])
                leads.append(steps)
                sec = r.get("lead_seconds")
                if sec is None:
                    # 评估时已按当时周期写死的记录直接用 lead_seconds；只有更老的记录才现算
                    sps, used_default = _record_sps(r, default_seconds_per_step)
                    fallback += 1 if used_default else 0
                    sec = steps * sps
                try:
                    lead_seconds.append(float(sec))
                except (TypeError, ValueError):
                    pass
        elif o == "false_positive":
            fp += 1
        elif o == "correct_normal":
            cn += 1
        elif o == "miss":
            miss += 1
    return {"hit": hit, "false_positive": fp, "correct_normal": cn, "miss": miss,
            "leads": leads, "lead_seconds": lead_seconds, "lead_seconds_fallback": fallback}


def _rate_block(c: dict, seconds_per_step: float):
    """把混淆矩阵换算成指标。

    seconds_per_step 现在只是"记录缺自己采样周期"时的兜底默认值（旧调用方按位置传参，签名保留），
    真正的提前时长用每条记录自己的周期换算后再平均。
    """
    hit, fp, miss, cn = c["hit"], c["false_positive"], c["miss"], c["correct_normal"]
    pred_pos = hit + fp
    actual_pos = hit + miss
    total = hit + fp + miss + cn
    leads = c["leads"]
    lead_secs = c.get("lead_seconds")
    if lead_secs is None:
        # 兼容外部自己构造 c 的旧调用：没有逐记录秒数时退回"当前周期 × 步数"
        lead_secs = [float(s) * float(seconds_per_step or 0) for s in leads]
    avg_lead = (sum(leads) / len(leads)) if leads else 0.0
    avg_sec = (sum(lead_secs) / len(lead_secs)) if lead_secs else 0.0
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
        "avg_lead_seconds": round(avg_sec, 1),
        "avg_lead_text": _lead_text_seconds(avg_sec),
        "max_lead_steps": int(max(leads)) if leads else 0,
        "max_lead_seconds": round(max(lead_secs), 1) if lead_secs else 0.0,
        # 兜底记录条数：指标页据此判断"这些提前时长里有多少条是按默认周期算的"
        "lead_seconds_fallback": int(c.get("lead_seconds_fallback") or 0),
    }


def _lead_text_seconds(sec: float):
    """秒数直接格式化（各记录周期已各自换算完，这里不再乘任何周期）。"""
    if not sec:
        return "—"
    if sec >= 3600:
        return f"{sec / 3600:.1f} 小时"
    if sec >= 60:
        return f"{sec / 60:.1f} 分钟"
    return f"{sec:.0f} 秒"


def _lead_text(steps: float, seconds_per_step: float):
    """旧签名保留（按"当前周期 × 步数"格式化）；新代码走 _lead_text_seconds。"""
    if not steps:
        return "—"
    return _lead_text_seconds(float(steps) * float(seconds_per_step or 0))


def metrics(seconds_per_step: float = 3.0, horizon: int = 12, model_ready: bool = False) -> dict:
    recs = load_ledger()
    # 每条记录的提前量用各自的 seconds_per_step 算；传进来的当前周期只作旧记录的兜底默认值
    model = _rate_block(_confusion([r for r in recs if r.get("method") == "model"],
                                   seconds_per_step), seconds_per_step)
    rule = _rate_block(_confusion([r for r in recs if r.get("method") == "rule_baseline"],
                                  seconds_per_step), seconds_per_step)

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
    fb_stat = {"total": len(fb),
               # 生效 / 未生效分开报：未生效的是"点过但没写进台账"（没有可写的未评估记录），
               # 它不参与重训导出，面板上必须能看出来，否则又成了"看起来已回流"的假象
               "applied": len([x for x in fb if _fb_applied(x)]),
               "not_applied": len([x for x in fb if not _fb_applied(x)])}
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
        # 落盘健康度：写失败不再静默，指标接口就能看出"内存有账、磁盘没账"
        "persist": _persist_status(),
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
_FEEDBACK_LABEL = {"确认异常": "hit", "误报": "false_positive",
                   "已处置": "hit", "无需处置": "correct_normal"}


def _fb_applied(fb: dict) -> bool:
    """这条反馈是否真的写进了台账。旧数据没有 applied 字段：老逻辑确实改过台账，按已生效算。"""
    return bool(fb.get("applied", True))


def _is_writable(r: dict) -> bool:
    """可写 = 既没被自动评估、也没被人工锁过。人工锁定的标签是重训真实标签，不许覆盖。"""
    return not r.get("locked") and not r.get("evaluated")


def _record_alert_key(r: dict) -> str:
    """记录所属的"每传感器恒定"预警号：新记录读 alert_base，旧记录直接就是 alert_id。"""
    return str(r.get("alert_base") or r.get("alert_id") or "")


def _rec_tick(r: dict) -> int:
    """记录 tick（仿真帧号）转 int；缺失 / 非法一律按 0 处理，别让旧数据把排序搞崩。"""
    try:
        return int(r.get("tick"))
    except (TypeError, ValueError):
        return 0


def _newest(seq: list):
    """取"最近一条"：先比 tick，再比写入先后（下标大者更新）。

    为什么不能直接取 seq[-1]：规则基线记录（alert_id 为空）夹在中间，被过滤后列表尾不一定是
    最新那条；按 tick + 写入顺序显式取最大，才和"该传感器最近一条记录"的语义一致。
    """
    best = None
    best_key = None
    for i, r in enumerate(seq):
        key = (_rec_tick(r), i)
        if best_key is None or key > best_key:
            best, best_key = r, key
    return best


def _match_alert(r: dict, alert_id: str) -> bool:
    """台账记录的预警号是否匹配调用方给的编号（唯一号 / alert_base / 唯一号前缀都认）。

    兼容旧口径：面板上那个"每传感器恒定"的编号没有 tick 后缀，用前缀反查唯一编号。
    """
    if not alert_id:
        return False
    if r.get("alert_id") == alert_id or r.get("alert_base") == alert_id:
        return True
    return str(r.get("alert_id") or "").startswith(alert_id + "-")


def _failed_feedback(alert_id: str, sensor_id: str, outcome: str, note: str = "",
                     user: str = "", extra: dict = None, reason: str = "") -> dict:
    """反馈没生效时：留档 + 明确返回失败（绝不静默返回 success=True）。

    为什么失败也要留档：人工确实点过，面板要能分清"点过但没生效"和"没点过"。
    这里写盘失败只记日志，不能盖掉原始的失败原因；真正要保证"不静默"的是台账落盘。
    """
    fb = {
        "id": f"FB-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}",
        "time": _now(),
        "alert_id": alert_id or "",
        "sensor_id": sensor_id or "",
        "outcome": outcome,
        "note": note or "",
        "by": user or "",
        "applied": False,          # 未写进台账：不参与重训导出
        "reason": reason,
    }
    if extra:
        fb["extra"] = extra
    try:
        items = load_feedback()
        items.append(fb)
        save_feedback(items)
    except Exception as e:
        logger.error("被拒反馈留档失败：%s", e)
    logger.warning("预测反馈未生效：alert=%s sensor=%s outcome=%s 原因=%s",
                   alert_id, sensor_id, outcome, reason)
    return {"success": False, "feedback": fb, "ledger_updated": 0,
            "reason": reason, "error": reason, "message": f"反馈未生效：{reason}"}


def add_feedback(alert_id: str, sensor_id: str, outcome: str, note: str = "",
                 user: str = "", extra: dict = None) -> dict:
    """outcome: 确认异常 / 误报 / 已处置 / 无需处置

    只允许改写「该传感器最近一条尚未评估（未 locked）」的预警记录。
    旧实现遍历整本台账去匹配 alert_id，而 alert_id 是每传感器恒定值、又不检查 evaluated，
    于是点一次反馈就把该传感器历史上所有记录（连已被自动评估成 hit/miss/false_positive 的）
    批量刷成同一个标签，前向评估被不可逆污染。现在的规则：

      1) 目标只能是该传感器最新一条未评估的预警记录，写完立即 locked；
      2) 已评估 / 已锁定的记录一概不改写，反馈也不允许回头改更早的批次；
      3) 找不到可写记录、或传进来的编号对不上最新那条 → success=False 明确失败，不静默成功。
    """
    alert_id = (alert_id or "").strip()
    sensor_id = (sensor_id or "").strip()
    recs = load_ledger()

    matched = [r for r in recs if _match_alert(r, alert_id)] if alert_id else []
    if not sensor_id and matched:
        sensor_id = str(_newest(matched).get("sensor_id") or "")   # 调用方只给预警号时反查传感器
    if not sensor_id:
        return _failed_feedback(alert_id, sensor_id, outcome, note, user, extra,
                                "缺少 sensor_id，且预警号未匹配到任何台账记录")

    # 候选必须是"预警记录"：模型记录才有预警号，规则基线 alert_id 为空，不参与人工反馈
    cand = [r for r in recs
            if str(r.get("sensor_id") or "") == sensor_id
            and _record_alert_key(r)
            and _is_writable(r)]
    if not cand:
        total = len([r for r in recs if str(r.get("sensor_id") or "") == sensor_id])
        return _failed_feedback(alert_id, sensor_id, outcome, note, user, extra,
                                f"该传感器没有尚未评估的台账记录（共 {total} 条，均已评估并加锁），"
                                f"历史标签不可追溯改写")
    target = _newest(cand)    # 该传感器最近一条尚未评估（未 locked）的记录

    # 编号必须指向最新那条：避免拿着台账里某条旧记录的编号，把标签写到另一条记录上
    if alert_id and not _match_alert(target, alert_id):
        return _failed_feedback(alert_id, sensor_id, outcome, note, user, extra,
                                f"预警 {alert_id} 不是该传感器最新的待标注记录"
                                f"（最新为 {target.get('alert_id')}），为避免追溯改写历史台账已拒绝")

    # 反馈只能沿 tick 向前推进：连点 / 仿真暂停后再点，不能让标签落到更早的批次上
    tgt_tick = _rec_tick(target)
    labeled_ticks = [fb.get("labeled_tick") for fb in load_feedback()
                     if _fb_applied(fb)
                     and str(fb.get("sensor_id") or "") == sensor_id
                     and isinstance(fb.get("labeled_tick"), int)
                     and not isinstance(fb.get("labeled_tick"), bool)]
    if labeled_ticks and tgt_tick <= max(labeled_ticks):
        return _failed_feedback(alert_id, sensor_id, outcome, note, user, extra,
                                f"tick {tgt_tick} 不晚于已反馈批次 tick {max(labeled_ticks)}，"
                                f"同一批次不重复标注，反馈只能向前推进")

    label = _FEEDBACK_LABEL.get(outcome)
    fb = {
        "id": f"FB-{int(time.time() * 1000)}-{uuid.uuid4().hex[:6]}",   # 同毫秒连点也不重号
        "time": _now(),
        "alert_id": alert_id or "",
        # 人工结论到底落在哪条台账记录上：可审计（台账编号唯一，能一条条对上）
        "ledger_alert_id": target.get("alert_id") or "",
        "record_id": target.get("id") or "",
        "sensor_id": sensor_id,
        "labeled_tick": tgt_tick,
        "outcome": outcome,
        "label": label,
        "note": note or "",
        "by": user or "",
        "applied": True,
    }
    if extra:
        fb["extra"] = extra

    # 先落台账、再落反馈：台账是唯一事实源，只要它锁上了，人工结论就不会被下一次反馈覆盖；
    # 反过来先写反馈文件的话，一旦台账落盘失败，就会出现"标签说生效、记录其实没锁"的裂口。
    target["feedback"] = {"outcome": outcome, "note": note, "by": user,
                          "time": fb["time"], "alert_id": fb["ledger_alert_id"]}
    target["locked"] = True          # 加锁：人工标签是重训的真实标签，后续反馈不得改写
    target["feedback_label"] = True
    if label:
        target["evaluated"] = True
        target["outcome"] = label
        target["actual_anomaly"] = label in ("hit",)
        target["evaluated_at"] = target.get("evaluated_at") or fb["time"]
    _flush_ledger()                  # 失败抛 LedgerPersistError（不再静默 pass），并留痕

    items = load_feedback()
    items.append(fb)
    save_feedback(items)
    logger.info("预测反馈已回流：%s → %s（台账记录 %s，tick %s）",
                fb["ledger_alert_id"], outcome, fb["record_id"], tgt_tick)
    return {"success": True, "feedback": fb, "ledger_updated": 1,
            "locked_record": fb["record_id"], "locked_alert_id": fb["ledger_alert_id"]}


def feedback_training_rows(include_unapplied: bool = False) -> list:
    """导出可直接喂给重训的标签行（sensor_id + 结论 + 时间）。

    默认只导出真正写进台账的反馈：没生效的（applied=False）在台账里没有对应记录，
    混进去等于拿没有证据的标签去训练。旧数据没有 applied 字段，按"已生效"处理，导出行为不变。
    """
    rows = []
    for fb in load_feedback():
        if not include_unapplied and not _fb_applied(fb):
            continue
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
    items = retrain_history()          # 损坏时在这里就抛出，不会拿空历史把文件覆盖掉
    items.append({"time": _now(), "status": status, **detail})
    _save(RETRAIN_FILE, items[-50:])
    return items[-1]


def retrain_history() -> list:
    """读重训历史。文件不存在 / 空 → []；有内容但不是数组 = 损坏，抛错而不是静默空历史。"""
    data = _load(RETRAIN_FILE, [])
    if not data:
        return []
    if not isinstance(data, list):
        raise LedgerCorruptError(f"重训历史文件结构异常 {RETRAIN_FILE}: 顶层不是数组")
    return data


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
