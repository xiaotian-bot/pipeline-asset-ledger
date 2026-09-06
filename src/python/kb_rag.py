#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG 知识库 - 向量化检索
========================
零新增依赖：sklearn TfidfVectorizer（字符级 n-gram，中文无需分词）+ 余弦相似度，
索引持久化到 models/kb_index.joblib，语料变更自动重建。

语料三个来源：
  1) KB_SEED            内置规范 / 运维手册 / 应急预案条目
  2) data/kb.json       用户自定义条目（可在界面增量维护）
  3) data/workorders.json  已闭环工单 → 沉淀为处置经验条目

注意：KB_SEED 是演示语料，条目摘要为便于检索而改写；正式环境应替换为
已获授权的规范原文切片，并核对标准号与版本号。
"""
import hashlib
import json
import os
import re
import time

import numpy as np

try:
    import joblib
except ImportError:
    joblib = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    HAVE_SKLEARN = True
except ImportError:
    HAVE_SKLEARN = False


HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
KB_FILE = os.path.join(DATA_DIR, "kb.json")
WORKORDER_FILE = os.path.join(DATA_DIR, "workorders.json")
INDEX_FILE = os.path.join(MODEL_DIR, "kb_index.joblib")


# ==============================================================================
# 语料：规范 / 手册 / 预案
# ==============================================================================
KB_SEED = [
    {
        "title": "球墨铸铁管检测周期与项目", "category": "检测规范", "kind": "standard",
        "source": "CJJ 140-2010《城镇供水管网运行、维护及安全技术规程》",
        "keywords": ["球墨铸铁", "检测周期", "内检测", "壁厚", "防腐层", "接口密封"],
        "content": (
            "球墨铸铁管常规内检测周期建议 3 年一次；处于腐蚀高风险区段、超龄运行（敷设年限超过 20 年）"
            "或曾发生爆管的管段应加密至 1 年一次。检测项目包括管壁剩余壁厚、防腐层完整性（水泥砂浆内衬与"
            "外涂层）、承插接口密封性、支墩与镇墩稳固性。检测前需完成停水降压、管道冲洗与通风检测，"
            "检测结果应录入资产台账并同步更新风险评分。"
        ),
    },
    {
        "title": "PE 管（聚乙烯）维护与连接要点", "category": "维护规范", "kind": "standard",
        "source": "CJJ 63-2018《聚乙烯燃气管道工程技术标准》",
        "keywords": ["PE管", "聚乙烯", "热熔", "电熔", "连接", "埋深", "第三方破坏"],
        "content": (
            "PE 管宜采用热熔对接或电熔连接，连接前必须清洁管端、校验加热板温度与吸热/冷却时间，"
            "并留存焊口编号与操作人记录。维护重点为接口渗漏巡查与第三方开挖破坏防控；"
            "埋深不足 0.8 米或穿越车行道的区段应增设保护套管与警示带。PE 管严禁明火烘烤调直，"
            "冬季施工环境温度低于 5℃ 时应延长冷却时间。"
        ),
    },
    {
        "title": "供水管网压力异常与爆管处置流程", "category": "应急处置", "kind": "playbook",
        "source": "《城市供水管网应急抢修预案》",
        "keywords": ["压力异常", "超压", "爆管", "降压", "关阀", "应急处置"],
        "content": (
            "监测到压力超标或压力骤降时：第一步立即上报值班领导并启动应急响应；第二步联系上游泵站降压，"
            "关闭相关分段阀门隔离疑似区段，关阀顺序应遵循由近及远、先干管后支管；第三步组织现场核查，"
            "结合分区计量（DMA）夜间最小流量与地面渗水、塌陷迹象判定爆管位置；第四步必要时启动应急预案，"
            "通知下游用户停水范围与预计恢复时间，同步调度应急供水车。抢修完成后需冲洗消毒并复检水质合格方可复水。"
        ),
    },
    {
        "title": "管网泄漏定位方法选型", "category": "检测技术", "kind": "manual",
        "source": "CJJ 92-2013《城镇供水管网漏损控制及评定标准》",
        "keywords": ["泄漏", "漏点", "定位", "音听法", "相关仪", "气体示踪", "DMA", "夜间最小流量"],
        "content": (
            "常用检漏方法与适用场景：音听法（阀栓听音、路面听音）适用于初步普查，成本低但依赖经验；"
            "声学相关仪适用于长距离干管精确定位，需两端可接触；噪声记录仪适合夜间大规模布点普查；"
            "气体示踪法（氢气/氦气）适用于非金属管与低压力管段；分区计量 DMA 夜间最小流量分析用于"
            "锁定漏损小区间。实际定位应结合管线竣工图、压力分布与地面异常综合判断，先分区后定点。"
        ),
    },
    {
        "title": "埋地钢质管道腐蚀风险评估与阴极保护", "category": "风险评估", "kind": "standard",
        "source": "GB/T 21448-2017《埋地钢质管道阴极保护技术规范》",
        "keywords": ["腐蚀", "壁厚", "阴极保护", "土壤电阻率", "杂散电流", "防腐层破损", "电位"],
        "content": (
            "腐蚀风险与土壤电阻率、含水量、氯离子与硫酸盐含量、杂散电流干扰、防腐层破损程度相关。"
            "高风险管段应开展壁厚超声检测与阴极保护有效性评估：断电电位应达到 -0.85V（相对铜/硫酸铜参比电极）"
            "或更负；存在硫酸盐还原菌时须达到 -0.95V。防腐层破损点应使用 DCVG 或 ACVG 定位并分级修复。"
            "年腐蚀速率超过 0.1mm/a 时应缩短检测周期并考虑换管。"
        ),
    },
    {
        "title": "燃气浓度超限报警处置", "category": "应急处置", "kind": "playbook",
        "source": "CJJ 51-2016《城镇燃气设施运行、维护和抢修安全技术规程》",
        "keywords": ["燃气", "浓度超限", "报警", "泄漏", "禁火", "警戒", "抢修"],
        "content": (
            "燃气浓度超限报警后：立即划定警戒区（一般不小于泄漏点周围 20 米）并疏散无关人员；"
            "警戒区内严禁一切火源、电器开关与手机使用，处置人员必须穿防静电工作服、使用防爆工具与可燃气体检测仪；"
            "由专业人员切断气源（关闭上下游阀门），采用防爆风机强制通风稀释；浓度降至爆炸下限 20% 以下方可进入作业。"
            "同步向燃气集团调度中心与应急管理部门报告，重大泄漏启动一级响应。抢修完成后须保压试验合格再恢复供气。"
        ),
    },
    {
        "title": "排水管道检测与结构缺陷评估", "category": "检测技术", "kind": "standard",
        "source": "CJJ 181-2012《城镇排水管道检测与评估技术规程》",
        "keywords": ["排水管道", "CCTV", "声呐", "结构性缺陷", "功能性缺陷", "QV", "评估"],
        "content": (
            "排水管道检测优先采用 CCTV 电视检测，管径较大且满水时辅以声呐检测，检查井内可用 QV 潜望镜快速普查。"
            "缺陷分结构性缺陷（破裂、变形、腐蚀、错口、脱节、渗漏、起伏、异物穿入）与功能性缺陷"
            "（沉积、结垢、障碍物、树根、浮渣）。评估采用结构性缺陷密度与修复指数计算，"
            "等级分为 1-4 级，3 级以上应安排修复。检测前应完成封堵、降水与通风，并做有限空间作业审批。"
        ),
    },
    {
        "title": "管网资产台账与一物一档要求", "category": "台账管理", "kind": "manual",
        "source": "《城镇排水管网资产管理办法》",
        "keywords": ["台账", "资产", "档案", "二维码", "一物一档", "权属", "数字化"],
        "content": (
            "每段管网资产应建立数字化档案（一物一档），字段至少包含：资产编号、管线类型、管径、材质、"
            "长度、敷设年代、埋深、权属单位、运维单位、所在镇街、经纬度起止点、风险评分、历次检测记录、"
            "维修与更换记录。可通过资产二维码扫码查看档案与生命周期时间线。台账变更（新增/修改/删除/审核）"
            "应留痕并可追溯，重要变更同步存证。资产盘点差异需生成盘点差异报告并限期核销。"
        ),
    },
    {
        "title": "季度巡检作业内容", "category": "巡检作业", "kind": "manual",
        "source": "《管网巡检作业指导书》",
        "keywords": ["巡检", "季度", "阀门", "井室", "盖板", "标识桩", "第三方施工", "巡线"],
        "content": (
            "季度巡检覆盖：阀门启闭灵活性与密封性（抽样启闭并记录扭矩）、井室结构与盖板完好、"
            "沿线第三方施工占压与开挖、标识桩与警示牌完好、明漏与地面塌陷渗水迹象、跨越与穿堤段稳固性。"
            "巡检应使用移动端按路线打卡并上传照片，发现问题当场登记工单并跟踪闭环；"
            "紧急缺陷（如井盖缺失、明漏）需 2 小时内上报并设临时围挡。"
        ),
    },
    {
        "title": "供暖管网水力失衡与温度异常排查", "category": "运行调节", "kind": "manual",
        "source": "CJJ 34-2010《城镇供热管网运行维护安全技术规程》",
        "keywords": ["供暖", "供热", "水力失衡", "供回水温差", "循环流量", "失水", "调节"],
        "content": (
            "供回水温差持续偏大提示循环流量不足，应检查循环泵运行台数与变频设定、除污器堵塞情况；"
            "温差持续偏小提示流量偏大或短路，需核查调节阀开度与旁通状态。水力失衡表现为近端过热远端不热，"
            "应通过平衡阀调节或加装自力式流量控制阀解决。系统失水率超过规范限值时排查泄漏点并检查补水装置，"
            "严禁大量补水导致水质恶化与腐蚀加剧。供热季前应完成水压试验与冲洗。"
        ),
    },
    {
        "title": "危废输送管道安全运行要求", "category": "安全管理", "kind": "standard",
        "source": "GB 18597《危险废物贮存污染控制标准》及配套输送要求",
        "keywords": ["危废", "危险废物", "输送", "防渗漏", "双套管", "联单", "应急池"],
        "content": (
            "危废输送管道应采用耐腐蚀材质并设双层套管或检漏通道，输送泵与阀门选用无泄漏型式（磁力泵、屏蔽泵）。"
            "沿线设置泄漏检测点与紧急切断阀，末端接入应急收集池。转运须执行危险废物转移联单制度，"
            "记录产生单位、类别代码、数量、承运与接收单位。作业人员须配备防护服、防毒面具与洗眼器；"
            "发生泄漏立即启动应急预案，围堵导流至应急池，禁止直接冲洗进入市政管网。"
        ),
    },
    {
        "title": "污水管网硫化氢与有限空间作业安全", "category": "安全管理", "kind": "playbook",
        "source": "CJJ 68-2016《城镇排水管道维护安全技术规程》",
        "keywords": ["污水", "硫化氢", "有限空间", "作业审批", "通风", "气体检测", "中毒"],
        "content": (
            "污水管网井室内易积聚硫化氢、甲烷与二氧化碳，属有限空间作业。作业必须执行"
            "「先通风、再检测、后作业」：机械通风不少于 30 分钟，检测氧含量（19.5%-23.5%）、"
            "硫化氢（≤10mg/m³）、一氧化碳与可燃气体浓度合格后方可进入；作业人员须系全身式安全带、"
            "佩戴正压式空气呼吸器（浓度超标时），井上必须设专人监护并保持通讯。作业前办理有限空间作业审批票，"
            "严禁未检测直接下井、严禁盲目施救。"
        ),
    },
    {
        "title": "地下管线探测与竣工测量", "category": "检测技术", "kind": "standard",
        "source": "CJJ 61-2017《城市地下管线探测技术规程》",
        "keywords": ["管线探测", "物探", "电磁感应", "探地雷达", "竣工测量", "坐标", "埋深"],
        "content": (
            "金属管线优先采用电磁感应法探测，非金属管线（PE、混凝土）采用探地雷达、示综法或声学法；"
            "复杂路段需多种物探方法组合验证并开挖样洞校核。探测成果应包含平面坐标、埋深、管径、材质、"
            "权属与走向，平面位置中误差不超过 ±0.15m、埋深中误差不超过 ±0.25m。"
            "新建管线竣工测量须在覆土前完成，成果纳入城市地下管线综合管理系统。"
        ),
    },
    {
        "title": "管道非开挖修复工艺选型", "category": "维修养护", "kind": "manual",
        "source": "CJJ/T 210-2014《城镇排水管道非开挖修复更新工程技术规程》",
        "keywords": ["非开挖", "修复", "CIPP", "紫外光固化", "点状修复", "螺旋缠绕", "管片内衬"],
        "content": (
            "整体修复工艺：CIPP 紫外光固化内衬适用于 DN200-DN1600 圆形管道，强度高、工期短；"
            "热水固化 CIPP 适用于大管径与异形断面；螺旋缠绕适用于带水作业与长距离；"
            "管片内衬适用于大断面方沟。局部修复：不锈钢双胀环、点状 CIPP、树脂灌浆适用于单个接口或破损点。"
            "修复前须完成清淤、CCTV 复检与缺陷定位，修复后须做 CCTV 验收与密闭性试验。"
        ),
    },
    {
        "title": "预测性维护：风险分级与处置时限", "category": "预测运维", "kind": "manual",
        "source": "《管网预测性维护作业指引（内部）》",
        "keywords": ["预测性维护", "风险分级", "RUL", "剩余寿命", "处置时限", "提前预警", "命中率"],
        "content": (
            "风险分 0-40 为正常，纳入常规巡检；40-60 为预警，安排 30 天内专项检测并加密监测频次；"
            "60-80 为高风险，7 天内完成现场核查与检测评估，制定修复方案；80 以上为危急，"
            "24 小时内到场处置并同步推送责任人。RUL（剩余寿命步数）低于预警窗口时应提前触发工单。"
            "预测结果须与规则阈值基线对比评估命中率、提前预警时长与误报率，误报率过高时应回调阈值或重训模型；"
            "处置反馈（确认异常/误报）应回流至训练样本，实现批式重训。"
        ),
    },
    {
        "title": "传感器监测指标阈值与告警判据", "category": "监测标准", "kind": "manual",
        "source": "《管网在线监测点位配置与阈值设定指引（内部）》",
        "keywords": ["传感器", "压力", "流量", "温度", "阈值", "告警", "误报", "漂移"],
        "content": (
            "供水管网压力正常区间 0.20-0.60MPa，超过 0.75MPa 判定超压告警，低于 0.10MPa 判定失压（疑似爆管）；"
            "温度 -5℃ 至 45℃ 为正常，冬季低于 0℃ 触发防冻告警；流量突变超过基线 30% 触发异常。"
            "燃气管网以甲烷浓度为主判据，达到爆炸下限 20% 触发一级告警。阈值应结合季节与工况动态调整，"
            "连续 3 帧越限才判定告警以避免抖动误报；传感器数据长期恒定或方差为 0 提示探头漂移或断线，应触发设备自检工单。"
        ),
    },
    {
        "title": "分区计量 DMA 建设与漏损评定", "category": "漏损控制", "kind": "standard",
        "source": "CJJ 92-2013《城镇供水管网漏损控制及评定标准》",
        "keywords": ["DMA", "分区计量", "漏损率", "产销差", "夜间最小流量", "水平衡"],
        "content": (
            "DMA 分区应遵循封闭性、可计量性与规模适度原则，单区供水户数一般 500-3000 户。"
            "每个分区须安装计量总表并具备夜间最小流量分析能力（取凌晨 2:00-4:00 稳定段均值）。"
            "漏损率 = (供水总量 - 注册用水量) / 供水总量 × 100%，评定应扣除消防用水与免费用水。"
            "夜间最小流量突增是新增漏损的敏感指标，应设阈值自动预警；"
            "分区漏损率超标时按「先水平衡分析、后物理检漏」顺序排查。"
        ),
    },
    {
        "title": "工单闭环与处置反馈要求", "category": "运维管理", "kind": "manual",
        "source": "《运维工单管理办法（内部）》",
        "keywords": ["工单", "闭环", "指派", "处置反馈", "验收", "时限", "回流"],
        "content": (
            "工单状态流转：待指派 → 已指派 → 处理中 → 待验收 → 已闭环。紧急工单 2 小时内响应、24 小时内处置；"
            "高优先级 4 小时响应、3 天完成；中优先级 1 天响应、7 天完成。"
            "处置反馈须包含现场照片、缺陷确认结论（确认异常/误报/已处置）、采取措施与耗材；"
            "验收由运维主管或属地负责人完成，验收通过后工单闭环。"
            "预测类工单的反馈结论必须回填到预测台账，作为模型重训的真实标签，形成「预测→预警→推送→建单→处置→回流」闭环。"
        ),
    },
]


# ==============================================================================
# 语料组装
# ==============================================================================
def _doc_id(kind, key):
    return f"{kind}-{hashlib.md5(str(key).encode('utf-8')).hexdigest()[:10]}"


def _load_json(path, default):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                v = json.load(f)
            return v if v else default
    except Exception:
        pass
    return default


def _seed_corpus():
    docs = []
    for i, e in enumerate(KB_SEED):
        docs.append({
            "doc_id": e.get("id") or _doc_id("seed", e["title"]),
            "kind": e.get("kind", "standard"),
            "title": e["title"],
            "category": e.get("category", ""),
            "source": e.get("source", ""),
            "content": e["content"],
            "keywords": e.get("keywords", []),
            "builtin": True,
        })
    return docs


def _user_corpus():
    """data/kb.json 中的自定义条目（兼容 keywords / tags 两种字段）。"""
    docs = []
    for e in _load_json(KB_FILE, []) or []:
        if not isinstance(e, dict):
            continue
        title = (e.get("title") or "").strip()
        content = (e.get("content") or "").strip()
        if not title and not content:
            continue
        docs.append({
            "doc_id": e.get("id") or _doc_id("user", title or content[:20]),
            "kind": e.get("kind", "custom"),
            "title": title or content[:20],
            "category": e.get("category", "") if isinstance(e.get("category"), str) else "",
            "source": e.get("source", "") or "用户自定义",
            "content": content or title,
            "keywords": e.get("keywords") or e.get("tags") or [],
            "builtin": False,
        })
    return docs


def _workorder_corpus(path=WORKORDER_FILE, limit=200):
    """已闭环工单 → 处置经验条目，让 Agent 能引用本项目的历史处置记录。"""
    docs = []
    items = _load_json(path, []) or []
    closed = [w for w in items if isinstance(w, dict) and w.get("status") == "已闭环"]
    if not closed:
        closed = [w for w in items if isinstance(w, dict) and w.get("title")]
    for w in closed[-limit:]:
        title = (w.get("title") or "").strip()
        if not title:
            continue
        hist = w.get("history") or []
        steps = "；".join(
            f"{h.get('time', '')} {h.get('action', '')}" + (f"（{h['note']}）" if h.get("note") else "")
            for h in hist if isinstance(h, dict)
        )
        content = " ".join(x for x in [
            f"历史工单：{title}。",
            f"所属区域 {w.get('region') or '—'}，优先级 {w.get('priority') or '—'}，状态 {w.get('status') or '—'}。",
            w.get("description") or "",
            f"处置流转：{steps}。" if steps else "",
        ] if x).strip()
        docs.append({
            "doc_id": w.get("workorder_id") or _doc_id("wo", title),
            "kind": "workorder",
            "title": f"历史工单 · {title}",
            "category": "处置经验",
            "source": f"运维工单 {w.get('workorder_id', '')}".strip(),
            "content": content,
            "keywords": [k for k in [w.get("region"), w.get("priority"), w.get("status")] if k],
            "builtin": False,
        })
    return docs


def _norm_title(title):
    return re.sub(r"[\s·\-—_()（）\[\]【】:：,，.。/、]+", "", (title or "")).lower()


def build_corpus(include_workorders=True):
    docs = _seed_corpus() + _user_corpus()
    seen, seen_titles = set(), set()
    out = []
    for d in docs:
        key = _norm_title(d["title"])
        if d["doc_id"] in seen or (key and key in seen_titles):
            continue
        seen.add(d["doc_id"])
        if key:
            seen_titles.add(key)
        out.append(d)
    if include_workorders:
        for d in _workorder_corpus():
            key = _norm_title(d["title"])
            if d["doc_id"] in seen or (key and key in seen_titles):
                continue
            seen.add(d["doc_id"])
            if key:
                seen_titles.add(key)
            out.append(d)
    return out


def _doc_text(d):
    """参与向量化的文本：关键词加权重复 + 标题 + 正文。"""
    kw = " ".join(d.get("keywords") or [])
    return f"{d['title']} {d['title']} {kw} {kw} {kw} {d['content']}"


# ==============================================================================
# 索引
# ==============================================================================
class KbIndex:
    def __init__(self, vectorizer, matrix, docs, built_at, corpus_hash, method):
        self.vectorizer = vectorizer
        self.matrix = matrix
        self.docs = docs
        self.built_at = built_at
        self.corpus_hash = corpus_hash
        self.method = method

    def size(self):
        return len(self.docs)


def corpus_hash(docs):
    payload = json.dumps(
        [[d["doc_id"], d["title"], d["category"], d["source"], d["content"], d.get("keywords")] for d in docs],
        ensure_ascii=False, sort_keys=True)
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def _corpus_fingerprint():
    """语料指纹：内置条目 + kb.json / workorders.json 的 mtime 与大小。"""
    parts = [str(len(KB_SEED))]
    for p in (KB_FILE, WORKORDER_FILE):
        try:
            st = os.stat(p)
            parts.append(f"{os.path.basename(p)}:{int(st.st_mtime)}:{st.st_size}")
        except Exception:
            parts.append(f"{os.path.basename(p)}:missing")
    return "|".join(parts)


_last_fingerprint = ""


def build_index(docs=None, persist=True):
    """构建 TF-IDF 字符 n-gram 向量索引。sklearn 缺失时降级为纯 Python 词袋。"""
    global _last_fingerprint
    docs = docs if docs is not None else build_corpus()
    _last_fingerprint = _corpus_fingerprint()
    if not docs or not HAVE_SKLEARN:
        idx = KbIndex(None, None, docs, time.strftime("%Y-%m-%d %H:%M:%S"), corpus_hash(docs),
                      "fallback_keyword" if docs else "empty")
        return idx

    texts = [_doc_text(d) for d in docs]
    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 3), min_df=1,
                                 sublinear_tf=True, norm="l2")
    matrix = vectorizer.fit_transform(texts)
    idx = KbIndex(vectorizer, matrix, docs, time.strftime("%Y-%m-%d %H:%M:%S"),
                  corpus_hash(docs), "tfidf_char_ngram")
    if persist and joblib is not None:
        try:
            os.makedirs(MODEL_DIR, exist_ok=True)
            joblib.dump({"vectorizer": vectorizer, "matrix": matrix, "docs": docs,
                         "built_at": idx.built_at, "corpus_hash": idx.corpus_hash,
                         "method": idx.method}, INDEX_FILE)
        except Exception:
            pass
    return idx


def _load_index():
    if joblib is None or not os.path.exists(INDEX_FILE):
        return None
    try:
        blob = joblib.load(INDEX_FILE)
        return KbIndex(blob["vectorizer"], blob["matrix"], blob["docs"], blob.get("built_at", ""),
                       blob.get("corpus_hash", ""), blob.get("method", "tfidf_char_ngram"))
    except Exception:
        return None


_index = None


def get_index(force=False):
    """获取索引；语料指纹变化时自动重建。"""
    global _index
    fp = _corpus_fingerprint()
    if _index is not None and not force and fp == _last_fingerprint:
        return _index
    if not force:
        cached = _load_index()
        if cached is not None:
            docs_now = build_corpus()
            if cached.corpus_hash == corpus_hash(docs_now):
                _index = cached
                return _index
    _index = build_index()
    return _index


# ==============================================================================
# 检索
# ==============================================================================
_SENT_SPLIT = re.compile(r"[。；;！!\n]")


def _sentences(text):
    return [s.strip() for s in _SENT_SPLIT.split(text or "") if s and s.strip()]


def _snippet(content, query, max_sent=2):
    """抽取与查询最相关的句子作为引用片段。"""
    sents = _sentences(content)
    if not sents:
        return (content or "")[:160]
    qchars = set(re.sub(r"\s", "", query or ""))
    if not qchars:
        return "。".join(sents[:max_sent])
    scored = sorted(sents, key=lambda s: len(qchars & set(s)) / max(len(s), 1), reverse=True)
    picked = scored[:max_sent]
    keep = [s for s in sents if s in picked]
    return "。".join(keep or picked) + "。"


def _fallback_search(docs, query, top_k, min_score):
    """sklearn 不可用时的字符重合度打分（保底，不让接口空转）。"""
    q = set(re.sub(r"\s", "", query or ""))
    if not q:
        return []
    scored = []
    for d in docs:
        hay = set(re.sub(r"\s", "", _doc_text(d)))
        s = len(q & hay) / max(len(q), 1)
        if s > 0:
            scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    out = []
    for s, d in scored[:top_k]:
        out.append({
            "doc_id": d["doc_id"], "title": d["title"], "category": d["category"],
            "source": d["source"], "kind": d["kind"], "content": d["content"],
            "snippet": _snippet(d["content"], query), "score": round(min(s, 1.0), 4),
            "retriever": "fallback_keyword",
        })
    return out


def retrieve(query, top_k=4, min_score=0.03):
    """向量检索知识库，返回带出处与引用片段的结果。"""
    query = (query or "").strip()
    if not query:
        return []
    idx = get_index()
    if idx is None or idx.vectorizer is None or idx.matrix is None or not idx.docs:
        return _fallback_search(build_corpus(), query, top_k, min_score)

    qv = idx.vectorizer.transform([query])
    try:
        sims = np.asarray((idx.matrix @ qv.T).todense()).ravel()
    except Exception:
        return _fallback_search(idx.docs, query, top_k, min_score)

    order = np.argsort(-sims)
    out = []
    for i in order:
        s = float(sims[i])
        if s < min_score or len(out) >= top_k:
            break
        d = idx.docs[int(i)]
        out.append({
            "doc_id": d["doc_id"], "title": d["title"], "category": d["category"],
            "source": d["source"], "kind": d["kind"], "content": d["content"],
            "snippet": _snippet(d["content"], query), "score": round(s, 4),
            "retriever": idx.method,
        })
    if not out:
        return _fallback_search(idx.docs, query, top_k, min_score)
    return out


def build_context(query, top_k=3, max_chars=1400):
    """把检索结果拼成注入大模型 prompt 的上下文块（含 [编号] 便于引用）。"""
    hits = retrieve(query, top_k=top_k)
    if not hits:
        return "", []
    lines, cites = [], []
    used = 0
    for n, h in enumerate(hits, 1):
        block = f"[{n}] {h['title']}（{h['source']}）\n{h['snippet']}"
        if used + len(block) > max_chars:
            break
        lines.append(block)
        cites.append({"n": n, "title": h["title"], "source": h["source"],
                      "score": h["score"], "doc_id": h["doc_id"], "category": h["category"],
                      "snippet": h.get("snippet") or ""})
        used += len(block)
    return "\n\n".join(lines), cites


def stats():
    idx = get_index()
    if idx is None:
        return {"available": False, "docs": 0}
    kinds, cats = {}, {}
    for d in idx.docs:
        kinds[d["kind"]] = kinds.get(d["kind"], 0) + 1
        c = d.get("category") or "未分类"
        cats[c] = cats.get(c, 0) + 1
    vocab = 0
    try:
        vocab = len(idx.vectorizer.vocabulary_) if idx.vectorizer is not None else 0
    except Exception:
        vocab = 0
    return {
        "available": bool(idx.docs),
        "docs": len(idx.docs),
        "method": idx.method,
        "vocabulary": vocab,
        "built_at": idx.built_at,
        "by_kind": kinds,
        "by_category": cats,
        "sklearn": HAVE_SKLEARN,
        "persisted": bool(joblib is not None and os.path.exists(INDEX_FILE)),
    }


def add_entries(entries):
    """向 data/kb.json 追加自定义条目并重建索引。"""
    cur = _load_json(KB_FILE, []) or []
    if isinstance(cur, dict):
        cur = []
    added = []
    for e in entries or []:
        if not isinstance(e, dict):
            continue
        title = (e.get("title") or "").strip()
        content = (e.get("content") or "").strip()
        if not title or not content:
            continue
        rec = {
            "id": _doc_id("user", title + str(time.time())),
            "title": title,
            "category": (e.get("category") or "自定义").strip(),
            "source": (e.get("source") or "用户自定义").strip(),
            "keywords": e.get("keywords") or e.get("tags") or [],
            "content": content,
            "kind": "custom",
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        cur.append(rec)
        added.append(rec)
    if added:
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(KB_FILE, "w", encoding="utf-8") as f:
                json.dump(cur, f, ensure_ascii=False, indent=2)
        except Exception:
            return []
        get_index(force=True)
    return added


def list_entries():
    return get_index().docs if get_index() else []
