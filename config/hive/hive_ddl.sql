-- ==============================================================================
-- 城市管网资产数字化台账 - 数仓建表语句
-- 分层：ODS（原始数据） → DWD（明细数据） → DWS（聚合数据） → ADS（应用数据）
-- 覆盖：供水、供暖、燃气、污水、危废 5类管网资产
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS ods;
CREATE DATABASE IF NOT EXISTS dwd;
CREATE DATABASE IF NOT EXISTS dws;
CREATE DATABASE IF NOT EXISTS ads;

-- ==============================================================================
-- ODS层：原始数据（JSON存储）
-- ==============================================================================
USE ods;

-- 管网资产原始数据
CREATE EXTERNAL TABLE IF NOT EXISTS ods.pipeline_asset_raw (
    raw_json STRING COMMENT '原始JSON记录字符串'
)
PARTITIONED BY (dt STRING COMMENT '日期分区 yyyy-MM-dd')
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/ods.db/pipeline_asset_raw';

-- 生命周期事件原始数据
CREATE EXTERNAL TABLE IF NOT EXISTS ods.lifecycle_event_raw (
    raw_json STRING COMMENT '原始JSON记录字符串'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/ods.db/lifecycle_event_raw';

-- 盘点记录原始数据
CREATE EXTERNAL TABLE IF NOT EXISTS ods.inventory_check_raw (
    raw_json STRING COMMENT '原始JSON记录字符串'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/ods.db/inventory_check_raw';

-- ==============================================================================
-- DWD层：明细数据
-- ==============================================================================
USE dwd;

-- 管网资产明细宽表
CREATE TABLE IF NOT EXISTS dwd.pipeline_asset_detail (
    asset_id STRING COMMENT '资产编号',
    pipeline_type_code STRING COMMENT '管网类型编码(WSP/HTP/GSP/SWP/HZP)',
    pipeline_type_name STRING COMMENT '管网类型名称',
    diameter STRING COMMENT '管径(DNxx)',
    material STRING COMMENT '管材',
    install_year INT COMMENT '安装年份',
    service_years INT COMMENT '已服役年限',
    design_life INT COMMENT '设计寿命(年)',
    remaining_life INT COMMENT '剩余寿命(年)',
    region_code STRING COMMENT '区域编码',
    region_name STRING COMMENT '区域名称',
    longitude DOUBLE COMMENT '经度',
    latitude DOUBLE COMMENT '纬度',
    segment_length_m DOUBLE COMMENT '管段长度(米)',
    burial_depth_m DOUBLE COMMENT '埋深(米)',
    pressure_level STRING COMMENT '压力等级',
    asset_status STRING COMMENT '资产状态(在用/停用/待检/报废)',
    original_value_yuan DOUBLE COMMENT '资产原值(元)',
    net_value_yuan DOUBLE COMMENT '资产净值(元)',
    depreciation_rate DOUBLE COMMENT '折旧率',
    risk_score INT COMMENT '风险评分(0-100)',
    ownership_unit STRING COMMENT '产权单位',
    oam_unit STRING COMMENT '运维单位',
    supervision_unit STRING COMMENT '监管单位',
    last_inspection_date STRING COMMENT '最近检测日期',
    qr_code STRING COMMENT '二维码编号'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 生命周期事件明细
CREATE TABLE IF NOT EXISTS dwd.lifecycle_event_detail (
    event_id STRING COMMENT '事件编号',
    asset_id STRING COMMENT '关联资产编号',
    pipeline_type_name STRING COMMENT '管网类型名称',
    event_type_code STRING COMMENT '事件类型编码',
    event_type_name STRING COMMENT '事件类型名称(采购/施工/运维/巡检/改造/报废)',
    event_date STRING COMMENT '事件日期',
    responsible_unit STRING COMMENT '责任主体',
    cost_yuan DOUBLE COMMENT '费用(元)',
    description STRING COMMENT '事件描述',
    operator STRING COMMENT '操作人'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 盘点明细
CREATE TABLE IF NOT EXISTS dwd.inventory_check_detail (
    check_id STRING COMMENT '盘点记录编号',
    batch_id STRING COMMENT '盘点批次号',
    asset_id STRING COMMENT '资产编号',
    pipeline_type_name STRING COMMENT '管网类型名称',
    region_name STRING COMMENT '区域名称',
    check_method STRING COMMENT '盘点方式(扫码/巡检/无人机/GIS比对)',
    check_date STRING COMMENT '盘点日期',
    diff_status STRING COMMENT '账实状态(一致/差异-缺失/差异-多余/差异-信息不符)',
    diff_description STRING COMMENT '差异描述',
    checker STRING COMMENT '盘点人',
    checker_unit STRING COMMENT '盘点单位'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- ==============================================================================
-- DWS层：聚合数据
-- ==============================================================================
USE dws;

-- 资产分类统计汇总表（按管网类型/管径/材质/年代/区域）
CREATE TABLE IF NOT EXISTS dws.asset_category_summary (
    pipeline_type_code STRING COMMENT '管网类型编码',
    pipeline_type_name STRING COMMENT '管网类型名称',
    diameter STRING COMMENT '管径',
    material STRING COMMENT '管材',
    decade STRING COMMENT '年代(如1990s)',
    region_name STRING COMMENT '区域名称',
    asset_count BIGINT COMMENT '资产数量(管段数)',
    total_length_m DOUBLE COMMENT '总长度(米)',
    total_original_value DOUBLE COMMENT '资产原值合计(元)',
    total_net_value DOUBLE COMMENT '资产净值合计(元)',
    avg_risk_score DOUBLE COMMENT '平均风险评分',
    in_service_count BIGINT COMMENT '在用数量',
    retired_count BIGINT COMMENT '报废数量'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 资产生命周期统计
CREATE TABLE IF NOT EXISTS dws.asset_lifecycle_summary (
    pipeline_type_name STRING COMMENT '管网类型名称',
    event_type_code STRING COMMENT '事件类型编码',
    event_type_name STRING COMMENT '事件类型名称',
    event_count BIGINT COMMENT '事件数量',
    total_cost DOUBLE COMMENT '总费用(元)',
    avg_cost DOUBLE COMMENT '平均费用(元)',
    max_cost DOUBLE COMMENT '最大费用(元)',
    min_cost DOUBLE COMMENT '最小费用(元)'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 资产风险评估
CREATE TABLE IF NOT EXISTS dws.asset_risk_assessment (
    asset_id STRING COMMENT '资产编号',
    pipeline_type_code STRING COMMENT '管网类型编码',
    pipeline_type_name STRING COMMENT '管网类型名称',
    region_name STRING COMMENT '区域名称',
    service_years INT COMMENT '已服役年限',
    design_life INT COMMENT '设计寿命',
    aging_index DOUBLE COMMENT '老化指数(服役/设计寿命)',
    risk_score INT COMMENT '综合风险评分',
    risk_level STRING COMMENT '风险等级(低/中/高/极高)',
    remaining_life INT COMMENT '剩余寿命(年)',
    material STRING COMMENT '管材',
    ownership_unit STRING COMMENT '产权单位',
    oam_unit STRING COMMENT '运维单位'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 盘点核对汇总
CREATE TABLE IF NOT EXISTS dws.inventory_reconciliation (
    batch_id STRING COMMENT '盘点批次号',
    pipeline_type_name STRING COMMENT '管网类型名称',
    region_name STRING COMMENT '区域名称',
    total_checked BIGINT COMMENT '盘点总数',
    match_count BIGINT COMMENT '一致数量',
    missing_count BIGINT COMMENT '缺失数量',
    extra_count BIGINT COMMENT '多余数量',
    mismatch_count BIGINT COMMENT '信息不符数量',
    diff_rate DECIMAL(10,4) COMMENT '差异率'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- ==============================================================================
-- ADS层：应用数据
-- ==============================================================================
USE ads;

-- 资产全景总览
CREATE TABLE IF NOT EXISTS ads.asset_overview (
    total_assets BIGINT COMMENT '资产总数(管段数)',
    total_length_km DOUBLE COMMENT '总里程(公里)',
    total_original_value DOUBLE COMMENT '资产总值-原值(元)',
    total_net_value DOUBLE COMMENT '资产总值-净值(元)',
    in_service_count BIGINT COMMENT '在用资产数',
    retired_count BIGINT COMMENT '报废资产数',
    pending_inspection_count BIGINT COMMENT '待检资产数',
    suspended_count BIGINT COMMENT '停用资产数',
    avg_risk_score DOUBLE COMMENT '平均风险评分',
    high_risk_count BIGINT COMMENT '高风险资产数',
    inventory_diff_rate DECIMAL(10,4) COMMENT '盘点差异率',
    water_supply_count BIGINT COMMENT '供水管网数量',
    heating_count BIGINT COMMENT '供暖管网数量',
    gas_count BIGINT COMMENT '燃气管网数量',
    sewage_count BIGINT COMMENT '污水管网数量',
    hazardous_count BIGINT COMMENT '危废管网数量'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 资产分布统计
CREATE TABLE IF NOT EXISTS ads.asset_distribution (
    dimension_type STRING COMMENT '统计维度(type/diameter/material/decade/region/ownership)',
    dimension_value STRING COMMENT '维度值',
    pipeline_type_name STRING COMMENT '管网类型名称',
    asset_count BIGINT COMMENT '资产数量',
    total_length_m DOUBLE COMMENT '总长度(米)',
    total_value DOUBLE COMMENT '资产总值(元)',
    avg_risk_score DOUBLE COMMENT '平均风险评分'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 生命周期时间线
CREATE TABLE IF NOT EXISTS ads.lifecycle_timeline (
    asset_id STRING COMMENT '资产编号',
    pipeline_type_name STRING COMMENT '管网类型名称',
    region_name STRING COMMENT '区域名称',
    event_count BIGINT COMMENT '事件总数',
    total_cost DOUBLE COMMENT '累计费用(元)',
    first_event_date STRING COMMENT '首次事件日期',
    last_event_date STRING COMMENT '最近事件日期',
    purchase_cost DOUBLE COMMENT '采购费用',
    construction_cost DOUBLE COMMENT '施工费用',
    maintenance_cost DOUBLE COMMENT '运维费用',
    renovation_cost DOUBLE COMMENT '改造费用'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 盘点差异报告
CREATE TABLE IF NOT EXISTS ads.inventory_diff_report (
    batch_id STRING COMMENT '盘点批次号',
    pipeline_type_name STRING COMMENT '管网类型名称',
    region_name STRING COMMENT '区域名称',
    check_method STRING COMMENT '盘点方式',
    total_checked BIGINT COMMENT '盘点总数',
    diff_count BIGINT COMMENT '差异数量',
    diff_rate DECIMAL(10,4) COMMENT '差异率',
    missing_count BIGINT COMMENT '缺失数',
    extra_count BIGINT COMMENT '多余数',
    mismatch_count BIGINT COMMENT '信息不符数'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 权属责任汇总
CREATE TABLE IF NOT EXISTS ads.ownership_summary (
    ownership_unit STRING COMMENT '产权单位',
    oam_unit STRING COMMENT '运维单位',
    supervision_unit STRING COMMENT '监管单位',
    pipeline_type_name STRING COMMENT '管网类型名称',
    region_name STRING COMMENT '区域名称',
    asset_count BIGINT COMMENT '资产数量',
    total_length_m DOUBLE COMMENT '总长度(米)',
    total_value DOUBLE COMMENT '资产总值(元)',
    avg_risk_score DOUBLE COMMENT '平均风险评分'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- ==============================================================================
-- 传感器实时数据（Kafka → Spark Streaming → Hive）
-- ==============================================================================

-- ODS层：传感器原始数据
CREATE EXTERNAL TABLE IF NOT EXISTS ods.sensor_raw (
    sensor_id STRING COMMENT '传感器编号',
    asset_id STRING COMMENT '关联资产编号',
    pipe_type STRING COMMENT '管网类型',
    metrics STRING COMMENT '指标JSON',
    status STRING COMMENT '状态 normal/warning/fault',
    alert_code INT COMMENT '告警代码',
    alert_desc STRING COMMENT '告警描述',
    timestamp STRING COMMENT '采集时间'
)
PARTITIONED BY (dt STRING COMMENT '日期分区 yyyy-MM-dd')
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/ods.db/sensor_raw';

-- DWD层：传感器明细（展开 metrics JSON 为独立字段）
CREATE TABLE IF NOT EXISTS dwd.sensor_detail (
    sensor_id STRING COMMENT '传感器编号',
    asset_id STRING COMMENT '关联资产编号',
    pipe_type STRING COMMENT '管网类型',
    status STRING COMMENT '状态',
    alert_code INT COMMENT '告警代码',
    alert_desc STRING COMMENT '告警描述',
    pressure DOUBLE COMMENT '压力(MPa)',
    flow DOUBLE COMMENT '流量(m³/h)',
    temperature DOUBLE COMMENT '温度(℃)',
    gas_concentration DOUBLE COMMENT '燃气浓度(ppm)',
    level DOUBLE COMMENT '液位(m)',
    vibration DOUBLE COMMENT '振动(mm/s)',
    timestamp STRING COMMENT '采集时间'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- DWS层：分钟级统计（Spark 滑动窗口聚合结果）
CREATE TABLE IF NOT EXISTS dws.sensor_minute_stats (
    window_start STRING COMMENT '窗口开始',
    window_end STRING COMMENT '窗口结束',
    pipe_type STRING COMMENT '管网类型',
    avg_pressure DOUBLE COMMENT '平均压力',
    avg_temperature DOUBLE COMMENT '平均温度',
    avg_flow DOUBLE COMMENT '平均流量',
    anomaly_count INT COMMENT '异常数量',
    sensor_count INT COMMENT '传感器总数',
    health_score DOUBLE COMMENT '健康度评分(0-100)'
)
PARTITIONED BY (dt STRING COMMENT '日期分区')
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- ADS层：大屏实时汇总（最新一帧快照）
CREATE TABLE IF NOT EXISTS ads.sensor_realtime (
    window_start STRING COMMENT '窗口开始',
    pipe_type STRING COMMENT '管网类型',
    avg_pressure DOUBLE COMMENT '平均压力',
    avg_temperature DOUBLE COMMENT '平均温度',
    avg_flow DOUBLE COMMENT '平均流量',
    anomaly_count INT COMMENT '异常数量',
    sensor_count INT COMMENT '传感器总数',
    health_score DOUBLE COMMENT '健康度评分'
)
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');
