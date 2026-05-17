# 中国近代史人物关系图谱

这是一个基于 Neo4j、Streamlit、Pyvis 和 DashScope Qwen 的 GraphRAG 示例项目，用于导入 `data-json.json` 中的近代史人物数据，展示 2 度关系图谱，并基于图谱事实进行问答。

## 目录结构

```text
KGagent/
  app.py                 # Streamlit 主应用
  etl_import.py          # JSON 入库脚本
  requirements.txt       # Python 依赖
  .env.example           # 环境变量模板
  data-json.json         # 原始人物数据
  src/
    config.py            # 配置读取
    etl.py               # JSON 解析和实体关系转换
    graph_store.py       # Neo4j 数据访问层
    graphrag.py          # DashScope Qwen GraphRAG 问答
    visualization.py     # Pyvis 可视化
```

## 使用方式

1. 安装依赖：

```bash
pip install -r requirements.txt
```

2. 复制环境变量模板，并填入 Neo4j 密码和 DashScope API Key：

```bash
copy .env.example .env
```

可选地修改 `DASHSCOPE_BASE_URL`，默认是百炼的 OpenAI 兼容地址。

3. 确保 Neo4j 已启动，然后导入数据：

```bash
python etl_import.py
```

4. 启动页面：

```bash
streamlit run app.py
```

## 注意

- ETL 使用 `中文名` 作为 Person 主键，并把 `附加名称` 合并进 `aliases` 列表。
- `相关人物` 中的 `未知` 会写为 `:UNTYPED_RELATION`。
- 人物关系类型会从中文关系名转换为安全的 Cypher relationship type，同时保留原始中文关系名到 `r.name`。
- aliases 去重使用纯 Cypher 实现，不依赖 APOC 插件。
