"""Streamlit application for the Chinese modern history knowledge graph."""

import streamlit as st
import streamlit.components.v1 as components

from src.config import Settings
from src.graphrag import GraphRAGService
from src.graph_store import Neo4jGraphStore
from src.visualization import build_pyvis_html


st.set_page_config(
    page_title="中国近代史人物关系图谱",
    page_icon="KG",
    layout="wide",
)


@st.cache_resource
def get_graph_store() -> Neo4jGraphStore:
    """Create one Neo4j driver instance for the Streamlit process."""
    settings = Settings.from_env()
    return Neo4jGraphStore(settings)


@st.cache_resource
def get_graphrag_service() -> GraphRAGService:
    settings = Settings.from_env()
    return GraphRAGService(get_graph_store(), settings)


graph = get_graph_store()
graphrag = get_graphrag_service()

st.title("中国近代史人物关系图谱")

with st.sidebar:
    st.header("图谱统计")
    stats = graph.get_basic_stats()
    st.metric("总人数", stats["people_count"])
    st.metric("总关系数", stats["relation_count"])

    st.subheader("核心历史人物 TOP 5")
    central_people = graph.get_top_degree_people(limit=5)
    if central_people:
        for index, item in enumerate(central_people, start=1):
            st.write(f"{index}. {item['name']} - Degree {item['degree']}")
    else:
        st.caption("暂无数据，请先运行 ETL 入库。")


tab_graph, tab_qa = st.tabs(["图谱检索", "GraphRAG 问答"])

with tab_graph:
    keyword = st.text_input("搜索人物", value="钱瑗", placeholder="输入中文名或别名")
    if keyword:
        subgraph = graph.get_person_neighborhood(keyword, depth=2, limit=150)
        if not subgraph["nodes"]:
            st.warning("没有找到匹配的人物或关联节点。")
        else:
            html = build_pyvis_html(subgraph)
            components.html(html, height=720, scrolling=True)

with tab_qa:
    question = st.text_area(
        "请输入问题",
        placeholder="例如：钱瑗和钱钟书是什么关系？她毕业于哪些学校？",
        height=120,
    )
    if st.button("基于图谱回答", type="primary") and question.strip():
        with st.spinner("正在检索图谱上下文并调用 Qwen..."):
            answer, context = graphrag.answer(question.strip())
        st.subheader("回答")
        st.write(answer)
        with st.expander("查看发送给模型的图谱上下文"):
            st.code(context or "未检索到相关图谱上下文。", language="text")
