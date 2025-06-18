from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from google.colab import userdata
from langgraph.graph import END, StateGraph, START
from typing import TypedDict, Annotated, List, Literal
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from sentence_transformers import SentenceTransformer
import pandas as pd
import os, re, pickle, hashlib, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import Image, display
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from langchain_core.language_models import BaseLanguageModel

class AgentState(TypedDict):
    question: str
    refined_question: str
    search_query: List[str]
    retrieved_memories: List[str]
    response: str

class InstructorEmbeddingsWrapper:
    def __init__(self, model_name="hkunlp/instructor-large"):
        self.model = SentenceTransformer(model_name, device='cuda')

    def embed_documents(self, texts):
        return self.model.encode(
            [["Represent the document for retrieval", t] for t in texts],
            show_progress_bar=False
        )

class CachedMemoryRetriever:
    def __init__(self, memory_dir, cache_dir, embedding_model):
        self.memory_sources = {
            "persona": ("persona.txt", "persona.pkl", "txt"),
            "short_term": ("short_term.txt", "short_term.pkl", "txt"),
            "long_term": ("long_term.csv", "long_term.pkl", "csv"),
        }
        self.memory_dir = memory_dir
        self.cache_dir = cache_dir
        self.embedder = embedding_model

    def load_memory_with_cache(self, txt_path, pkl_path, file_type):

        if file_type == "txt":
            if not os.path.exists(txt_path):
                return [], []

            with open(txt_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f if line.strip()]
        elif file_type == "csv":
            if not os.path.exists(txt_path):
                return [], []

            df = pd.read_csv(txt_path)
            if df.empty or "content" not in df.columns:
                return [], []
            lines = df["content"].astype(str).tolist()
        else:
            raise ValueError(f"Unsupported file_type: {file_type}")

        hashes = [hashlib.sha256(line.encode()).hexdigest() for line in lines]

        if os.path.exists(pkl_path):
            with open(pkl_path, "rb") as f:
                cache = pickle.load(f)
        else:
            cache = {"lines": [], "hashes": [], "vectors": []}

        new_lines, new_hashes = [], []
        for line, h in zip(lines, hashes):
            if h not in cache["hashes"]:
                new_lines.append(line)
                new_hashes.append(h)

        if new_lines:
            new_vectors = self.embedder.embed_documents(new_lines)
            if len(cache["vectors"]) == 0:
                cache["vectors"] = new_vectors
            else:
                cache["vectors"] = np.vstack([cache["vectors"], new_vectors])
            cache["lines"].extend(new_lines)
            cache["hashes"].extend(new_hashes)
            with open(pkl_path, "wb") as f:
                pickle.dump(cache, f)

        return cache["lines"], cache["vectors"]

    def retrieve(self, search_queries, k_per_file=3):
        results = []

        query_vecs = self.embedder.embed_documents(search_queries)
        for name, (file_path, pkl_path, file_type) in self.memory_sources.items():
            txt_path = os.path.join(self.memory_dir, file_path)
            pkl_path = os.path.join(self.cache_dir, pkl_path)

            lines, vectors = self.load_memory_with_cache(txt_path, pkl_path, file_type)
            if not lines:
                continue

            sim_matrix = cosine_similarity(query_vecs, vectors)
            best_scores = sim_matrix.max(axis=0)
            top_indices = best_scores.argsort()[-k_per_file:][::-1]

            top_memories = [f"[{name}] {lines[i]}" for i in top_indices]
            results.extend(top_memories)

        return results

    def embed_query(self, query):
        return self.model.encode(
            [["Represent the query for retrieval", query]],
            show_progress_bar=False
        )[0]

def build_langgraph(
    file_path: str,
    analysis_llm: BaseLanguageModel,
    response_llm: BaseLanguageModel,
    summary_llm: BaseLanguageModel,
    analysis_name: str,
    response_name: str,
    summary_name: str,
    me: str,
    you: str
) -> StateGraph:
    # 1) 경로 설정
    MEMORY_DIR = os.path.join(file_path, "memory")
    CACHE_DIR  = os.path.join(file_path, "memory_cache")
    PROMPT_DIR = os.path.join(os.path.dirname(file_path), "prompt")
    os.makedirs(MEMORY_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR,  exist_ok=True)

    # 2) 프롬프트 로드
    def load_prompt(dir, name):  # persona, question_analysis, ...
        with open(os.path.join(dir, f"{name}.txt"), "r", encoding="utf-8") as f:
            return f.read()
    persona_content          = load_prompt(MEMORY_DIR, "persona")
    question_analysis_prompt = PromptTemplate.from_template(load_prompt(PROMPT_DIR, f"question_analysis_{analysis_name}")).partial(persona=persona_content)
    response_prompt          = PromptTemplate.from_template(load_prompt(PROMPT_DIR, f"response_{response_name}")).partial(persona=persona_content)
    summary_prompt           = PromptTemplate.from_template(load_prompt(PROMPT_DIR, f"summary_{summary_name}")).partial(persona=persona_content)

    embedder = InstructorEmbeddingsWrapper()
    retriever = CachedMemoryRetriever(
        memory_dir=MEMORY_DIR,
        cache_dir=CACHE_DIR,
        embedding_model=embedder
    )

    question_analysis_chain = (
        question_analysis_prompt
        | analysis_llm
        | JsonOutputParser()
    )

    response_chain  = (
        response_prompt
        | response_llm
        | StrOutputParser()
    )

    summary_chain = (
        summary_prompt
        | summary_llm
        | StrOutputParser()
    )

    def analyze_question_node(state: AgentState) -> AgentState:
        result = question_analysis_chain.invoke({"question": state["question"]})
        return {
            **state,
            "refined_question": result["refined_question"],
            "search_query": result["search_query"]
        }
    
    def retrieve_memory_node(state: AgentState) -> AgentState:
        memories = retriever.retrieve(state["search_query"], k_per_file=3)
        return {
            **state,
            "retrieved_memories": memories
        }
    
    def generate_response_node(state: AgentState) -> AgentState:
        response = response_chain.invoke({
            "refined_question": state["refined_question"],
            "retrieved_memories": "\n".join(state["retrieved_memories"])
        })
        return {
            **state,
            "response": response
        }
    
    def update_short_term_node(state: AgentState) -> AgentState:
        short_term_path = os.path.join(MEMORY_DIR, "short_term.txt")

        # 대화 포맷 작성
        dialogue = [
            f"{you}: {state['question']}",
            f"{me}: {state['response']}"
        ]

        # 파일에 append
        with open(short_term_path, "a", encoding="utf-8") as f:
            f.write("\n".join(dialogue) + "\n")

        return state  # 그대로 상태 전달
    
    def check_short_term_threshold(state: AgentState) -> Literal["summarize", "end"]:
        short_term_path = os.path.join(MEMORY_DIR, "short_term.txt")

        if not os.path.exists(short_term_path):
            return "end"

        with open(short_term_path, "r") as f:
            lines = [line.strip() for line in f if line.strip()]

        # 줄 수가 30줄 이상이면 요약
        if len(lines) >= 30:
            return "summarize"
        else:
            return "end"

    def summarize_short_term_node(state: AgentState) -> AgentState:
        short_term_path = os.path.join(MEMORY_DIR, "short_term.txt")
        long_term_path = os.path.join(MEMORY_DIR, "long_term.csv")

        if not os.path.exists(short_term_path):
            return state

        # 1. 단기 기억 읽기
        short_term_path = os.path.join(MEMORY_DIR, "short_term.txt")
        with open(short_term_path, "r", encoding="utf-8") as f:
            conversation = f.read()

        # 2. 장기 기억 생성
        summary_text = summary_chain.invoke({"dialogue": conversation})

        # 3. 요약된 기억 파싱
        new_rows = []
        new_contents = set()
        for line in summary_text.strip().splitlines():
            match = re.match(r"[-•]?\s*(.+?)\s*\((\d+)\)", line.strip())
            if match:
                content = match.group(1)
                importance = int(match.group(2))
                new_rows.append([content, importance, 5])  # freshness = 5
                new_contents.add(content)

        if not new_rows:
            open(short_term_path, "w").close()
            return state

        new_df = pd.DataFrame(new_rows, columns=["content", "importance", "freshness"])

        # 4. 기존 long_term 로드
        if os.path.exists(long_term_path):
            old_df = pd.read_csv(long_term_path)
        else:
            old_df = pd.DataFrame(columns=["content", "importance", "freshness"])

        # 5. 기존 기억들 freshness 감소 (새 요약 제외)
        old_df["freshness"] = old_df.apply(
            lambda row: row["freshness"] if row["content"] in new_contents else max(0, row["freshness"] - 1),
            axis=1
        )

        # 6. 병합
        merged_df = pd.concat([old_df, new_df], ignore_index=True)

        # 7. 총 점수 계산 및 삭제
        merged_df["score_sum"] = merged_df["importance"] + merged_df["freshness"]
        total = merged_df["score_sum"].sum()

        if total > 250:
            merged_df = merged_df.sort_values(by=["score_sum", "freshness"], ascending=[True, True])
            excess = total - 250
            removed = 0
            drop_indices = []
            for idx, row in merged_df.iterrows():
                removed += row["score_sum"]
                drop_indices.append(idx)
                if removed >= excess:
                    break
            merged_df = merged_df.drop(index=drop_indices)

        # 8. 저장
        merged_df.drop(columns=["score_sum"]).to_csv(long_term_path, index=False)

        # 9. short_term 초기화
        open(short_term_path, "w").close()

        return state
    
    # 그래프 빌더 생성
    graph_builder = StateGraph(AgentState)

    # 일반 노드
    graph_builder.add_node("analyze_question", analyze_question_node)
    graph_builder.add_node("retrieve_memory", retrieve_memory_node)
    graph_builder.add_node("generate_response", generate_response_node)
    graph_builder.add_node("update_short_term", update_short_term_node)
    graph_builder.add_node("summarize_short_term", summarize_short_term_node)

    # ✅ 조건 분기 노드: 요약 판단
    graph_builder.add_conditional_edges(
        "update_short_term",
        check_short_term_threshold,
        {
            "summarize": "summarize_short_term",
            "end": END
        }
    )

    # 노드 연결
    graph_builder.add_edge(START, "analyze_question")
    graph_builder.add_edge("analyze_question", "retrieve_memory")
    graph_builder.add_edge("retrieve_memory", "generate_response")
    graph_builder.add_edge("generate_response", "update_short_term")
    graph_builder.add_edge("summarize_short_term", END)

    # 그래프 컴파일
    graph = graph_builder.compile()

    return graph