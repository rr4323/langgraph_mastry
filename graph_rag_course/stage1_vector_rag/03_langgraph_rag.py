"""Simple LangGraph RAG: Retrieve → Generate"""
import os
from pathlib import Path
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage
from pinecone import Pinecone

load_dotenv()
BASE_DIR = Path(__file__).parent
INDEX_DIR = BASE_DIR / "faiss_index"
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# retrieval helper
def top_k(query: str, k: int = 4):
    if PINECONE_API_KEY:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index = pc.Index(os.getenv("PINECONE_INDEX", "stage1-demo"))
        vec = embeddings.embed_query(query)
        res = index.query(vector=vec, top_k=k, include_metadata=True)
        return [(m["metadata"]["source"], m["score"]) for m in res["matches"]]
    else:
        vs = FAISS.load_local(INDEX_DIR, embeddings)
        docs = vs.similarity_search_with_score(query, k=k)
        return [(d.metadata["source"], s) for d, s in docs]

# LangGraph nodes
chat_model = ChatOpenAI(temperature=0.0)

def retrieve_node(state):
    q = state["question"]
    hits = top_k(q)
    print("\nRetrieved chunks:")
    for src, score in hits:
        print(f"  {src}  (score={score:.3f})")
    return {**state, "context": "\n".join(src for src, _ in hits), "next": "generate"}

def generate_node(state):
    prompt = f"Answer the user query using only the provided context.\n\nContext:\n{state['context']}\n\nUser: {state['question']}"
    answer = chat_model([HumanMessage(content=prompt)]).content
    print("\nAnswer:\n", answer)
    return {"next": END}

# build graph
sg = StateGraph(dict)
sg.add_node("retrieve", retrieve_node)
sg.add_node("generate", generate_node)
sg.add_conditional_edges("retrieve", lambda s: s["next"], {"generate": "generate"})
sg.add_conditional_edges("generate", lambda s: s["next"], {"end": END})
sg.set_entry_point("retrieve")
app = sg.compile()

while True:
    try:
        q = input("\nAsk a question (or 'exit'): ")
        if q.lower().strip() == "exit":
            break
        app.invoke({"question": q})
    except KeyboardInterrupt:
        break
