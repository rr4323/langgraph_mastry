"""Graph-RAG pipeline using Neo4j vector search + entity expansion."""
import os
from pathlib import Path
from dotenv import load_dotenv
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from langchain.chat_models import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain.schema import HumanMessage

load_dotenv()
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
chat = ChatOpenAI(temperature=0.0)

uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
user = os.getenv("NEO4J_USER", "neo4j")
pwd = os.getenv("NEO4J_PASSWORD", "pass1234")

def vector_query(tx, vector, k=4):
    return tx.run(
        """
        MATCH (d:Document)
        WITH d, gds.similarity.cosine(d.embedding, $v) AS score
        ORDER BY score DESC LIMIT $k
        RETURN d.id AS id, d.text AS text, score
        """,
        v=vector,
        k=k,
    ).data()

def expand_entities(tx, doc_ids):
    return tx.run(
        """
        MATCH (d:Document)-[:MENTIONS]->(e:Entity)<-[:MENTIONS]-(d2:Document)
        WHERE d.id IN $ids AND d2.id <> d.id
        RETURN DISTINCT d2.text AS text LIMIT 10
        """,
        ids=doc_ids,
    ).value()

def retrieve(state):
    q = state["question"]
    vec = embed_model.encode(q).tolist()
    with GraphDatabase.driver(uri, auth=(user, pwd)) as driver:
        with driver.session(database="neo4j") as sess:
            res = sess.execute_read(vector_query, vec)
            top_ids = [r["id"] for r in res]
            extra = sess.execute_read(expand_entities, top_ids)
    context = "\n".join(r["text"] for r in res) + "\n" + "\n".join(extra)
    return {"context": context, "question": q, "next": "generate"}

def generate(state):
    prompt = f"Answer the question only using the context.\n\nContext:\n{state['context']}\n\nQuestion: {state['question']}"
    ans = chat([HumanMessage(content=prompt)]).content
    print("\nAnswer:\n", ans)
    return {"next": END}

sg = StateGraph(dict)
sg.add_node("retrieve", retrieve)
sg.add_node("generate", generate)
sg.add_conditional_edges("retrieve", lambda s: s["next"], {"generate": "generate"})
sg.add_conditional_edges("generate", lambda s: s["next"], {"end": END})
sg.set_entry_point("retrieve")
app = sg.compile()

while True:
    q = input("\nAsk (or exit): ")
    if q.lower() == "exit":
        break
    app.invoke({"question": q})
