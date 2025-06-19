"""Simple Streamlit chat UI calling Stage3 LangGraph and upload API."""
import requests, streamlit as st
from 31_graph import app as graph_app

st.title("Graph-RAG Chat")

if "history" not in st.session_state:
    st.session_state.history = []

query = st.chat_input("Ask something…")
if query:
    st.session_state.history.append(("user", query))
    res = graph_app.invoke({"question": query})
    st.session_state.history.append(("assistant", res["answer"]))

for role, msg in st.session_state.history:
    st.chat_message(role).write(msg)

st.divider()
"""File upload"""
file = st.file_uploader("Upload .txt/.md to augment knowledge", type=["txt", "md"])
if file and st.button("Upload"):
    resp = requests.post("http://localhost:8000/upload", files={"file": (file.name, file.getvalue())})
    st.success(f"Uploaded – {resp.json().get('chunks')} chunks ingested")
