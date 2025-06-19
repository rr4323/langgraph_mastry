"""Defines specialist agents used in Stage 3."""
import os
from typing import List
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate

load_dotenv()
llm = ChatOpenAI(temperature=0.0)

# ----- Researcher: gather facts via Graph-RAG (calls Stage2 retriever) -----
from stage2_neo4j_graph_rag.12_graph_rag import retrieve as graph_retrieve

def researcher_agent(query: str) -> str:
    state = graph_retrieve({"question": query})
    return state["context"]

# ----- Analyst: reason over context, list key points -----
analyst_prompt = PromptTemplate.from_template(
    """You are an analyst. From the provided context, extract the 3 most relevant facts that answer the question.\n\nContext:\n{context}\n\nQuestion: {question}\n\nBullet list of facts:"""
)

def analyst_agent(question: str, context: str) -> str:
    return llm(analyst_prompt.format_prompt(question=question, context=context)).content

# ----- Writer: craft final answer -----
writer_prompt = PromptTemplate.from_template(
    """You are a technical writer. Formulate a clear answer using the bullet point facts.\n\nFacts:\n{facts}\n\nAnswer:"""
)

def writer_agent(facts: str) -> str:
    return llm(writer_prompt.format_prompt(facts=facts)).content
