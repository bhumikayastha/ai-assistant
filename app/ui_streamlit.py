import streamlit as st
import requests
import os

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="AI Assistant", page_icon="🤖")
st.title("AI Assistant")

query = st.text_input("Ask something:")

if st.button("Send") and query:
    with st.spinner("Thinking..."):
        resp = requests.post(f"{API_URL}/chat", json={"query": query})
    if resp.status_code == 200:
        st.write(resp.json()["answer"])
    else:
        st.error(f"Error: {resp.status_code}")

mode = st.radio("Mode", ["Simple RAG", "Agentic (self-check)"])
endpoint = "/chat" if mode == "Simple RAG" else "/chat/agentic"

if st.button("Send") and query:
    with st.spinner("Thinking..."):
        resp = requests.post(f"{API_URL}{endpoint}", json={"query": query})
    if resp.status_code == 200:
        data = resp.json()
        if data.get("clarifying_question"):
            st.warning(data["clarifying_question"])
        else:
            st.write(data.get("answer") or data)
        if "trace" in data:
            st.caption(f"Iterations: {len(data['trace'])}, tokens: {data['tokens']}")