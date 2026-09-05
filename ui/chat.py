import os

import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="PDF RAG Conversational Assistant", page_icon="🤖", layout="wide")

st.title("🤖 PDF RAG Conversational Assistant")
st.caption("Діалоговий агент із підтримкою сесій та автоматичним викликом Hybrid RAG Tool")


def fetch_history(sess_id: str):
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/sessions/{sess_id}/history", timeout=10)
        if res.status_code == 200:
            return res.json().get("messages", [])
    except Exception as e:
        st.sidebar.error(f"Помилка з'єднання з API: {e}")
    return []


with st.sidebar:
    session_id = "default_session"

    st.header("📄 Завантаження PDF")
    uploaded_file = st.file_uploader("Оберіть PDF-файл", type=["pdf"])

    if uploaded_file is not None and st.button("Завантажити та проіндексувати"):
        with st.spinner("Індексація документа..."):
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                response = requests.post(f"{API_BASE_URL}/api/v1/upload", files=files, timeout=60)
                if response.status_code == 201:
                    data = response.json()
                    st.success(f"Файл {data['filename']} успішно додано! Чанків: {data['chunks_created']}")
                else:
                    st.error(f"Помилка: {response.json().get('detail')}")
            except Exception as e:
                st.error(f"Помилка підключення: {e}")

if "last_session_id" not in st.session_state or st.session_state.last_session_id != session_id:
    st.session_state.messages = fetch_history(session_id)
    st.session_state.last_session_id = session_id

for msg in st.session_state.messages:
    with st.chat_message(msg.get("role", "user")):
        st.write(msg.get("content", ""))

if prompt := st.chat_input("Запитайте щось або попросіть знайти інформацію в PDF..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Агент міркує..."):
            try:
                res = requests.post(
                    f"{API_BASE_URL}/api/v1/chat",
                    json={"session_id": session_id, "message": prompt},
                    timeout=60
                )
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    used_tools = data.get("used_tools", [])

                    if "search_pdf_documents" in used_tools:
                        st.info("🛠️ **Агент викликав інструмент:** `search_pdf_documents` (Hybrid Search)")

                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    detail = res.json().get("detail", res.text) if res.headers.get("content-type") == "application/json" else res.text
                    st.error(f"Помилка API ({res.status_code}): {detail}")
            except Exception as e:
                st.error(f"Не вдалося виконати запит: {e}")