import streamlit as st
import requests

API_BASE_URL = "http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAG Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 RAG Assistant")
st.caption("Інтелектуальна система пошуку відповідей у документації на базі LangGraph & Gemini")


def check_api_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except Exception:
        return False


api_online = check_api_health()

# Бокова панель
with st.sidebar:
    st.header("⚙️ Статус системи")
    
    if api_online:
        st.success("FastAPI Сервер: ONLINE 🟢")
    else:
        st.error("FastAPI Сервер: OFFLINE 🔴")
        st.info("Переконайся, що в терміналі запущено:\n`uvicorn main:app --reload`")

    st.divider()
    st.markdown("### 📊 Стек технологій")
    st.markdown("- **Orchestration:** LangGraph")
    st.markdown("- **LLM:** Gemini API")
    st.markdown("- **VectorStore:** ChromaDB")
    st.markdown("- **Backend:** FastAPI")
    st.markdown("- **Telemetry:** LangSmith")

    st.divider()
    if st.button("🧹 Очистити історію чату", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


tab_chat, tab_upload = st.tabs(["💬 Чат з Асистентом", "📂 Завантаження документів"])

# Розділ з чатом
with tab_chat:
    if not api_online:
        st.warning("⚠️ Для початку роботи запустіть FastAPI backend сервер.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()

    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                if "sources" in message and message["sources"]:
                    with st.expander("📚 Переглянути джерела контексту"):
                        for idx, src in enumerate(message["sources"], 1):
                            st.markdown(f"**Джерело #{idx}** (`{src.get('source_file')}`):")
                            st.info(src.get("content_preview"))

    if prompt := st.chat_input("Введіть ваше запитання по документах..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Шукаю інформацію у базі знань..."):
                    try:
                        payload = {
                            "query": prompt,
                            "session_id": "streamlit_user_session"
                        }
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/query",
                            json=payload,
                            timeout=60
                        )

                        if response.status_code == 200:
                            data = response.json()
                            answer = data.get("answer", "Не вдалося отримати відповідь.")
                            sources = data.get("sources", [])
                        else:
                            answer = f"⚠️ Помилка сервера ({response.status_code}): {response.text}"
                            sources = []

                    except Exception as e:
                        answer = f"⚠️ Не вдалося з'єднатися з API сервером: {e}"
                        sources = []

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

        st.rerun()

# Розділ завантаження файлів
with tab_upload:
    st.header("📂 Завантаження та індексація PDF")
    st.write("Завантажте нові PDF-файли, щоб автоматично нарізати їх на чанки та додати у ChromaDB.")

    uploaded_file = st.file_uploader("Оберіть PDF-документ", type=["pdf"])

    if uploaded_file is not None:
        st.write(f"📄 Обрано файл: **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
        
        if st.button("🚀 Проіндексувати документ", type="primary", use_container_width=True):
            if not api_online:
                st.error("Неможливо завантажити: FastAPI сервер офлайн.")
            else:
                with st.spinner("Завантаження, парсинг та збереження у ChromaDB..."):
                    try:
                        files = {
                            "file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")
                        }
                        response = requests.post(
                            f"{API_BASE_URL}/api/v1/upload",
                            files=files,
                            timeout=60
                        )

                        if response.status_code in (200, 201):
                            res_data = response.json()
                            st.success(f"✅ Документ **{res_data.get('filename')}** успішно оброблено!")
                            
                            col1, col2 = st.columns(2)
                            col1.metric("Створено чанків", res_data.get("chunks_created", 0))
                            col2.metric("Статус", res_data.get("status", "success"))
                        else:
                            st.error(f"Помилка індексації: {response.json().get('detail')}")

                    except Exception as e:
                        st.error(f"Помилка під час відправки файлу: {e}")