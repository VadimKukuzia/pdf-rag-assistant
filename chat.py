import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Conversational Assistant", page_icon="🤖", layout="wide")

st.title("🤖 PDF RAG Conversational Assistant")
st.caption("Діалоговий агент із підтримкою сесій та автоматичним викликом Hybrid RAG Tool")

# Sidebar
with st.sidebar:
    st.header("⚙️ Налаштування сесії")
    
    session_id = st.text_input("Session ID", value="default_session")
    
    if st.button("🔄 Оновити / Завантажити історію"):
        st.session_state.messages = []
        try:
            res = requests.get(f"{API_BASE_URL}/api/v1/sessions/{session_id}/history")
            if res.status_code == 200:
                history_data = res.json().get("messages", [])
                st.session_state.messages = history_data
                st.success("Історію завантажено!")
            else:
                st.warning("Сесія порожня або не знайдена.")
        except Exception as e:
            st.error(f"Помилка з'єднання з API: {e}")

    st.divider()
    st.header("📄 Завантаження PDF")
    uploaded_file = st.file_uploader("Оберіть PDF-файл", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Завантажити та проіндексувати"):
            with st.spinner("Індексація документа..."):
                try:
                    files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                    response = requests.post(f"{API_BASE_URL}/api/v1/upload", files=files)
                    if response.status_code == 201:
                        data = response.json()
                        st.success(f"Файл {data['filename']} успішно додано! Чанків: {data['chunks_created']}")
                    else:
                        st.error(f"Помилка: {response.json().get('detail')}")
                except Exception as e:
                    st.error(f"Помилка підключення: {e}")

# Ініціалізація стану чату
if "messages" not in st.session_state:
    st.session_state.messages = []

# Відображення повідомлень
for msg in st.session_state.messages:
    role = msg.get("role", "user")
    content = msg.get("content", "")
    with st.chat_message(role):
        st.write(content)

# Поле введення нового повідомлення
if prompt := st.chat_input("Запитайте щось або попросіть знайти інформацію в PDF..."):
    # Додаємо повідомлення користувача в інтерфейс
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # Відправляємо запит до Агента
    with st.chat_message("assistant"):
        with st.spinner("Агент міркує..."):
            try:
                payload = {"session_id": session_id, "message": prompt}
                res = requests.post(f"{API_BASE_URL}/api/v1/chat", json=payload)
                
                if res.status_code == 200:
                    data = res.json()
                    answer = data.get("answer", "")
                    used_tools = data.get("used_tools", [])

                    # Індикатор виклику тулзи
                    if "search_pdf_documents" in used_tools:
                        st.info("🛠️ **Агент визвав інструмент:** `search_pdf_documents` (Hybrid Search)")

                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error(f"Помилка API: {res.json().get('detail')}")
            except Exception as e:
                st.error(f"Не вдалося виконати запит: {e}")