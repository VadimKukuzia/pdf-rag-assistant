import streamlit as st
import requests

API_BASE_URL = "http://localhost:8000"

st.set_page_config(page_title="PDF RAG Conversational Assistant", page_icon="🤖", layout="wide")

st.title("🤖 PDF RAG Conversational Assistant")
st.caption("Діалоговий агент із підтримкою сесій та автоматичним викликом Hybrid RAG Tool")


def fetch_history(sess_id: str):
    """Отримує історію повідомлень з бекенду за session_id."""
    try:
        res = requests.get(f"{API_BASE_URL}/api/v1/sessions/{sess_id}/history")
        if res.status_code == 200:
            return res.json().get("messages", [])
    except Exception as e:
        st.sidebar.error(f"Помилка з'єднання з API: {e}")
    return []


# Sidebar
with st.sidebar:
    st.header("⚙️ Налаштування сесії")
    
    session_id = st.text_input("Session ID", value="default_session")
    
    # Кнопка для ручного примусового оновлення
    if st.button("🔄 Оновити історію"):
        st.session_state.messages = fetch_history(session_id)
        st.session_state.last_session_id = session_id
        st.success("Історію оновлено!")

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

# АВТОМАТИЧНЕ ПІДВАНТАЖЕННЯ: спрацьовує при першому заході або зміні Session ID
if "last_session_id" not in st.session_state or st.session_state.last_session_id != session_id:
    st.session_state.messages = fetch_history(session_id)
    st.session_state.last_session_id = session_id

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
                        st.info("🛠️ **Агент викликав інструмент:** `search_pdf_documents` (Hybrid Search)")

                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    try:
                        error_detail = res.json().get("detail", res.text)
                    except Exception:
                        error_detail = res.text or f"HTTP Status {res.status_code}"
                    st.error(f"Помилка API ({res.status_code}): {error_detail}")
            except Exception as e:
                st.error(f"Не вдалося виконати запит: {e}")