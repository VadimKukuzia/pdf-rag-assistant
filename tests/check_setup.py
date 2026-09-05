import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("🔍 Перевірка конфігурації оточення...")

gemini_key = os.getenv("GEMINI_API_KEY")
langsmith_key = os.getenv("LANGCHAIN_API_KEY")

if not gemini_key or "тут_твій" in gemini_key:
    print("❌ Помилка: GEMINI_API_KEY не вказано в файлі .env!")
    exit(1)

if not langsmith_key or "тут_твій" in langsmith_key:
    print("⚠️ Попередження: LANGCHAIN_API_KEY не вказано. Трейсинг в LangSmith буде вимкнено.")
else:
    print("✅ Ключі оточення успішно зчитано.")

print("\n🚀 Тестування з'єднання з Gemini API та відправка трейсу в LangSmith...")

try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-flash-latest",
        google_api_key=gemini_key,
        temperature=0
    )

    response = llm.invoke("Привіт! Напиши коротко в 1 речення: 'Інфраструктура готова до роботи!'.")

    if isinstance(response.content, list):
        text_response = "".join([item if isinstance(item, str) else str(item) for item in response.content])
    else:
        text_response = str(response.content)

    print("\n🟢 Відповідь від Gemini API:")
    print(f"👉 {text_response.strip()}")
    print("\n✨ Крок 1 пройдено ідеально! Перевір свій дашборд на https://smith.langchain.com/ — там має з'явитися перший трейс.")

except Exception as e:
    print(f"\n❌ Помилка під час запиту до API: {e}")