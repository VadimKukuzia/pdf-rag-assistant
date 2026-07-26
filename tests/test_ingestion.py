import os
from ingestion import ingest_pdf, get_vectorstore

print("🔍 Перевірка модуля індексації та ChromaDB...")

TEST_PDF_PATH = "./data/sample.pdf" 

try:
    print(f"\n📥 Завантаження та нарізка файлу '{TEST_PDF_PATH}'...")
    result = ingest_pdf(TEST_PDF_PATH)
    print("🟢 Успішно збережено в ChromaDB:")
    print(f"   • Файл: {result['filename']}")
    print(f"   • Створено чанків: {result['chunks_created']}")

    print("\n🔎 Тестуємо семантичний векторний пошук у ChromaDB...")
    vectorstore = get_vectorstore()
    
    query = "картка"
    docs = vectorstore.similarity_search(query, k=2)

    print(f"🟢 Знайдено {len(docs)} релевантних чанки за запитом '{query}':")
    for i, doc in enumerate(docs, 1):
        print(f"\n--- Чанк #{i} ---")
        print(f"Зміст: {doc.page_content[:150]}...")

    print("\n✨ Крок 3 пройдено ідеально!")

except Exception as e:
    print(f"\n❌ Помилка під час індексації: {e}")