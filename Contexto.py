from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import pandas as pd



CSV_FILE = "btc_context.csv"
DB_DIR = "./chrome_langchain_db"


embeddings = OllamaEmbeddings(model="mxbai-embed-large")



try:
    
    df = pd.read_csv(CSV_FILE)
    print("Creando o cargando base de conocimiento desde:", CSV_FILE)

    documents = []
    ids = []

    for i, row in df.iterrows():
        content = f"{row['Title']}. {row['Content']}"
        doc = Document(page_content=content, metadata={"source": row['Title']}, id=str(i))
        documents.append(doc)
        ids.append(str(i))

    
    vector_store = Chroma(
        collection_name="btc_knowledge",
        persist_directory=DB_DIR,
        embedding_function=embeddings
    )

    
    if len(vector_store.get()['ids']) == 0:
        vector_store.add_documents(documents=documents, ids=ids)
        print("Base de conocimiento creada y guardada en:", DB_DIR)
    else:
        print("Base de conocimiento ya existente, cargada correctamente.")

except Exception as e:
    print("Error al crear la base de conocimiento:", e)
    raise e


def csm():
    """Devuelve el retriever para usar en main.py"""
    return vector_store.as_retriever(search_kwargs={"k": 5})
