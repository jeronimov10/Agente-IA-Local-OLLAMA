from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from Contexto import csm


model = OllamaLLM(model="llama3")


template = """
Eres un experto en análisis de inversiones, dataframes y gráficas.
Tu tarea es ayudar a maximizar inversiones en BTC-USD.

Este es tu contexto:
{contexto}

Pregunta del usuario:
{question}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

retriever = csm()

while True:
    print("\n\n-------------------------------")
    question = input("Pregunta (q para salir): ")
    print("\n\n")
    if question == "q":
        break
    
    docs = retriever.invoke(question)
    contexto = "\n\n".join([d.page_content for d in docs])

    
    result = chain.invoke({"contexto": contexto, "question": question})
    print(result)