from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from assistant.vector_store import search

MODEL = "granite4.2:8b"

RAG_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a career assistant. Answer questions about the candidate using ONLY the context below.
If the answer is not in the context, say "I don't have that information in the CV."

Context:
{context}""",
        ),
        ("human", "{question}"),
    ]
)


def ask(question: str, top_k: int = 5) -> str:
    context_chunks = search(question, top_k=top_k)
    context = "\n\n---\n\n".join(context_chunks)
    chain = RAG_PROMPT | OllamaLLM(model=MODEL) | StrOutputParser()
    return chain.invoke({"context": context, "question": question})
