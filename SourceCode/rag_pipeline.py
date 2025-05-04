
from langchain.prompts import PromptTemplate



def get_relevant_docs_basic(user_query):
    vectordb = Chroma(persist_directory="vector_store",
                      embedding_function=embeddings)
    retriever = vectordb.as_retriever(score_threshold=0.5)
    relevant_docs = retriever.invoke(user_query)
    return relevant_docs




def make_rag_prompt(query, relevant_passage):
    task_guidance = """
    You are an intelligent, formal, and trustworthy AI assistant for a local bank. 
    Use the retrieved context below to answer the customer's question accurately and concisely. 
    Do not repeat the question or include any extra information, explanations, or reasoning. 
    Just provide the direct answer in English, along with the source of the information.
    
    If the answer is not present in the context, respond with: "I'm sorry, I do not have that information at the moment."
"""

    template = PromptTemplate(
        input_variables=["query", "relevant_passage"],
        template=(
            "{task_guidance}\n\n"
            "Context:\n{relevant_passage}\n\n"
            "Question: {query}\n"
            "Answer (include source):"
        )
    )

    return template.format(
        query=query,
        relevant_passage=relevant_passage,
        task_guidance=task_guidance.strip()
    )
