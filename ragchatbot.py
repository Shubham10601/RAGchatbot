import os
from http.client import responses
import pdfplumber
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter

st.title("My First Chat Bot")

with st.sidebar:
    st.title("Your documents")
    file = st.file_uploader("Upload a PDF file and start asking questions", type="PDF")

#Extract the contents from the PDF and chunk it
if file is not None:
    #extract text from it
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"
    # st.write(text)

    #Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ". ", " ", ""],
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_text(text)
    st.write(f"Total chunks created: {len(chunks)}")

    # generating embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # store embeddings in vector DB
    vector_store = FAISS.from_texts(chunks,embeddings)

    #get user question
    user_question = st.text_input("Type your question here")

    #generate answer
    #question -> embedding -> similiarity search -> result to LLM -> response (CHAIN)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    retriever = vector_store.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k":4}
    )

    #define the LLM and prompts
    os.environ["GROQ_API_KEY"] = "YOUR_API_KEY"

    llm = ChatGroq(
        model_name="llama-3.1-8b-instant",
        temperature=0.3,
        max_tokens=1000
    )

    # provide the prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant answering questions about a PDF document.\n\n"
         "Guidelines:\n"
         "1. Provide complete, well-explained answers using the context below.\n"
         "2. Include relevant details, numbers, and explanations to give a thorough response.\n"
         "3. If the context mentions related information, include it to give fuller picture.\n"
         "4. If the answer is not found in the context, clearly say the information is unavailable.\n"
         "5. Summarize long information, ideally in bullets where needed.\n"
         "6. If the information is not in the context, say so politely.\n\n"
         "Context:\n{context}"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


    if user_question:
        response = chain.invoke(user_question)
        st.write(response)

















