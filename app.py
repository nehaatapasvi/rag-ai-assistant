"""
RAG AI Assistant
A full Retrieval-Augmented Generation application using Streamlit, LangChain, and Groq.
"""

import streamlit as st
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import tempfile
import os

# Load env
load_dotenv()

# Config
LLM_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


# ================= LLM =================
@st.cache_resource
def get_llm():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY not set")

    return ChatGroq(
        model=LLM_MODEL,
        api_key=api_key,
        temperature=0.1
    )


# ================= PDF PROCESS =================
def process_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        path = tmp.name

    try:
        loader = PyMuPDFLoader(path)
        documents = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )

        chunks = splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        vectorstore = FAISS.from_documents(chunks, embeddings)

        return vectorstore, len(chunks)

    finally:
        os.unlink(path)


# ================= RAG CHAIN =================
def create_qa_chain(vectorstore):
    llm = get_llm()

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": 4}
    )

    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant.

Answer ONLY from the context below.
If the answer is not found, say:
"I don't have enough information."

Context:
{context}

Question:
{question}

Answer:
""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


# ================= UI =================
def main():
    st.set_page_config(
        page_title="RAG AI Assistant",
        page_icon="🤖"
    )

    st.title("🤖 RAG AI Assistant")
    st.write("Upload a PDF and ask questions")

    # session
    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = None

    # upload
    file = st.file_uploader("Upload PDF", type=["pdf"])

    if file:
        if st.session_state.vectorstore is None:
            with st.spinner("Processing PDF..."):
                vs, count = process_pdf(file)
                st.session_state.vectorstore = vs
                st.session_state.qa_chain = create_qa_chain(vs)

                st.success(f"{count} chunks created")

        if st.button("Clear"):
            st.session_state.vectorstore = None
            st.session_state.qa_chain = None
            st.rerun()

    # ask
    st.subheader("Ask Question")

    if st.session_state.qa_chain:
        question = st.text_input("Enter your question")

        if st.button("Ask"):
            if not question.strip():
                st.warning("Enter a question")
            else:
                with st.spinner("Thinking..."):
                    try:
                        answer = st.session_state.qa_chain.invoke(question)

                        st.success("Answer")
                        st.write(answer)

                    except Exception as e:
                        st.error(str(e))
    else:
        st.info("Upload a PDF first")


# ================= RUN =================
if __name__ == "__main__":
    main()