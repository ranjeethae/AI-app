import streamlit as st
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters  import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

loader = PyPDFLoader(r"d:\Downloads\On-the-Job Training Program.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

docs = text_splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings()

vectorstore = FAISS.from_documents(docs, embeddings)

retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

generator = pipeline("text-generation", model="gpt2")

llm = HuggingFacePipeline(pipeline=generator)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)
st.title("AI PDF Assistant")

question = st.text_input("Ask a question about the document:")

if question:
    result = qa_chain.run(question)
    st.write(result)