from langchain_core.messages import AIMessage, HumanMessage
from langchain_classic.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from langchain_ollama import OllamaEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import bs4

groq_api_key = st.secrets["GROQ_API_KEY"]
os.environ['HF_TOKEN'] = st.secrets["HF_TOKEN"]

st.set_page_config(page_title="AI analyzer",page_icon=":robot_face:")
st.title("AI Blog Analyzer")
st.caption("Ask questions about Lilian Weng's Agent blog")

@st.cache_resource()
def get_vector_store():
            
        #     loader = WebBaseLoader(
        #     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
        #     bs_kwargs=dict(
        #         parse_only=bs4.SoupStrainer(
        #             class_=("post-content", "post-title", "post-header")
        #         )
        #     ),
        # )
            loader = PyPDFLoader("attention.pdf")
            document = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
            doc = text_splitter.split_documents(document)
            vectorstore= FAISS.from_documents(doc,HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
            return vectorstore

with st.spinner("Loading and indexing data..."):
    vector_store = get_vector_store()

llm =  ChatGroq(model="llama-3.3-70b-versatile",groq_api_key=groq_api_key,temperature=0.7)
k= st.slider("Select number of similar documents to retrieve",min_value=1,max_value=10,value=3) # slider to select k value
retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k":k}
)


if st.button("Clear Chat"):
    st.session_state.chat_history = []
    
@st.cache_resource()
def get_rag_chain():
        contextualize_qa_propmt = ChatPromptTemplate.from_messages(
        [
            ("system","Given a chat history and a new question which might some refrences to the context, make it a standalone question. do not answer"),
            (MessagesPlaceholder("chat_history")),
            ("human","{input}")
        ]
)

        history_aware_retrievr = create_history_aware_retriever(llm,retriever,contextualize_qa_propmt)

        system_propmt =  ChatPromptTemplate.from_messages(
                [
                    ("system","""You are an helpful assistant.Answer the question based on the following context. 
                    If you don't know the answer, say you don't know. do not try to make up an answer.
                    <context>
                    {context}
                    </context>
                    """),
                    MessagesPlaceholder("chat_history"),
                    ("human","{input}")
                ]
        )

        document_chain =  create_stuff_documents_chain(llm,system_propmt)
        rag_chain = create_retrieval_chain(history_aware_retrievr,document_chain)
        return rag_chain

rag_chain = get_rag_chain()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
        if isinstance(msg,HumanMessage):
                st.chat_message("user")
                st.markdown(msg.content)
        elif isinstance(msg,AIMessage):
                st.chat_message("assistant")
                st.markdown(msg.content)

user_input = st.chat_input("Ask question about the blog post")

if user_input:
        st.chat_message("user")
        st.markdown(user_input)

        with st.chat_message("assistant"):
           with st.spinner("thinking...🤔"):
            try:
                response = rag_chain.invoke({"input":user_input,"chat_history":st.session_state.chat_history})
                answer= response["answer"]
                st.markdown(answer)
            except Exception as e:
                answer = "Sorry, something went wrong. Please try again later."
                st.markdown(answer)
                print(e)

        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=answer))



