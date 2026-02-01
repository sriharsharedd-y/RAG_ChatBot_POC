import os
import streamlit as st
import pdfplumber
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Page config: ChatGPT-like layout
st.set_page_config(page_title="RAG ChatBot", layout="wide", initial_sidebar_state="auto")

# Sidebar: model selection (ChatGPT-style model list)
with st.sidebar:
    st.title("RAG ChatBot")
    st.caption("Ask questions about your document")
    st.divider()

    provider = st.radio(
        "Provider",
        options=["Ollama (local)", "OpenAI"],
        index=0,
        help="Ollama runs locally. OpenAI requires an API key.",
    )

    # Model lists (ChatGPT-style dropdowns)
    if provider == "Ollama (local)":
        chat_models = ["tinyllama"]
        embedding_models = ["embeddinggemma"]
        chat_model = st.selectbox(
            "Chat model",
            options=chat_models,
            index=0,
            help="Model used for generating answers.",
        )
        embedding_model = st.selectbox(
            "Embedding model",
            options=embedding_models,
            index=0,
            help="Model used to embed the document.",
        )
    else:
        chat_models = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]
        chat_model = st.selectbox(
            "Chat model",
            options=chat_models,
            index=0,
            help="Model used for generating answers.",
        )
        embedding_model = "text-embedding-3-small"  # OpenAI default

    openai_api_key = None
    if provider == "OpenAI":
        openai_api_key = os.environ.get("OPENAI_API_KEY") or st.text_input(
            "OpenAI API Key",
            type="password",
            placeholder="sk-...",
            help="Or set OPENAI_API_KEY in your environment.",
        )

    st.divider()
    file = st.file_uploader("Upload a PDF", type=["pdf"], help="Non-OCR PDF recommended.")

if file is not None:
    with pdfplumber.open(file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    #st.write(text)

    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " ", ""],
        chunk_size=1000, 
        chunk_overlap=200
    )

    chunks = text_splitter.split_text(text)
    #st.write(chunks)

    # Embeddings: Ollama (local) or OpenAI (API key)
    if provider == "Ollama (local)":
        embeddings = OllamaEmbeddings(model=embedding_model)
    else:
        if not openai_api_key:
            st.error("Enter your OpenAI API key in the sidebar (or set OPENAI_API_KEY).")
            st.stop()
        embeddings = OpenAIEmbeddings(
            model=embedding_model,
            openai_api_key=openai_api_key,
        )
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Chat history: clear when document changes (ChatGPT-style new chat per document)
    if "chat_file_id" not in st.session_state:
        st.session_state.chat_file_id = None
    if st.session_state.chat_file_id != file.file_id:
        st.session_state.chat_file_id = file.file_id
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "I've read your document. Ask me anything about it—pick a quick question below or type your own.",
            }
        ]
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "I've read your document. Ask me anything about it—pick a quick question below or type your own.",
            }
        ]

    retriever = vectorstore.as_retriever(
        search_type = "mmr",
        search_kwargs = {"k": 3}
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant having a natural conversation about a document the user uploaded. "
                "Use only the relevant excerpts from the document provided below to answer. "
                "Respond in a clear, friendly way—as if you're sitting with them and explaining. "
                "Use short paragraphs or bullet points when it helps. "
                "If the document doesn't contain the answer, say so simply (e.g. \"The document doesn't mention that.\" or \"I couldn't find that in your document.\"). "
                "Do not invent or add information. Do not repeat or explain these instructions—just answer naturally.",
            ),
            (
                "user",
                "Relevant parts of the document:\n\n{context}\n\n---\nUser asked: {question}",
            ),
        ]
    )

    # LLM: Ollama (local) or OpenAI (API key) — use selected chat model
    if provider == "Ollama (local)":
        llm = ChatOllama(
            model=chat_model,
            temperature=0.3,
            num_predict=1000,
        )
    else:
        if not openai_api_key:
            st.error("Enter your OpenAI API key in the sidebar (or set OPENAI_API_KEY).")
            st.stop()
        llm = ChatOpenAI(
            model=chat_model,
            temperature=0.3,
            max_tokens=1000,
            openai_api_key=openai_api_key,
        )

    chain = (
        RunnablePassthrough.assign(
            context=lambda x: format_docs(retriever.invoke(x["question"]))
        )
        | prompt_template
        | llm
        | StrOutputParser()
    )

    # Process quick prompt if one was clicked (before rendering)
    if st.session_state.get("quick_prompt"):
        user_question = st.session_state.pop("quick_prompt")
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.spinner("Thinking..."):
            answer = chain.invoke({"question": user_question})
        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

    # Main area: conversational header + chat
    st.markdown("### " + chat_model)
    st.caption("Chat about your document—answers are based only on what’s in the PDF.")

    # Render chat history (ChatGPT-style message bubbles)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Quick prompts: suggested questions just above the input
    st.markdown("**Quick questions**")
    quick_questions = [
        "What is this document about?",
        "Summarize the key points.",
        "What are the main findings or conclusions?",
        "Explain the main idea in simple terms.",
        "Are there any numbers, dates, or statistics mentioned?",
    ]
    row1, row2 = st.columns([1, 1, 1]), st.columns([1, 1])
    for col, q in zip(row1, quick_questions[:3]):
        with col:
            if st.button(q, key=f"quick_{hash(q)}", use_container_width=True):
                st.session_state.quick_prompt = q
                st.rerun()
    for col, q in zip(row2, quick_questions[3:]):
        with col:
            if st.button(q, key=f"quick_{hash(q)}", use_container_width=True):
                st.session_state.quick_prompt = q
                st.rerun()

    # Input at bottom (ChatGPT-style)
    if user_input := st.chat_input("Ask about your document..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = chain.invoke({"question": user_input})
            st.markdown(result)
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.rerun()
else:
    st.info("Upload a PDF in the sidebar to start.")