import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import HuggingFaceHub

st.set_page_config(page_title="RAG Support Chatbot", layout="centered")
st.title("📞 RAG Chatbot – Local Version (Free)")

user_query = st.text_input("Ask a question:")

if user_query:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("vectorstore", embeddings)

    # Local lightweight LLM (acts as a placeholder)
    from langchain.llms import OpenAI  # Still needed for compatibility in LangChain 0.1
    from langchain.chains.question_answering import load_qa_chain
    from langchain.chains import RetrievalQA

    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    chain = RetrievalQA.from_chain_type(
        llm=FakeLLM(),  # This will be replaced by local logic or stub
        retriever=retriever,
        return_source_documents=True
    )

    result = chain(user_query)
    answer = result["result"]
    sources = result["source_documents"]

    if not sources or answer.strip().lower() in ["", "i don't know"]:
        st.warning("🤷 I don't know. Please ask something from the provided documentation.")
    else:
        st.success(answer)
        with st.expander("Sources"):
            for doc in sources:
                st.markdown(f"• `{doc.metadata['source']}`")
