from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
import streamlit as st
from streamlit_option_menu import option_menu

def load_documents(file_path):
    '''
    Load data from a text file and create a list of Langchain documents
    '''
    with open(file_path, 'r', encoding='utf-8') as f:
        movie_text = f.read()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=300)

    docs = text_splitter.create_documents([movie_text])

    return docs

def initialize_models():
    '''
    Initialize embedding model and large language model
    '''
    embed_model = HuggingFaceBgeEmbeddings(
        model_name="BAAI/bge-small-en-v1.5", 
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True} 
    )
    llm = ChatGroq(
        temperature=0,
        model="mixtral-8x7b-32768",
        api_key="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"
    )
    return embed_model, llm

def hybrid_retrieval(keyword_retriever, vector_retriever):
    '''
    Performs hybrid search (keyword + vector search)
    '''
    ensemble_retriever = EnsembleRetriever(retrievers=[keyword_retriever, vector_retriever], weights=[0.4, 0.6])
    return ensemble_retriever

def create_prompt():
    '''
    Define a prompt template for the Q&A system
    '''
    
    prompt_template = """
    You are a helpful assistant who can generate insightful answers for questions related to movies covering everything from
    plot summaries and cast details to release dates and reviews.
    Strictly answer the given question only based on the following context

    {context}
    Question: {question}

    Helpful Answers:
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return prompt

def main():
    st.set_page_config(page_title="Movie Whisperer", layout="wide")
    
    with st.sidebar:
        selected = option_menu(
            menu_title="",
            options=["Home", "Q&A"],
            icons=["house", "question-circle"],
            default_index=0,
        )

    if selected == "Home":
        st.title("Get to Know More about Recently-released Movies!")
        st.write("Shoot your questions in the 'Q&A' section!\n\n")
        st.write("""I'll be able to answer questions about the Highest-Grossing Hollywood and Mollywood Movies released in 2024.""")

    if selected == "Q&A":
        st.title("Ask a Question")
        st.write("""I'll be able to answer questions about the Highest-Grossing Hollywood and Mollywood Movies released in 2024.\n""")
        
        if "documents_loaded" not in st.session_state:
            st.session_state.documents_loaded = False

        if not st.session_state.documents_loaded:
            with st.spinner('Loading documents...'):
                file_path = "C:\\Users\\Aditya\\OneDrive\\movie_qa\\movie_data.txt"
                final_documents = load_documents(file_path)

                embed_model, llm = initialize_models()

                bm25_retriever = BM25Retriever.from_documents(final_documents)
                bm25_retriever.k = 3

                #load previously created vector index
                vector_store = FAISS.load_local('C:\\Users\\Aditya\\OneDrive\\movie_qa\\faiss_index_movies', embed_model, allow_dangerous_deserialization=True)
                vector_retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})

                ensemble_retriever = hybrid_retrieval(bm25_retriever, vector_retriever)

                prompt = create_prompt()

                retrievalQA = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=ensemble_retriever,
                    return_source_documents=True,
                    chain_type_kwargs={"prompt": prompt}
                )
                st.session_state.retrievalQA = retrievalQA
                st.session_state.documents_loaded = True

        if st.session_state.documents_loaded:
            query = st.text_input("")
            if query:
                result = st.session_state.retrievalQA.invoke({"query": query})
                st.write("\n", result['result'])

if __name__ == "__main__":
    main()
