from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv

load_dotenv()

# Load your OpenAI API key from the environment
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OpenAI API key is missing. Please set it in the environment variables.")

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Define the path to the directory containing PDF files
pdf_directory_path = "docs"

# Function to load and process PDF data from all files in the directory
def load_pdf_data():
    all_pages = []
    for filename in os.listdir(pdf_directory_path):
        if filename.endswith(".pdf"):
            pdf_file_path = os.path.join(pdf_directory_path, filename)
            loader = PyPDFLoader(pdf_file_path, extract_images=True)
            pages = loader.load_and_split()
            all_pages.extend(pages)
    return all_pages

# Function to create a vector store from the documents
def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(documents)
    if not docs:
        raise ValueError("No documents were split. Please check the input documents.")
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store

# Function to query the vector store
def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query, k=k)
    docs_page_content = " ".join([doc.page_content for doc in docs])

    llm = OpenAI(temperature=0.5, openai_api_key=openai_api_key)
    prompt = PromptTemplate(
        input_variables=["questions", "docs"],
        template="""You are a helpful assistant that can answer questions based on the provided data.

        Answer the following questions: {questions}
        By searching the following data: {docs}

        Only use the factual information from the data to answer the question.

        If you feel like you don't have enough information to answer the question, you can say "I don't have enough information to answer that question."

        Your answer should be detailed.
        """,
    )

    chain = prompt | llm
    response = chain.invoke(
        {
            "questions": query,
            "docs": docs_page_content
        }
    )
    response = response.replace("\n", " ")
    return response

# Function to validate the API key
def validate_openai_api_key(openai_api_key):
    if openai_api_key is None or openai_api_key == "":
        raise Exception("OpenAI API key is missing.")
    if openai_api_key != os.getenv("OPENAI_API_KEY"):
        raise Exception("Invalid OpenAI API key.")
    return True

# Load the PDF data and create the vector store at the start
documents = load_pdf_data()
if not documents:
    raise ValueError("No documents were loaded. Please check the PDF directory path and files.")
vector_store = create_vector_store(documents)