import os

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import CharacterTextSplitter

load_dotenv()

if __name__ == "__main__":
    print("ingesting data ....")
    loader = TextLoader("medium_blog_on_vecDbs.txt")
    documents = loader.load()

    print("splitting data ....")
    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = splitter.split_documents(documents)
    print(f"Number of chunks: {len(chunks)}")

    embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

    print("putting embeddings into vector store ....")
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embedding_model,
        index_name=os.getenv("INDEX_NAME"),
    )
    print("done")
