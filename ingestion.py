from dotenv import load_dotenv
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def ingest_docs():
    print("Ingesting docs...")

    def custom_bs_loader(file_path):
        return BSHTMLLoader(file_path, open_encoding="utf-8", bs_kwargs={"features": "lxml"})

    loader = DirectoryLoader(
        "langchain-docs/python.langchain.com/docs",
        glob="**/*.html",
        loader_cls=custom_bs_loader
    )

    raw_documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_documents)

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs\\python.langchain.com\\", "https://python.langchain.com/")
        doc.metadata.update({"source": new_url})

    PineconeVectorStore.from_documents(
        documents,
        embeddings,
        index_name="langchain-docs-index"
    )

    print("*** Done ***")

if __name__ == "__main__":
    ingest_docs()