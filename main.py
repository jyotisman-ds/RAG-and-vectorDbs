import os

from langchain import hub
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

if __name__ == "__main__":
    print("Hello from rag-and-vectordbs!")

    embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))
    llm = ChatOpenAI(model="gpt-4o")

    question = "What is SingleStore in the context of LLMs?"
    # chain = PromptTemplate.from_template(template=question) | llm
    # response = chain.invoke(input={})
    # print(response.content)

    vector_store = PineconeVectorStore(
        index_name=os.getenv("INDEX_NAME"),
        embedding=embeddings,
    )
    # get the chat_template from the hub
    retrieval_qa_chat = hub.pull("langchain-ai/retrieval-qa-chat")
    # create langchain object to help combine relevant vector_store documents
    # along with the chat_template and sending it to the LLM
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=retrieval_qa_chat)

    retrieval_chain = create_retrieval_chain(
        retriever=vector_store.as_retriever(), combine_docs_chain=combine_docs_chain
    )
    result = retrieval_chain.invoke(input={"input": question})
    print(result)
