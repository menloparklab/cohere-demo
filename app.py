import urllib.request
import os
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings
from langchain.chains import VectorDBQAWithSourcesChain, VectorDBQA
from langchain.llms import Cohere
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.chains.question_answering import load_qa_chain
from qdrant_client import QdrantClient
from langchain.docstore.document import Document
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import RetrievalQA

def download_file(url, user_id):
    # Path to the local mounted folder on the Azure VM
    folder_path = f'/content/{user_id}/'

    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)

    # Filename for the downloaded file
    filename = url.split('/')[-1]

    # Full path to the downloaded file
    file_path = os.path.join(folder_path, filename)

    # Download the file and save it to the local folder
    urllib.request.urlretrieve(url, file_path)

    print(f'Successfully downloaded file from {url} to {file_path}')
    
    return file_path

def load_docs(filetype, userid, **kwargs):
    if filetype == 'url':
        #need to provide a list of urls to the next step
        url_list = [kwargs['input_url']]
        loader = UnstructuredURLLoader(url_list)
    else:
        save_path = download_file(kwargs['s3_path'], userid)
        loader = UnstructuredFileLoader(save_path, mode="elements")
    docs = loader.load()
    return docs, loader

def generate_embeddings(scope, department, userid, filetype, **kwargs):
    docs, loader = load_docs(filetype, userid, **kwargs)
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
    texts = text_splitter.split_documents(docs)

    # for i, text in enumerate(texts):
    #     text.metadata['scope'] = scope
    #     text.metadata['department'] = department
    
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)

    qdrant = Qdrant.from_documents(
        texts,
        embeddings,
        location=":memory:",
        collection_name="cohere_docs",
    )

    # qdrant = Qdrant.from_documents(texts, embeddings, host='localhost', collection_name=userid, prefer_grpc=True)

    return qdrant.collection_name



def qdrant_search_completion(query, collection_name, filter_dict,k,with_source):

    client = QdrantClient("localhost", prefer_grpc=True)
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    compressor = CohereRerank()
    qdrant = Qdrant(client,collection_name, embeddings)
    retriever = qdrant.as_retriever()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    chain = RetrievalQA.from_chain_type(
        llm=Cohere(temperature=0, cohere_api_key=cohere_api_key), 
        retriever=compression_retriever,
        return_source_documents=True)

    docs = qdrant.similarity_search(query=query, k=k, filter=filter_dict, embedding_func=embeddings.embed_query, collection_name=collection_name,client=client)
    chain = load_qa_with_sources_chain(OpenAI(temperature=0,openai_api_key=openai_api_key), chain_type="stuff")
    result = chain({"input_documents": docs, "question": query}, return_only_outputs=False)

    result = chain({"input_documents": docs, "query": query}, return_only_outputs=False)    
    return result