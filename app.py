import urllib.request
import os
from langchain.document_loaders import UnstructuredURLLoader, UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import CohereEmbeddings
from langchain.llms import Cohere
from qdrant_client import QdrantClient
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CohereRerank
from langchain.chains import RetrievalQA


from dotenv import load_dotenv
load_dotenv()


cohere_api_key = os.environ.get('cohere_api_key')

def download_file(url, user_id):
    # Path to the local mounted folder on the Azure VM
    folder_path = f'/home/azureuser/mydrive/user_files/{user_id}/'

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
        url_list = kwargs['url']
        loader = UnstructuredURLLoader(urls=url_list)
    else:
        save_path = download_file(kwargs['url'], userid)
        loader = UnstructuredFileLoader(save_path, mode="elements")
    docs = loader.load()
    # print(docs)
    return docs, loader

def generate_embeddings(docName, group, userid, filetype, **kwargs):
    docs, loader = load_docs(filetype, userid, **kwargs)
    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=30)
    texts = text_splitter.split_documents(docs)

    for i, text in enumerate(texts):
        text.metadata['docName'] = docName
        text.metadata['group'] = group
    
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)


    qdrant = Qdrant.from_documents(texts, embeddings, host='localhost', collection_name=group, prefer_grpc=True)
    joined_content = ' '.join([docu.page_content for docu in docs])
    print("embed complete")
    return qdrant.collection_name, joined_content



def qdrant_search_completion(query, collection_name, filter_dict,k, with_source):

    client = QdrantClient("localhost", prefer_grpc=True)
    embeddings = CohereEmbeddings(model="multilingual-22-12", cohere_api_key=cohere_api_key)
    compressor = CohereRerank(top_n=4)
    print(collection_name)
    qdrant = Qdrant(client,collection_name=collection_name, embeddings=embeddings)
    retriever = qdrant.as_retriever()
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=retriever
    )
    chain = RetrievalQA.from_chain_type(
        llm=Cohere(temperature=0.75, cohere_api_key=cohere_api_key, model="command-nightly", max_tokens="2000", truncate="END"), 
        retriever=compression_retriever,
        return_source_documents=with_source)

    docs = qdrant.similarity_search(query=query, k=k, filter=filter_dict)
    print(docs)
    result = chain({"input_documents": docs, "query": query}, return_only_outputs=True)    
    return result


from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get('/')
def hello_world():
    return {"Hello": "World"}

@app.post('/embed')
async def embed(request: Request):
    data = await request.json()
    docName = data.get("docName")
    group = data.get("group")
    userid = data.get("userid")
    filetype = data.get("filetype")
    url = data.get("url")
    # s3_path = data.get("s3_path")

    collection_name, docs = generate_embeddings(docName, group,  userid, filetype, url=url)
    return {"collection_name": collection_name, "extracted_text":docs}

@app.post('/qsearch')
async def search(request: Request):
    data = await request.json()
    query = data.get("query")
    collection_name = data.get("collection_name")
    filter_dict = data.get("filter_dict")
    k = data.get("k")
    with_source = data.get("with_source")

    search_result = qdrant_search_completion(query, collection_name, filter_dict, k, with_source)
    return search_result
