# API Documentation

This API provides functionality to generate embeddings for documents, perform similarity searches, and retrieve search results using FastAPI.

## Installation
Install the required dependencies using pip and the requirements.txt file.

```bash
pip install -r requirements.txt
```

## API Routes

### Hello World

- **Route:** `/`
- **Method:** GET
- **Description:** Returns a greeting message.
- **Example:**
  - **Request:** `GET /`
  - **Response:**
    ```json
    {
        "Hello": "World"
    }
    ```

### Generate Embeddings

- **Route:** `/embed`
- **Method:** POST
- **Description:** Generates embeddings for documents and returns the collection name and extracted text.
- **Parameters:**
  - `docName` (string): Name of the document.
  - `group` (string): Group name for the collection.
  - `userid` (string): User ID associated with the document.
  - `filetype` (string): Type of file (`url` or `file`).
  - `url` (string): URL of the document or file.
- **Example:**
  - **Request:** `POST /embed`
    ```json
    {
        "docName": "Document 1",
        "group": "Group 1",
        "userid": "User 1",
        "filetype": "url",
        "url": "https://example.com/"
    }
    ```
  - **Response:**
    ```json
    {
        "collection_name": "Group 1",
        "extracted_text": "This is the extracted text from the document."
    }
    ```

### Perform Search

- **Route:** `/qsearch`
- **Method:** POST
- **Description:** Performs a similarity search based on the provided query and returns the search results.
- **Parameters:**
  - `query` (string): Search query.
  - `collection_name` (string): Name of the collection to search in.
  - `filter_dict` (object): Filter dictionary for the search.
  - `k` (integer): Number of search results to retrieve.
  - `with_source` (boolean): Whether to include source documents in the search results.
- **Example:**
  - **Request:** `POST /qsearch`
    ```json
    {
        "query": "example query",
        "collection_name": "Group 1",
        "filter_dict": {},
        "k": 10,
        "with_source": true
    }
    ```
  - **Response:**
    ```json
    {
        "result": "Search results"
    }
    ```

## Starting the Gunicorn Server

To start the Gunicorn server with the API, use the following command:

```bash
gunicorn -w 4 -k uvicorn.workers.UvicornWorker app:app --timeout 600
```

- `-w 4` specifies the number of worker processes (adjust as needed).
- `-k uvicorn.workers.UvicornWorker` specifies the worker class to use.
- `main:app` specifies the location of the FastAPI application instance.

Make sure to install the necessary dependencies and provide the required environment variables (`openai_api_key` and `cohere_api_key`) before starting the server.

Please note that you may need to modify the server configuration and other deployment details based on your specific requirements and environment.

---

Feel free to customize the documentation as per your needs, adding more details or examples if necessary.