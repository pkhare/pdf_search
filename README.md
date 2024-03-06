# PDF Search
A short exercise demonstrating a pipeline for document ingestion, search, and query.

Dependencies (also highlighted in requirements.txt):
* PyMuPDF
* chromadb #chromadb==0.4.14
* langchain
* langchain_openai
* protobuf #conda install protobuf 

The two sample PDFs are in **doc_files** folder. First execute **data_process.py** through command line in a suited Python environment. This will process all the **PDF** files in the folder and generate a vector database in the **chroma** folder, using Open AI embeddings (default model "text-embedding-ada-002") and Chroma DB (vector database).

While executing the **data_process.py**, you will need your **Open AI API key** ready for usage. The CLI will ask you to insert your key.

Once data is processed, you execute **make_query.py** to run an a query service in a loop (unless typed _quit_ or forcefully terminated). For this service again you need your **Open AI API key** handy.
