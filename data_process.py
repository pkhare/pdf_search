from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
import os
import shutil
import time
import fitz
import getpass

class data_create:

  def __init__(self) -> None:

    self.VECTORDB_PATH = "./chroma"
    self.DATA_PATH = "./doc_files/"
    self.main()

  def main(self):
      self.generate_data_store()

  def generate_data_store(self):

      pdf_files = []
      for file in os.listdir(self.DATA_PATH):
        if file.endswith(".pdf"):
          pdf_files.append(os.path.join(self.DATA_PATH, file))
      print(pdf_files)
      chunks = []
      for pdf_file in pdf_files:
        doc = fitz.open(pdf_file)
        if doc.page_count < 1:
          print("ERROR",pdf_file,doc.page_count)
          continue
        text = ""
        for i in range(doc.page_count):
          page = doc.load_page(i)
          text += page.get_text()

        text = text.replace("\n"," ")

        chunks.extend(self.split_text(text, pdf_file))
        print(pdf_file)
      print("LEN",len(chunks))
      self.vectorise_in_chroma(chunks)

  def split_text(self, text: str, file_name:str):
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=400,
          chunk_overlap=100,
          length_function=len,
          add_start_index=True,
      )

      chunks = text_splitter.split_text(text)

      print(f"Split into {len(chunks)} chunks.")

      chunksDocs = []
      for chunk in chunks:
        doc =  Document(page_content=chunk, metadata={"file": file_name})
        chunksDocs.append(doc)

      return chunksDocs


  def vectorise_in_chroma(self, chunks: list[Document]):
      # Clear out the database first.
      if os.path.exists(self.VECTORDB_PATH):
        shutil.rmtree(self.VECTORDB_PATH)

      time.sleep(5)

      # Create a new DB from the documents.
      db = Chroma.from_documents(
          chunks, OpenAIEmbeddings(), persist_directory=self.VECTORDB_PATH
      )
      db.persist()
      print(f"Saved {len(chunks)} chunks to {self.VECTORDB_PATH}.")

if __name__ == "__main__":
   os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
   dataInitiate = data_create()