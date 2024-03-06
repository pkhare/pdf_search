import argparse
from dataclasses import dataclass
from langchain_community.vectorstores.chroma import Chroma
#from langchain.embeddings import OpenAIEmbeddings
#from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
#from langchain.chat_models import ChatOpenAI
#from langchain_community.chat_models import ChatOpenAI
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
import os
import getpass
import sys


VECTORDB_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context, if the answer is not in the context then say that you do not have enough information on this:

{context}

---

Answer the question based on the above context: {question}
"""


def main():
    # Create CLI.
    while True:
        sys.stdout.write('Input Query (type "quit" to end)> ')
        query_text = input()
        if query_text.lower() == 'quit':
            break
    
        #SAMPLE QUERIES
        #query_text = "Who are the authors of the paper Attention Is All You Need?"
        #query_text = "How many identical layers does encode have?"
        #query_text = "What is the complexity per layer in self-attention?"
        #query_text = "What functions are used for positional embedding?"
        #query_text = "How to unlock the doors of the vehicle/car?"
        #query_text = "How to unlock the doors of the vehicle/car with remote?"
        #query_text = "How to adjust seats?"
        #query_text = "Where is the horn present?"
        #query_text = "How to increase the speed of the wiper?"
        #query_text = "How to handle manual transmission of the vehicle?"

        # Prepare the DB.
        embedding_function = OpenAIEmbeddings()
        db = Chroma(persist_directory=VECTORDB_PATH, embedding_function=embedding_function)

        # Search the DB.
        results = db.similarity_search_with_relevance_scores(query_text, k=3)
        #print(results)
        if len(results) == 0 or results[0][1] < 0.5:
            print(f"Unable to find matching results.")

        else:
            context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
            prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
            prompt = prompt_template.format(context=context_text, question=query_text)
            #print(prompt)

            model = ChatOpenAI()
            #response_text = model.predict(prompt)
            response_text = model.invoke(prompt)

            sources = set([doc.metadata.get("file", None) for doc, _score in results])
            #formatted_response = f"Response: {response_text}\nSources: {sources}"
            formatted_response = f"Response: {response_text.content}\nSources: {sources}"
            print(formatted_response)


if __name__ == "__main__":
    os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
    main()