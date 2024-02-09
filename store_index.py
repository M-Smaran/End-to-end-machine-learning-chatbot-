from src.helper import load_PDF,text_split,download_hugging_face_embedding
import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY") 
PINECONE_API_ENV = os.environ.get("PINECONE_API_ENV")


#print(PINECONE_API_KEY)
#print(PINECONE_API_ENV)

extracted_data = load_PDF("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embedding()

#Initializing the pinecone
pinecone.init(api_key=PINECONE_API_KEY,environment = PINECONE_API_ENV)

#Creating Embeddings for Each of the Text Chunks and storing 
index_name = "medical-chatbot"
docsearch = Pinecone.from_texts([t.page_content for t in text_chunks],embeddings,index_name=index_name)


