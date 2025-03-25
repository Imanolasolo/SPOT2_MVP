import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
import os

st.set_page_config(page_title="Spot2 Conversational AI", page_icon="🏡", layout="wide")

# Configure OpenAI API Key from secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Load and process the PDF with property examples
pdf_loader = PyPDFLoader("properties.pdf")
documents = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Create a vector index with FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Configure the conversational model with memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Initial state for required fields
if "conversation_data" not in st.session_state:
    st.session_state.conversation_data = {}
if "missing_fields" not in st.session_state:
    st.session_state.missing_fields = ["budget", "size", "type", "city"]

# Function to check missing fields
def check_missing_fields():
    return [key for key in ["budget", "size", "type", "city"] if key not in st.session_state.conversation_data]

# Streamlit UI
st.title("🏡 Spot2 Conversational AI")
st.write("Chat with me to find the best commercial property for you.")

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
txt = st.chat_input("Hi, I'm Spoty, your assistant. Write your message here...")

if txt:
    # Display the user's message
    st.session_state.messages.append({"role": "user", "content": txt})
    with st.chat_message("user"):
        st.markdown(txt)

    # Check if there are missing fields
    if st.session_state.missing_fields:
        current_field = st.session_state.missing_fields[0]
        st.session_state.conversation_data[current_field] = txt
        st.session_state.missing_fields = check_missing_fields()

        if st.session_state.missing_fields:
            next_field = st.session_state.missing_fields[0]
            response = f"Thank you. Now, please provide your {next_field}."
        else:
            # All fields are complete, search the database
            query = (
                f"I'm looking for a property with the following characteristics:\n"
                f"Budget: {st.session_state.conversation_data['budget']} USD\n"
                f"Size: {st.session_state.conversation_data['size']} m²\n"
                f"Type: {st.session_state.conversation_data['type']}\n"
                f"City: {st.session_state.conversation_data['city']}\n"
            )
            response = qa_chain.run(query)

            # Check if there are results
            if "No matching properties" in response or not response.strip():
                response += "\nSorry, we couldn't find a property with those characteristics in our database. Please contact our team at contacto@empresa.com for further assistance."
    else:
        # If there are no missing fields, process directly
        response = qa_chain.run(txt)

    # Display the chatbot's response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
