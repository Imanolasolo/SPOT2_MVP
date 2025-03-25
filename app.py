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

# Configurar API Key de OpenAI desde secrets
OPENAI_API_KEY = st.secrets["openai"]["api_key"]
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Cargar y procesar el PDF con ejemplos de propiedades
pdf_loader = PyPDFLoader("properties.pdf")
documents = pdf_loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Crear un índice vectorial con FAISS
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(texts, embeddings)

# Configurar el modelo conversacional con memoria
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo"),
    retriever=vectorstore.as_retriever(),
    memory=memory
)

# Estado inicial para los campos requeridos
if "conversation_data" not in st.session_state:
    st.session_state.conversation_data = {}
if "missing_fields" not in st.session_state:
    st.session_state.missing_fields = ["budget", "size", "type", "city"]

# Función para verificar campos faltantes
def check_missing_fields():
    return [key for key in ["budget", "size", "type", "city"] if key not in st.session_state.conversation_data]

# UI en Streamlit
st.title("🏡 Spot2 Conversational AI")
st.write("Chatea conmigo para encontrar la mejor propiedad comercial para ti.")

# Historial de chat
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Entrada de usuario
txt = st.chat_input("Hola, soy Spoty, tu asistente,escribe tu mensaje aquí...")

if txt:
    # Mostrar el mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": txt})
    with st.chat_message("user"):
        st.markdown(txt)

    # Verificar si faltan campos
    if st.session_state.missing_fields:
        current_field = st.session_state.missing_fields[0]
        st.session_state.conversation_data[current_field] = txt
        st.session_state.missing_fields = check_missing_fields()

        if st.session_state.missing_fields:
            next_field = st.session_state.missing_fields[0]
            response = f"Gracias. Ahora, por favor proporciona tu {next_field}."
        else:
            # Todos los campos están completos, buscar en la base de datos
            query = (
                f"Estoy buscando una propiedad con las siguientes características:\n"
                f"Presupuesto: {st.session_state.conversation_data['budget']} USD\n"
                f"Tamaño: {st.session_state.conversation_data['size']} m²\n"
                f"Tipo: {st.session_state.conversation_data['type']}\n"
                f"Ciudad: {st.session_state.conversation_data['city']}\n"
            )
            response = qa_chain.run(query)

            # Verificar si hay resultados
            if "No matching properties" in response or not response.strip():
                response += "\nLo siento, no encontramos una propiedad con esas características en nuestra base de datos. Por favor, contacta a nuestro equipo en contacto@empresa.com para más ayuda."
    else:
        # Si no hay campos faltantes, procesar directamente
        response = qa_chain.run(txt)

    # Mostrar la respuesta del chatbot
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant"):
        st.markdown(response)
