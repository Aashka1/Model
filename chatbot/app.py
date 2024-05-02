from dotenv import load_dotenv
import streamlit as st
import os
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import speech_recognition as sr
import pyttsx3
import sounddevice as sd
import soundfile as sf
import pygame
import time

# Load environment variables
load_dotenv()

# Configure Google API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to load Gemini Pro model and start chat
model_chat = genai.GenerativeModel("gemini-pro")
chat = model_chat.start_chat(history=[])

# Function to get Gemini response for Q&A chatbot
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response

# Function to get Gemini response for image description
def get_gemini_response_image(input, image):
    model = genai.GenerativeModel('gemini-pro-vision')
    if input != "":
        response = model.generate_content([input, image]).text
    else:
        response = model.generate_content(image).text
    return response

# Function to get text from PDF files
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

# Function to create and save vector store
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function to initialize conversational chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Function to handle user input and provide response for PDF chat
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]

# Function to record voice
def record_voice(filename, duration):
    samplerate = 44100  # Adjust as needed
    with st.spinner("Recording..."):
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=2, dtype="float64")
        sd.wait()
        sf.write(filename, recording, samplerate)

# Function to convert speech to text using Google Speech Recognition
def speech_to_text(filename):
    r = sr.Recognizer()
    try:
        with sr.AudioFile(filename) as source:
            audio_data = r.record(source)
            text = r.recognize_google(audio_data)
            return text
    except sr.UnknownValueError:
        return "Speech recognition could not understand audio"
    except sr.RequestError as e:
        return f"Could not request results from Google Speech Recognition service; {e}"
# Streamlit app initialization
st.set_page_config(page_title="Gemini Multi-Purpose App")

# Sidebar for file uploaders and mode selection
st.sidebar.header("Options")
mode = st.sidebar.radio("Mode:", ("Image Description", "Text Chatbot", "Chat with PDF"))

if mode == "Image Description":
    uploaded_file = st.sidebar.file_uploader("Upload Image:", type=["jpg", "jpeg", "png"])
elif mode == "Chat with PDF":
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files:", accept_multiple_files=True)

# Main content area for input prompt, submit button, and chat history
st.header("Gemini Application")
if 'input_prompt' not in st.session_state:
    st.session_state['input_prompt'] = ""
input_prompt = st.text_input("Input Prompt:", key="input", value=st.session_state['input_prompt'])
submit_button = st.button("Submit")

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

# Microphone recording functionality
if st.button("Start Recording"):
    filename = "recorded_audio.wav"
    record_voice(filename, duration=5)  # Adjust duration as needed
    text = speech_to_text(filename)
    st.success("Recording Completed!")
    st.session_state['input_prompt'] = text  # Update main input prompt with recorded text

# Main app code
if submit_button:
    # Add user query to chat history
    st.session_state['chat_history'].append(("You", input_prompt))

    if mode == "Image Description" and uploaded_file:
        image = Image.open(uploaded_file)
        response = get_gemini_response_image(input_prompt, image)
        st.subheader("Response:")
        st.write(response)
        # Add bot response to chat history
        st.session_state['chat_history'].append(("Bot", response))
    elif mode == "Text Chatbot":
        if input_prompt.strip():  # Check if there's text input
            response = get_gemini_response(input_prompt)
            st.subheader("Response:")
            for chunk in response:
                st.write(chunk.text)
                # Add bot response to chat history
                st.session_state['chat_history'].append(("Bot", chunk.text))
        else:  # Convert voice input to text
            try:
                filename = "recorded_audio.wav"
                record_voice(filename, duration=5)  # Record for 5 seconds
                text = speech_to_text(filename)
                st.subheader("Voice Input:")
                st.write(text)
                st.session_state['input_prompt'] = text  # Update main input prompt with recorded text
                response = get_gemini_response(text)
                st.subheader("Response:")
                for chunk in response:
                    st.write(chunk.text)
                    # Add bot response to chat history
                    st.session_state['chat_history'].append(("Bot", chunk.text))
            except Exception as e:
                st.error(f"An error occurred: {e}")
    elif mode == "Chat with PDF" and pdf_docs:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        get_vector_store(text_chunks)
        response = user_input(input_prompt)
        st.subheader("Response:")
        st.write(response)
        # Add bot response to chat history
        st.session_state['chat_history'].append(("Bot", response))

# Display chat history
st.subheader("Chat History")
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
