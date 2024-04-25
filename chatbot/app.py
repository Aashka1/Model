from dotenv import load_dotenv
load_dotenv() ## loading all the environment variables

import streamlit as st
import os
import google.generativeai as genai
from PIL import Image

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

## function to load Gemini Pro model and get repsonses
model=genai.GenerativeModel("gemini-pro") 
chat = model.start_chat(history=[])
def get_gemini_response(question):
    
    response=chat.send_message(question,stream=True)
    return response
def get_gemini_pdfresponse(input,image,prompt):
    response=model.generate_content([input,image[0],prompt])
    return response.text

##initialize our streamlit app

st.set_page_config(page_title="Vision")

st.header("Gemini LLM Application")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []

input=st.text_input("Input: ",key="input")
File=st.file_uploader("choose an image....",type=["jpg","jpeg","png"])
img=""
submit=st.button("Submit")
if File is not None:
    img=Image.open(File)
    st.image(img,caption="File uploaded",use_column_width=True)
send=st.button("Send")
def input_image_setup(uploaded_file):
    # Check if a file has been uploaded
    if uploaded_file is not None:
        # Read the file into bytes
        bytes_data = uploaded_file.getvalue()

        image_parts = [
            {
                "mime_type": uploaded_file.type,  # Get the mime type of the uploaded file
                "data": bytes_data
            }
        ]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")

input_prompt = """
               You are an expert in understanding invoices.
               You will receive input images as invoices &
               you will have to answer questions based on the input image
               """

## If ask button is clicked

if send:
    image_data = input_image_setup(File)
    response=get_gemini_pdfresponse(input_prompt,image_data,input)
    st.subheader("The Response is")
    st.write(response)
    
if submit and input:
    response=get_gemini_response(input)
    # Add user query and response to session state chat history
    st.session_state['chat_history'].append(("User", input))
    st.subheader("Output")
    for chunk in response:
        st.write(chunk.text)
        st.session_state['chat_history'].append(("Model", chunk.text))
st.subheader("History")
    
for role, text in st.session_state['chat_history']:
    st.write(f"{role}: {text}")
    



    
