import base64
import soundfile as sf
import requests
import json
import textwrap
import chromadb
import numpy as np
import pandas as pd
import os
import google.generativeai as genai
import google.ai.generativelanguage as glm
import gradio as gr
from chromadb import Documents, EmbeddingFunction, Embeddings
API_KEY1=os.getenv("API_KEY1") 
genai.configure(api_key=API_KEY1)
API_KEY=os.getenv("API_KEY")
API_URL=os.getenv("API_URL")
def audio_to_base64(audio):
    sr, data = audio
    # Save audio data to a temporary file
    temp_file = "temp.wav"
    sf.write(temp_file, data, sr, format='wav')

    # Read the temporary file as binary and encode it to base64
    with open(temp_file, "rb") as audio_file:
        base64_audio = base64.b64encode(audio_file.read()).decode("utf-8")

    # Remove the temporary file
    os.remove(temp_file)

    response_text = send_to_api(base64_audio)
    response_json = json.loads(response_text)
    output_text = response_json["output"]["segments"][0]["text"]

    final_answer=qna(output_text)
    return final_answer

def send_to_api(base64_audio):
    payload = {
        "input": {
            "audio_base64": base64_audio,
            "model": "medium",
            "transcription": "plain text",
            "translate": True,
            "language": "en",
            "temperature": 0,
            "best_of": 5,
            "beam_size": 5,
            "patience": 1,
            "suppress_tokens": "-1",
            "condition_on_previous_text": False,
            "temperature_increment_on_fallback": 0.2,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1,
            "no_speech_threshold": 0.6,
            "word_timestamps": False,
            "initial_prompt": "You are a voice assistant for Bhuvan Portal , Users will use indian english and indian accent and sometimes hindi words , some mispelt words - ruin - Bhuvan , Amritsar , Aadhaar , Anganwadi , Ganga , Pradhan Mantri , Awas , Yojna , Gram , Sadak "
        },
        "enable_vad": True
    }

    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": API_KEY
    }

    response = requests.post(API_URL, json=payload, headers=headers)
    print("done")

    return response.text

def read_text_files_in_directory(directory_path):
    documents = []

    try:
        # List all files in the directory
        files = [f for f in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, f))]

        for file_name in files:
            file_path = os.path.join(directory_path, file_name)

            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    documents.append(content)
            except Exception as e:
                print(f"An error occurred for {file_path}: {str(e)}")

    except FileNotFoundError:
        print(f"Directory not found: {directory_path}")

    return documents

directory_path = 'Bhuvan'
documents = read_text_files_in_directory(directory_path)

# Now 'documents' is a list containing the content of each text file in the "dataset" directory.
class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/embedding-001'
    title = "Custom query"
    return genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)["embedding"]
def create_chroma_db(documents, name):
  chroma_client = chromadb.Client()
  db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())

  for i, d in enumerate(documents):
    db.add(
      documents=d,
      ids=str(i)
    )
  return db
db = create_chroma_db(documents, "bhuvan1")
def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage
def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  prompt = ("""You are a helpful and informative chat bot that answers questions regarding navigating the various Bhuvan website sub sites using text from the reference passage included below. Bhuvan is a web-based platform that provides access to a wide range of geospatial data, including satellite imagery, maps, and GIS data. It is used by a variety of users, including government agencies, businesses, and individuals. Bhuvan is a valuable resource for anyone who needs to access or use Indian geospatial data.\
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  strike a friendly and converstional tone. \
  make sure to provide a website link with the answer , the website link should be related to the answer \
  if the user is asking a very niche topic tell them to check https://bhuvan-app1.nrsc.gov.in/sitemap/ , this website has all the Sub-sites listed and indexed \
  if the user wants support or wants to discuss any topic ask them to check https://bhuvan.nrsc.gov.in/forum/ , this is a forum where the user can disuss all things related to bhuvan with experts and peers \
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)
  return prompt
def qna(query):
  prompt_ini=query
  context=get_relevant_passage(prompt_ini,db)
  prompt=make_prompt(prompt_ini,context)
  print("done2")
  model = genai.GenerativeModel('gemini-pro')
  answer = model.generate_content(prompt)
  return answer.text

demo = gr.Interface(
    fn=audio_to_base64,
    inputs=["microphone"],
    outputs="text",
    title="Voice Assistant for Bhuvan **Team UnderGod** (Submit Twice If Error Pops Up!)",
    description="Speak into the microphone and see the quick response!. (Faster Whisper + Google Gemini + Vector Embeddings of Bhuvan Websites)",
    theme='WeixuanYuan/Soft_dark'
)

if __name__ == "__main__":
    demo.launch()


