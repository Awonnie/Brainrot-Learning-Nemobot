from fastapi import FastAPI, Request, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse

from langchain.docstore.document import Document
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain.prompts import PromptTemplate
from langchain_google_vertexai import VertexAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

from google.cloud import aiplatform
from pptx import Presentation
from elevenlabs.client import ElevenLabs
from pydub import AudioSegment

import os
import json
import aiofiles
import requests
import uuid
import uvicorn
import re

# === Vertex AI Init ===
print("[Startup] Initializing Vertex AI...")
aiplatform.init(
    project="brainrot-learning-4052",
    location="us-central1"
)
print("[Startup] Vertex AI initialized ✅")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# === ElevenLabs Voice Settings ===
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
client = ElevenLabs(api_key=ELEVENLABS_API_KEY)

VOICE_IDS = {
    "Blur Sotong": "cgSgspJ2msm6clMCkdW9",
    "Chao Mugger": "CwhRBWXzGAHq8TQ4Fs17"
}

# === Speech Synthesis ===
def synthesize_speech(text, speaker):
    voice_id = VOICE_IDS.get(speaker)
    if not voice_id:
        print(f"[ERROR] No voice ID found for speaker: {speaker}")
        return None

    try:
        audio_gen = client.text_to_speech.convert(
            voice_id=voice_id,
            model_id="eleven_multilingual_v2",
            text=text,
            output_format="mp3_44100_128"
        )

        audio_bytes = b"".join(audio_gen)
        audio_path = f"static/audio/{uuid.uuid4()}.mp3"
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        return audio_path
    except Exception as e:
        print(f"[ERROR] ElevenLabs synthesis failed: {e}")
        return None

# === Vertex AI LLM ===
ques_parameters = {
    "model_name": "gemini-1.0-pro",
    "max_output_tokens": 2048,
    "temperature": 0.3,
    "top_p": 0.8,
    "top_k": 40,
    "verbose": True,
}
question_llm = VertexAI(**ques_parameters)

# === Helper: Extract .pptx text ===
def extract_pptx_text(filepath):
    prs = Presentation(filepath)
    return "\n".join(shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text"))

# === Generate Brainrot Conversation ===
def generate_brainrot_convo(file_path):
    print(f"[Brainrot] Processing file: {file_path}")

    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

    elif file_path.endswith(".pptx"):
        raw_text = extract_pptx_text(file_path)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        docs = text_splitter.create_documents([raw_text])

    else:
        raise ValueError("Unsupported file type. Please upload a .pdf or .pptx file.")

    convo_prompt_template = """
You are creating a fun and educational conversation between two students:
- Blur Sotong: A very confused student who doesn't understand anything.
- Chao Mugger: A hardworking student who explains concepts clearly and helpfully.

They are studying the following lecture notes together:
------------
{text}
------------

Generate a *natural-sounding*, back-and-forth conversation like this:

Blur Sotong: [confused question]
Chao Mugger: [clear explanation]
Blur Sotong: [follow-up confusion or curiosity]
Chao Mugger: [further clarification]

DO NOT include scene directions or markdown syntax like ** or ##.
Only produce lines that begin with either "Blur Sotong:" or "Chao Mugger:", followed by text.
End the conversation when you have explained all the key ideas.
"""

    prompt = PromptTemplate(template=convo_prompt_template, input_variables=["text"])
    conversation_data = []

    for i, chunk in enumerate(docs):
        print(f"[LLM] Processing chunk {i+1}/{len(docs)}")
        try:
            formatted_prompt = prompt.format(text=chunk.page_content)
            response = question_llm.invoke(formatted_prompt)

            lines = [line.strip() for line in response.split("\n") if ":" in line]
            for line in lines:
                speaker, message = line.split(":", 1)
                speaker = speaker.strip()
                message = message.strip()
                clean_speaker = re.sub(r"^[#\s*]+|[\s*]+$", "", speaker)
                audio_path = synthesize_speech(message, clean_speaker)
                conversation_data.append({
                    "speaker": clean_speaker,
                    "text": message,
                    "audio_path": audio_path
                })
        except Exception as e:
            print(f"[ERROR] Failed on chunk {i+1}: {e}")
            continue

    return conversation_data

@app.get("/favicon.ico")
async def favicon():
    return FileResponse("static/favicon.ico")

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("brainrot.html", {"request": request})

@app.post("/brainrot_chatbot")
async def brainrot_chatbot(request: Request, pdf_file: UploadFile = File(...)):
    base_folder = "static/docs/"
    os.makedirs(base_folder, exist_ok=True)
    os.makedirs("static/audio/", exist_ok=True)

    file_path = os.path.join(base_folder, pdf_file.filename)
    async with aiofiles.open(file_path, "wb") as f:
        await f.write(await pdf_file.read())

    convo_output = generate_brainrot_convo(file_path)

    with open("static/output/brainrot_convo.json", "w", encoding="utf-8") as f:
        json.dump({"conversation": convo_output}, f, indent=2)

    return templates.TemplateResponse("brainrot.html", {
        "request": request,
        "conversation": convo_output
    })

@app.get("/generate_full_audio")
async def generate_full_audio():
    print("[🔊] Generating full podcast-style audio with alternating voices...")

    json_path = "static/output/brainrot_convo.json"
    with open(json_path, "r", encoding="utf-8") as f:
        convo_data = json.load(f)

    conversation = convo_data.get("conversation", [])
    podcast_audio = AudioSegment.empty()

    for idx, entry in enumerate(conversation):
        audio_path = entry.get("audio_path")
        if audio_path and os.path.exists(audio_path):
            print(f"[🎧] Adding clip {idx+1} to final audio: {audio_path}")
            segment = AudioSegment.from_mp3(audio_path)
            podcast_audio += segment + AudioSegment.silent(duration=300)

    final_audio_path = "static/audio/brainrot_full_audio.mp3"
    podcast_audio.export(final_audio_path, format="mp3")

    return {
        "status": "success",
        "message": "Podcast audio generated with alternating voices.",
        "audio_file": final_audio_path
    }

if __name__ == "__main__":
    print("[Startup] Launching FastAPI server...")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
