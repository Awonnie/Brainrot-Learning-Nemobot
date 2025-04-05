# Brainrot Learning Chatbot

Upload your lecture slides (PDF or PPTX), and get a funny, conversational-style summary between a blur sotong and a chao mugger — perfect for brainrot learning and memory reinforcement. The chatbot even speaks to you using real voices, powered by ElevenLabs!

---

## Tech Stack

- **FastAPI** — backend server  
- **Vertex AI (Gemini 1.0 Pro)** via `langchain-google-vertexai`  
- **LangChain** — document parsing + prompt chaining  
- **Jinja2** — frontend templating  
- **PyPDFLoader / python-pptx** — PDF and PPTX file parsing  
- **Pydub** — audio segment stitching  
- **ElevenLabs API** — realistic voice synthesis  

---

## Features

- Upload any **PDF** or **PPTX** lecture file  
- Extracts slide content automatically  
- Generates a full **chat-style explanation** between:
  - **Blur Sotong** (confused student)
  - **Chao Mugger** (hardworking student)
- Automatically generates **realistic voice audio** for both characters  
- Stitches audio into a **playable conversation podcast**

---

## Local Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/yourusername/brainrot-learning-chatbot.git
cd brainrot-learning-chatbot
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # or `.\venv\Scripts\activate` on Windows
```

### 3. Install all dependencies
```bash
pip install -r requirements.txt
```

> If you don't have a `requirements.txt`, use:

```bash
pip install fastapi uvicorn jinja2 aiofiles PyPDF2 \
langchain langchain-community langchain-google-vertexai \
google-cloud-aiplatform
```

### 4. Install FFmpeg (required for audio merging)
```bash
brew install ffmpeg
```
> or

```bash
sudo apt install ffmpeg
```
---

## ☁️ Vertex AI Setup (Required for LLM)

### 1. Install Google Cloud SDK
If you don’t have `gcloud` installed:

```bash
brew install --cask google-cloud-sdk
```

Or follow the manual installer:  
https://cloud.google.com/sdk/docs/install

### 2. Authenticate your machine
```bash
gcloud auth application-default login
```

This opens a browser. Sign in with your Google account that has Vertex AI enabled.

### 3. Set your Google Cloud project
Your project info:

- **Project ID:** `brainrot-learning-4052`
- **Region:** `us-central1`

In your `app.py`, initialize Vertex AI like so:

```python
from google.cloud import aiplatform

aiplatform.init(
    project="brainrot-learning-4052",
    location="us-central1"
)
```

### 4. ElevenLabs Setup

1. Create an account: https://www.elevenlabs.io
2. Get your API key
3. Set it as an environment variable:

In your `.env` file, declare this:
```bash
export ELEVENLABS_API_KEY="your_key_here"
```

---

## ▶️ Running the App Locally

```bash
python app.py
```

## ▶️ Deploying the App on Google Cloud

```bash
gcloud app deploy
```
```bash
gcloud app browse
```

## Folder Structure

```
.
├── app.py
├── static/
│   ├── docs/        # Uploaded files
│   ├── audio/       # Generated MP3 clips & final stitched audio
│   └── output/      # Output JSON of chat
├── templates/
│   └── brainrot.html
├── requirements.txt
└── README.md
```

---

## 🤝 Contributors

- Angie Wong
- Guo Chenrui
- Keith Heng
