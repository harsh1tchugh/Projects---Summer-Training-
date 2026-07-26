import os
import random
import io
import re
import requests
import torch
import nltk
import streamlit as st

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from gtts import gTTS
from transformers import pipeline

# Download NLTK datasets quietly
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------
# Page Setup & Customized Styling Theme
# ---------------------------------------------------------
st.set_page_config(
    page_title="Aura AI Studio",
    page_icon="⚡",
    layout="centered"
)

st.markdown("""
<style>
    .stApp {
        background-color: #0B0E14;
        color: #E6EDF3;
    }
    .brand-header {
        text-align: center;
        padding: 1.2rem 0;
        border-bottom: 1px solid #1F293D;
        margin-bottom: 1.8rem;
    }
    .brand-header h1 {
        background: linear-gradient(135deg, #00F2FE 0%, #4FACFE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    .brand-header p {
        color: #7D8B9E;
        font-size: 0.95rem;
        letter-spacing: 0.5px;
    }
    section[data-testid="stSidebar"] {
        background-color: #11151C;
        border-right: 1px solid #1E2430;
    }
    .quick-card {
        background-color: #161B26;
        padding: 0.75rem 1rem;
        border-radius: 8px;
        border-left: 3px solid #00F2FE;
        margin-bottom: 0.6rem;
    }
    .quick-card code {
        color: #00F2FE;
    }
    .footer-credits {
        text-align: center;
        color: #57657A;
        font-size: 0.8rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
        border-top: 1px solid #1E2430;
    }
</style>
""", unsafe_allow_html=True)

RESPONSES = {
    "greeting": [
        "Greetings! How can Aura assist your workflow today?",
        "Hello there! What are we creating next?",
        "Welcome back. Systems online and ready!"
    ],
    "how_are_you": [
        "I'm performing at 100%! All systems operational. How are you doing today?",
        "Doing great and ready to assist! What project are we tackling today?",
        "I'm operating smoothly! Thanks for asking. How can I help you right now?"
    ],
    "who_are_you": [
        "I am Aura, your multi-modal AI assistant capable of text generation, image creation, speech synthesis, and audio transcription."
    ],
    "goodbye": [
        "Goodbye! Have an insightful day ahead.",
        "Session ended. Reach back out anytime!"
    ],
    "help": [
        "Aura can assist with text generation, image creation from prompts, text-to-speech audio synthesis, or audio transcription."
    ],
    "thank": [
        "You're most welcome!",
        "Glad to be of service!",
        "Anytime! Let me know if you need anything else."
    ]
}

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def preprocess(text):
    text = text.lower()
    words = word_tokenize(text)
    return [stemmer.stem(w) for w in words if w.isalnum() and w not in stop_words]

# ---------------------------------------------------------
# Computation Engines & Resource Caching
# ---------------------------------------------------------
DEVICE_ID = 0 if torch.cuda.is_available() else -1

@st.cache_resource
def load_llm_pipeline():
    return pipeline("text-generation", model="distilgpt2", device=DEVICE_ID)

@st.cache_resource
def load_asr_pipeline():
    return pipeline("automatic-speech-recognition", model="openai/whisper-tiny", device=DEVICE_ID)

@st.cache_resource
def load_sd_pipeline():
    from diffusers import StableDiffusionPipeline
    model_id = "runwayml/stable-diffusion-v1-5"
    
    if torch.cuda.is_available():
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16)
        return pipe.to("cuda")
    else:
        pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        return pipe.to("cpu")

# ---------------------------------------------------------
# Online HuggingFace Inference API Handler
# ---------------------------------------------------------
def query_hf_api(prompt):
    API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
    payload = {
        "inputs": f"<|system|>\nYou are Aura, an intelligent and helpful AI assistant. Answer clearly and concisely.</s>\n<|user|>\n{prompt}</s>\n<|assistant|>\n",
        "parameters": {"max_new_tokens": 150, "temperature": 0.7, "return_full_text": False}
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=8)
        if response.status_code == 200:
            res_json = response.json()
            if isinstance(res_json, list) and "generated_text" in res_json[0]:
                return res_json[0]["generated_text"].strip()
    except Exception:
        pass
    return None

# ---------------------------------------------------------
# Intent Parsing Engine
# ---------------------------------------------------------
def detect_intent(text, words):
    text_lower = text.lower().strip()
    
    if text_lower in ["exit", "quit", "close", "stop"]:
        return "EXIT"

    # Vision requests & intent detection
    image_keywords = ["image", "img", "picture", "pic", "photo", "draw", "paint", "render"]
    action_keywords = ["generate", "genrate", "create", "make", "draw", "render"]
    
    is_image_request = any(k in text_lower for k in image_keywords) or \
                       (any(a in text_lower for a in action_keywords) and ("cat" in text_lower or "of" in text_lower))

    if is_image_request:
        return "IMAGE_GEN"
    
    # Audio synthesis requests
    tts_triggers = ["say out loud", "text to speech", "speak", "read this out", "convert to audio", "tts", "convert", "speech", "say"]
    if any(trigger in text_lower for trigger in tts_triggers):
        return "TTS"

    # Natural conversation intents
    if any(phrase in text_lower for phrase in ["how are you", "how r u", "how do you do"]):
        return "HOW_ARE_YOU"
    if any(phrase in text_lower for phrase in ["who are you", "what is your name", "what are you"]):
        return "WHO_ARE_YOU"
    if any(w in words for w in ["hi", "hello", "hey", "greetings"]):
        return "GREETING"
    if "help" in words:
        return "HELP"
    if any(w in words for w in ["bye", "goodby", "see"]):
        return "GOODBYE"
    if "thank" in words or "thanks" in words:
        return "THANK"

    return "LLM_GEN"

def clean_image_prompt(prompt):
    cleaned = re.sub(
        r"^(please\s+)?(can\s+you\s+)?(gen\w*|make|create|draw|paint|render|show)(\s+\w+)*?\s+(an?\s+)?(image|picture|photo|pic|img)?\s*(of\s+)?",
        "", 
        prompt, 
        flags=re.IGNORECASE
    ).strip()
    return cleaned if cleaned else prompt

# ---------------------------------------------------------
# Interface Header & Reset Action
# ---------------------------------------------------------
st.markdown("""
<div class="brand-header">
    <h1>⚡ Aura AI Studio</h1>
    <p>Next-Gen Intelligence Engine • Speech, Vision & Generation</p>
</div>
""", unsafe_allow_html=True)

col_main, col_clear = st.columns([5, 1])
with col_clear:
    if st.button("🧹 Reset", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ---------------------------------------------------------
# Control Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## 🧭 Workspace Navigation")
    st.caption("Interact with multi-modal capabilities via prompts or standard file uploads.")
    
    st.divider()
    
    st.markdown("### 💡 Quick Commands")
    st.markdown("""
    <div class="quick-card">
        <b>Image:</b> <code>render an image of a cybernetic wolf</code>
    </div>
    <div class="quick-card">
        <b>Voice:</b> <code>Say out loud: Systems online</code>
    </div>
    <div class="quick-card">
        <b>Text:</b> <code>Explain quantum computing in simple terms</code>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("### 🎙️ Speech Transcriber")
    uploaded_audio = st.file_uploader("Upload recording (.wav, .mp3, .flac)", type=["wav", "mp3", "m4a", "flac"])
    
    if uploaded_audio is not None:
        if st.button("Process Audio Track", use_container_width=True):
            with st.spinner("Decoding audio via Whisper..."):
                import soundfile as sf
                import librosa
                
                audio_bytes = uploaded_audio.read()
                data, samplerate = sf.read(io.BytesIO(audio_bytes))
                
                # Convert stereo to mono
                if len(data.shape) > 1:
                    data = data.mean(axis=1)

                # Resample to 16000Hz (Whisper native rate) to avoid torchaudio requirement
                if samplerate != 16000:
                    data = librosa.resample(y=data, orig_sr=samplerate, target_sr=16000)
                    samplerate = 16000

                asr = load_asr_pipeline()
                res = asr({"raw": data, "sampling_rate": samplerate})
                
                st.success("Transcription Result:")
                st.info(res["text"])

    st.markdown('<div class="footer-credits">Crafted with precision by Harshit</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# Chat Interface Loop
# ---------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "audio" in message:
            st.audio(message["audio"], format="audio/mp3")
        if "image" in message:
            st.image(message["image"])

# ---------------------------------------------------------
# User Interaction Handler
# ---------------------------------------------------------
user_prompt = st.chat_input("Ask Aura or issue a command...")

if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    tokens = preprocess(user_prompt)
    intent = detect_intent(user_prompt, tokens)
    bot_message = {}

    with st.chat_message("assistant"):
        # 1. Exit Session
        if intent == "EXIT":
            farewell = random.choice(RESPONSES["goodbye"]) + " *(Workspace cleared)*"
            st.markdown(farewell)
            st.session_state.messages = []
            bot_message = {"role": "assistant", "content": farewell}

        # 2. Image Generation Workflow
        elif intent == "IMAGE_GEN":
            target_prompt = clean_image_prompt(user_prompt)
            mode_text = "GPU" if torch.cuda.is_available() else "CPU"
            with st.spinner(f"🎨 Rendering visual artwork for '{target_prompt}' on {mode_text}..."):
                pipe = load_sd_pipeline()
                generated_img = pipe(target_prompt).images[0]
                
                resp_text = f"Visual synthesis complete for: **\"{target_prompt}\"**"
                st.markdown(resp_text)
                st.image(generated_img, use_container_width=True)
                
                bot_message = {"role": "assistant", "content": resp_text, "image": generated_img}

        # 3. Text-To-Speech Synthesis
        elif intent == "TTS":
            with st.spinner("🔊 Generating voice audio..."):
                speech_text = user_prompt
                
                # Carefully strip command prefixes/suffixes without stripping content words like 'to' or 'in'
                speech_text = re.sub(r"^(convert|say out loud|text to speech|speak|read this out|say|tts)\s+", "", speech_text, flags=re.IGNORECASE)
                speech_text = re.sub(r"\s+(to speech|in speech|to audio|in audio)$", "", speech_text, flags=re.IGNORECASE)
                speech_text = speech_text.strip()

                speech_text = speech_text if speech_text else "Audio stream initialized."
                
                tts = gTTS(text=speech_text, lang='en')
                audio_fp = io.BytesIO()
                tts.write_to_fp(audio_fp)
                audio_bytes = audio_fp.getvalue()
                
                resp_text = f"Audio clip generated for: *\"{speech_text}\"*"
                st.markdown(resp_text)
                st.audio(audio_bytes, format="audio/mp3")
                
                bot_message = {"role": "assistant", "content": resp_text, "audio": audio_bytes}

        # 4. Standard Conversational Responses
        elif intent in ["GREETING", "HELP", "HOW_ARE_YOU", "WHO_ARE_YOU", "GOODBYE", "THANK"]:
            key = intent.lower()
            resp_text = random.choice(RESPONSES[key])
            st.markdown(resp_text)
            bot_message = {"role": "assistant", "content": resp_text}

        # 5. Language Model Generation (Smart Online Model + Safe Fallback)
        else:
            with st.spinner("⚡ Aura thinking..."):
                api_response = query_hf_api(user_prompt)
                
                if api_response:
                    clean_reply = api_response
                else:
                    generator = load_llm_pipeline()
                    formatted_prompt = f"Question: {user_prompt}\nAnswer:"
                    outputs = generator(
                        formatted_prompt, 
                        max_new_tokens=60, 
                        do_sample=True, 
                        temperature=0.6, 
                        top_k=40,
                        top_p=0.85,
                        no_repeat_ngram_size=2,
                        pad_token_id=50256,
                        return_full_text=False
                    )
                    raw_reply = outputs[0]["generated_text"].strip()
                    
                    last_punct = max(raw_reply.rfind('.'), raw_reply.rfind('!'), raw_reply.rfind('?'))
                    if last_punct != -1:
                        clean_reply = raw_reply[:last_punct + 1]
                    else:
                        clean_reply = raw_reply

                    if not clean_reply or len(clean_reply) < 5:
                        clean_reply = "I am Aura AI. How can I assist you today?"

                st.markdown(clean_reply)
                bot_message = {"role": "assistant", "content": clean_reply}

    if intent != "EXIT":
        st.session_state.messages.append(bot_message)
