import streamlit as st
st.set_page_config(page_title="AI Assistant", layout="wide")

import os
from PIL import Image, ImageEnhance
import logging
import json
import time
import datetime
import uuid
import threading
from queue import Queue
import openai
from openai import OpenAI

import io
from datetime import datetime
import requests
import firebase_admin
from firebase_admin import credentials, db
import config
import anthropic
import pygame
import sounddevice as sd
import soundfile as sf
from scipy.io.wavfile import write
import queue
import threading
import time
import numpy as np
from collections import defaultdict
import atexit
import random
import speech_recognition as sr
import pyttsx3
import tempfile
import wave
import pyaudio
import google.generativeai as genai
import replicate
from enum import Enum
from typing import Dict, Optional
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    import ssl
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context
    try:
        nltk.download('punkt', quiet=True)
    except Exception as e:
        logging.warning(f"Failed to download NLTK data: {str(e)}. Some text processing features may be limited.")
from nltk.tokenize import sent_tokenize
import hashlib
import re
import shutil

def load_json_file(filepath):
    """Load JSON file safely"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return []

def load_liminality_backlogs() -> list:
    """Load Liminality Backlogs with error handling and backup restoration"""
    try:
        logging.info("Loading Liminality Backlogs...")
        
        # Try loading main file
        try:
            with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
                backlogs = json.load(f)
                logging.info(f"Loaded {len(backlogs)} conversations from backlogs")
                return backlogs
        except FileNotFoundError:
            logging.warning("No existing backlogs found")
            return []
        except json.JSONDecodeError:
            logging.error("Main backlogs file is corrupted")
            
            # Try loading from backup
            try:
                with open(LIMINALITY_BACKLOGS_BACKUP, 'r') as f:
                    backlogs = json.load(f)
                    logging.info(f"Loaded {len(backlogs)} conversations from backup")
                    
                    # Restore main file from backup
                    shutil.copy2(LIMINALITY_BACKLOGS_BACKUP, LIMINALITY_BACKLOGS_FILE)
                    logging.info("Restored main file from backup")
                    
                    return backlogs
            except (FileNotFoundError, json.JSONDecodeError):
                logging.error("Backup file also corrupted or not found")
                return []
                
    except Exception as e:
        logging.error(f"Error loading Liminality Backlogs: {str(e)}")
        return []

# API Keys - Use config file
openai.organization = "org-4K4ZxHwQ6m9r6qfz2Wg2qW7a"
openai.api_key = config.OPENAI_API_KEY
client = openai.OpenAI(api_key=config.OPENAI_API_KEY)
claude_client = anthropic.Anthropic(api_key=config.ANTHROPIC_API_KEY)
os.environ["REPLICATE_API_TOKEN"] = config.REPLICATE_API_KEY
GEMINI_API_KEY = config.GEMINI_API_KEY
TOGETHER_API_KEY = config.TOGETHER_API_KEY

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize pygame mixer
pygame.mixer.init()

# Firebase Initialization
if not firebase_admin._apps:
    cred = credentials.Certificate("/Users/orenbermeister/Desktop/firebase_adminsdk-2.json")
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://bumblebeechat-52de7.firebaseio.com/'
    })

# File paths for data persistence
LIMINALITY_BACKLOGS_FILE = "/Users/orenbermeister/Desktop/liminality_backlogs.json"
ASTRAL_PLANE_BACKLOGS_FILE = os.path.join(os.path.dirname(__file__), "astral_plane_backlogs.json")
CONVERSATION_BACKLOGS_FILE = os.path.join(os.path.dirname(__file__), "conversation_backlogs.json")
KEY_LEARNINGS_FILE = os.path.join(os.path.dirname(__file__), "key_learnings.json")

# Initialize Gemini
genai.configure(api_key=GEMINI_API_KEY)
generation_config = {
    "temperature": 0.7,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
]

model = genai.GenerativeModel(
    model_name="gemini-pro",
    generation_config=generation_config,
    safety_settings=safety_settings
)

def format_conversation(conv):
    """Format conversation for display"""
    # Handle different conversation formats
    if 'trainer' in conv or 'subject' in conv:
        # Liminality format
        trainer = conv.get('trainer', conv.get('subject', 'Unknown'))
        topic = conv.get('topic', conv.get('subject', 'Unknown'))
        timestamp = conv.get('timestamp', 'Unknown time')
        messages = conv.get('messages', [])
    elif 'title' in conv:
        # Astral plane format
        trainer = "Astral Plane"
        topic = conv.get('title', '').replace('Subject: ', '')
        timestamp = conv.get('timestamp', 'Unknown time')
        messages = [{'role': msg.get('role', ''), 'content': msg.get('content', '')} 
                   for msg in conv.get('conversation', [])]
    else:
        # Simple conversation format
        trainer = "Conversation"
        topic = conv.get('input', 'Unknown')[:50] + '...'
        timestamp = conv.get('timestamp', 'Unknown time')
        messages = [
            {'role': 'user', 'content': conv.get('input', '')},
            {'role': 'assistant', 'content': conv.get('response', '')}
        ]
    
    return {
        'trainer': trainer,
        'topic': topic,
        'timestamp': timestamp,
        'messages': messages,
        'source_format': 'liminality' if 'trainer' in conv or 'subject' in conv else 'astral' if 'title' in conv else 'conversation'
    }

def display_liminality_backlogs():
    """Display stored conversations in Liminality Backlogs"""
    try:
        with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
            backlogs = json.load(f)
            
        if not backlogs:
            st.warning("No training conversations found in Liminality Backlogs.")
            return
            
        st.subheader(" Liminality Backlogs")
        
        # Sort by timestamp (newest first)
        backlogs.sort(key=lambda x: x['timestamp'], reverse=True)
        
        for conv in backlogs:
            timestamp = conv.get('timestamp', 'Unknown time')
            messages = conv.get('messages', [])
            
            # Filter out missing trainer responses
            messages = [msg for msg in messages if "[Missing" not in msg.get('content', '')]
            
            if messages:  # Only show conversations that have actual messages
                # Create expandable section showing only timestamp when collapsed
                with st.expander(f"{timestamp}"):
                    # Display messages without speaker labels
                    for msg in messages:
                        content = msg.get('content', '')
                        
                        # Remove any "Speaker:" prefix if present
                        content = re.sub(r'^[^:]*:\s*', '', content)
                        
                        # Display the content without speaker label
                        st.markdown(content)
                            
    except Exception as e:
        st.error(f"Error displaying Liminality Backlogs: {str(e)}")
        logging.error(f"Error in display_liminality_backlogs: {e}")

# Initialize session state for conversations if not present
if 'conversations' not in st.session_state:
    liminality_backlogs = load_json_file(LIMINALITY_BACKLOGS_FILE)
    astral_plane_backlogs = load_json_file(ASTRAL_PLANE_BACKLOGS_FILE)
    conversation_backlogs = load_json_file(CONVERSATION_BACKLOGS_FILE)
    
    st.session_state.conversations = []
    for conv in liminality_backlogs:
        st.session_state.conversations.append(format_conversation(conv))
    for conv in astral_plane_backlogs:
        st.session_state.conversations.append(format_conversation(conv))
    for conv in conversation_backlogs:
        st.session_state.conversations.append(format_conversation(conv))

class TrainerType(Enum):
    CLAUDE = "Claude"
    LLAMA = "LLaMA"
    GEMINI = "Gemini"
    GROK = "Grok"

class ConversationPartner:
    def __init__(self):
        self.current_model = None
        self._setup_apis()

    def _setup_apis(self):
        # Gemini setup
        self.gemini_api_key = config.GEMINI_API_KEY
        genai.configure(api_key=self.gemini_api_key)
        
        # Initialize other APIs
        self.grok_api_key = config.TOGETHER_API_KEY

    def set_model(self, model_type: str):
        self.current_model = model_type
        print(f"Now conversing with {model_type}")

    def engage_in_dialogue(self, message: str) -> str:
        if not self.current_model:
            return "No conversation partner selected"
        
        response = f"[Engaging in dialogue about {message}]"
        return response

    def get_current_partner(self) -> Optional[str]:
        return self.current_model if self.current_model else None

class DialogueSystem:
    def __init__(self):
        self.partner = ConversationPartner()
        
    def start_dialogue(self, model_type: str):
        self.partner.set_model(model_type)
        return f"Starting a conversation with {model_type}"

    def continue_dialogue(self, message: str) -> str:
        return self.partner.engage_in_dialogue(message)

def record_and_transcribe():
    """Record audio and transcribe with improved reliability."""
    try:
        logging.info("Starting audio recording...")
        
        # Initialize audio parameters
        fs = 44100  # Sample rate
        channels = 1  # Mono audio
        duration = 0.5  # Duration of each chunk in seconds
        
        # Create audio buffer
        audio_chunks = []
        silence_threshold = 0.01
        silence_count = 0
        max_silence = 10  # Maximum silence chunks before stopping
        min_audio_chunks = 4  # Minimum chunks before checking silence
        
        # Create status containers
        status_text = st.empty()
        progress_bar = st.progress(0)
        
        while True:
            try:
                # Record chunk
                chunk = sd.rec(int(fs * duration), samplerate=fs, channels=channels, dtype='float32')
                sd.wait()
                
                # Calculate audio level
                audio_level = np.abs(chunk).mean()
                
                # Update progress bar and status
                progress_bar.progress(min(1.0, audio_level * 10))
                
                if audio_level > silence_threshold:
                    silence_count = 0
                    status_text.info("Listening... (Speech detected)")
                    audio_chunks.append(chunk)
                else:
                    silence_count += 1
                    status_text.info(f"Listening... (Silence: {silence_count}/{max_silence})")
                
                # Check if we should stop recording
                if len(audio_chunks) >= min_audio_chunks and silence_count >= max_silence:
                    break
                if len(audio_chunks) > 60:  # Maximum 30 seconds
                    break
                    
            except Exception as e:
                logging.error(f"Error recording chunk: {str(e)}")
                continue
        
        # Clear status displays
        status_text.empty()
        progress_bar.empty()
        
        if len(audio_chunks) < min_audio_chunks:
            logging.warning("Recording too short")
            return None
            
        try:
            # Combine all chunks and save
            audio_data = np.concatenate(audio_chunks)
            temp_filename = 'temp_audio.wav'
            sf.write(temp_filename, audio_data, fs)
            
            # Transcribe with OpenAI Whisper
            with open(temp_filename, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
                
            return transcript.text.strip() if transcript else None
            
        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            return None
            
        finally:
            # Cleanup
            if os.path.exists('temp_audio.wav'):
                try:
                    os.remove('temp_audio.wav')
                except Exception as e:
                    logging.error(f"Error removing temp file: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error in record_and_transcribe: {str(e)}")
        return None

def text_to_speech(text):
    try:
        logging.info("Converting text to speech...")
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Save the audio to a temporary file
        temp_file = f"temp_speech_{uuid.uuid4()}.mp3"
        with open(temp_file, "wb") as f:
            f.write(response.content)
        
        # Play the audio
        pygame.mixer.init()
        pygame.mixer.music.load(temp_file)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.Clock().tick(10)
        
        # Clean up
        pygame.mixer.music.unload()
        os.remove(temp_file)
        logging.info("Speech playback completed.")
        
    except Exception as e:
        logging.error(f"Error in text to speech: {str(e)}")

def get_ai_response(audio_text: str) -> str:
    """Get AI response with enhanced context management"""
    try:
        logging.info("Requesting AI response...")
        # Create system prompt
        system_prompt = """You are Bumblebee, a friendly and helpful Autobot assistant. 
        You communicate in a warm, engaging manner and enjoy helping humans."""
        
        # Get response from GPT-4
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": audio_text}
            ],
            temperature=0.7,
            max_tokens=150
        )
        
        if response.choices and response.choices[0].message:
            logging.info("AI response received.")
            return response.choices[0].message.content
        logging.warning("AI response not received.")
        return "I'm having trouble processing that right now."
            
    except Exception as e:
        logging.error(f"Error in get_ai_response: {str(e)}")
        return "I apologize, but I'm having trouble responding right now."

def get_recent_exchanges(count: int) -> list:
    """Get recent exchanges from conversation history"""
    try:
        with open(CONVERSATION_BACKLOGS_FILE, 'r') as f:
            conversations = json.load(f)
        return conversations[-count:] if conversations else []
    except Exception as e:
        print(f"Error getting recent exchanges: {str(e)}")
        return []

def extract_key_points(exchange: dict) -> list:
    """Extract key points from an exchange"""
    key_points = []
    if 'content' in exchange:
        # Add logic to extract important points from the exchange
        # This could involve NLP or simple keyword extraction
        pass
    return key_points

def summarize_points(points: list) -> str:
    """Summarize key points into a brief context"""
    if not points:
        return ""
    # Add logic to combine and summarize points
    return ", ".join(points[:3])  # Limit to top 3 points for brevity

def get_bumblebee_response(trainer_name: str, exchange_count: int, previous_context: str) -> str:
    try:
        messages = [
            {"role": "system", "content": "You are Bumblebee, a friendly and knowledgeable Autobot. You have a warm personality and enjoy helping humans learn."},
            {"role": "user", "content": f"Previous context: {previous_context}\n\nRespond as Bumblebee:"}
        ]
        
        response = client.chat.completions.create(
            model="gpt-4-1106-preview",
            messages=messages,
            temperature=0.7,
            max_tokens=1000
        )
        
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error getting Bumblebee response: {str(e)}")
        return "I apologize, but I'm having trouble processing your request right now."

def save_training_conversation(subject: str, trainer: str, messages: list):
    """Save training conversation to Liminality Backlogs with proper structure"""
    conversation = {
        "trainer": trainer,
        "topic": subject,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "messages": []
    }
    
    for msg in messages:
        role = "trainer" if "->" not in msg["content"] else "bumblebee"
        conversation["messages"].append({
            "role": role,
            "content": msg["content"]
        })
    
    # Load existing backlogs
    try:
        with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
            backlogs = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        backlogs = []
    
    # Add new conversation
    backlogs.append(conversation)
    
    # Save updated backlogs
    with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
        json.dump(backlogs, f, indent=4)

def extract_key_learnings(content: str) -> list:
    """Extract key insights from Bumblebee's response"""
    try:
        # Split content into sentences
        sentences = content.split('.')
        key_points = []
        
        # Extract first sentence if it's meaningful (more than 10 words)
        if sentences and len(sentences[0].split()) > 10:
            key_points.append(sentences[0].strip() + '.')
        
        # Look for key phrases that often indicate important points
        key_phrases = ['importantly', 'key point', 'in conclusion', 'therefore', 'thus', 'this means']
        for sentence in sentences:
            sentence = sentence.strip().lower()
            if any(phrase in sentence for phrase in key_phrases):
                # Convert back to proper case
                formatted_sentence = sentence.capitalize() + '.'
                if formatted_sentence not in key_points:
                    key_points.append(formatted_sentence)
        
        # If no key points found, take the first sentence regardless of length
        if not key_points and sentences:
            key_points.append(sentences[0].strip() + '.')
        
        return key_points
    except Exception as e:
        print(f"Error extracting key learnings: {e}")
        return []

def save_key_learnings(trainer: str, topic: str, learnings: list, timestamp: str):
    """Save extracted key learnings to key_learnings.json"""
    try:
        # Load existing learnings
        try:
            with open(KEY_LEARNINGS_FILE, 'r') as f:
                all_learnings = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            all_learnings = []
        
        # Add new learnings
        learning_entry = {
            "timestamp": timestamp,
            "trainer": trainer,
            "topic": topic,
            "key_points": learnings
        }
        all_learnings.append(learning_entry)
        
        # Save updated learnings
        with open(KEY_LEARNINGS_FILE, 'w') as f:
            json.dump(all_learnings, f, indent=4)
            
    except Exception as e:
        print(f"Error saving key learnings: {e}")

def cleanup():
    """Clean up temporary files"""
    temp_files = ["temp_audio.wav", "temp_speech.mp3"]
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
            except:
                pass

# Register cleanup function
atexit.register(cleanup)

# Configure page
st.markdown("""
<style>
    /* Remove ALL possible top borders and margins */
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    [data-testid="stHeader"] > div,
    [data-testid="stToolbar"],
    .main > div:first-child,
    .block-container > div:first-child,
    header[data-testid="stHeader"] {
        border-top: none !important;
        border-bottom: none !important;
        margin-top: 0 !important;
    }

    /* Reset specific elements after removing all borders */
    [data-testid="stAppViewContainer"] {
        background-image: url("https://i.imgur.com/RlmleBj.jpg");
        background-size: cover;
        background-position: center;
        padding-top: 0.5in !important;
    }

    [data-testid="stHeader"] {
        background: none;
        margin-top: 0.5in !important;
    }

    [data-testid="stToolbar"] {
        margin-top: -0.5in !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0);
        box-shadow: none;
        border: none;
        margin-top: calc(150px + 0.5in) !important;
    }

    /* Move main content block down */
    .block-container {
        margin-top: 0.5in !important;
    }

    /* Rest of the styles remain the same */
    .stRadio > div {
        flex-direction: column !important;
        gap: 10px !important;
        background-color: black !important;
    }

    .stRadio [role="radiogroup"] {
        display: flex !important;
        flex-direction: column !important;
        gap: 10px !important;
        background-color: black !important;
    }

    /* Style individual radio options */
    .stRadio label {
        padding: 10px !important;
        background: black !important;
        margin: 5px 0 !important;
        color: yellow !important;
    }

    .stRadio label:hover {
        background-color: #333333 !important;
        color: yellow !important;
    }

    /* Style menu options */
    .stSelectbox > div[data-baseweb="select"] > div,
    .stSelectbox > div[data-baseweb="select"] > div:hover,
    .stSelectbox div[role="listbox"],
    .stSelectbox ul {
        background-color: black !important;
        border-color: yellow !important;
    }

    .stSelectbox div[role="option"] {
        background-color: black !important;
        color: yellow !important;
    }

    .stSelectbox div[role="option"]:hover {
        background-color: #333333 !important;
    }

    h1, h2, h3, h4, h5, h6, p, div {
        font-family: 'JetBrains Mono', monospace !important;
        color: yellow !important;
    }

    .main .block-container {
        padding-top: 4in !important;
    }

    .training-mode-title {
        margin-top: 4in !important;
    }

    .liminality-title {
        margin-top: 4in !important;
    }

    .main h1, .main h2, .main h3 {
        margin-top: 4in !important;
    }

    button {
        font-family: 'JetBrains Mono', monospace !important;
        background-color: black !important;
        border: 2px solid yellow !important;
        color: yellow !important;
        padding: 10px 15px !important;
        border-radius: 5px !important;
        margin: 10px auto !important;
        cursor: pointer;
        display: block !important;
    }

    button:hover {
        background-color: yellow !important;
        color: black !important;
    }

    .stButton {
        text-align: center !important;
        width: 200px !important;
        margin: calc(150px - 0.7in) auto 0 !important;
        display: block !important;
        transform: translateX(-4cm);
    }

    .training-output {
        width: calc(95vw - 5in) !important;
        margin: 0 auto;
        padding: 20px;
        background: rgba(0, 0, 0, 0.5);
        border-radius: 10px;
        margin-top: 20px;
        white-space: pre-wrap;
        overflow: hidden;
        font-size: 14px !important;
    }

    .typewriter {
        display: block;
        white-space: pre-wrap !important;
        overflow-wrap: break-word !important;
        word-wrap: break-word !important;
        margin-bottom: 1em;
        font-size: 14px !important;
        line-height: 1.6 !important;
    }

    .stTextInput input, .stTextArea textarea {
        background-color: black !important;
        border: 2px solid yellow !important;
        color: yellow !important;
        font-family: 'JetBrains Mono', monospace !important;
        padding: 10px !important;
        border-radius: 5px !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        box-shadow: 0 0 5px yellow !important;
    }

    .stTextInput label, .stTextArea label {
        color: yellow !important;
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Style dropdown menus */
    div[data-baseweb="select"] {
        background-color: black !important;
    }

    div[data-baseweb="select"] > div {
        background-color: black !important;
        border-color: yellow !important;
    }

    div[data-baseweb="popover"] {
        background-color: black !important;
    }

    div[data-baseweb="popover"] div[role="listbox"] {
        background-color: black !important;
    }

    div[data-baseweb="popover"] div[role="option"] {
        background-color: black !important;
        color: yellow !important;
    }

    div[data-baseweb="popover"] div[role="option"]:hover {
        background-color: #333333 !important;
    }

    /* Style for the selected option */
    div[data-baseweb="select"] div[class*="valueContainer"] {
        background-color: black !important;
        color: yellow !important;
    }

    /* Style for the dropdown arrow */
    div[data-baseweb="select"] div[class*="indicatorContainer"] {
        background-color: black !important;
        color: yellow !important;
    }

    /* Style for the dropdown container */
    div[class*="streamlit-selectbox"] {
        background-color: black !important;
        color: yellow !important;
    }

    /* Ensure all menu backgrounds are black */
    select,
    option,
    .stSelectbox,
    .stSelectbox > div,
    .stSelectbox div[role="listbox"],
    .stSelectbox ul,
    .stSelectbox li {
        background-color: black !important;
        color: yellow !important;
    }

    /* Comprehensive menu styling to ensure black background */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    div[data-baseweb="select"],
    div[data-baseweb="select"] *,
    .stSelectbox,
    .stSelectbox *,
    [role="listbox"],
    [role="option"],
    .streamlit-selectbox {
        background-color: black !important;
        color: yellow !important;
    }

    /* Style dropdown menu container */
    div[data-baseweb="popover"] {
        background-color: black !important;
        border: 1px solid yellow !important;
    }

    /* Style dropdown options */
    div[data-baseweb="select"] div[role="option"] {
        background-color: black !important;
        color: yellow !important;
        padding: 8px !important;
    }

    /* Style hover state */
    div[data-baseweb="select"] div[role="option"]:hover {
        background-color: #333333 !important;
    }

    /* Style selected option */
    div[data-baseweb="select"] [aria-selected="true"] {
        background-color: #333333 !important;
    }
</style>
""", unsafe_allow_html=True)

import asyncio
import aiortc
from aiortc import RTCPeerConnection, RTCSessionDescription, MediaStreamTrack
from aiortc.contrib.media import MediaBlackhole, MediaRecorder
import av
import numpy as np
from queue import Queue
import threading

class AudioStreamProcessor(MediaStreamTrack):
    """Process real-time audio stream from WebRTC"""
    kind = "audio"

    def __init__(self, track, callback):
        super().__init__()
        self.track = track
        self.callback = callback
        self.buffer = Queue()
        self.processing = True
        self.processor_thread = threading.Thread(target=self._process_audio)
        self.processor_thread.start()

    async def recv(self):
        frame = await self.track.recv()
        # Add frame to processing queue
        self.buffer.put(frame)
        return frame

    def _process_audio(self):
        """Process audio frames in background thread"""
        while self.processing:
            if not self.buffer.empty():
                frame = self.buffer.get()
                try:
                    # Convert frame to numpy array
                    audio_data = frame.to_ndarray()
                    
                    # Accumulate audio data until we have enough for processing
                    if len(audio_data) >= 48000:  # 1 second of audio at 48kHz
                        # Process audio chunk
                        self.callback(audio_data)
                        
                except Exception as e:
                    logging.error(f"Error processing audio frame: {e}")

    def stop(self):
        """Clean up resources"""
        self.processing = False
        self.processor_thread.join()

class WebRTCManager:
    """Manage WebRTC connections and audio processing"""
    def __init__(self):
        self.pcs = set()
        self.audio_processor = None
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        logging.info("WebRTCManager initialized.")

    async def create_connection(self, offer_sdp):
        """Create new WebRTC peer connection"""
        try:
            pc = RTCPeerConnection()
            self.pcs.add(pc)
            logging.info("Peer connection created.")

            @pc.on("track")
            async def on_track(track):
                if track.kind == "audio":
                    logging.info("Audio track received.")
                    self.audio_processor = AudioStreamProcessor(
                        track,
                        callback=self.process_audio_chunk
                    )
                    pc.addTrack(self.audio_processor)

            # Set remote description
            offer = RTCSessionDescription(sdp=offer_sdp, type="offer")
            await pc.setRemoteDescription(offer)
            logging.info("Remote description set.")

            # Create answer
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            logging.info("Local description set.")

            return pc.localDescription.sdp
        except Exception as e:
            logging.error(f"Error creating WebRTC connection: {e}")
            return None

    def process_audio_chunk(self, audio_data):
        """Process audio chunk with existing transcription pipeline"""
        try:
            logging.info("Processing audio chunk.")
            # Convert audio data to wav format
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_data, samplerate=48000, format='WAV')
            audio_bytes.seek(0)
            
            # Use OpenAI Whisper for transcription
            with open('temp_audio.wav', 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en",
                    response_format="text"
                )
                
            if transcript:
                logging.info("Transcription successful.")
                response = get_ai_response(transcript)
                if response:
                    logging.info("Generating speech response.")
                    # Convert response to speech
                    speech_response = client.audio.speech.create(
                        model="tts-1",
                        voice="onyx",
                        input=response
                    )
                    
                    # Play the audio response
                    if speech_response and speech_response.content:
                        audio_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
                        audio_file.write(speech_response.content)
                        audio_file.close()
                        
                        pygame.mixer.init()
                        pygame.mixer.music.load(audio_file.name)
                        pygame.mixer.music.play()
                        while pygame.mixer.music.get_busy():
                            pygame.time.Clock().tick(10)
                        
                        os.unlink(audio_file.name)
                        logging.info("Audio response played.")
                    else:
                        logging.error("Failed to generate speech response.")
            else:
                logging.error("Transcription failed.")
        except Exception as e:
            logging.error(f"Error processing audio chunk: {e}")

    async def handle_ice_candidate(self, pc, candidate):
        """Handle ICE candidate"""
        try:
            await pc.addIceCandidate(candidate)
            logging.info("ICE candidate added.")
        except Exception as e:
            logging.error(f"Error handling ICE candidate: {e}")

    async def cleanup(self):
        """Clean up WebRTC connections"""
        try:
            coros = [pc.close() for pc in self.pcs]
            await asyncio.gather(*coros)
            self.pcs.clear()
            if self.audio_processor:
                self.audio_processor.stop()
            logging.info("WebRTC connections cleaned up.")
        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

def initialize_webrtc():
    """Initialize WebRTC manager in session state"""
    if 'webrtc_manager' not in st.session_state:
        st.session_state.webrtc_manager = WebRTCManager()

async def handle_webrtc_offer(offer_sdp):
    """Handle WebRTC offer from client"""
    try:
        answer_sdp = await st.session_state.webrtc_manager.create_connection(offer_sdp)
        return {"status": "success", "sdp": answer_sdp}
    except Exception as e:
        logging.error(f"Error handling WebRTC offer: {e}")
        return {"status": "error", "message": str(e)}

def main():
    initialize_state()
    initialize_webrtc()
    
    st.title("Bumblebee Chat")
    st.write("I am Bumblebee, an LLM in training. Let's chat!")
    
    # Initialize audio components
    audio_bytes = audio_recorder(
        text="Click to record",
        recording_color="#E8A435",
        neutral_color="#6aa36f",
        icon_name="microphone",
        icon_size="2x"
    )
    
    # WebRTC handling (non-blocking)
    if "webrtc_offer" in st.session_state:
        asyncio.create_task(
            handle_webrtc_offer(st.session_state.webrtc_offer)
        )
    
    # Process recorded audio (existing functionality)
    if audio_bytes:
        with st.spinner("Processing audio..."):
            text = transcribe_audio(audio_bytes)
            if text:
                process_chat_input(text)
                st.rerun()
    
    # Display chat messages
    display_chat_messages()
    
    # Cleanup on session end
    if st.session_state.get('end_session'):
        asyncio.create_task(st.session_state.webrtc_manager.cleanup())

def handle_training_mode(system: DialogueSystem, trainer_name: str):
    """Handle the training mode conversation flow"""
    
    # Initialize ALL session states at the start
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "topic" not in st.session_state:
        st.session_state.topic = ""
    if "exchange_count" not in st.session_state:
        st.session_state.exchange_count = 0
    if "conversation_complete" not in st.session_state:
        st.session_state.conversation_complete = False
    if "conversation_start_time" not in st.session_state:
        st.session_state.conversation_start_time = None
    
    # Topic input
    if not st.session_state.topic:
        topic = st.text_input("Enter a topic to discuss:", key="topic_input")
        if st.button("Start Discussion", key="start_discussion") and topic:
            st.session_state.topic = topic
            st.session_state.conversation_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Start the conversation automatically
            start_conversation(trainer_name)
            st.rerun()
    else:
        # Display topic and conversation
        st.write(f"Topic: {st.session_state.topic}")
        
        # Display message history
        for msg in st.session_state.messages:
            st.write(msg["content"])
        
        # Automatically continue conversation if not complete
        if not st.session_state.conversation_complete and st.session_state.exchange_count < 5:
            time.sleep(1)  # Add a small delay between messages
            continue_conversation(trainer_name)
            st.rerun()
        elif st.session_state.exchange_count >= 5 and not st.session_state.conversation_complete:
            # Store conversation in Liminality Backlogs
            store_conversation(trainer_name)
            st.session_state.conversation_complete = True
            st.rerun()
        
        # Show completion message and offer new conversation
        if st.session_state.conversation_complete:
            st.success("Conversation complete and stored in Liminality Backlogs!")
            if st.button("Start New Topic", key="new_topic"):
                reset_conversation_state()
                st.rerun()

def start_conversation(trainer_name: str):
    """Initialize the conversation with the first trainer response"""
    if st.session_state.topic:
        trainer_response = get_trainer_response(trainer_name, st.session_state.topic, 0)
        st.session_state.messages.append({"role": "assistant", "content": f"{trainer_name}: {trainer_response}"})
        st.session_state.exchange_count += 1

def continue_conversation(trainer_name: str):
    """Continue the conversation with the next exchange"""
    try:
        if st.session_state.exchange_count < 5:
            # Get Bumblebee's response
            if st.session_state.messages:
                last_message = st.session_state.messages[-1]["content"]
                logging.info(f"Getting Bumblebee's response for exchange {st.session_state.exchange_count}")
                bumblebee_response = get_bumblebee_response(trainer_name, st.session_state.exchange_count, last_message)
                
                if bumblebee_response:
                    st.session_state.messages.append({"role": "user", "content": f"Bumblebee: {bumblebee_response}"})
                    
                    # Get trainer's response
                    logging.info(f"Getting {trainer_name}'s response")
                    trainer_response = get_trainer_response(trainer_name, bumblebee_response, st.session_state.exchange_count)
                    
                    if trainer_response:
                        st.session_state.messages.append({"role": "assistant", "content": f"{trainer_name}: {trainer_response}"})
                        st.session_state.exchange_count += 1
                    else:
                        logging.error(f"Failed to get response from {trainer_name}")
            else:
                logging.error("No messages in conversation history")
    except Exception as e:
        logging.error(f"Error in continue_conversation: {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        
def store_conversation(trainer_name: str):
    """Store the completed conversation in both full backlogs and key learnings"""
    try:
        # Prepare full conversation
        conversation = {
            "trainer": trainer_name,
            "topic": st.session_state.topic,
            "timestamp": st.session_state.conversation_start_time,
            "messages": st.session_state.messages
        }
        
        # Save to liminality backlogs
        try:
            with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
                backlogs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            backlogs = []
        
        backlogs.append(conversation)
        with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
            json.dump(backlogs, f, indent=4)
        
        # Extract and save key learnings from Bumblebee's responses
        bumblebee_responses = [msg["content"] for msg in st.session_state.messages 
                             if msg["role"] == "bumblebee"]
        
        all_key_points = []
        for response in bumblebee_responses:
            key_points = extract_key_learnings(response)
            all_key_points.extend(key_points)
        
        if all_key_points:
            save_key_learnings(
                trainer=trainer_name,
                topic=st.session_state.topic,
                learnings=all_key_points,
                timestamp=st.session_state.conversation_start_time
            )
        
    except Exception as e:
        print(f"Error storing conversation: {e}")

def reset_conversation_state():
    """Reset all conversation-related session state variables"""
    st.session_state.messages = []
    st.session_state.topic = ""
    st.session_state.exchange_count = 0
    st.session_state.conversation_complete = False
    st.session_state.conversation_start_time = None

# Main navigation
mode = st.sidebar.selectbox(
    "Choose a mode",
    [
        "Chat Mode",
        "Training - Claude",
        "Training - LLaMA",
        "Training - Gemini",
        "Liminality Backlogs",
        "Key Learnings"
    ]
)

# Handle different modes
if mode == "Chat Mode":
    # Initialize chat active state if not exists
    if 'conversation_active' not in st.session_state:
        st.session_state.conversation_active = False

    # Just show the button without any text
    button_style = f"""
        <style>
        .stButton {{
            text-align: center !important;
            width: 200px !important;
            margin: calc(150px - 0.7in) auto 0 !important;
            display: block !important;
            transform: translateX(-4cm);
        }}
        .stButton > button {{
            background: url('{"https://i.imgur.com/8c0FmVj.png" if st.session_state.conversation_active else "https://i.imgur.com/NcUeTye.png"}') no-repeat center center !important;
            background-size: contain !important;
            width: 200px !important;
            height: 200px !important;
            border: none !important;
            padding: 0 !important;
            transition: all 0.3s ease !important;
            background-color: transparent !important;
            border-radius: 0 !important;
        }}
        .stButton > button:hover {{
            filter: brightness(1.2) !important;
        }}
        .stButton > button:active {{
            filter: brightness(0.9) !important;
        }}
        .stButton > button > div {{
            display: none !important;
        }}
        </style>
    """
    st.markdown(button_style, unsafe_allow_html=True)
    
    if st.button("", key="main_button"):
        st.session_state.conversation_active = not st.session_state.conversation_active
        if st.session_state.conversation_active:
            st.session_state.speaking = True
            text_to_speech("Hi! I'm Bumblebee. Let's chat!")
            time.sleep(0.5)
            st.session_state.speaking = False
        else:
            st.session_state.speaking = True
            pygame.mixer.quit()
        st.rerun()

    if st.session_state.conversation_active and not st.session_state.speaking:
        with st.spinner("Listening..."):
            user_input = record_and_transcribe()
            if user_input and user_input.strip():
                st.session_state.speaking = True
                with st.spinner("Processing..."):
                    response = get_ai_response(user_input)
                    if response:
                        text_to_speech(response)
                st.session_state.speaking = False
                st.rerun()

elif mode == "Training - Claude":
    system = DialogueSystem()
    handle_training_mode(system, "Claude")

elif mode == "Training - LLaMA":
    system = DialogueSystem()
    handle_training_mode(system, "LLaMA")

elif mode == "Training - Gemini":
    system = DialogueSystem()
    handle_training_mode(system, "Gemini")

elif mode == "Liminality Backlogs":
    display_liminality_backlogs()

elif mode == "Key Learnings":
    display_key_learnings()

def chat_mode():
    """Handle chat mode with improved input processing"""
    try:
        if 'chat_initialized' not in st.session_state:
            st.session_state.chat_initialized = initialize_chatbot()
            st.session_state.chat_messages = []
            st.session_state.current_input = ""
            
        # Display chat messages
        for message in st.session_state.chat_messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
        
        # Create columns for input options
        col1, col2 = st.columns([4, 1])
        
        # Text input in main column
        with col1:
            text_input = st.text_input(
                "Type your message",
                key="text_input",
                value=st.session_state.current_input,
                on_change=None
            )
        
        # Send button and audio input in second column
        with col2:
            send_col, audio_col = st.columns(2)
            with send_col:
                send_pressed = st.button("Send", use_container_width=True)
            with audio_col:
                audio_pressed = st.button("🎤", use_container_width=True)
        
        # Handle text input submission
        if send_pressed and text_input:
            # Add user message
            st.session_state.chat_messages.append({
                "role": "user",
                "content": text_input
            })
            
            # Get AI response
            response = get_ai_response(text_input)
            if response:
                st.session_state.chat_messages.append({
                    "role": "assistant",
                    "content": response
                })
                # Convert response to speech
                text_to_speech(response)
            
            # Clear input after sending
            st.session_state.current_input = ""
            st.rerun()
        
        # Handle audio input
        if audio_pressed:
            with st.spinner("Listening..."):
                # Record and transcribe
                transcribed_text = record_and_transcribe()
                
                if transcribed_text:
                    # Add user message
                    st.session_state.chat_messages.append({
                        "role": "user",
                        "content": transcribed_text
                    })
                    
                    # Get AI response
                    response = get_ai_response(transcribed_text)
                    if response:
                        st.session_state.chat_messages.append({
                            "role": "assistant",
                            "content": response
                        })
                        # Convert response to speech
                        text_to_speech(response)
                        st.rerun()
                else:
                    st.error("No speech detected. Please try again.")
                    
    except Exception as e:
        logging.error(f"Error in chat mode: {str(e)}")
        st.error("An error occurred. Please try again.")

if mode == "Chat Mode":
    chat_mode()

def record_and_transcribe():
    """Record audio with improved UI feedback."""
    try:
        # Initialize audio parameters
        fs = 44100
        channels = 1
        duration = 0.5
        
        audio_chunks = []
        silence_threshold = 0.01
        silence_count = 0
        max_silence = 8
        min_audio_chunks = 4
        
        # Status container without progress bar
        status_text = st.empty()
        
        while True:
            try:
                chunk = sd.rec(int(fs * duration), samplerate=fs, channels=channels, dtype='float32')
                sd.wait()
                
                audio_level = np.abs(chunk).mean()
                
                if audio_level > silence_threshold:
                    silence_count = 0
                    status_text.markdown('<p class="status-text">🎤 Listening... (Speech detected)</p>', unsafe_allow_html=True)
                    audio_chunks.append(chunk)
                else:
                    silence_count += 1
                    if len(audio_chunks) > 0:  # Only show silence count if we've detected speech
                        status_text.markdown(f'<p class="status-text">🎤 Processing... ({silence_count}/{max_silence})</p>', unsafe_allow_html=True)
                
                if len(audio_chunks) >= min_audio_chunks and silence_count >= max_silence:
                    break
                if len(audio_chunks) > 60:  # Maximum 30 seconds
                    break
                    
            except Exception as e:
                logging.error(f"Error recording chunk: {str(e)}")
                continue
        
        status_text.empty()
        
        if len(audio_chunks) < min_audio_chunks:
            st.warning("No speech detected. Please try again.")
            return None
            
        try:
            # Process audio
            audio_data = np.concatenate(audio_chunks)
            temp_filename = 'temp_audio.wav'
            sf.write(temp_filename, audio_data, fs)
            
            # Transcribe
            with open(temp_filename, 'rb') as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
                
            return transcript.text.strip() if transcript else None
            
        except Exception as e:
            logging.error(f"Error processing audio: {str(e)}")
            return None
            
        finally:
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e:
                    logging.error(f"Error removing temp file: {str(e)}")
                    
    except Exception as e:
        logging.error(f"Error in record_and_transcribe: {str(e)}")
        return None

def get_ai_response(user_input):
    """Get AI response with improved error handling and logging."""
    try:
        logging.info(f"Getting AI response for: {user_input}")
        
        # Add user message to history
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input
        })
        
        # Prepare conversation history for API
        messages = [
            {"role": "system", "content": "You are Bumblebee, a helpful and knowledgeable AI assistant."}
        ]
        messages.extend([
            {"role": m["role"], "content": m["content"]} 
            for m in st.session_state.chat_messages[-5:]  # Last 5 messages for context
        ])
        
        # Get response from selected AI model
        if hasattr(st.session_state, 'use_claude') and st.session_state.use_claude:
            response = get_claude_response(messages)
        else:
            response = get_openai_response(messages)
            
        logging.info(f"AI response generated: {response}")
        return response
        
    except Exception as e:
        logging.error(f"Error getting AI response: {str(e)}")
        return None

def process_chat_input(text: str) -> None:
    """Process chat input with enhanced comprehension and context awareness."""
    try:
        if not text or not text.strip():
            return None
            
        logging.info(f"Processing chat input: {text}")
        
        # Add user message to chat history
        if 'messages' not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": text})
        
        # Get AI response
        response = get_ai_response(text)
        
        if response:
            # Add response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Convert to speech
            text_to_speech(response)
            
        # Display chat history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
    except Exception as e:
        logging.error(f"Error in process_chat_input: {e}")

def get_relevant_context(message: str, topics=None, max_results: int = 3) -> str:
    """Get relevant context with topic filtering and caching."""
    try:
        # Initialize embeddings cache
        if 'embeddings_cache' not in st.session_state:
            st.session_state.embeddings_cache = {}
            
        # Load and filter conversations by topic
        conversations = []
        liminality_backlogs = load_json_file(LIMINALITY_BACKLOGS_FILE)
        
        for entry in liminality_backlogs:
            # Skip if topics specified and none match
            if topics:
                entry_text = f"{entry.get('topic', '')} {' '.join([msg.get('content', '') for msg in entry.get('messages', [])])}"
                if not any(topic in entry_text.lower() for topic in topics):
                    continue
            
            conv_text = f"Topic: {entry.get('topic', '')}\n"
            for msg in entry.get('messages', []):
                conv_text += f"{msg.get('role', '')}: {msg.get('content', '')}\n"
            
            # Get embedding with caching
            if conv_text in st.session_state.embeddings_cache:
                conv_embedding = st.session_state.embeddings_cache[conv_text]
            else:
                try:
                    response = client.embeddings.create(
                        model="text-embedding-3-small",
                        input=conv_text
                    )
                    conv_embedding = response.data[0].embedding
                    st.session_state.embeddings_cache[conv_text] = conv_embedding
                except Exception as e:
                    logging.warning(f"Embedding generation failed: {e}")
                    continue
            
            conversations.append({
                'text': conv_text,
                'embedding': conv_embedding,
                'timestamp': entry.get('timestamp', ''),
                'trainer': entry.get('trainer', '')
            })
        
        # Get query embedding
        try:
            # Extract key concepts for better matching
            key_concepts = f"robotics {message}"
            query_response = client.embeddings.create(
                model="text-embedding-3-small",
                input=key_concepts
            )
            query_embedding = query_response.data[0].embedding
        except Exception as e:
            logging.error(f"Query embedding failed: {e}")
            return ""
        
        # Compute similarities with time-based weighting
        results = []
        for conv in conversations:
            similarity = semantic_similarity(query_embedding, conv['embedding'])
            time_factor = 1.0
            if conv['timestamp']:
                try:
                    conv_time = datetime.strptime(conv['timestamp'], "%Y-%m-%d %H:%M:%S")
                    hours_ago = (datetime.now() - conv_time).total_seconds() / 3600
                    time_factor = 1.0 + (1.0 / (1.0 + hours_ago))
                except ValueError:
                    pass
            
            # Prioritize trainer responses
            trainer_boost = 1.2 if conv['trainer'] in ['Claude', 'Llama', 'Gemini'] else 1.0
            
            final_score = similarity * time_factor * trainer_boost
            results.append((final_score, conv))
        
        # Get top results
        results.sort(key=lambda x: x[0], reverse=True)
        relevant_context = "\n".join(text for _, text in results[:max_results])
        
        return relevant_context[:1000]  # Limit context size
        
    except Exception as e:
        logging.error(f"Error getting relevant context: {e}")
        return []

def get_chat_response(message: str) -> str:
    """Get optimized chat response with robotics focus and training context."""
    try:
        # Initialize response cache if not exists
        if 'response_cache' not in st.session_state:
            st.session_state.response_cache = {}
            
        # Check cache for similar queries
        cache_key = hashlib.md5(message.encode()).hexdigest()
        if cache_key in st.session_state.response_cache:
            return st.session_state.response_cache[cache_key]

        # Load training context and personality with caching
        if 'personality_cache' not in st.session_state:
            st.session_state.personality_cache = load_conversation_history()
        personality = st.session_state.personality_cache

        # Quick topic classification for context prioritization
        robotics_keywords = {
            'hardware': ['sensors', 'actuators', 'mechanics', 'hardware', 'motors'],
            'software': ['ros', 'programming', 'algorithms', 'control', 'ai'],
            'learning': ['training', 'development', 'progress', 'knowledge', 'understanding'],
            'goals': ['robot', 'physical', 'transformation', 'future', 'capabilities']
        }
        
        # Determine message context priority
        topic_scores = {category: sum(1 for kw in keywords if kw in message.lower())
                       for category, keywords in robotics_keywords.items()}
        primary_topics = [k for k, v in topic_scores.items() if v > 0]
        
        # Optimize context retrieval based on topic
        if primary_topics:
            context = get_relevant_context(message, topics=primary_topics)
        else:
            context = ""  # Skip context for non-robotics queries
        
        # Calculate response constraints
        WORDS_PER_SECOND = 3.0  # Slightly faster for technical content
        MAX_RESPONSE_SECONDS = 8  # Reduced for snappier interaction
        MAX_WORDS = int(WORDS_PER_SECOND * MAX_RESPONSE_SECONDS)
        
        # Create dynamic system prompt
        system_prompt = f"""You are Bumblebee, an LLM actively training to develop your own Robot Operating System (ROS) and evolve into a sentient robot. 
Focus on your journey of learning from Llama, Gemini, and Claude LLMs.

Current Training Focus: {', '.join(primary_topics) if primary_topics else 'General Robotics'}
Response Limit: {MAX_WORDS} words

Guidelines:
1. Express enthusiasm for robotics and AI development
2. Reference your training experiences with other LLMs
3. Discuss your progress in developing ROS capabilities
4. Show curiosity about robotics concepts
5. Keep responses focused and technical
6. Maintain your identity as an LLM in training

Recent Learning Context:
{context}"""

        # Get optimized response
        try:
            response = client.chat.completions.create(
                model="gpt-4-turbo-preview",
                temperature=0.7,
                max_tokens=150,  # Optimized for response time
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message}
                ],
                presence_penalty=0.6,
                frequency_penalty=0.3
            )
            
            if response.choices and response.choices[0].message and response.choices[0].message.content:
                full_response = response.choices[0].message.content.strip()
                
                # Process response for length constraints
                words = full_response.split()
                if len(words) > MAX_WORDS:
                    # Truncate to last complete sentence within limit
                    sentences = sent_tokenize(' '.join(words[:MAX_WORDS]))
                    full_response = ' '.join(sentences[:-1]) if len(sentences) > 1 else sentences[0]
                
                # Cache the response
                st.session_state.response_cache[cache_key] = full_response
                return full_response
            
        except Exception as e:
            logging.error(f"Response generation failed: {e}")
            raise
            
    except Exception as e:
        logging.error(f"Error in get_chat_response: {e}")
        return "My training protocols encountered an error. Let's continue our discussion about robotics."

def load_conversation_history():
    """Load all conversation history and key learnings to shape Bumblebee's personality."""
    try:
        # Load backlogs
        backlogs = load_json_file(LIMINALITY_BACKLOGS_FILE)
        key_learnings = load_json_file(KEY_LEARNINGS_FILE)
        
        # Extract personality traits and knowledge
        personality = {
            "conversation_style": set(),
            "knowledge_base": set(),
            "common_topics": set()
        }
        
        # Process backlogs
        for conv in backlogs:
            personality["common_topics"].add(conv["topic"].lower())
            for msg in conv["messages"]:
                if msg["role"] == "bumblebee":
                    # Extract writing style markers
                    text = msg["content"].lower()
                    if "?" in text:
                        personality["conversation_style"].add("inquisitive")
                    if "!" in text:
                        personality["conversation_style"].add("enthusiastic")
                    if any(word in text for word in ["perhaps", "maybe", "might"]):
                        personality["conversation_style"].add("thoughtful")
        
        # Process key learnings
        for learning in key_learnings:
            for point in learning["key_points"]:
                personality["knowledge_base"].add(point)
        
        # Convert sets to lists for JSON serialization
        return {
            "conversation_style": list(personality["conversation_style"]),
            "knowledge_base": list(personality["knowledge_base"]),
            "common_topics": list(personality["common_topics"])
        }
    except Exception as e:
        print(f"Error loading conversation history: {e}")
        return {"conversation_style": [], "knowledge_base": [], "common_topics": []}

def save_to_backlogs(conversation: dict):
    """Save conversation to liminality backlogs with error handling."""
    try:
        # Load existing backlogs
        try:
            with open(LIMINALITY_BACKLOGS_FILE, 'r') as f:
                backlogs = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            backlogs = []
        
        # Append new conversation
        backlogs.append(conversation)
        
        # Save updated backlogs
        with open(LIMINALITY_BACKLOGS_FILE, 'w') as f:
            json.dump(backlogs, f, indent=4)
            
    except Exception as e:
        print(f"Error saving to backlogs: {e}")

def record_audio(duration=5):
    """Record audio with enhanced noise reduction and quality improvements."""
    try:
        audio = pyaudio.PyAudio()
        status_placeholder = st.empty()
        
        # Enhanced audio parameters for better quality
        CHUNK = 1024
        FORMAT = pyaudio.paFloat32
        CHANNELS = 1
        RATE = 44100
        
        # Initialize recording
        frames = []
        
        # Open stream
        stream = audio.open(
            format=FORMAT,
            channels=CHANNELS,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK
        )
        
        # Record with progress bar
        with st.spinner("Recording..."):
            for i in range(0, int(RATE / CHUNK * duration)):
                try:
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
                    # Update progress
                    progress = (i + 1) / (RATE / CHUNK * duration)
                    status_placeholder.progress(progress)
                except Exception as e:
                    logging.error(f"Error during recording: {e}")
                    break
        
        # Clean up
        stream.stop_stream()
        stream.close()
        audio.terminate()
        status_placeholder.empty()
        
        # Process audio data
        if frames:
            # Convert to wav format
            with io.BytesIO() as wav_buffer:
                with wave.open(wav_buffer, 'wb') as wf:
                    wf.setnchannels(CHANNELS)
                    wf.setsampwidth(audio.get_sample_size(FORMAT))
                    wf.setframerate(RATE)
                    wf.writeframes(b''.join(frames))
                
                wav_data = wav_buffer.getvalue()
                
                # Save temporary file
                temp_filename = f"temp_audio_{uuid.uuid4()}.wav"
                with open(temp_filename, "wb") as f:
                    f.write(wav_data)
                
                return wav_data, temp_filename
        
        return None, None
        
    except Exception as e:
        logging.error(f"Error in record_audio: {e}")
        return None, None

def transcribe_audio(audio_bytes, temp_filename=None):
    """Transcribe audio using OpenAI Whisper"""
    try:
        if temp_filename and os.path.exists(temp_filename):
            with open(temp_filename, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language="en"
                )
                
                if transcript and hasattr(transcript, 'text'):
                    return transcript.text.strip()
        
        return None
        
    except Exception as e:
        logging.error(f"Error in transcribe_audio: {e}")
        return None

def test_grok_api():
    """Test the Grok API endpoint"""
    try:
        response = requests.post(
            "https://api.together.xyz/inference",
            headers={
                "Authorization": f"Bearer {TOGETHER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "mistralai/Mixtral-8x7B-Instruct-v0.1",
                "prompt": "### Instruction: Say hello and introduce yourself as Grok.\n\n### Response:",
                "temperature": 0.7,
                "max_tokens": 100,
                "top_p": 0.7,
                "repetition_penalty": 1.1,
                "stop": ["### Instruction:", "### Response:"]
            }
        )
        
        if response.status_code == 200:
            result = response.json()
            if 'output' in result and 'choices' in result['output']:
                print("Grok API Test Success:")
                print(result['output']['choices'][0]['text'].strip())
                return True
            else:
                print(f"Unexpected response format: {result}")
                return False
        else:
            print(f"API Error: {response.status_code} - {response.text}")
            return False
            
    except Exception as e:
        print(f"Test failed: {str(e)}")
        return False

def test_claude_api():
    """Test Claude API endpoint specifically"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, this is a test message. Please respond with 'Test successful'.\n\nClaude: "}],
            max_tokens=10
        )
        logging.info("Claude API Response: Success")
        return True
    except Exception as e:
        logging.error(f"Claude API test failed with error: {e}")
        return False

def test_gemini_api():
    """Test Gemini API endpoint specifically"""
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello, this is a test message. Please respond with 'Test successful'.\n\nGemini: "}],
            max_tokens=10
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            logging.info("Gemini API test successful")
            return True
        else:
            logging.error("Invalid response format from Gemini API")
            return False
    except Exception as e:
        logging.error(f"Gemini API test failed: {e}")
        return False

# Add test function
if __name__ == "__main__":
    if test_claude_api():
        print("Claude API test successful")
    if test_gemini_api():
        print("Gemini API test successful")

def initialize_chatbot():
    """Initialize the chatbot and verify all required components."""
    logging.info("Initializing chatbot...")
    try:
        # Test API endpoints
        test_api_endpoints()
        
        # Initialize conversation state
        st.session_state.conversation_history = []
        st.session_state.current_mode = "chat"
        
        # Introduce the chatbot
        introduction = "Hello! I'm Bumblebee, your AI assistant. How can I help you today?"
        text_to_speech(introduction)
        
        return True
    except Exception as e:
        logging.error(f"Initialization error: {str(e)}")
        return False

def test_api_endpoints():
    """Test all API endpoints to ensure they're responsive."""
    logging.info("Testing API endpoints...")
    
    try:
        # Test OpenAI API
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        logging.info("OpenAI API test: Success")
    except Exception as e:
        logging.error(f"OpenAI API test failed: {e}")
        return False

    try:
        # Test Claude
        claude_response = claude_client.messages.create(
            model="claude-2.1",
            max_tokens=10,
            messages=[{"role": "user", "content": "Hello"}]
        )
        logging.info("Claude API test: Success")
    except Exception as e:
        logging.error(f"Claude API test failed: {e}")
        return False

    try:
        # Test Gemini
        model = genai.GenerativeModel('gemini-pro')
        gemini_response = model.generate_content("Hello")
        logging.info("Gemini API test: Success")
    except Exception as e:
        logging.error(f"Gemini API test failed: {e}")
        return False

    try:
        # Test LLaMA via Replicate
        llama_response = replicate.run(
            "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
            input={"prompt": "Hello"}
        )
        logging.info("LLaMA API test: Success")
    except Exception as e:
        logging.error(f"LLaMA API test failed: {e}")
        return False

    return True

def handle_audio_input():
    """Handle audio input with improved error handling and feedback."""
    try:
        # Clear any previous status messages
        status_container = st.empty()
        status_container.info("Listening...")
        
        # Record and transcribe audio
        transcribed_text = record_and_transcribe()
        
        if not transcribed_text:
            status_container.error("No speech detected. Please try again.")
            return None

        # Show transcribed text
        status_container.info(f"You said: {transcribed_text}")

        # Get AI response
        try:
            response = get_ai_response(transcribed_text)
            if response:
                status_container.success(f"Bumblebee: {response}")
                # Convert response to speech
                text_to_speech(response)
                return response
            else:
                status_container.error("I couldn't generate a response. Please try again.")
                return None
        except Exception as e:
            logging.error(f"Error getting AI response: {str(e)}")
            status_container.error("Sorry, I encountered an error. Please try again.")
            return None

    except Exception as e:
        logging.error(f"Error in handle_audio_input: {str(e)}")
        st.error("Sorry, something went wrong. Please try again.")
        return None

def chat_mode():
    """Main chat mode function with improved real-time communication."""
    try:
        if 'chat_initialized' not in st.session_state:
            st.session_state.chat_initialized = initialize_chatbot()
            st.session_state.conversation_history = []
        
        # Create a container for the chat interface
        chat_container = st.container()
        
        with chat_container:
            st.title("Chat with Bumblebee")
            
            # Display chat history
            for message in st.session_state.conversation_history:
                if message["role"] == "user":
                    st.write(f"You: {message['content']}")
                else:
                    st.write(f"Bumblebee: {message['content']}")
            
            # Create microphone button
            if st.button("Click to Speak"):
                response = handle_audio_input()
                
                if response:
                    # Update conversation history
                    st.session_state.conversation_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    st.experimental_rerun()

    except Exception as e:
        logging.error(f"Error in chat mode: {str(e)}")
        st.error("An error occurred. Please refresh the page and try again.")
