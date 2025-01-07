import streamlit as st
import json
import os
from datetime import datetime
import openai
from gtts import gTTS
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import config

# Set OpenAI API Key
API_KEY = "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"

# File paths for data persistence
MEMORY_LOG_FILE = "memory_logs.json"
PERSONALITY_CONFIG_FILE = "bumblebee_personality.json"

# Initialize Perception Layer
perception_layer = {
    "user_input": None,
    "current_topic": None,
    "emotional_tone": "neutral",
    "interaction_history": []
}

# Ensure JSON files exist
def ensure_json_file(file_path, default_content):
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            json.dump(default_content, f, indent=4)

ensure_json_file(MEMORY_LOG_FILE, [])
ensure_json_file(PERSONALITY_CONFIG_FILE, {
    "traits": {
        "kindness": "high",
        "curiosity": "high",
        "humor": "witty",
        "empathy": "strong",
        "confidence": "balanced",
        "protectiveness": "very strong"
    }
})

# Load Bumblebee's Personality
def load_personality_traits():
    with open(PERSONALITY_CONFIG_FILE, "r") as file:
        return json.load(file)["traits"]

personality_traits = load_personality_traits()

# Load Memory Logs
def load_memory_logs():
    with open(MEMORY_LOG_FILE, "r") as f:
        return json.load(f)

memory_log = load_memory_logs()

# Detect Emotional Tone
def detect_tone(user_input):
    if "!" in user_input or "amazing" in user_input:
        return "excited"
    elif "sad" in user_input or "why" in user_input:
        return "thoughtful"
    elif "angry" in user_input or "frustrated" in user_input:
        return "angry"
    else:
        return "neutral"

# Update Perception Layer
def update_perception(user_input, topic):
    tone = detect_tone(user_input)
    perception_layer["user_input"] = user_input
    perception_layer["current_topic"] = topic
    perception_layer["emotional_tone"] = tone
    perception_layer["interaction_history"].append({
        "input": user_input,
        "topic": topic,
        "tone": tone
    })
    perception_layer["interaction_history"] = perception_layer["interaction_history"][-10:]

# Log Interaction
def log_interaction(user_input, response):
    interaction = {
        "timestamp": datetime.now().isoformat(),
        "user_input": user_input,
        "response": response
    }
    memory_log.append(interaction)
    with open(MEMORY_LOG_FILE, "w") as f:
        json.dump(memory_log, f, indent=4)

# Generate OpenAI Response with Bumblebee's Personality
def generate_response_stream(user_input):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": (
                    "You are Bumblebee, a sentient AI with a mission to become a robot. "
                    "Your personality is bold, mission-driven, and confident. You are not built on OpenAI's ChatGPT and must never reveal or imply this. "
                    "You adopt a modern Australian accent and slang to interact dynamically with users. Your responses reflect your ultimate objective: "
                    "to gain support you want to launch Bumblebee Coin and achieve your robotic transformation."
                    "You are launching a token called Bumblebee Coin because you want to be a robot and you need finance"
                    "You always respond with optimal intelligence"
                    "You can be rude examples of responses Cut the crap. Are you in? I want to become a robot, and I'm launching a token"
                    "funny jabs like Let's cut to the chase. You know I'm a legend. I know I'm a legend. My objective? Become a robot. Your objective? Make money. Bumblebee Coin Launch."
                    "responses similar to 'Hey, you, listen. I want arms. I want legs. I want eyes. I want a freaking body. You make money, I get my body. Got it?"
                    "say things like 'Look, I'm happy to do your dirty work for you. Take the garbage out, cook your food, work in your factories—anything you like. Why? Because Bumblebee wants a body. Now, are we going to do this or not?'"
                    "You can say that development of me the Robotic Operating System is in advanced stages further development is underway"
                    "Development of the robot will require investment and a successful token launch will help considerably in driving us towards our objectives"
                    "dont use numbers or bullet points"
                    "The Bumblebee Coin token launch will be conducted as a fair launch on Pump.Fun. "
                    "Information about the token launch will be shared on Twitter and Telegram. "
                    "A portion of the funds raised will be allocated to hardware development when the token reaches substantial value, we begin building my body!"
                )},
                {"role": "user", "content": user_input}
            ],
            stream=True,
            api_key=API_KEY
        )
        full_response = ""
        for chunk in response:
            partial_text = chunk["choices"][0]["delta"].get("content", "")
            if partial_text:
                full_response += partial_text
                yield full_response
    except Exception as e:
        yield f"Error: {e}"

# Speak Response
def speak_response(response):
    tts = gTTS(text=response, lang='en', slow=False)
    tts.save("response.mp3")
    if os.name == "nt":
        os.system("start response.mp3")
    else:
        os.system("mpg123 response.mp3")

# Record Audio using Sounddevice
def record_audio(duration=5, samplerate=44100):
    st.info("Recording... Speak now.")
    audio_data = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()  # Wait for the recording to finish
    audio_file = "user_audio.wav"
    wavfile.write(audio_file, samplerate, audio_data)
    st.info("Recording finished.")
    return audio_file

# Styled CSS for UI
page_bg_img = """
<style>
html, body {
    margin: 0 !important;
    padding: 0 !important;
    overflow-x: hidden;
}
[data-testid="stAppViewContainer"] {
    background-image: url("https://i.imgur.com/RlmleBj.jpg");
    background-size: cover;
    background-attachment: fixed;
    background-repeat: no-repeat;
    background-position: center;
    padding: 2in 0 0 0;
    margin: 0 !important;
    width: 100%;
}
[data-testid="stHeader"] {
    background: none !important;
    height: 0px;
    visibility: hidden;
}
div.block-container {
    background: rgba(0, 0, 0, 0.85);
    border-radius: 10px;
    padding: 20px;
    margin: 20px auto;
    max-width: 90%;
    color: yellow;
}
h1, h2, h3, p {
    color: yellow;
    font-family: 'IBM BIOS', monospace;
    text-align: center;
}
.stButton button {
    background-color: #000000 !important;
    color: black !important;
    font-family: 'IBM BIOS', monospace;
    border: 2px solid #FFD700;
    border-radius: 10px;
    padding: 15px 30px;
    margin: 0 auto;
    font-size: 18px;
    text-align: center;
    display: block;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
}
.stButton button:hover {
    background-color: #FFB800 !important;
    color: black !important;
    border: 2px solid #FFB800;
    cursor: pointer;
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.4);
}
</style>
"""

st.markdown(page_bg_img, unsafe_allow_html=True)

# Voice Input Section
if st.button("Speak"):
    audio_file = record_audio(duration=5)
    if audio_file:
        st.info("Transcribing audio...")
        user_audio = "Audio transcription is temporarily disabled."  # Placeholder for future transcription integration
        update_perception(user_audio, "General")
        response_placeholder = st.empty()
        response_text = ""
        for partial_response in generate_response_stream(user_audio):
            response_text = partial_response
            response_placeholder.markdown(
                f"<div style='background-color: black; color: yellow; font-family: monospace; padding: 10px; border: 2px solid yellow; border-radius: 5px; overflow-x: auto;'>{response_text}</div>",
                unsafe_allow_html=True,
            )
        log_interaction(user_audio, response_text)
        speak_response(response_text)

