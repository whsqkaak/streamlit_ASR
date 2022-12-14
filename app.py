import time
import datetime
import logging
import soundfile
import streamlit as st

from pathlib import Path

from asr import load_model, inference

LOG_DIR = "./logs"
DATA_DIR = "./data"
logger = logging.getLogger(__name__)

def upload_audio() -> Path:
    # Upload audio file
    uploaded_file = st.file_uploader("Choose a audio file(wav, mp3, flac)", type=['wav','mp3','flac'])
    if uploaded_file is not None:
        # Save audio file
        audio_data, samplerate = soundfile.read(uploaded_file)
        
        # Make save directory
        now = datetime.datetime.now()
        now_time = now.strftime('%Y-%m-%d-%H:%M:%S')
        audio_dir = Path(DATA_DIR) / f"{now_time}"
        audio_dir.mkdir(parents=True, exist_ok=True)
        
        audio_path = audio_dir / uploaded_file.name
        soundfile.write(audio_path, audio_data, samplerate)
        
        # Show audio file
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        st.audio(audio_bytes, format=uploaded_file.type)
        
        return audio_path

@st.experimental_singleton(show_spinner=False)
def call_load_model():
    generator = load_model()
    return generator

def main():
    st.header("Speech-to-Text app with streamlit")
    st.markdown(
        """
This STT app is using [Wav2Vec2.0-Korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean).
This app only process Korean.
        """
    )
    
    audio_path = upload_audio()
    logger.info(f"Uploaded audio file: {audio_path}")
    
    with st.spinner(text="Wait for loading ASR Model..."):
        generator = call_load_model()
    
    if audio_path is not None:
        start_time = time.time()
        with st.spinner(text='Wait for inference...'):
            output = inference(generator, audio_path)

        end_time = time.time()

        process_time = time.gmtime(end_time - start_time)
        process_time = time.strftime("%H hour %M min %S secs", process_time)

        st.success(f"Inference finished in {process_time}.")
        st.write(f"output: {output['text']}")
    

if __name__ == "__main__":
    # Setting logger
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(levelname)8s %(asctime)s %(name)s %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    now = datetime.datetime.now()
    now_time = now.strftime('%Y-%m-%d-%H:%M:%S')
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{now_time}.log"
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info('Start App')
    
    main()