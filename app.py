import datetime
import logging
import pydub
import soundfile
import streamlit as st

from pathlib import Path

from asr import inference

logger = logging.getLogger(__name__)

def upload_audio() -> Path:
    # TODO: refactoring
    # Upload audio file
    uploaded_file = st.file_uploader("Choose a audio file(wav, mp3, flac)", type=['wav','mp3','flac'])
    if uploaded_file is not None:
        # Save audio file
        audio_data, samplerate = soundfile.read(uploaded_file)
        now_time = now.strftime('%Y-%m-%d-%H:%M:%S')
        audio_dir = Path('./data') / f"{now_time}"
        audio_dir.mkdir(parents=True, exist_ok=True)
        audio_path = audio_dir / uploaded_file.name
        soundfile.write(audio_path, audio_data, samplerate)
        
        # Show audio file
        with open(audio_path, 'rb') as audio_file:
            audio_bytes = audio_file.read()
        
        st.audio(audio_bytes, format=uploaded_file.type)
        
        return audio_path

    
def main():
    st.header("Speech-to-Text app with streamlit")
    st.markdown(
        """
This STT app is using [Wav2Vec2.0-Korean](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean).
This app only process Korean.
        """
    )
    
    audio_path = upload_audio()
    st.write(f"audio_path: {audio_path}")
    print(audio_path)
    output = inference(audio_path)
    # TODO Write model loading
    st.write(f"output: {output}")
    

if __name__ == "__main__":
    # Setting logger
    logger.setLevel(logging.INFO)
    
    formatter = logging.Formatter("%(levelname)8s %(asctime)s %(name)s %(message)s")
    
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    
    now = datetime.datetime.now()
    now_time = now.strftime('%Y-%m-%d-%H:%M:%S')
    log_dir = Path("./logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{now_time}.log"
    file_handler = logging.FileHandler(str(log_file), encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    logger.info('Start App')
    
    main()