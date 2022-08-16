from pathlib import Path

from transformers import (
    Wav2Vec2ForCTC,
    Wav2Vec2Processor,
    pipeline,
)

def inference(audio_path: Path):
    model = Wav2Vec2ForCTC.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    processor = Wav2Vec2Processor.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    
    generator = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=2.0
    )

    output = generator(str(audio_path.absolute()))
    return output