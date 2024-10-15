import whisper
import text2text
model = whisper.load_model("base")

from IPython.display import Audio
Audio("Ar_F114_Rania (mp3cut.net).mp3")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("Ar_F114_Rania (mp3cut.net).mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
# text2text.translate("Tunisian Arabic","English",result.text)
