from fastapi import FastAPI, HTTPException, UploadFile, File
from torchvision import transforms
import torchaudio.transforms as T
import streamlit as st
from PIL import Image
import torch.nn as nn
import torchaudio
import librosa
import numpy as np
import uvicorn
import tempfile
import torch
import io
import os


classes = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

transform = T.MelSpectrogram(
     sample_rate = 22050,
     n_mels = 64
)

class CheckAudio(nn.Module):
    def __init__(self):
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.second = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, audio):
        audio = audio.unsqueeze(1)
        audio = self.first(audio)
        audio = self.second(audio)
        return audio

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CheckAudio()
model.load_state_dict(torch.load('audioGTZAN.pth', map_location=device))
model.to(device)
model.eval()

st.title('Audio Genre Classifier')
st.text('Загрузите аудио, и модель попробует её распознать.')

mnist_audio = st.file_uploader('Выберите аудио', type=['wav', 'mp3', 'flac', 'ogg'])

if not mnist_audio:
    st.info('Загрузите аудио')
else:
    st.audio(mnist_audio)

    if st.button('Распознать'):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                tmp_file.write(mnist_audio.read())
                tmp_path = tmp_file.name

            waveform, sample_rate = librosa.load(tmp_path, sr=22050)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
            os.unlink(tmp_path)

            mel_spec = transform(waveform)
            mel_spec = mel_spec.mean(dim=0) if mel_spec.dim() == 3 else mel_spec
            mel_spec = mel_spec.unsqueeze(0).to(device)

            with torch.no_grad():
                y_prediction = model(mel_spec)
                prediction = y_prediction.argmax(dim=1).item()

            st.success(f'Модель думает, что это: {classes[prediction]}')

        except Exception as e:
            st.error(f'Ошибка: {str(e)}')




# app = FastAPI()
#
# @app.post('/predict')
# async def check_image(file:UploadFile = File(...)):
#     try:
#         data = await file.read()
#         if not data:
#             raise HTTPException(status_code=400, detail='File not Found')
#
#         img = Image.open(io.BytesIO(data))
#         img_tensor = transform(img).unsqueeze(0).to(device)
#
#         with torch.no_grad():
#             prediction = model(img_tensor)
#             result = prediction.argmax(dim=1).item()
#             return {f'class': classes[result]}
#
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f'{e}')
#
# if __name__ == '__main__':
#     uvicorn.run(app, host='127.0.0.1', port=8000)