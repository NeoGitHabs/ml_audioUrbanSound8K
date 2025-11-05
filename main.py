from fastapi import FastAPI, HTTPException, UploadFile, File
import streamlit as st
import torch.nn as nn
import torchaudio
import torch.nn.functional as F
import torch
import tempfile
import soundfile as sf
import os

# Классы Urban Sound 8K
classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling',
           'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

# Параметры из notebook
SAMPLE_RATE = 22050
N_MELS = 64
MAX_LEN = 500

transform = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS)


class UrbanAudio(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# Загрузка модели
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = UrbanAudio()
model.load_state_dict(torch.load('audioUrbanSound8K.pth', map_location=device))
model.to(device)
model.eval()

st.title('Audio Urban Sound 8K')
st.text('Загрузите аудио, и модель попробует его распознать.')

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

            # Загрузка через soundfile
            waveform, sr = sf.read(tmp_path)
            waveform = torch.from_numpy(waveform).float()

            # Преобразование в формат [channels, samples]
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            else:
                waveform = waveform.T

            os.unlink(tmp_path)

            # Моно + ресемплинг
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != SAMPLE_RATE:
                resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
                waveform = resampler(waveform)

            # Спектрограмма + паддинг
            spec = transform(waveform)
            spec = spec[..., :MAX_LEN] if spec.shape[-1] > MAX_LEN else F.pad(spec, (0, MAX_LEN - spec.shape[-1]))
            spec = spec.unsqueeze(0).to(device)

            with torch.no_grad():
                y_prediction = model(spec)
                prediction = y_prediction.argmax(dim=1).item()

            st.success(f'Модель думает, что это: {classes[prediction]}')

        except Exception as e:
            st.error(f'Ошибка: {str(e)}')