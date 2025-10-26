import warnings

warnings.filterwarnings('ignore')

from fastapi import FastAPI, HTTPException, UploadFile, File
import uvicorn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import io
import soundfile as sf

# Конфигурация
SAMPLE_RATE = 22050
N_MELS = 64
MAX_LEN = 1500
MIN_AUDIO_LENGTH = 1000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

app = FastAPI(title="Urban Sound Classifier")


# Модель
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
print("Загрузка модели...")
model = UrbanAudio().to(DEVICE)
model.load_state_dict(torch.load('urban_model.pth', map_location=DEVICE, weights_only=True))
model.eval()

# Загрузка меток
try:
    labels = torch.load('urban_labels.pth', map_location='cpu', weights_only=False)
    if isinstance(labels, torch.Tensor):
        labels = labels.tolist()
    labels = [str(l) for l in labels]
    print(f"Метки загружены из файла: {labels}")
except Exception as e:
    print(f"Ошибка загрузки меток: {e}")
    # Стандартные метки UrbanSound8K по порядку classID
    labels = [
        'air_conditioner', 'car_horn', 'children_playing', 'dog_bark',
        'drilling', 'engine_idling', 'gun_shot', 'jackhammer',
        'siren', 'street_music'
    ]
    print(f"Используются стандартные метки: {labels}")

print(f"Модель загружена. Классов: {len(labels)}")

# Трансформации
transform = nn.Sequential(
    torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLE_RATE, n_mels=N_MELS),
    torchaudio.transforms.AmplitudeToDB()
)


def preprocess_audio(waveform: torch.Tensor, sr: int) -> torch.Tensor:
    """Предобработка аудио"""
    if waveform.ndim > 1:
        waveform = waveform.mean(dim=0)

    if waveform.numel() == 0:
        raise ValueError("Аудиофайл пустой")

    if len(waveform) < MIN_AUDIO_LENGTH:
        waveform = F.pad(waveform, (0, MIN_AUDIO_LENGTH - len(waveform)))

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
        waveform = resampler(waveform.unsqueeze(0)).squeeze(0)

    spec = transform(waveform.unsqueeze(0)).squeeze(0)

    if spec.shape[1] > MAX_LEN:
        spec = spec[:, :MAX_LEN]
    elif spec.shape[1] < MAX_LEN:
        spec = F.pad(spec, (0, MAX_LEN - spec.shape[1]))

    return spec.unsqueeze(0).unsqueeze(0)


@app.post('/predict/')
async def predict(file: UploadFile = File(...)):
    """Предсказание класса аудио"""
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail='Пустой файл')

        waveform, sr = sf.read(io.BytesIO(data), dtype='float32')
        waveform = torch.tensor(waveform, dtype=torch.float32)

        spec = preprocess_audio(waveform, sr).to(DEVICE)

        with torch.no_grad():
            logits = model(spec)
            pred_idx = logits.argmax(dim=1).item()
            confidence = torch.softmax(logits, dim=1)[0, pred_idx].item()

        if pred_idx >= len(labels):
            raise ValueError(f"Индекс {pred_idx} вне диапазона (доступно {len(labels)} классов)")

        label = labels[pred_idx]

        return {
            "index": pred_idx,
            "sound": label,
            "confidence": round(confidence, 4),
            "all_scores": {labels[i]: round(torch.softmax(logits, dim=1)[0, i].item(), 4)
                           for i in range(len(labels))}
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка: {str(e)}")


@app.get('/')
async def root():
    return {
        "message": "Urban Sound Classifier API",
        "endpoint": "/predict/",
        "classes": labels,
        "num_classes": len(labels)
    }


@app.get('/health')
async def health():
    return {"status": "ok", "device": str(DEVICE), "num_classes": len(labels)}


if __name__ == '__main__':
    print(f"Документация: http://127.0.0.1:8000/docs")
    uvicorn.run(app, host='127.0.0.1', port=8000)