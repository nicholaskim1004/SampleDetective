import librosa
import matplotlib.pyplot as plt
import numpy as np

y, sr = librosa.load("/Users/nicholaskim/Documents/Repositories/sample detector/songs/Ostavi trag_September.mp3")

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram of Ostavi Trag')
plt.show()

y, sr = librosa.load("/Users/nicholaskim/Documents/Repositories/sample detector/songs/DUCKWORTH_Kendrick_Lamar.mp3")

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,fmax=8000)

fig, ax = plt.subplots()
S_dB = librosa.power_to_db(S, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr,
                         fmax=8000, ax=ax)
fig.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram of DUCKWORTH')
plt.show()