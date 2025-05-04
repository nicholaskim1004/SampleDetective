import librosa
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict,OrderedDict

y1, sr1 = librosa.load("/Users/nicholaskim/Documents/Repositories/sample detector/songs/Ostavi trag_September.mp3")

Ostavi = librosa.feature.melspectrogram(y=y1, sr=sr1, n_mels=128,fmax=4000)

fig1, ax = plt.subplots()
S_dB = librosa.power_to_db(Ostavi, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr1,
                         fmax=4000, ax=ax)
fig1.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram of Ostavi Trag')
plt.show()

y2, sr2 = librosa.load("/Users/nicholaskim/Documents/Repositories/sample detector/songs/DUCKWORTH_Kendrick_Lamar.mp3")

Duck = librosa.feature.melspectrogram(y=y2, sr=sr2, n_mels=128,fmax=4000)

fig2, ax = plt.subplots()
S_dB = librosa.power_to_db(Duck, ref=np.max)
img = librosa.display.specshow(S_dB, x_axis='time',
                         y_axis='mel', sr=sr2,
                         fmax=4000, ax=ax)
fig2.colorbar(img, ax=ax, format='%+2.0f dB')
ax.set(title='Mel-frequency spectrogram of DUCKWORTH')
plt.show()

def peak_finder(song):
    
    y, sr = librosa.load(song)

    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))
    frequencies = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
    bands= [[20,60],[60,250],[250,500],[500,2000],[2000,6000]]

    constellation = []
    
    for band in bands:
        start = band[0]
        end = band[1]
            
        band_mask = (frequencies >= start) & (frequencies < end)
        band_energy = S[band_mask, :]
        peaks = np.max(band_energy, axis=1) 
        peak_ind = np.argmax(band_energy, axis=1)

        peak_times = librosa.frames_to_time(peak_ind)
        
        for i in range(len(peak_times)):
            constellation.append((peak_times[i],peaks[i]))
        
        sorted_constellation = sorted(constellation)
            
    return sorted_constellation
        
ostava_peaks = peak_finder("songs/Ostavi trag_September.mp3")
duckworth_peaks = peak_finder("songs/DUCKWORTH_Kendrick_Lamar.mp3")

def hasher(points, target_size=5, quantize=20):
    fingerprint = defaultdict(list)
    for i in range(len(points)):
        t_anchor, f_anchor = points[i]
        for j in range(1,target_size+1):
            if i+j >= len(points):
                break
            t_target, f_target = points[i+j]
            
            delta_t = t_target - t_anchor
            
            quant_anchor = int(f_anchor // quantize)
            quant_target = int(f_target // quantize)
            
            h = hash((int(quant_target),int(quant_anchor),round(delta_t,3)))
            fingerprint[h] = (t_target,f_target)
    return fingerprint

o_fingerprint = hasher(ostava_peaks,5)
d_fingerprint = hasher(duckworth_peaks,5)

match = []
match_loc = []

for id in o_fingerprint:
    if id in d_fingerprint:
        f_a = o_fingerprint[id][1]
        f_tar = d_fingerprint[id][1]
        
        delta = f_tar - f_a 
        
        match.append(delta)
        
        if np.abs(delta) < 1:
            t_a = o_fingerprint[id][0]
            t_tar = d_fingerprint[id][0]
            match_loc.append((id,t_a,t_tar))
match_loc