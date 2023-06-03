import matplotlib.pyplot as plt
import librosa.core as lc
import numpy as np
import librosa.display
from scipy.io import wavfile

path = "test.wav"
fs, y_ = wavfile.read(path)
n_fft = 1024
y, sr = librosa.load(path, sr=fs)
plt.figure(figsize=(17, 8))
#获取宽带声谱图
mag = np.abs(lc.stft(y, n_fft=n_fft, hop_length=10, win_length=40, window='hamming'))
D = librosa.amplitude_to_db(mag, ref=np.max)
librosa.display.specshow(D, sr=fs, hop_length=10, x_axis='s', y_axis='linear')


plt.colorbar(format='%+2.0f dB')
plt.title('broadband spectrogram',fontsize = 14)
plt.tick_params(axis = 'both', which = 'major', labelsize = 13)
plt.xlabel('Time',fontsize = 14)
plt.ylabel('Hz',fontsize = 14)
plt.show()

plt.figure(figsize=(17,8))
#获取窄带声谱图
mag1 = np.abs(lc.stft(y, n_fft=n_fft, hop_length=100, win_length=400, window='hamming'))
mag1_log = 20*np.log(mag1)
D1 = librosa.amplitude_to_db(mag1, ref=np.max)
librosa.display.specshow(D1, sr=fs, hop_length=100, x_axis='s', y_axis='linear')
plt.colorbar(format='%+2.0f dB')
plt.title('narrow spectrogram', fontsize=14)
plt.tick_params(axis='both', which='major', labelsize=13)
plt.xlabel('Time', fontsize=14)
plt.ylabel('Hz', fontsize=14)
plt.show()
