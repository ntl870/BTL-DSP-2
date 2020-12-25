import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy.fftpack import fft
Fs, data = read('./Resources/TinHieuMau/lab_female.wav')

def Normalize(data, min, max):  # Chuẩn hóa data về 0,1
    res = []    # Tạo LIST res rỗng để chứa kết quả
    for i in range(0, len(data)):   # Cho i chạy hết qua data
        res.append((data[i]-min)/(max-min))  # Đẩy kết quả vào LIST res
    return res

hamm = np.hamming(len(data))
y = hamm*data
y = abs(fft(y, n=16384))
y = y[:y.size//2]

altY = []
for i in range(75, 350):
    altY.append(y[i])
altY = Normalize(altY,min(altY),max(altY))
peaks, _ = find_peaks(altY,height=0.6)
max = 0
pos = 0
if(len(peaks) == 1):
    pos = peaks[0]
else:
    for i in range(len(peaks)):
        if peaks[i] - pos > 30 :
             break
    if altY[peaks[i]] > max:
        max = altY[peaks[i]]
        pos = peaks[i]
    

print(pos + 75)
print(peaks)
plt.subplot(2,1,1)
plt.title("FFT+Hamming lab_female")
plt.plot(y, color='r')
plt.subplot(2,1,2)
plt.title("Data")
plt.plot(data)
# plt.stem(peaks, altY[peaks])
# plt.subplot(2,1,2)
# plt.plot(y)

plt.show()