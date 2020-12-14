import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import read
from scipy.signal import find_peaks



def Normalize(data, min, max):  # Chuẩn hóa data về 0,1
    res = []    # Tạo LIST res rỗng để chứa kết quả
    for i in range(0, len(data)):   # Cho i chạy hết qua data
        res.append((data[i]-min)/(max-min))  # Đẩy kết quả vào LIST res
    return res


def CalculateAC(data):
    AC = np.zeros(int(len(data)))
    for i in range(0, int(len(data))):
        for k in range(i, int(len(data) -1)):
            AC[i] += data[k]*data[k-i]
    return AC


# def autocorr(x):
#     result = np.correlate(x, x, mode='full')
#     return result[result.size/2:]

# def autocorr1(x):
#     r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
#     return r2[:len(x)//2]

# def autocorr(x):
#     result = np.correlate(x, x, mode='full')
#     return result[result.size/2:]

Fs, data = read('./Resources/TinHieuMau/lab_male.wav')

altdata = Normalize(data, min(data), max(data))
# def autocorr2(x):
#     r2=np.fft.ifft(np.abs(np.fft.fft(x))**2).real
#     c=(r2/x.shape-np.mean(x)**2)/np.std(x)**2

#     return c[:len(x)//2]



def autocorr(data):
    y = data - np.mean(data)
    norm = np.sum(y ** 2)
    correlated = np.correlate(y, y, mode='same')/norm
    correlated = correlated[correlated.size //2:]
    return correlated

AC = autocorr(data)

mark = []

for i in range(0,len(AC)):
    if AC[i] > 0.018:
        mark.append(i)

# mark.append(len(AC))
sum = 0
pitch = []
for i in range(0,len(mark)):
 if(i+1 < len(mark)):
     if(mark[i+1] - mark[i] < 1000):
        sum += (mark[i+1] - mark[i])/len(mark)
        pitch.append(1/((mark[i + 1] - mark[i])/Fs))




print(1/(sum/Fs))


arrayX = []
for i in range(0,len(pitch)):
    arrayX.append(i)


plt.subplot(2,1,1)
plt.plot(AC)

plt.subplot(2,1,2)
plt.stem(pitch)
plt.show()
