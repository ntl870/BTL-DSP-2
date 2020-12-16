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
    AC = []
    temp = 0
    for k in range(0, len(data) - 1):
        temp += data[k]*data[k+1]
        AC.append(temp)
        temp = 0
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

Fs, data = read('./Resources/TinHieuMau/studio_male.wav')

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

peaks, _ = find_peaks(AC, height=0.01)


print(max(AC[peaks]))


index = np.where(AC[peaks] == max(AC[peaks]))

freq = Fs/(peaks[index[0] + 1] - peaks[index[0]])[0]

print(freq)
# freq = Fs/(AC[peaks][index + 1] - AC[peaks][index])
# print(freq)
# freq = []
# for i in range(0,len(peaks) - 1):
#     temp = peaks[i+1] - peaks[i]
#     temp = Fs/temp
#     if(temp < 400 and temp > 80):
#         freq.append(temp)


plt.stem(peaks ,AC[peaks])
plt.plot(AC,color='r')
plt.show()

# plt.subplot(2,1,2)
# plt.plot(data)
# plt.show()

# plt.subplot(2,1,2)
# plt.stem(pitch)
# plt.show()
