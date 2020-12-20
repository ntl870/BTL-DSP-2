import numpy as np
import matplotlib.pyplot as plt
import scipy
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
    # correlated = correlated[correlated.size // 2:]
    return correlated


AC = autocorr(data)


def getMax(arr, a, b):
    max_index = a
    for i in range(a, b + 1):
        if(arr[max_index] < arr[i]):
            max_index = i
    return max_index

def Framing(Fs, data):
    num_frames = int(Fs*0.01)
    res = []
    max_i = 0
    x = 0
    while(x < len(data) - num_frames):
        res.append(getMax(data, int(x), int(num_frames + x)))
        x = x + num_frames
    return res


frames = Framing(Fs, AC)
plotfreq = np.zeros(len(frames)-1)
for i in range(0, len(frames) - 1):
    temp = frames[i+1] - frames[i]
    if(temp >= Fs/400 and temp <= Fs/80):
        plotfreq[i] = (Fs/temp)
    else:
        plotfreq[i] = np.nan





arrayFreq = []
for i in range(0, len(plotfreq)):
    arrayFreq.append(int(i*Fs*0.01))




plotfreq = scipy.signal.medfilt(plotfreq,5)


print(np.nanmean(plotfreq))


plt.subplot(3, 1, 1)
plt.plot(AC, color='r')

plt.subplot(3, 1, 2)
plt.plot(arrayFreq,plotfreq, '.')

plt.subplot(3, 1, 3)
plt.plot(data)

plt.show()

# peaks, _ = find_peaks(freq, rel_height=0.01)

# index = np.where(AC[peaks] == max(AC[peaks]))

# freq = peaks[index[0][0] + 1] - peaks[index[0][0]]
# freq = Fs/freq
# print(index[0][0])


# arrayX = []
# for i in range(len(peaks)):
#     arrayX.append(i)


# print(np.mean(freq))
