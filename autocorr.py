import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from scipy.signal import medfilt
import scipy.stats.stats as st
def Normalize(data, min, max):  # Chuẩn hóa data về 0,1
    res = []    # Tạo LIST res rỗng để chứa kết quả
    for i in range(0, len(data)):   # Cho i chạy hết qua data
        res.append((data[i]-min)/(max-min))  # Đẩy kết quả vào LIST res
    return res


Fs, data = read('./Resources/TinHieuMau/lab_male.wav')

altdata = []
for i in range(len(data)):
    altdata.append(data[i])

splited_data = np.array_split(altdata, len(data)/(Fs*0.03))


def autocorr(data):
    y = data - np.mean(data)
    norm = np.sum(y ** 2)
    correlated = np.correlate(y, y, mode='same')/norm
    # correlated = correlated[correlated.size // 2:]
    return correlated


def getF0(data):
    peaks, _ = find_peaks(data, height=0.0)
    altdata = data[peaks]
    altdata = np.sort(altdata)
    for i in range(1, len(altdata)):
        temp = altdata[i]
        index, = np.where(data == temp)
        for i in range(0, len(index)):
            if(index[i] <= 200 and index[i] >= 40):
                return index[i]
                break


res = []
for i in range(0, len(splited_data)):
    temp = []
    temp = autocorr(splited_data[i])
    temp = getF0(temp)
    res.append(temp)

for i in range(0,len(res)):
    if(res[i] == None):
        res[i] = 120

res = medfilt(res,19)

print(Fs/np.mean(res))

# def Framing(Fs, data):
#     num_frames = int(Fs*0.03)
#     res = []
#     x = 0
#     while(x < len(data) - num_frames):
#         temp = autocorr
#         res.append(getMax(data, int(x), int(num_frames + x)))
#         x = x + num_frames
#     return res


# frames = Framing(Fs, AC)
# plotfreq = np.zeros(len(frames)-1)
# for i in range(0, len(frames) - 1):
#     temp = frames[i+1] - frames[i]
#     if(temp >= Fs/400 and temp <= Fs/80):
#         plotfreq[i] = (Fs/temp)
#     else:
#         plotfreq[i] = np.nan


arrayFreq = []
for i in range(0, len(res)):
    arrayFreq.append(int(i*Fs*0.03))





# print(np.nanmean(plotfreq))

# print(ACtest[peaks])

# plt.subplot(3, 1, 1)
# plt.plot(res,'.')
# plt.stem(peaks,ACtest[peaks])


plt.subplot(3, 1, 2)
plt.plot(arrayFreq,res, '.')

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
