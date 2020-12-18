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
    correlated = correlated[correlated.size // 2:]
    return correlated


AC = autocorr(data)


# def Framing(Fs, data): # Hàm tính STE
#     dur = len(data) / Fs  # Độ dài của tín hiệu âm thanh
#     t = np.arange(0, dur, 0.02)  # Chia khoảng 0.02s bằng hàm arange của numpy
#     # Tạo mảng E toàn 0 với độ dài của t với hàm zeros của numpy với kiểu dữ liệu là float64
#     n1 = 0  # Biến n1 phụ để chạy qua mỗi khung
#     res = []
#     max_i = 0
#     # Xác định tương đối 1 khoảng chia gặp bao nhiêu thơi gian lấy mẫu ( = 0.02/T)
#     x = int(0.02*Fs)  # x là độ dài 1 khung
#     for n in range(len(data)):  # Cho biến n chạy hết qua E
#         while ((n*x + n1) < len(data)):  # Đảm bảo vẫn ở trong khoảng của data
#             if(data[n*x + n1] > data[max_i]):
#                 max_i = n*x + n1
#             n1 += 1  # Tăng biến n1 lên 1 đơn vị
#             if (n1 == x):   # Nếu n1 bằng x thì đang duyệt đến vị trí cuối khung
#                 n1 = 0  # Đưa n1 = 0 để đến khung tiếp theo
#                 break   # break để dừng
#         res.append(max_i)
#     return res    # Trả lại E cho hàm

# print(Framing(Fs,AC))
def getMax(arr,a,b):
    max_index = a
    for i in range(a,b + 1):
        if(arr[max_index] < arr[i]):
            max_index = i
    return max_index

def Framing(Fs, data):
    num_frames = Fs*0.01
    res = []
    max_i = 0
    x = 0
    while(x < len(data) - num_frames):
        res.append(getMax(data,int(x),int(num_frames + x)))
        x = x + num_frames
    return res


frames = Framing(Fs,AC)
freq = []
for i in range(0,len(frames) - 1):
    temp = frames[i+1] - frames[i]
    if(temp != 0):
        temp = Fs/temp
        if(temp < 400 and temp > 80):
            freq.append(temp)

print(np.mean(freq))
# peaks, _ = find_peaks(AC, height=0.01)

# index = np.where(AC[peaks] == max(AC[peaks]))

# freq = peaks[index[0][0] + 1] - peaks[index[0][0]]
# freq = Fs/freq
# print(index[0][0])


# arrayX = []
# for i in range(len(peaks)):
#     arrayX.append(i)


# print(np.mean(freq))

plt.stem(frames,AC[frames])
plt.plot(AC, color='r')
plt.show()
