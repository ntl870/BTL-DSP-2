import numpy as np
import matplotlib.pyplot as plt
import scipy
from scipy.io.wavfile import read
from scipy.signal import find_peaks
from scipy.signal import medfilt
from scipy.fftpack import fft


def Normalize(data, min, max):  # Chuẩn hóa data về 0,1
    res = []    # Tạo LIST res rỗng để chứa kết quả
    for i in range(0, len(data)):   # Cho i chạy hết qua data
        res.append((data[i]-min)/(max-min))  # Đẩy kết quả vào LIST res
    return res


def CalculateMA_SplitFrame(Fs, data):  # Hàm tính MA
    dur = len(data) / Fs  # Độ dài của tín hiệu âm thanh
    t = np.arange(0, dur, 0.03)  # Chia khoảng 0.03s bằng hàm arange của numpy
    # Tạo mảng MA toàn 0 với độ dài của t với hàm zeros của numpy với kiểu dữ liệu là float64
    MAtemp = 0
    n1 = 0
    x = int(0.03*Fs)  # x là độ dài 1 khung
    Element = np.zeros(x)

    # Xác định tương đối 1 khoảng chia gặp bao nhiêu thơi gian lấy mẫu ( = 0.03/T)

    for n in range(len(t)):  # Cho biến n chạy hết qua E
        while ((n*x + n1) < len(data)):  # Đảm bảo vẫn ở trong khoảng của data
            MAtemp += abs(data[n*x + n1])     # Công thức tính của MA
            Element[n1] = data[n*x+n1]
            n1 += 1  # Tăng biến n1 lên 1 đơn vị
            if (n1 == x):  # Nếu n1 bằng x thì đang duyệt đến vị trí cuối khung
                n1 = 0  # Đưa n1 = 0 để đến khung tiếp theo
                SplitData.append(Element)
                Element = np.zeros(x)
                MA.append(MAtemp)
                MAtemp = 0
                break  # break để dừng

# Lọc MA với ngưỡng cho trước, xác định các khoảng lặng và gán giá trị khoảng lặng với [0,0]


def AltData(threshold):
    for i in range(len(MA)):
        if (MA[i] < 0.1):
            for j in range(int(Fs*0.03)):
                SplitData[i] = [0, 0]


def correlate(arra, arrb):
    cor = []
    sum = 0
    for i in range(len(arra)):
        position = abs(round((len(arra)-1)/2)-i)
        for j in range(len(arra) - position):
            sum += arra[j] * arrb[j+position]
        cor.append(sum)
        sum = 0
    cor = cor[len(cor)//2:]
    return cor


def AMDF(arra, arrb):
    cor = []
    sum = 0
    for i in range(len(arra)):
        position = abs(round((len(arra)-1)/2)-i)
        for j in range(len(arra) - position):
            sum += abs(arra[j] - arrb[j+position])
        cor.append(sum)
        sum = 0
    cor = cor[len(cor)//2:]
    return cor


def AMDFunction():
    F0_amdf = [0]
    for n in range(len(MA)-1):
        if (len(SplitData[n]) == 2):
            F0_amdf.append(None)
        else:
            amdf = AMDF(SplitData[n], SplitData[n])
            for i in range(len(amdf)):
                amdf[i] = -amdf[i]
            peaks, _ = find_peaks(amdf, threshold=-1)
            positionmax = 1
            for i in range(2, len(peaks)-1):
                if(peaks[i] >= Fs/400 and peaks[i] <= Fs/80):
                    if amdf[peaks[i]] > amdf[peaks[positionmax]]:
                        positionmax = i
            F0_amdf.append(peaks[positionmax])
    F0_amdf.append(0)
    return F0_amdf


def autocorr():
    F0_autocorr = [0]
    for n in range(len(MA)-1):
        if (len(SplitData[n]) == 2):
            F0_autocorr.append(None)
        else:
            correlated = correlate(SplitData[n], SplitData[n])
            peaks, _ = find_peaks(correlated, height=0.0)
            # Selection sort mảng peaks theo chiều giảm dần của correlated[peaks]
            for i in range(len(peaks)):
                max_idx = i
                for j in range(i+1, len(peaks)):
                    if correlated[peaks[max_idx]] < correlated[peaks[j]]:
                        max_idx = j
                peaks[i], peaks[max_idx] = peaks[max_idx], peaks[i]
            # Kiểm tra điều kiện các peak để xác định tần số chính xác
            for i in range(1, len(peaks)):
                if(abs(peaks[i]-peaks[0]) >= Fs/400 and abs(peaks[i]-peaks[0]) <= Fs/80):
                    F0_autocorr.append(Fs/abs(peaks[i]-peaks[0]))
                    break
    F0_autocorr.append(0)
    return F0_autocorr


def FFT(data):
    res = abs(fft(data, n=16384))
    res = res[:res.size//2]
    return res


def CalculateFFT():
    F0_FFT = [0]
    for n in range(len(MA)-1):
        if (len(SplitData[n]) == 2):
            F0_FFT.append(None)
        else:
            ffted = FFT(SplitData[n])
            ffted = Normalize(ffted, min(ffted), max(ffted))
            peaks, _ = find_peaks(ffted, height=0.3)
            for i in range(1, len(peaks)):
                if(abs(peaks[i]-peaks[i-1]) >= 75 and abs(peaks[i]-peaks[i - 1]) <= 350):
                    F0_FFT.append(abs(peaks[i]-peaks[i-1]))
                    break
    F0_FFT.append(0)
    return F0_FFT



def Average(freq):
    count = 0
    sum = 0
    for i in range(len(freq)):
        if (freq[i] != None and freq[i] != 0):
            sum += freq[i]
            count += 1
    return sum/count

# ------------------------------------------MAIN---------------------------------------


Fs, data = read('./Resources/TinHieuMau/lab_female.wav')

MA = []
SplitData = []
CalculateMA_SplitFrame(Fs, data)

MA = Normalize(MA, min(MA), max(MA))
AltData(0.1)


ListFreq_autocorr = autocorr()
ListFreq_autocorr = medfilt(ListFreq_autocorr, 9)
ListFreq_autocorr = np.insert(ListFreq_autocorr, 0, 0)
print(Average(ListFreq_autocorr))

ListFreq_amdf = AMDFunction()
ListFreq_amdf = medfilt(ListFreq_amdf, 9)
ListFreq_amdf = np.insert(ListFreq_amdf, 0, 0)
print(Average(ListFreq_amdf))


ListFreq_FFT = CalculateFFT()
ListFreq_FFT = medfilt(ListFreq_FFT, 9)
ListFreq_FFT = np.insert(ListFreq_FFT, 0, 0)
print(Average(ListFreq_FFT))


plt.subplot(4, 1, 1)
plt.plot(ListFreq_autocorr, '.')

plt.subplot(4, 1, 2)
plt.plot(ListFreq_amdf, '.')

plt.subplot(4, 1, 3)
plt.plot(ListFreq_FFT, '.')

plt.subplot(4, 1, 4)
plt.plot(data)

plt.show()
