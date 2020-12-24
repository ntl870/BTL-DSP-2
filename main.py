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


def UnVoicedToZero(threshold):
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
            amdf = Normalize(amdf, min(amdf), max(amdf))
            peaks, _ = find_peaks(amdf, height=0.3)

            # Selection sort mảng peaks theo chiều tawng dần của amdf[peaks]
            for i in range(len(peaks)):
                min_idx = i
                for j in range(i+1, len(peaks)):
                    if amdf[peaks[min_idx]] < amdf[peaks[j]]:
                        min_idx = j
                peaks[i], peaks[min_idx] = peaks[min_idx], peaks[i]
            # Kiểm tra điều kiện các peak để xác định tần số chính xác
            for i in range(1, len(peaks)):
                if(abs(peaks[i]-peaks[0]) >= Fs/350 and abs(peaks[i]-peaks[0]) <= Fs/75):
                    F0_amdf.append(Fs/abs(peaks[i]-peaks[0]))
                    break
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
                if(abs(peaks[i]-peaks[0]) >= Fs/350 and abs(peaks[i]-peaks[0]) <= Fs/75):
                    F0_autocorr.append(Fs/abs(peaks[i]-peaks[0]))
                    break
    F0_autocorr.append(0)
    return F0_autocorr

# Hàm tính FFT


def FFT(data):      # Đầu vào là data
    # Dùng hàm fft của scipy để tính với n = 16384 = 2^14
    res = abs(fft(data, n=16384))
    # Vì fft sinh ra hàm chẵn nên ta lấy nửa đầu để tính toán
    res = res[:res.size//2]
    return res  # Trả về mảng giá trị res

def FFT_Hamming(data):      # Đầu vào là data
    # Dùng hàm fft của scipy để tính với n = 16384 = 2^14
    res = fft(data, n=16384)
    # Vì fft sinh ra hàm chẵn nên ta lấy nửa đầu để tính toán
    res = res[:res.size//2]
    return res  # Trả về mảng giá trị res

# ------------------------------------------------

# Hàm tính FFT với từng khung


def CalculateFFT():
    # Tạo mảng chứa giá trị F0 của khung với 1 giá trị là 0 để cân chỉnh đồ thị
    F0_FFT = [0]
    for n in range(len(MA)-1):  # Cho biến n chạy qua hết MA
        # Nếu độ dài của Splitdata là 2 thì Splitdata là khoảng lặng
        if (len(SplitData[n]) == 2):
            F0_FFT.append(None)     # Cho vào mảng F0 giá trị None
        else:   # Nếu không thì Splitdata là tiếng nói
            # Tạo mảng fftvalue chưa giá trị của FFT
            fftvalue = FFT(SplitData[n])
            fftvalue = Normalize(fftvalue, min(fftvalue), max(
                fftvalue))    # Chuẩn hóa fftvalue
            # Dùng hàm find_peaks với ngưỡng là 0.3 để tìm các cực đại
            peaks, _ = find_peaks(fftvalue, height=0.3)
            for i in range(1, len(peaks)):  # Duyệt qua mảng peaks
                # Điều kiện tần số cơ bản từ 75 Hz - 350 Hz
                if(abs(peaks[i]-peaks[i-1]) >= 75 and abs(peaks[i]-peaks[i - 1]) <= 350):
                    # Đưa giá trị tần số
                    F0_FFT.append(abs(peaks[i]-peaks[i-1]))
                    break   # Nếu có giá trị break
    F0_FFT.append(0)    # Trả giá trị 0 vào cuối mảng để cân chỉnh đồ thị
    return F0_FFT   # Trả về mảng chứa f0
# ------------------------------------------------

# Hàm tính FFT + Hamming với từng khung
def Calculate_FFT_Hamming():
    # Tạo mảng chứa giá trị F0 của khung với 1 giá trị là 0 để cân chỉnh đồ thị
    F0_FFT_Hamming = [0]
    for n in range(len(MA) - 1):  # Cho biến n chạy qua hết MA
        # Nếu độ dài của Splitdata là 2 thì Splitdata là khoảng lặng
        if (len(SplitData[n]) == 2):
            F0_FFT_Hamming.append(None)  # Cho vào mảng F0 giá trị None
        else:  # Nếu không thì Splitdata là tiếng nói
            # Tạo vector cửa sổ hamming với độ dài là bằng 1 khung độ dài 30ms
            hamming = np.hamming(int(Fs*0.03))
            # Nhân 2 vector hamming và Splitdata[n]
            FFTHamming = np.convolve(hamming, SplitData[n])
            FFTHamming = FFT_Hamming(FFTHamming)  # Tính toán FFT của FFTHamming
            FFTHamming = Normalize(FFTHamming, min(FFTHamming), max(
                FFTHamming))  # Chuẩn hóa FFTHamming về 0,1
            # Dùng findpeaks tìm các cực đại lớn hơn 0
            peaks, _ = find_peaks(FFTHamming, height=0.0)
            positionMax = 1  # Tạo biến chứa vị trí của cực đại cao nhất
            MAX = 0  # Tạo biến MAX chứa giá trị của cực đại cao nhất
            for i in range(len(peaks)):  # Duyệt qua mảng peaks
                # Điều kiện tần số cơ bản từ 75 Hz - 350 Hz
                if (peaks[i] >= Fs/350 and peaks[i] <= Fs/75):
                    # Tìm max trong khoảng
                    if (FFTHamming[peaks[i]] > MAX):
                        MAX = FFTHamming[peaks[i]]
                        positionMax = peaks[i]
                    # ----------------
            F0_FFT_Hamming.append(positionMax)  # Sau khi tìm ra vị trí max đưa vào mảng chứa F0
    F0_FFT_Hamming.append(0)    # Đưa 0 vào cuối mảng để cân chỉnh đồ thị
    return F0_FFT_Hamming # Trả về mảng chưa F0
#------------------------------------------------------------------

# Hàm tính trung bình F0
def AverageF0(freq):  # Đầu vào mảng freq
    count = 0 # Biến đếm phần tử count
    sum = 0 # Biến sum chứa tổng
    for i in range(len(freq)):  # Duyệt qua mảng F0
        if (freq[i] != None and freq[i] != 0):  # Nếu giá trị tần số khác None và 0 thì hợp lệ
            sum += freq[i]  # Tính tổng  
            count += 1 # Tăng số lần tính lên 1
    return sum/count # Trả về giá trị trung bình là thương sum và count

#-------------------------------------------------------------------


# ------------------------------------------MAIN---------------------------------------


Fs, data = read('./Resources/TinHieuMau/lab_male.wav') # Đọc tính hiệu đầu vào

# Block code sau đây để phân biện khoảng lặng tiếng nói bằng phương pháp MA với ngưỡng đã khảo sát là 0.1
MA = [] # Tạo mảng MA rỗng
SplitData = []  # Mảng Splitdata để chưa từng khung 
CalculateMA_SplitFrame(Fs, data)    # Cắt data thành nhiều khung lưu vào mảng 2 chiều
MA = Normalize(MA, min(MA), max(MA)) # Chuẩn hóa MA về 0,1
UnVoicedToZero(0.1)    # Đưa những khoảng lặng về 0 với ngưỡng 0.1
#------------------------------------

# Tính toán trả về list F0 sử dụng hàm tự tương quan
ListFreq_autocorr = autocorr() # Gán giá trị của hàm tự tương quan vào mảng
ListFreq_autocorr = medfilt(ListFreq_autocorr, 21)  # Lọc trung vị với độ dài là 21
ListFreq_autocorr = np.insert(ListFreq_autocorr, 0, 0)  # Đưa vào đầu giá trị 0 để cân đồ thị
print("Tan so co ban trung binh cua TTQ: ", AverageF0(ListFreq_autocorr)) # In ra giá trị trung bình của F0
#-----------------------------------------


# Tính toán trả về list F0 sử dụng AMDF
ListFreq_amdf = AMDFunction()    # Gán giá trị của hàm AMDF vào mảng
ListFreq_amdf = medfilt(ListFreq_amdf, 21) # Lọc trung vị với độ dài là 21
ListFreq_amdf = np.insert(ListFreq_amdf, 0, 0) # Đưa vào đầu giá trị 0 để cân đồ thị
print("Tan so co ban trung binh cua AMDF: ", AverageF0(ListFreq_amdf)) # In ra giá trị trung bình của F0
#-----------------------------------------

# Tính toán trả về list F0 sử dụng FFT
ListFreq_FFT = CalculateFFT() # Gán giá trị của FFT vào mảng
ListFreq_FFT = medfilt(ListFreq_FFT, 21)  # Lọc trung vị với độ dài là 21
ListFreq_FFT = np.insert(ListFreq_FFT, 0, 0) # Đưa vào đầu giá trị 0 để cân đồ thị
print("Tan so co ban trung binh cua FFT: ", AverageF0(ListFreq_FFT)) # In ra giá trị trung bình của F0
#-----------------------------------------

# Tính toán trả về list F0 sử dụng FFT + Hamming
ListFreq_FFT_Hamming = Calculate_FFT_Hamming() # Gán giá trị của FFT+Hamming vào mảng
ListFreq_FFT_Hamming = medfilt(ListFreq_FFT_Hamming, 21) # Lọc trung vị với độ dài là 21
ListFreq_FFT_Hamming = np.insert(ListFreq_FFT_Hamming, 0, 0) # Đưa vào đầu giá trị 0 để cân đồ thị
print("Tan so co ban trung binh cua FFT + Hamming: ",AverageF0(ListFreq_FFT_Hamming))  # In ra giá trị trung bình của F0
#-----------------------------------------


# In ra đồ thị
plt.subplot(5, 1, 1)
plt.title("Studio_male")
plt.plot(ListFreq_autocorr, '.')
plt.ylabel("Frequencies")

plt.subplot(5, 1, 2)
plt.plot(ListFreq_amdf, '.')
plt.ylabel("Frequencies")

plt.subplot(5, 1, 3)
plt.plot(ListFreq_FFT, '.')
plt.ylabel("Frequencies")

plt.subplot(5, 1, 4)
plt.plot(ListFreq_FFT_Hamming, '.')
plt.ylabel("Frequencies")

plt.subplot(5, 1, 5)
plt.plot(data)
plt.xlabel("Samples")

plt.show()
#-----------------