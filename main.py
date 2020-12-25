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
    MAtemp = 0  # tạo giá trị để lưu tạm MA
    n1 = 0  # Khởi tạo giá trị ban đầu của chỉ số trong mảng
    x = int(0.03*Fs)  # x là độ dài 1 khung
    Element = np.zeros(x)  # tạo mảng tạm để lưu data trong một khung

    # Xác định tương đối 1 khoảng chia gặp bao nhiêu thơi gian lấy mẫu ( = 0.03/T)

    for n in range(len(t)):  # Cho biến n chạy hết qua E
        while ((n*x + n1) < len(data)):  # Đảm bảo vẫn ở trong khoảng của data
            MAtemp += abs(data[n*x + n1])     # Công thức tính của MA
            Element[n1] = data[n*x+n1]  # Lưu data vào mảng tạm
            n1 += 1  # Tăng biến n1 lên 1 đơn vị
            if (n1 == x):  # Nếu n1 bằng x thì đang duyệt đến vị trí cuối khung
                n1 = 0  # Đưa n1 = 0 để đến khung tiếp theo
                # Hết khung, push mảng tạm vào mảng 2 chiều
                # push Element vào mảng 2 chiều lưu các khung SplitData
                SplitData.append(Element)
                Element = np.zeros(x)  # xóa giá trị trong Element
                MA.append(MAtemp)  # push giá trị MAtemp vào mảng chứa MA
                MAtemp = 0  # trả lại giá trị temp =0
                break  # break để dừng

# Lọc MA với ngưỡng cho trước, xác định các khoảng lặng và gán giá trị khoảng lặng với [0,0]


def UnVoicedToZero(threshold):  # Dùng phương pháp MA để phân biệt khoảng lặng và tiếng nói
    for i in range(len(MA)):  # Duyệt tất cả phần tử trong MA
        if (MA[i] < threshold):  # Nếu bé hơn ngưỡng thì gán 0 cho khoảng lặng
            SplitData[i] = [0, 0]


# Hàm nhân tương quan 2 ma trận
def correlate(arra, arrb):
    cor = []  # tạo mảng tạm để lưu mảng kết quả
    sum = 0  # khởi tạo giá trị lưu phần tử trong mảng
    for i in range(len(arra)):  # Chạy tất cả các phần tử trong mảng A
        # đánh dấu vị trí để băts đầu và kết thúc
        position = abs(round((len(arra)-1)/2)-i)
        for j in range(len(arra) - position):
            sum += arra[j] * arrb[j+position]
        cor.append(sum)  # push phần tử vào mảng tạm
        sum = 0  # trả lại giá trị 0 cho biến tạm
    cor = cor[len(cor)//2:]  # hàm đối xứng nên xứng, chỉ duyệt nửa sau
    return cor  # trả về giá trị correlate

# Hàm tự tương quan


def autocorr():
    # Tạo mảng F lưu các tần số cơ bản tại mỗi khung, tạo giá trị đầu bằng 0
    F0_autocorr = [0]
    for n in range(len(MA)-1):  # Số lượng phần tử trong MA bằng số lượng khung lưu trong SplitData
        # Vì sau khi dùng hàm AltData với ngưỡng, các khung trong khoảng tiếng nói sẽ trở thành mảng 2 phần tử [0,0]
        if (len(SplitData[n]) == 2):
            F0_autocorr.append(None)  # Đặt giá trị NULL vào mảng
        else:
            # Mảng tạm để lưu giá trị phép bình phương correlate của khung SplitData[n]
            correlated = correlate(SplitData[n], SplitData[n])
            # xác định các peaks xuất hiện trong mảng correlated
            peaks, _ = find_peaks(correlated, height=0.0)
            # Selection sort mảng peaks theo chiều giảm dần của correlated[peaks]
            for i in range(len(peaks)):
                max_idx = i
                for j in range(i+1, len(peaks)):
                    if correlated[peaks[max_idx]] < correlated[peaks[j]]:
                        max_idx = j
                # Swap 2 giá trị với nhau
                peaks[i], peaks[max_idx] = peaks[max_idx], peaks[i]
            # Kiểm tra điều kiện các peak để xác định tần số chính xác
            for i in range(1, len(peaks)):
                # Điều kiện tần số tiếng nói trong khoảng 75 tới 350
                if(abs(peaks[i]-peaks[0]) >= Fs/350 and abs(peaks[i]-peaks[0]) <= Fs/75):
                    # Nếu thỏa mãn điều kiện thì push giá trị vào mảng F
                    F0_autocorr.append(Fs/abs(peaks[i]-peaks[0]))
                    break  # Ngắt vòng lặp vì đã xác định được F0 của khung
    F0_autocorr.append(0)  # tạo giá trị cuối bằng 0
    return F0_autocorr

# Hàm tính AMDF của 2 mảng a và b


def AMDF(arra, arrb):
    amdf = []  # tạo mảng tạm để lưu kết quả trả vể
    sum = 0  # tạo phần tử tạm để lưu giá trị trong mảng
    for i in range(len(arra)):  # Duyệt tất cả phần tử trong mảng, tính cor
        position = abs(round((len(arra)-1)/2)-i)
        for j in range(len(arra) - position):
            sum += abs(arra[j] - arrb[j+position])
        amdf.append(sum)  # push giá trị tính được vào mảng tạm
        sum = 0  # trả lại giá trị 0 cho biến tạm
    amdf = amdf[len(amdf)//2:]  # Hàm đối xứng nên chỉ lấy một nửa để xử lí
    return amdf

# Hàm tính AMDF của cả data


def AMDFunction():
    F0_amdf = [0]  # tạo mảng 0 để lưu giá trị F0 của mỗi khung
    # Duyệt các phần tử trong mảng SpitData, có len(MA) bằng số mẫu trong SplitData
    for n in range(len(MA)-1):
        # Vì sau khi dùng hàm AltData với ngưỡng, các khung trong khoảng tiếng nói sẽ trở thành mảng 2 phần tử [0,0]
        if (len(SplitData[n]) == 2):
            F0_amdf.append(None)  # Đặt giá trị NULL vào mảng
        else:
            # Mảng tạm để lưu giá trị phép bình phương amdf của khung SplitData[n]
            amdf = AMDF(SplitData[n], SplitData[n])
            for i in range(len(amdf)):  # lật ngược mảng amdf để tìm peaks
                amdf[i] = -amdf[i]
            amdf = Normalize(amdf, min(amdf), max(amdf)
                             )  # chuẩn hóa amdf về 0,1
            # tìm peaks với ngưỡng là 0.3
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
                # Điều kiện tần số tiếng nói trong khoảng 75 tới 350
                if(abs(peaks[i]-peaks[0]) >= Fs/350 and abs(peaks[i]-peaks[0]) <= Fs/75):
                    # Nếu thỏa mãn điều kiện thì push giá trị vào mảng F
                    F0_amdf.append(Fs/abs(peaks[i]-peaks[0]))
                    break  # Ngắt vòng lặp vì đã xác định được F0 của khung
    F0_amdf.append(0)  # tạo giá trị cuối bằng 0
    return F0_amdf  # trả về mảng các F0 của từng khung

# Hàm tính FFT


def FFT(data):      # Đầu vào là data
    # Dùng hàm fft của scipy để tính với n = 16384 = 2^14
    res = abs(fft(data, n=16384))
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

# Hàm tính FFT + Hamming


def Calculate_FFT_Hamming():
    # Tạo biến hamm chứa vector cửa sổ hamming chiều dài bằng data
    hamm = np.hamming(len(data))
    y = hamm*data  # Biến y là tích của vector hamm và data
    y = abs(fft(y, n=16384))  # FFT vector y với n = 16384 = 2^14
    y = y[:y.size//2]  # Vì y là hàm chẵn nên ta lấy nửa đầu để xét
    altY = []  # Tạo mảng phụ rỗng
    for i in range(75, 350):  # Duyệt trong khoảng 75hz - 350hz
        altY.append(y[i])  # Cho giá trị vào mảng
    altY = Normalize(altY, min(altY), max(altY))  # Chuẩn hóa altY
    # Dùng hàm find peaks để tìm những cực đại với ngưỡng 0.6
    peaks, _ = find_peaks(altY, height=0.6)
    MAX = 0  # Biến tạm MAX để lưu giá trị của cực đại
    pos = 0  # Biến tạm pos để lưu vị trí của cực đại
    if(len(peaks) == 1):  # Nếu mảng peaks có đọ dài bằng 1 thì trả về trả vị trí đó
        pos = peaks[0]
    else:
        for i in range(len(peaks)):  # Duyệt qua mảng peaks
            if peaks[i] - pos > 30:  # Nếu không có vị trí cực đại nào trong khoảng 30hz thì break
                break
        if altY[peaks[i]] > MAX:  # Tìm vị trí cực đại
            MAX = altY[peaks[i]]
            pos = peaks[i]
    return pos + 75  # Trả về vị trị

# ------------


# Hàm để plot đồ thị
def PlotHamming(data):
    hamm = np.hamming(len(data))
    y = hamm*data
    y = abs(fft(y, n=16384))
    y = y[:y.size//2]
    return y
# ------------------------------------------------------------------

# Hàm tính trung bình F0


def AverageF0(freq):  # Đầu vào mảng freq
    count = 0  # Biến đếm phần tử count
    sum = 0  # Biến sum chứa tổng
    for i in range(len(freq)):  # Duyệt qua mảng F0
        if (freq[i] != None and freq[i] != 0):  # Nếu giá trị tần số khác None và 0 thì hợp lệ
            sum += freq[i]  # Tính tổng
            count += 1  # Tăng số lần tính lên 1
    return sum/count  # Trả về giá trị trung bình là thương sum và count

# -------------------------------------------------------------------


# ------------------------------------------MAIN---------------------------------------


# Đọc tính hiệu đầu vào
Fs, data = read('./Resources/TinHieuMau/studio_female.wav')

# Block code sau đây để phân biện khoảng lặng tiếng nói bằng phương pháp MA với ngưỡng đã khảo sát là 0.1
MA = []  # Tạo mảng MA rỗng
SplitData = []  # Mảng Splitdata để chưa từng khung
# Cắt data thành nhiều khung lưu vào mảng 2 chiều
CalculateMA_SplitFrame(Fs, data)
MA = Normalize(MA, min(MA), max(MA))  # Chuẩn hóa MA về 0,1
UnVoicedToZero(0.1)    # Đưa những khoảng lặng về 0 với ngưỡng 0.1
# ------------------------------------

# Tính toán trả về list F0 sử dụng hàm tự tương quan
ListFreq_autocorr = autocorr()  # Gán giá trị của hàm tự tương quan vào mảng
# Lọc trung vị với độ dài là 21
ListFreq_autocorr = medfilt(ListFreq_autocorr, 9)
# Đưa vào đầu giá trị 0 để cân đồ thị
ListFreq_autocorr = np.insert(ListFreq_autocorr, 0, 0)
print("Tan so co ban trung binh cua TTQ: ", AverageF0(
    ListFreq_autocorr))  # In ra giá trị trung bình của F0
# -----------------------------------------


# Tính toán trả về list F0 sử dụng AMDF
ListFreq_amdf = AMDFunction()    # Gán giá trị của hàm AMDF vào mảng
ListFreq_amdf = medfilt(ListFreq_amdf, 9)  # Lọc trung vị với độ dài là 21
# Đưa vào đầu giá trị 0 để cân đồ thị
ListFreq_amdf = np.insert(ListFreq_amdf, 0, 0)
print("Tan so co ban trung binh cua AMDF: ", AverageF0(
    ListFreq_amdf))  # In ra giá trị trung bình của F0
# -----------------------------------------

# Tính toán trả về list F0 sử dụng FFT
ListFreq_FFT = CalculateFFT()  # Gán giá trị của FFT vào mảng
# ListFreq_FFT = medfilt(ListFreq_FFT, 21)  # Lọc trung vị với độ dài là 21
# Đưa vào đầu giá trị 0 để cân đồ thị
ListFreq_FFT = medfilt(ListFreq_FFT,9)
ListFreq_FFT = np.insert(ListFreq_FFT, 0, 0)
print("Tan so co ban trung binh cua FFT: ", AverageF0(
    ListFreq_FFT))  # In ra giá trị trung bình của F0
# -----------------------------------------

# Tính toán trả về list F0 sử dụng FFT + Hamming
# Gán giá trị của FFT+Hamming vào mảng
Freq_FFT_Hamming = Calculate_FFT_Hamming()
print("Tan so co ban tinh bang ket hop FFT + Hamming: ", Freq_FFT_Hamming)
# Lọc trung vị với độ dài là 21
# ListFreq_FFT_Hamming = medfilt(ListFreq_FFT_Hamming, 9)
# Đưa vào đầu giá trị 0 để cân đồ thị
# -----------------------------------------


# In ra đồ thị
plt.subplot(5, 1, 1)
plt.title("studio_female")
plt.plot(ListFreq_autocorr, '.')
plt.legend(['Autocorrelation'])
plt.ylabel("Frequencies")

plt.subplot(5, 1, 2)
plt.plot(ListFreq_amdf, '.')
plt.legend(['AMDF'])
plt.ylabel("Frequencies")

plt.subplot(5, 1, 3)
plt.plot(ListFreq_FFT, '.')
plt.legend(['FFT'])
plt.ylabel("Frequencies")

plt.subplot(5, 1, 4)
plt.plot(PlotHamming(data))
plt.legend(['FFT + Hamming'])
plt.ylabel("Frequencies")

plt.subplot(5, 1, 5)
plt.plot(data)
plt.legend(['Data'])
plt.xlabel("Samples")


plt.show()
# -----------------
