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


def CalculateMA(Fs, data):  # Hàm tính MA
    dur = len(data) / Fs  # Độ dài của tín hiệu âm thanh
    t = np.arange(0, dur, 0.02)  # Chia khoảng 0.02s bằng hàm arange của numpy
    # Tạo mảng MA toàn 0 với độ dài của t với hàm zeros của numpy với kiểu dữ liệu là float64
    MA = np.zeros(len(t))
    n1 = 0
    # Xác định tương đối 1 khoảng chia gặp bao nhiêu thơi gian lấy mẫu ( = 0.02/T)
    x = int(0.02*Fs)  # x là độ dài 1 khung
    for n in range(len(MA)):  # Cho biến n chạy hết qua E
        while ((n*x + n1) < len(data)):  # Đảm bảo vẫn ở trong khoảng của data
            MA[n] += abs(data[n*x + n1])     # Công thức tính của MA
            n1 += 1  # Tăng biến n1 lên 1 đơn vị
            if (n1 == x):  # Nếu n1 bằng x thì đang duyệt đến vị trí cuối khung
                n1 = 0  # Đưa n1 = 0 để đến khung tiếp theo
                break  # break để dừng
    return MA  # Trả lại MA cho hàm


def GetEdges(data, threshold):  # Hàm tìm biên tham số vào là data và ngưỡng đã khảo sát

    # tạo mảng altData toàn giá trị 0 với độ dài bằng độ dài data
    altData = np.zeros(len(data))
    for i in range(0, len(data)):  # Cho biên i duyệt qua hết data
        if data[i] <= threshold:  # Nếu biên độ tại i nhỏ hơn ngưỡng ta đặt biên độ tại đó bằng 0
            altData[i] = 0
        else:   # Nếu không trả lại giá trị nguyên vẹn
            altData[i] = data[i]

# Tiếp theo dùng LIST altData để xử lý tìm biên
    res = []    # Tạo mảng res
    for i in range(0, len(altData)-1):  # Cho i chạy hết qua data
        if(altData[i+1] > 0.0 and altData[i] == 0.0):  # Lấy biên bên trái
            res.append(i)   # Cho biên vào mảng res
        elif(altData[i-1] > 0.0 and altData[i] == 0.0):  # Lấy biên bên phải
            res.append(i)   # Cho biên vào mảng res

# Tại đây khi đã có các biên ta cần lọc những biên bị sai do sự nhiễu giữa các tiếng nói

# Đây là dãy code để lọc những biên ở quá sát nhau trong khoảng 12 đoạn
    u = 0       # Cho biến phụ u = 0
    while(u < len(res)):    # Điều kiện để u trong khoảng độ dài là len(res)
        temp = res[u]   # Cho biến phụ temp bằng res[u]
        for k in range(1, 13):  # Cho biến k chạy từ 1 đến 12
            if temp + k in res:  # Nếu temp + k có trong res thì xóa temp + k bằng hàm .remove
                res.remove(temp + k)
            if temp - k in res:  # Nếu temp - k có trong res thì xóa temp - k bằng hàm .remove
                res.remove(temp - k)
        u += 1  # Tăng biến u lên 1 đơn vị
# -------------------------------------------------------

# Sau khi đã lọc những khoảng sát nhau thì các tiếng nói vẫn còn dư những biên còn lại chứ kho
# Block code sau sẽ loại những biên đó
    temp1 = []  # Tạo mảng temp1 rỗng để chứa những phần tử cần xóa
    for i in range(0, len(res)):    # Cho biến i duyệt qua res
        for k in range(1, 13):  # Cho biến k
            if(res[i] - k > 1 and res[i] + k < len(altData) - 1):
                if altData[res[i] + k] > 0 and altData[res[i] - k] > 0:
                    temp1.append(res[i])
                    break
# Tiến hành xóa những biên lỗi đi bằng hàm .remove
    for i in range(0, len(temp1)):
        res.remove(temp1[i])

# -----------------------------------------------------------
    return res
def GetRealEdges(edges, Fs):
    res = []  # Khởi tạo mảng để lưu các biên trên data
    for i in range(len(edges)):  # Duyệt các phần tử trong edges
        # Theo định nghĩa của tần số lấy mẫu với độ dài mỗi khung = 0.02
        res.append(edges[i]*int(0.02*Fs))
    return res


Fs, data = read('./Resources/TinHieuMau/studio_male.wav')


MA = CalculateMA(Fs, data)
MA = Normalize(MA, min(MA), max(MA))
edge = GetEdges(MA, 0.1)
edge = np.array_split(edge,int(len(edge)/2))
edge = GetRealEdges(edge,Fs)

def SlicingArray(data,x):  
    altdata = data
    res = []
    for i in range(len(x)):
        temp = altdata[x[i][0]:x[i][1] + 1]    
        res.append(temp)
    return res

sliced = SlicingArray(data,edge)

# splited_data = np.array_split(altdata, int(len(data)/(Fs*0.03)))
def correlate(arra, arrb):
    cor = []
    sum = 0
    for i in range(len(arra)):
        position = abs((round(len(arra)-1)/2)-i)
        for j in range(len(arra) - position):
            sum += arra[j] * arrb[j+position]
        cor.append(sum)
        sum = 0
    return cor


def autocorr(data):
    altdata = np.array_split(data,int(len(data)/(Fs*0.03)))
    res = []
    for i in range(0,len(altdata)):
        y = data - np.mean(data)
        norm = np.sum(y ** 2)
        # correlated = np.correlate(y, y, mode='same')/norm
        correlated = correlate(y,y)/norm
        res.append(correlated)
    return res

def getF0(data):
    peaks, _ = find_peaks(data, height=0.0)
    altdata = data[peaks]
    altdata = np.sort(altdata)
    for i in range(1, len(altdata)):
        temp = altdata[i]
        index, = np.where(np.isclose(data, temp))
        for i in range(0, len(index)):
            if(index[i] <= 200 and index[i] >= 40):
                return index[i]
                break


res = []
for i in range(0, len(sliced)):
    temp = []
    temp = autocorr(sliced[i])
    for k in range(0,len(temp)):
        res.append(getF0(temp[k]))

res = medfilt(res, 9)

print(res)






print(Fs/np.mean(res))



arrayFreq = []
for i in range(0, len(res)):
    arrayFreq.append(int(i*Fs*0.03))

plt.subplot(3, 1, 2)
plt.plot(arrayFreq,res,'.')
# plt.stem(peaks,ACtest[peaks])


# plt.subplot(3, 1, 2)
# plt.plot(arrayFreq, res, '.')

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
