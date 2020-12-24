def AMDFunction():
    F0_amdf =[0]
    for n in range(len(MA)-1):
        if (len(SplitData[n])==2 ):
            F0_amdf.append(None)
        else:
            amdf = AMDF(SplitData[n],SplitData[n])
            amdf=amdf[int(len(amdf)/2): -1]
            for i in range(len(amdf)):
                amdf[i] = -amdf[i]
            peaks, _ = find_peaks(amdf, threshold=-1)
            positionmax = 1
            for i in range(2,len(peaks)-1):
                if(peaks[i] >= Fs/400 and peaks[i] <= Fs/80 ):
                    if amdf[peaks[i]] > amdf[peaks[positionmax]]:
                        positionmax = i
            F0_amdf.append(peaks[positionmax])
    F0_amdf.append(0)
    return F0_amdf