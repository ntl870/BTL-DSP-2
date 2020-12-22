def correlate(arra,arrb):
    cor = []
    sum =0
    for i in range(len(arra)):
        position =  abs(round(0(len(arra)-1)/2)-i)
        print()
        print(position,i)

        for j in range(len(arra)- position):
            print(arra[j],arrb[j+position])
            sum+= arra[j]* arrb[j+position]
        cor.append(sum)
        sum =
    return cor