from only5Funcs import *

num=50
startG=1e-2
stopG=1e-1
gnIndAll = np.linspace(start=np.log10(startG), stop=np.log10(stopG), num=num)
gAll = [10 ** elem for elem in gnIndAll]

threadNum = 24
# energyLevelMax = 4
levelStart=0
levelEnd=3
levelsAll = range(levelStart, levelEnd + 1)
inDataAll=[]

for nTmp in levelsAll:
    for gTmp in gAll:
        EEst=(nTmp+1/2)*np.pi
        inDataAll.append([nTmp,gTmp,EEst])



# ###########parallel computation part for adj, may be memory consuming
tWKBParalStart = datetime.now()
pool1 = Pool(threadNum)
retAllAdj=pool1.map(computeOneSolutionWith5AdjPairs,inDataAll)
tWKBParalEnd = datetime.now()
print("parallel WKB time for adj pairs: ", tWKBParalEnd - tWKBParalStart)

#############end of parallel computation

tPltStart = datetime.now()

# # plot WKB
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_ylabel("E")
ax.set_xlabel("g")
ax.set_title("Eigenvalues for potential $V(x)=x^{2}-igx^{5}$")

# data serialization for adj
nSctValsAdj = []
gSctValsAdj= []
ERealSctValsAdj = []
EImagSctValsAdj = []

#data serialization for adj
for itemTmp in retAllAdj:
    nTmp, gTmp, ERe, EIm = itemTmp
    if np.abs(EIm)>1e-2:
        continue
    nSctValsAdj.append(nTmp)
    gSctValsAdj.append(gTmp)
    ERealSctValsAdj.append(ERe)
    EImagSctValsAdj.append(EIm)


sctRealPartWKBAdj = ax.scatter(gSctValsAdj, ERealSctValsAdj, color="red", marker=".", s=50, label="WKB real part adj")
plt.legend()
# dirName="/home/users/nus/e0385051/Documents/pyCode/wkb/wkbp4/"
plt.savefig("AdjSeplevelStart"+str(levelStart)+"levelEnd"+str(levelEnd)+"NonLogstart"+str(startG)+"stop"+str(stopG)+"num"+str(num)+"tmp100.png")

tPltEnd = datetime.now()
print("plotting time: ", tPltEnd - tPltStart)