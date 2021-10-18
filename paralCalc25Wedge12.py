from funcs25Wedges12 import *

num=20
startG=1e-3
stopG=1e-2
gnIndAll = np.linspace(start=np.log10(startG), stop=np.log10(stopG), num=num)
gAll = [10 ** elem for elem in gnIndAll]
EWKB = []


threadNum = 24
# energyLevelMax = 4
levelStart=0
levelEnd=3
levelsAll = range(levelStart, levelEnd + 1)
inDataAll=[]

for nTmp in levelsAll:
    for gTmp in gAll:
        EEst=(nTmp+1/2)*np.pi+0.01j
        inDataAll.append([nTmp,gTmp,EEst])


tWKBParalStart = datetime.now()
pool1 = Pool(threadNum)
retAll=pool1.map(computeOneSolutionWith5AdjacentPairs,inDataAll)
tWKBParalEnd = datetime.now()
print("WKB time: ", tWKBParalEnd - tWKBParalStart)

tPltStart = datetime.now()

# plot WKB
fig, ax = plt.subplots(figsize=(20, 20))
ax.set_ylabel("E")
ax.set_xlabel("g")
ax.set_title("Eigenvalues for potential $V(x)=x^{2}-igx^{5}$")

# data serialization
nSctVals = []
gSctVals = []
ERealSctVals = []
EImagSctVals = []
#data serialization
for itemTmp in retAll:
    nTmp, gTmp, ERe, EIm = itemTmp
    nSctVals.append(nTmp)
    gSctVals.append(gTmp)
    ERealSctVals.append(ERe)
    EImagSctVals.append(EIm)

sctRealPartWKB = ax.scatter(gSctVals, ERealSctVals, color="red", marker=".", s=50, label="WKB real part")
plt.legend()
# dirName="/home/users/nus/e0385051/Documents/pyCode/wkb/wkbp4/"
plt.savefig("SeplevelStart"+str(levelStart)+"levelEnd"+str(levelEnd)+"NonLogstart"+str(startG)+"stop"+str(stopG)+"num"+str(num)+"tmp125.png")

tPltEnd = datetime.now()
print("plotting time: ", tPltEnd - tPltStart)