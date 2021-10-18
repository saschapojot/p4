from only5Funcs import *



g=0.1

nVals=list(range(0,10))

fig,ax=plt.subplots(figsize=(20,20))

targetVals=[(nTmp+1/2)*np.pi for nTmp in nVals]
ax.scatter(targetVals,[0]*len(targetVals),color="red",marker="+",s=40)

r1=2#adj
r2=4#sep
tStart=datetime.now()
thetaAll=np.linspace(start=-np.pi,stop=np.pi,num=100)
int5Adj=[]
int5AdjOpp=[]
int5Sep=[]
int5SepOpp=[]
for j in range(0,5):
    int5Adj.append([])
    int5AdjOpp.append([])
    int5Sep.append([])
    int5SepOpp.append([])

for thTmp in thetaAll:
    ETmpAdj=r1*np.exp(1j*thTmp)
    ETmpSep=r2*np.exp(1j*thTmp)
    rtsAdjPairs=return5AdjacentPairs(g,ETmpAdj)
    # rtsSepPairs=return5SeparatedPairs(g,ETmpSep)
    for j in range(0,5):
        x2Adj,x1Adj=rtsAdjPairs[j]
        int5Adj[j].append(integralQuadrature(g,ETmpAdj,x1Adj,x2Adj))
        # print(int5Adj)
        # int5AdjOpp.append(integralQuadrature(g,ETmpAdj,x2Adj,x1Adj))
    # for j in range(0,5):
    #     x2Sep,x1Sep=rtsSepPairs[j]
    #     int5Sep.append(integralQuadrature(g,ETmpSep,x1Sep,x2Sep))
    #     int5SepOpp.append(integralQuadrature(g,ETmpSep,x2Sep,x1Sep))

tEnd=datetime.now()
print("time : ",tEnd-tStart)

intRealAdj=[]
intImagAdj=[]
adjColor="k"
for j in range(0,5):
    for intTmp in int5Adj[j]:
        intRealAdj.append(np.real(intTmp))
        intImagAdj.append(np.imag(intTmp))
ax.scatter(intRealAdj,intImagAdj,color=adjColor)
plt.savefig("tmp.png")