import numpy as np, pandas as pd
CSV = r"C:\Users\manhn\Desktop\open-econs\tests\stata\fixtures\inputs\df_panel.csv"
df = pd.read_csv(CSV)
entities = sorted(df["entity"].unique())
YD = {e: df[df["entity"]==e].sort_values("time")["y"].to_numpy(float) for e in entities}
XD = {e: df[df["entity"]==e].sort_values("time")["x"].to_numpy(float) for e in entities}
ZD = {e: df[df["entity"]==e].sort_values("time")["z"].to_numpy(float) for e in entities}

def build_ni(js, dg0=1, ndg=4, lgn=2, form="D"):
    nz=ndg+2+lgn+2+1; Yd,Xd,Zd,Yl,Xl,Zl,E=[],[],[],[],[],[],[]
    for e in entities:
        y,x,z=YD[e],XD[e],ZD[e]
        for j in js:
            Yd.append(y[j]-y[j-1]); Xd.append([y[j-1],x[j]-x[j-1],z[j]-z[j-1],0.0])
            zr=np.zeros(nz)
            for d in range(ndg):
                idx=j-1-dg0-d; zr[d]=y[idx] if idx>=0 else 0.0
            zr[ndg]=x[j]-x[j-1]; zr[ndg+1]=z[j]-z[j-1]; Zd.append(zr); E.append(e)
        for j in js:
            Yl.append(y[j]); Xl.append([y[j-1],x[j],z[j],1.0])
            zr=np.zeros(nz); base=ndg+2
            if form=="D":
                zr[base]=y[j-1]-y[j-2] if j-2>=0 else 0.0
                if lgn>=2: zr[base+1]=y[j-2]-y[j-3] if j-3>=0 else 0.0
            else:
                zr[base]=y[j-1] if j-1>=0 else 0.0
                if lgn>=2: zr[base+1]=y[j-2] if j-2>=0 else 0.0
            zr[base+lgn]=x[j]; zr[base+lgn+1]=z[j]; zr[base+lgn+2]=1.0; Zl.append(zr); E.append(e)
    return np.array(Yd+Yl), np.array(Xd+Xl), np.array(Zd+Zl), np.array(E)

A1=pd.read_csv(r"C:\Users\manhn\Desktop\open-econs\tests\stata\generate-fixtures\archive\A1.csv").values
for (dg0,ndg,form) in [(1,4,"D"),(1,4,"L"),(2,3,"D")]:
    Yb,Xb,Zb,Eb=build_ni([2,3,4],dg0,ndg,2,form)
    ZtX=Zb.T@Xb; ZtY=Zb.T@Yb
    A1pi=np.linalg.pinv(A1)
    M=ZtX.T@A1pi@ZtX
    b=np.linalg.pinv(M)@(ZtX.T@A1pi@ZtY)
    print("dg0=%d ndg=%d form=%s b_Ly=%.6f b_x=%.6f b_z=%.6f b_cons=%.6f (target .009464 1.134976 -.442064 .090758)"%(dg0,ndg,form,b[0],b[1],b[2],b[3]))
