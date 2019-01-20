import numpy as np
import pickle
import struct

def mul_shift(max_u):
    for i in range(1,28):
        max_int=127.5*2**i
        if max_int>max_u:
            mul=round(max_int/max_u)
            temp=max_u*mul/2**i
            if(temp>127 and temp<127.5):
                shift=i
                return mul,shift
    return mul,i
def mul_shift_f(ratio):
    for i in range(10,28):
        max_int=2**i
        if max_int>ratio:
            temp=max_int/ratio
            mul=round(temp)
            if abs(max_int/mul-ratio)<0.02*ratio:
                shift=i
                return mul,shift
    return mul,i
def quant_qfp_layer(ratio,stepw,blu):
    blu_q=round(blu*ratio/stepw)
    mul,shifts=mul_shift(blu_q)
    # blu_adj*ratio/stepw*mul/2**shift=127
    blu_adj=127*2**shifts/mul*stepw/ratio
    blu_q=round(blu_adj*ratio/stepw)
    return [stepw,ratio,blu_adj,blu_q,mul,shifts]
def quant_qfp_concat(ratio,stepw1,blu1,stepw2,blu2):
    if blu1<blu2:
        blu1=blu2
    else:
        blu2=blu1
    blu_q1=round(blu1*ratio/stepw1)
    blu_q2=round(blu2*ratio/stepw2)
    mul1,shift1=mul_shift(blu_q1)
    mul2,shift2=mul_shift(blu_q2)

    if mul1/stepw1/2**shift1>mul2/stepw2/2**shift2:
        stepw1=stepw2*2**shift2/mul2*mul1/2**shift1
    else:
        stepw2=stepw1*2**shift1/mul1*mul2/2**shift2

    blu1_adj=127*2**shift1/mul1*stepw1/ratio
    blu2_adj=127*2**shift2/mul2*stepw2/ratio
    return [[stepw1,ratio,blu1_adj,blu_q1,mul1,shift1],[stepw2,ratio,blu2_adj,blu_q2,mul2,shift2]]
def quant_qfp_last(ratio,stepw):#最终ratio=255
    mul,shift=mul_shift_f(ratio/255/stepw)
    stepw_adj=ratio*mul/2**shift/255
    return [stepw_adj,ratio,0,0,mul,shift]

def adjust_quant(stepw_in,blu_in):
    ratio=255
    param1=quant_qfp_layer(ratio,stepw_in[0],blu_in[0])
    ratio=ratio/param1[0]*param1[4]/2**param1[5]
    [param2_1,param2_2]=quant_qfp_concat(ratio,stepw_in[1],blu_in[1],stepw_in[2],blu_in[2])
    ratio=ratio/param2_1[0]*param2_1[4]/2**param2_1[5]
    [param3_1,param3_2]=quant_qfp_concat(ratio,stepw_in[3],blu_in[3],stepw_in[4],blu_in[4])
    ratio=ratio/param3_1[0]*param3_1[4]/2**param3_1[5]
    param4=quant_qfp_last(ratio,stepw_in[5])
    return [param1,param2_1,param2_2,param3_1,param3_2,param4]

def quantNsave(wf,QP):
    #quant model
    stepw=[]
    if QP == 22:
        blu_init = [0.1111, 0.05, 0.05, 0.022, 0.022,0]#[-10,10]
    elif QP == 27:# observed 3sigma
        blu_init = [0.294,0.172,0.172,0.101,0.101, 0]#[-15,15]
    elif QP == 32:
        blu_init = [0.316,0.198,0.198,0.125,0.125,0]#[-20,20]
    else:# QP == 37
        blu_init = [0.349,0.243,0.243,0.169,0.169,0]#[-30,30]
    for w in wf:
        max=np.max(w)
        min=np.min(w)
        if max/127>-min/128:
            stepw.append(max / 127)
        else:
            stepw.append(-min / 128)

        #sigma=np.std(w)
        #stepw.append(sigma*4/127)
    print("init stepw:",stepw)
    quant_params=adjust_quant(stepw,blu_init)
    #save as list
    with open("quant_params%d.data"%QP,"wb") as fp:
        pickle.dump(quant_params,fp)
    # save as bin
    with open("quant_params_cpp_%d.data"%QP,"wb") as fp:
        for item in quant_params:
            print(item)
            fp.write(struct.pack('6d', *item))
    print("quant parameters saved:QP%d"%QP)
    return quant_params
def quant_w_b(wf,bf,stepw,ratio):
    wq=[]
    bq=[]
    for i in range(6):
        wq[i] = np.around(wf[i] / stepw[i]) * stepw[i]
        bq[i] = np.around(bf[i]*ratio[i] / stepw[i]) * stepw[i]/ratio
    return wq,bq

def loadInitBlu(QP):
    if QP == 22:
        blu_init = [0.265,0.140,0.140,0.0742,0.0742, 0]#[-10,10]
    elif QP == 27:  # observed 3sigma
        blu_init = [0.294,0.172,0.172,0.101,0.101, 0]  # [-15,15]
    elif QP == 32:
        blu_init = [0.316, 0.198, 0.198, 0.125, 0.125, 0]  # [-20,20]
    else:  # QP == 37
        blu_init = [0.349, 0.243, 0.243, 0.169, 0.169, 0]  # [-30,30]
    print("blu_init:",blu_init)
    return blu_init
def loadQpara(QP):
    stepw=[]
    blu=[]
    ratio=[]
    with open("quant_params%d.data"%QP,"rb") as fp:
        quant_params=pickle.load(fp)
    print("quant parameters loaded:QP%d"%QP)
    for param in quant_params:
        stepw.append(param[0])
        ratio.append(param[1])
        blu.append(param[2])
        print(param)
    return stepw,blu,ratio

def quant_blu_save(QP):
    if QP == 22:
        blu_init = [0.1111, 0.05, 0.05, 0.022, 0.022, 0]
    elif QP == 27:  # observed 3sigma
        blu_init = [0.294,0.172,0.172,0.101,0.101, 0]  # [-15,15]
    elif QP == 32:
        blu_init = [0.316, 0.198, 0.198, 0.125, 0.125, 0]  # [-20,20]
    else:  # QP == 37
        blu_init = [0.349, 0.243, 0.243, 0.169, 0.169, 0]  # [-30,30]
    stepw, blu, ratio=loadQpara(QP)
    quant_params = adjust_quant(stepw, blu_init)
    # save as list
    with open("quant_params%d.data" % QP, "wb") as fp:
        pickle.dump(quant_params, fp)
    # save as bin
    with open("quant_params_cpp_%d.data" % QP, "wb") as fp:
        for item in quant_params:
            print(item)
            fp.write(struct.pack('6d', *item))
    print("quant parameters saved:QP%d" % QP)
    for item in quant_params:
        print(item)
    return quant_params

if __name__ == '__main__':

    res=10
    a=128.0/res
    b=pow(a,1.0/4)
    print(res*b/255)
    print(res*b*b/255)
    print(res*b*b*b/255)
    print(res*b*b*b*b/255)
