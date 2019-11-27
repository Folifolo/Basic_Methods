import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from dataset import num_to_diag


def plt_top_hists(b, Y, diag, num_comp, plot = True):
    ret = []

    for k in diag:
        for j in num_comp:
            tmp1 = []
            tmp0 = []
            maxPN = 0
            maxppv = 0
            maxnpv = 0
            v_max = 0
            for i in range(700):
                if Y[i, k] == 1:
                    tmp1.append(b[i,j])
                else:
                    tmp0.append(b[i,j])
            tmp1.sort()
            tmp0.sort()

            tmp = tmp1+tmp0
            tmp.sort()
            for v in tmp:
                l0 = l1 = r0 = r1 = 0
                for v0 in tmp0:
                    if v0<=v:
                        l0 += 1
                    else:
                        r0 = len(tmp0) - l0
                        break

                for v1 in tmp1:
                    if v1<=v:
                        l1 += 1
                    else:
                        r1 = len(tmp1) - l1
                        break
                if l1/(r1+1) + r0/(l0+1) > l0/(r0+1) + r1/(l1+1):
                    tp = l1
                    tn = r0
                    fp = r1
                    fn = l0
                else:
                    tp = r1
                    tn = l0
                    fp = l1
                    fn = r0
                PPV = tp/(tp+fp)
                NPV = tn/(tn+fn)
                if PPV+NPV > maxPN and PPV > 0.5 and NPV >0.5:
                    maxPN = PPV+NPV
                    maxppv = PPV
                    maxnpv = NPV
                    v_max = v
            ret.append(maxPN)
            bins = np.linspace(min(min(tmp0), min(tmp1)), max(max(tmp0), max(tmp1)), 30)

            if plot:
                y_axes = np.arange(0, 1, 0.5)
                figure(figsize=(12,5))
                plt.plot([v_max]*2, y_axes, color = 'k')
                plt.hist(tmp0, bins=bins, weights=np.ones(len(tmp0)) / len(tmp0), color = 'r', alpha = 0.5)
                plt.hist(tmp1, bins=bins, weights=np.ones(len(tmp1)) / len(tmp1), color = 'b', alpha = 0.5)
                plt.title("PPV = " + str(round(maxppv, 4))+", NPV = " + str(round(maxnpv,4)) + ", diag = " +num_to_diag(k)+ ", num = " + str(j))
                plt.savefig("C:\\Users\\donte_000\\PycharmProjects\\exper1\\pics2\\hist_component_" + str(j)+"diag_"+str(k) + ".png")
                #plt.show()
                plt.clf()
    return ret



#if __name__ == "__main__":
