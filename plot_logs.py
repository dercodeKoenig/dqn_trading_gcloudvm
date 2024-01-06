import matplotlib.pyplot as plt
import numpy as np


def get_values(fn):
    f = open(fn)
    lines = f.readlines()
    f.close()

    lines = [float(x.replace("\n","")) for x in lines]
    for i in range(len(lines)):
        if np.isnan(lines[i]):
            lines[i] = 0
         
    return lines

avg_n=50
k=2

def plot_logs(losses, qs, name):
    fig, ax = plt.subplots(1,2,figsize=(14,5))
    ax[0].set_title("loss")
    lower, upper = np.percentile(losses, [k, 100-k])
    ax[0].set_ylim(lower, upper)
    ax[0].plot(losses, alpha=0.3)
    ax[0].plot([np.mean(losses[max(0,i-avg_n):i+1]) for i in range(len(losses))], c="r")
    ax[1].set_title("estimated q values")
    lower, upper = np.percentile(qs, [k, 100-k])
    ax[1].set_ylim(lower, upper)
    ax[1].plot(qs, alpha=0.3)
    ax[1].axhline(0)
    ax[1].plot([np.mean(qs[max(0,i-avg_n):i+1]) for i in range(len(qs))], c="r")
    plt.savefig(name)
            

losses = get_values("t1/loss.txt")
qs = get_values("t1/qv.txt")
plot_logs(losses, qs, "t1.jpg")

losses = get_values("t2/loss.txt")
qs = get_values("t2/qv.txt")
plot_logs(losses, qs, "t2.jpg")

losses = get_values("t3/loss.txt")
qs = get_values("t3/qv.txt")
plot_logs(losses, qs, "t3.jpg")

losses = get_values("t4/loss.txt")
qs = get_values("t4/qv.txt")
plot_logs(losses, qs, "t4.jpg")