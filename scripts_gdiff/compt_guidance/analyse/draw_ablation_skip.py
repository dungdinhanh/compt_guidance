
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure


skips=[2, 3, 4, 5,  6, 7, 8, 9, 10, 11]
IS=[99.17, 85.51, 78.42, 73.15, 69.85, 67.29, 65.46, 64.73, 63.57, 62.72]
FID=[5.15, 3.49, 2.71, 2.27, 2.09, 2.05, 1.96, 1.9, 1.84, 1.9]
Recall=[0.52, 0.55, 0.57, 0.59, 0.59, 0.6, 0.61, 0.61, 0.61,0.61]

def draw_figure(skips, values, name ,file_name):
    # fig = plt.figure()
    # fig.tight_layout()
    # figure(figsize=(7, 7))
    plt.plot(skips, values)
    plt.xlabel("Compact rate $\\frac{T}{|G|}$", fontsize=18)
    plt.ylabel(f"{name}", fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    # plt.subplots_adjust(left=1, right=1.1, top=1.1, bottom=1)
    plt.subplots_adjust(left=0.143, right=0.99, top=0.99, bottom=0.16)
    plt.savefig(file_name)
    plt.close()

def draw_tradeoff(values1, values2, name1, name2 ,file_name):
    plt.plot(values1, values2)
    # plt.plot()
    plt.xlabel(f"{name1}", fontsize=18)
    plt.ylabel(f"{name2}", fontsize=18)
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    fig = plt.figure()
    fig.tight_layout()
    plt.savefig(file_name)
    plt.close()


draw_figure(skips, IS, "IS" ,"runs/IS.pdf")
draw_figure(skips, FID, "FID" ,"runs/FID.pdf")
draw_figure(skips, Recall, "Recall" ,"runs/Rec.pdf")
draw_tradeoff(IS, FID, "IS", "FID", "runs/tradeoff_is_fid.pdf")