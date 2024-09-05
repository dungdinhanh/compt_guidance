import os

import matplotlib.pyplot as plt
import numpy as np


def read_file(guidance_file="../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting/IMN64_withxt/unconditional/scale4.0_skip1/reference/loss.npz",
              compact_file="../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting/IMN64_withxt/unconditional/scale4.0_skip1/reference/loss.npz"):

    guidance_losses = np.load(guidance_file)
    guidance_losses_training = guidance_losses["arr_0"]
    guidance_losses_testing = guidance_losses["arr_1"]

    compact_losses = np.load(compact_file)
    compact_losses_training = compact_losses["arr_0"]
    compact_losses_testing = compact_losses["arr_1"]

    return guidance_losses_training, guidance_losses_testing, compact_losses_training, compact_losses_testing

def visualize(guidance_losses_training, guidance_losses_testing, compact_losses_training, compact_losses_testing, file_name):
    timesteps = np.arange(guidance_losses_training.shape[0])[::-1]
    plt.plot(timesteps[100:], guidance_losses_training[100:],  label="G on-sampling loss", color='C2', linewidth=2.5)
    plt.plot(timesteps[100:], guidance_losses_testing[100:], '--', label="G w/Resnet152 off-sampling loss", color='C2', linewidth=2.5)
    plt.plot(timesteps[100:], compact_losses_training[100:], label="CompG on-sampling loss", color='C3', linewidth=2.5)
    plt.plot(timesteps[100:], compact_losses_testing[100:], '--', label="CompG w/Resnet152 off-sampling loss", color='C3', linewidth=2.5)
    plt.legend()

    plt.xlabel("Timestep $t$", fontsize=18)
    plt.ylabel("CE loss", fontsize=18)
    plt.legend(prop={'size': 14})
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    # plt.subplots_adjust(left=1, right=1.1, top=1.1, bottom=1)
    plt.subplots_adjust(left=0.12, right=0.99, top=0.99, bottom=0.125)

    plt.gca().invert_xaxis()

    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':
    # guidance_file = "../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting_imn64test/IMN64_withxt/unconditional/scale4.0_skip1/reference/loss.npz"
    # compact_file =  "../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting_imn64test/IMN64_withxt/unconditional/scale20.0_skip5/reference/loss.npz"
    #
    guidance_file = "../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting_imn64test2/IMN64_withxt_resnet/unconditional/scale4.0_skip1/reference/loss.npz"
    compact_file =  "../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting_imn64test2/IMN64_withxt_resnet/unconditional/scale20.0_skip5/reference/loss.npz"
    glt, gltest, clt, cltest = read_file(guidance_file, compact_file)
    out_folder = "runs/analayse"
    os.makedirs(out_folder, exist_ok=True)

    out_file = os.path.join(out_folder, "loss_visualize_xt_partial_20.pdf")
    visualize(glt, gltest, clt, cltest, out_file)

