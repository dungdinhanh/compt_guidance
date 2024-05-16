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

def visualize(guidance_losses_training, guidance_losses_testing, compact_losses_training, compact_losses_testing, timesteps, file_name):
    timesteps = np.arange(guidance_losses_training.shape[0])[::-1]
    plt.plot(timesteps, guidance_losses_training,  label="G training loss", color='C2')
    plt.plot(timesteps, guidance_losses_testing, '--', label="G testing loss", color='C2')
    plt.plot(timesteps, compact_losses_training, label="CG training loss", color='C1')
    plt.plot(timesteps, compact_losses_testing, '--', label="CG testing loss", color='C1')
    plt.legend()



    plt.gca().invert_xaxis()

    plt.savefig(file_name)
    plt.close()


if __name__ == '__main__':
    guidance_file = "../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting/IMN64_withxt/unconditional/scale4.0_skip1/reference/loss.npz"
    compact_file = "../selfsup-guidance/runs/sampling_compt2quad_visualize_overfitting/IMN64_withxt/unconditional/scale17.0_skip5/reference/loss.npz"

    glt, gltest, clt, cltest = read_file(guidance_file, compact_file)
    out_folder = "runs/analayse"
    os.makedirs(out_folder, exist_ok=True)

    out_file = os.path.join(out_folder, "loss_visualize.png")
    visualize(glt, gltest, clt, cltest, out_file)

