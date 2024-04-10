"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import torch as th
# import torch.distributed as dist
import hfai.nccl.distributed as dist
import torch.nn.functional as F
import hfai
from guided_diffusion import dist_util, logger
from scripts_gdiff.compt_guidance.analyse.scripts_vis import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion_wx0,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
)
import datetime
from PIL import Image
import hfai.client
from torchvision import utils
import matplotlib.pyplot as plt
import torchvision.models as models

def center_crop_arr(images, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    y_size = images.shape[2]
    x_size = images.shape[3]
    crop_y = (y_size - image_size) // 2
    crop_x = (x_size - image_size) // 2
    return images[:, :, crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def custom_normalize(images, mean, std):
    # print(images.shape)
    # Check if the input tensor has the same number of channels as the mean and std
    if images.size(1) != len(mean) or images.size(1) != len(std):
        raise ValueError("The number of channels in the input tensor must match the length of mean and std.")
    images = images.to(th.float)
    # Normalize the tensor
    for c in range(images.size(1)):
        images[:, c, :, :] = (images[:, c, :, :] - mean[c]) / std[c]

    return images

def main(local_rank):
    args = create_argparser().parse_args()

    dist_util.setup_dist(local_rank)

    if args.fix_seed:
        import random
        seed = 23333 + dist.get_rank()
        np.random.seed(seed)
        th.manual_seed(seed)  # CPU随机种子确定
        th.cuda.manual_seed(seed)  # GPU随机种子确定
        th.cuda.manual_seed_all(seed)  # 所有的GPU设置种子

        th.backends.cudnn.benchmark = False  # 模型卷积层预先优化关闭
        th.backends.cudnn.deterministic = True  # 确定为默认卷积算法

        random.seed(seed)
        np.random.seed(seed)

        os.environ['PYTHONHASHSEED'] = str(seed)
    base_folder = args.base_folder
    save_folder = os.path.join(
        base_folder,
        args.logdir,
        "logs",
    )

    logger.configure(save_folder, rank=dist.get_rank())

    output_images_folder = os.path.join(base_folder, args.logdir, "reference")
    os.makedirs(output_images_folder, exist_ok=True)
    vis_images_folder = os.path.join(output_images_folder, "sample_images")
    os.makedirs(vis_images_folder, exist_ok=True)
    vis_images_folder_xt = os.path.join(vis_images_folder, "xt")
    os.makedirs(vis_images_folder_xt, exist_ok=True)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_wx0(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.join(base_folder, args.model_path), map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(
        dist_util.load_state_dict(os.path.join(base_folder, args.classifier_path), map_location="cpu")
    )
    classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    resnet_address = os.path.join(base_folder, 'eval_models/resnet50-19c8e357.pth')
    resnet = models.resnet50()
    resnet.load_state_dict(th.load(resnet_address))
    resnet.eval()
    resnet.cuda()
    # use off-the-shelf classifier for visualize overfitting
    mean_imn = [0.485, 0.456, 0.406]
    std_imn = [0.229, 0.224, 0.225]

    df_steps = int(args.diffusion_steps)
    timespace = int(args.timestep_respacing)
    steps_skipped_diff = int(df_steps/timespace)
    skip = int(args.skip)
    loss_items = []
    loss_testing = []
    loss_testing_xt = []
    list_t = []

    def cond_fn(inputs, t, y=None):
        assert y is not None
        with th.enable_grad():
            convert_t = int(t[0]/steps_skipped_diff) + 1
            x_in = inputs[0].detach().requires_grad_(True)
            pred_xstart = inputs[1].detach().requires_grad_(True)

            pred_xstart_r = ((pred_xstart + 1) * 127.).clamp(0, 255) / 255.0
            pred_xstart_r = center_crop_arr(pred_xstart_r, args.image_size)
            pred_xstart_r = custom_normalize(pred_xstart_r, mean_imn, std_imn)

            x_in_r = ((x_in + 1) * 127.).clamp(0, 255)/255.0
            x_in_r = center_crop_arr(x_in_r, args.image_size)
            x_in_r = custom_normalize(x_in_r, mean_imn, std_imn)

            x_in_r_0 = resnet(x_in_r)
            log_x_in_probs = F.log_softmax(x_in_r_0, dim=-1)
            selected_x_in = log_x_in_probs[range(len(x_in_r_0)), y.view(-1)]
            loss_testing_xt.append(-selected_x_in.mean().detach().cpu().numpy())

            p_x_0 = resnet(pred_xstart_r)
            log_probs_x0 = F.log_softmax(p_x_0, dim=-1)
            selected_x_0 = log_probs_x0[range(len(p_x_0)), y.view(-1)]
            loss_testing.append(-selected_x_0.mean().detach().cpu().numpy())

            logits = classifier(x_in, t)
            log_probs = F.log_softmax(logits, dim=-1)
            selected = log_probs[range(len(logits)), y.view(-1)]
            loss_items.append(-selected.mean().detach().cpu().numpy())
            if convert_t % skip == 0 or convert_t <=2:
                # print("call guidance:-------------->", convert_t)
                list_t.append(convert_t)
                return th.autograd.grad(selected.sum(), x_in)[0] * args.classifier_scale
            else:
                # print("skip: ", convert_t)
                return th.zeros_like(x_in)

    def model_fn(x, t, y=None):

        assert y is not None
        return model(x, t, y if args.class_cond else None)

    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")

    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    # if os.path.isfile(final_file):
    #     dist.barrier()
    #     logger.log("sampling complete")
    #     return
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
        all_labels = list(npzfile['arr_1'])
    else:
        all_images = []
        all_labels = []
    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")
    if args.image_size == 28:
        img_channels = 1
        num_class = 10
    else:
        img_channels = 3
        num_class = NUM_CLASSES
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = th.randint(
            low=0, high=num_class, size=(args.batch_size,), device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        sample = sample_fn(
            model_fn,
            (args.batch_size, img_channels, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )

        if args.save_imgs_for_visualization and dist.get_rank() == 0 and (
                len(all_images) // dist.get_world_size()) < 10:
            save_img_dir = vis_images_folder
            utils.save_image(
                sample.clamp(-1, 1),
                os.path.join(save_img_dir, "samples_{}.png".format(len(all_images))),
                nrow=4,
                normalize=True,
                range=(-1, 1),
            )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        batch_images = [sample.cpu().numpy() for sample in gathered_samples]
        all_images.extend(batch_images)
        gathered_labels = [th.zeros_like(classes) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, classes)
        batch_labels = [labels.cpu().numpy() for labels in gathered_labels]
        all_labels.extend(batch_labels)
        if dist.get_rank() == 0:
            if hfai.client.receive_suspend_command():
                print("Receive suspend - good luck next run ^^")
                hfai.client.go_suspend()
            logger.log(f"created {len(all_images) * args.batch_size} samples")
            np.savez(checkpoint, np.stack(all_images), np.stack(all_labels))

    # draw classifier losses
    losses_list = np.asarray(loss_items)
    losses_testing_list = np.asarray(loss_testing)
    timesteps = np.arange(losses_list.shape[0])[::-1]
    list_t = np.asarray(list_t)
    plt.plot(timesteps, losses_list)
    plt.plot(timesteps, losses_testing_list)
    plt.vlines(list_t, -5, 0.0)
    plt.gca().invert_xaxis()
    loss_file = os.path.join(output_images_folder, "cls_loss.png")
    plt.savefig(loss_file)
    plt.close()

    plt.plot(timesteps, losses_list)
    loss_testing_xt_list = np.asarray(loss_testing_xt)
    plt.plot(timesteps, loss_testing_xt)
    plt.vlines(list_t, -5, 0.0)
    plt.gca().invert_xaxis()
    loss_file_xt = os.path.join(output_images_folder, "cls_loss_xt.png")
    plt.savefig(loss_file_xt)
    plt.close()


    # arr = np.concatenate(all_images, axis=0)
    # arr = arr[: args.num_samples]
    # label_arr = np.concatenate(all_labels, axis=0)
    # label_arr = label_arr[: args.num_samples]
    # if dist.get_rank() == 0:
    #     shape_str = "x".join([str(x) for x in arr.shape])
    #     out_path = os.path.join(output_images_folder, f"samples_{shape_str}.npz")
    #     logger.log(f"saving to {out_path}")
    #     np.savez(out_path, arr, label_arr)
    #     os.remove(checkpoint)
    #
    # dist.barrier()
    # logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
        save_imgs_for_visualization=True,
        fix_seed=False,
        specified_class=None,
        logdir="",
        base_folder="./",
        skip=5
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    th.multiprocessing.spawn(main, args=(), nprocs=ngpus)
