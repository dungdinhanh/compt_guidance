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
from guided_diffusion.script_util_classifier_free import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion_classifier_free2_compt,
    create_classifier,
    add_dict_to_argparser,
    args_to_dict,
    create_model,
    create_model_diffusion_unconditional
)
import datetime
from PIL import Image
import hfai.client
from torchvision import utils
import time


def main(local_rank):
    args = create_argparser().parse_args()

    dist_util.setup_dist(local_rank)
    base_folder = args.base_folder
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



    save_folder = os.path.join(
        base_folder,
        args.logdir,
        "logs",
    )

    logger.configure(save_folder, rank=dist.get_rank())

    output_images_folder = os.path.join(base_folder, args.logdir, "reference")
    os.makedirs(output_images_folder, exist_ok=True)

    logger.log("creating unconditional model and diffusion...")

    model, diffusion = create_model_and_diffusion_classifier_free2_compt(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.skip_compt = args.skip
    model.load_state_dict(
        dist_util.load_state_dict(os.path.join(base_folder, args.model_path), map_location="cpu")
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("loading conditional model...")
    uncond_model = create_model_diffusion_unconditional(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    uncond_model.load_state_dict(
        dist_util.load_state_dict(os.path.join(base_folder, args.uncond_model_path), map_location="cpu")
    )
    uncond_model.to(dist_util.dev())
    if args.use_fp16:
        uncond_model.convert_to_fp16()
    uncond_model.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        return uncond_model(x, t)

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t,  y)

    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    checkpoint_temp = os.path.join(output_images_folder, "samples_temp.npz")
    vis_images_folder = os.path.join(output_images_folder, "sample_images")
    os.makedirs(vis_images_folder, exist_ok=True)
    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{args.image_size}x{args.image_size}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        logger.log("sampling complete")
        return
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
    start = time.time()
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.specified_class is not None:
            classes = th.randint(
                low=int(args.specified_class), high=int(args.specified_class) + 1, size=(args.batch_size,),
                device=dist_util.dev()
            )
        else:
            classes = th.randint(
                low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev()
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
            w_cond=args.cond_model_scale
        )


        # if args.save_imgs_for_visualization and dist.get_rank() == 0 and (
        #         len(all_images) // dist.get_world_size()) < 10:
        #     save_img_dir = vis_images_folder
        #     utils.save_image(
        #         sample.clamp(-1, 1),
        #         os.path.join(save_img_dir, "samples_{}.png".format(len(all_images))),
        #         nrow=4,
        #         normalize=True,
        #         range=(-1, 1),
        #     )


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
    end = time.time()
    duration = end - start
    print(f"execution time {duration}")




def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        uncond_model_path="",
        cond_model_scale=1.0,
        save_imgs_for_visualization=True,
        fix_seed=False,
        specified_class=None,
        logdir="",
        num_classes=1000,
        base_folder="./",
        skip=5
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    th.multiprocessing.spawn(main, args=(), nprocs=ngpus)