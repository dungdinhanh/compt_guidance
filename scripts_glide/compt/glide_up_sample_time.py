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
from eds_guided_diffusion.script_util import (
    NUM_CLASSES,
    add_dict_to_argparser,
    args_to_dict,
)

from glide_text2im.clip.model_creation_compt import create_clip_compt_model
from glide_text2im.download import load_checkpoint
from glide_text2im.model_creation import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    model_and_diffusion_defaults_upsampler,
)

import datetime
from PIL import Image
from torchvision import utils
import hfai.client
import hfai.multiprocessing

from datasets.coco_helper import load_data_caption, load_data_caption_hfai
import time


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

    logger.log("creating model and diffusion...")
    options_model = args_to_dict(args, model_and_diffusion_defaults().keys())
    options_model['use_fp16'] = args.use_fp16
    model, diffusion = create_model_and_diffusion(
        **options_model
    )
    model.load_state_dict(
        load_checkpoint('base', th.device("cpu"))
    )
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()
    logger.log('total base parameters', sum(x.numel() for x in model.parameters()))

    options_up = model_and_diffusion_defaults_upsampler()
    options_up['timestep_respacing'] = 'fast27'
    options_up['use_fp16'] = args.use_fp16
    model_up, diffusion_up = create_model_and_diffusion(**options_up)
    model_up.load_state_dict(load_checkpoint('upsample', th.device("cpu")))
    model_up.to(dist_util.dev())
    if args.use_fp16:
        model_up.convert_to_fp16()
    model_up.eval()
    # print('total upsampler parameters', sum(x.numel() for x in model_up.parameters()))


    logger.log("loading clip...")
    clip_model = create_clip_compt_model(device=dist_util.dev())
    clip_model.image_encoder.load_state_dict(load_checkpoint('clip/image-enc', th.device("cpu")))
    clip_model.text_encoder.load_state_dict(load_checkpoint('clip/text-enc', th.device("cpu")))
    clip_model.image_encoder.to(dist_util.dev())
    clip_model.text_encoder.to(dist_util.dev())
    # classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    # classifier.load_state_dict(
    #     dist_util.load_state_dict(args.classifier_path, map_location="cpu")
    # )
    # classifier.to(dist_util.dev())
    # if args.classifier_use_fp16:
    #     classifier.convert_to_fp16()
    # classifier.eval()

    # cond_fn = clip_model.cond_fn
    # cond_fn = clip_model.cond_fn([prompt] * batch_size, guidance_scale)


    logger.log("Looking for previous file")
    checkpoint = os.path.join(output_images_folder, "samples_last.npz")
    checkpoint_temp = os.path.join(output_images_folder, "samples_temp.npz")
    vis_images_folder = os.path.join(output_images_folder, "sample_images")
    os.makedirs(vis_images_folder, exist_ok=True)
    ims = options_up["image_size"]
    final_file = os.path.join(output_images_folder,
                              f"samples_{args.num_samples}x{ims}x{ims}x3.npz")
    if os.path.isfile(final_file):
        dist.barrier()
        logger.log("sampling complete")
        return
    if os.path.isfile(checkpoint):
        npzfile = np.load(checkpoint)
        all_images = list(npzfile['arr_0'])
    else:
        all_images = []
    logger.log(f"Number of current images: {len(all_images)}")
    logger.log("sampling...")

    guidance_scale = args.guidance_scale

    caption_loader = load_data_caption_hfai(split="val", batch_size=args.batch_size)

    caption_iter = iter(caption_loader)

    upsample_temp = 0.997

    skip = args.skip
    skip_type = args.skip_type
    respace_gap = int(args.diffusion_steps / int(args.timestep_respacing))
    if skip_type == "linear":
        guidance_timesteps = get_guidance_timesteps_linear(int(args.timestep_respacing), skip)
    else:
        guidance_timesteps = get_guidance_timesteps_with_weight(int(args.timestep_respacing), skip)
    start = time.time()
    while len(all_images) * args.batch_size < args.num_samples:
        prompts = next(caption_iter)
        while len(prompts) != args.batch_size:
            prompts = next(caption_iter)
        #

        # print(len(prompts))
        cond_fn = clip_model.cond_fn(prompts, guidance_scale, guidance_timesteps, respace_gap)

        tokens = model.tokenizer.encode_batch(prompts)
        tokens, mask = model.tokenizer.padded_tokens_and_mask_batch(
            tokens, options_model['text_ctx']
        )

        model_kwargs = dict(
            tokens=th.tensor(tokens, device=dist_util.dev()),
            mask=th.tensor(mask, dtype=th.bool, device=dist_util.dev()),
        )

        sample_fn = (
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        model.del_cache()
        out = sample_fn(
            model,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=dist_util.dev(),
        )  # (B, 3, H, W)
        model.del_cache()
        sample = out

        tokens = model.tokenizer.encode_batch(prompts)
        tokens, mask = model.tokenizer.padded_tokens_and_mask_batch(
            tokens, options_model['text_ctx']
        )

        model_up_kwargs = dict(
            low_res=((sample + 1) * 127.5).round() / 127.5 - 1,
            # Text tokens
            tokens=th.tensor(tokens, device=dist_util.dev()),
            mask=th.tensor(
                            mask,
                            dtype=th.bool,
                            device=dist_util.dev())
        )

        model_up.del_cache()
        up_shape = (args.batch_size, 3, options_up["image_size"], options_up["image_size"])
        up_samples = diffusion_up.ddim_sample_loop(
            model_up,
            up_shape,
            noise=th.randn(up_shape, device=dist_util.dev()) * upsample_temp,
            device=dist_util.dev(),
            clip_denoised=True,
            progress=False,
            model_kwargs=model_up_kwargs,
            cond_fn=None,
        )[:args.batch_size]
        model_up.del_cache()

        sample = up_samples


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
        sample = sample.permute(0, 2, 3, 1)  # (B, H, W, 3)
        sample = sample.contiguous()

        gathered_samples = [th.zeros_like(sample) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        all_images.extend([sample.cpu().numpy() for sample in gathered_samples])


    end = time.time()
    duration = end - start
    logger.log(f"running time {duration}")


    logger.log("sampling complete")


def get_guidance_timesteps_linear(n=250, skip=5):
    # T = n - 1
    # max_steps = int(T/skip)
    guidance_timesteps = np.zeros((n,), dtype=int)
    for i in range(n):
        timestep = i + 1
        if timestep % skip == 0:
            guidance_timesteps[i] = 1
        pass
    guidance_timesteps[0] = 1
    guidance_timesteps[1] = 1
    return guidance_timesteps
    pass


def get_guidance_timesteps_with_weight(n=250, skip=5):
    # c * i^2
    T = n - 1
    max_steps = int(n/skip)
    c = n/(max_steps**2)
    guidance_timesteps = np.zeros((n,), dtype=int)
    for i in range(max_steps):
        guidance_index = - int(c * (i ** 2)) + T
        if 0 <= guidance_index and guidance_index <= T:
            guidance_timesteps[guidance_index] += 1
        else:
            print(f"guidance index: {guidance_index}")
            print(f"constant c: {c}")
            print(f"faulty index: {i}")
            print(f"timesteps {T}")
            print(f"compressd by {skip} times")
            print(f"error in index must larger than 0 or less than {T}")
            exit(0)
    guidance_timesteps[1] = 1
    guidance_timesteps[0] = 1
    # print(guidance_timesteps)
    # print(np.sum(guidance_timesteps))
    # exit(0)
    return guidance_timesteps


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=16,
        use_ddim=False,
        guidance_scale=1.0,
        save_imgs_for_visualization=True,
        fix_seed=False,
        specified_class=None,
        logdir="",
        skip=5,
        skip_type="linear",
        base_folder="./",
    )
    defaults.update(model_and_diffusion_defaults())
    # defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser



if __name__ == "__main__":
    ngpus = th.cuda.device_count()
    hfai.multiprocessing.spawn(main, args=(), nprocs=ngpus, bind_numa=False)
