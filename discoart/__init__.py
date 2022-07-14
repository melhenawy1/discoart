<<<<<<< HEAD
<<<<<<< HEAD
__version__ = "0.0.23"
__all__ = ["create"]

import os
#import sys
import gc
import copy
import random
import warnings
import clip
import torch
import yaml
import lpips
import numpy as np
import torchvision.transforms as T
import hashlib
import logging
import subprocess
import math
import torchvision.transforms.functional as TF

from guided_diffusion.script_util import model_and_diffusion_defaults, create_model_and_diffusion
from docarray import Document
from types import SimpleNamespace
from typing import overload, List, Optional, Dict, Any
from yaml import Loader
from PIL import ImageOps
from IPython import display
from ipywidgets import Output
from torch.nn import functional as F
from resize_right import resize
from torch import nn
from pathlib import Path
=======
import os
import warnings
from types import SimpleNamespace
>>>>>>> parent of a1ba0ed (Update)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__version__ = '0.0.23'

__all__ = ['create']

import sys

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get(__package__).__file__ if __package__ in
        sys.modules else __file__),
    'resources',
)

import gc

# check if GPU is available
import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    raise RuntimeError(
        'CUDA is not available. DiscoArt is unbearably slow on CPU. '
        'Please switch to GPU device, if you are using Google Colab, then free tier would work.'
    )

# download and load models, this will take some time on the first load

from .helper import load_all_models, load_diffusion_model, load_clip_models

model_config, secondary_model = load_all_models(
    '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model=True,
    device=device,
)

from typing import TYPE_CHECKING, overload, List, Optional

if TYPE_CHECKING:
    from docarray import DocumentArray, Document

_clip_models_cache = {}
<<<<<<< HEAD
cache_dir = cache_dir = f'/workspace/'

def _get_logger():
    logger = logging.getLogger(__package__)
    _log_level = os.environ.get('DISCOART_LOG_LEVEL', 'INFO')
    logger.setLevel(_log_level)
    ch = logging.StreamHandler()
    ch.setLevel(_log_level)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

logger = _get_logger()
if not os.path.exists(cache_dir):
    logger.info(f'Downloading models....')
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

logger.debug(f'`.cache` dir is set to: {cache_dir}')

check_model_SHA = False

def _wget(url, cache_dir): res = subprocess.run(['wget', url, '-q', '-P', f'{cache_dir}'])

def load_clip_models(device,enabled: List[str], clip_models: Dict[str, Any] = {}):

    import clip
    # load enabled models
    for k in enabled:
        if k not in clip_models:
            clip_models[k] = clip.load(
                k, jit=False)[0].eval().requires_grad_(False).to(device)
    # disable not enabled models to save memory
    for k in clip_models:
        if k not in enabled:
            clip_models.pop(k)
    return list(clip_models.values())

def load_all_models(diffusion_model, use_secondary_model, fallback=False, device=torch.device('cuda:0'),):
    model_512_downloaded = False
    model_512_SHA = '9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648'
    model_512_link = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_512_link_fb = 'https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt'
    model_512_path = f'{cache_dir}/512x512_diffusion_uncond_finetune_008100.pt'

    if fallback:
        model_512_link = model_512_link_fb
    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        if os.path.exists(model_512_path) and check_model_SHA:
            logger.debug('Checking 512 Diffusion File')
            with open(model_512_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_512_SHA:
                logger.debug('512 Model SHA matches')
                if os.path.exists(model_512_path): model_512_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(diffusion_model, use_secondary_model,
                                    False)
            else:
                logger.debug("512 Model SHA doesn't match, redownloading...")
                _wget(model_512_link, cache_dir)
                if os.path.exists(model_512_path): model_512_downloaded = True
                else:
                    logger.debug('First URL Failed using FallBack')
                    load_all_models(diffusion_model, use_secondary_model,
                                    False)
        elif (os.path.exists(model_512_path) and not check_model_SHA
              or model_512_downloaded == True):
            logger.debug(
                '512 Model already downloaded, check check_model_SHA if the file is corrupt'
            )
        else:
            _wget(model_512_link, cache_dir)
            model_512_downloaded = True

    model_config = model_and_diffusion_defaults()

    if diffusion_model == '512x512_diffusion_uncond_finetune_008100':
        model_config.update({
            'attention_resolutions': '32, 16, 8',
            'class_cond': False,
            'diffusion_steps':
            1000,  # No need to edit this, it is taken care of later.
            'rescale_timesteps': True,
            'timestep_respacing':
            250,  # No need to edit this, it is taken care of later.
            'image_size': 512,
            'learn_sigma': True,
            'noise_schedule': 'linear',
            'num_channels': 256,
            'num_head_channels': 64,
            'num_res_blocks': 2,
            'resblock_updown': True,
            'use_fp16': device != 'cpu',
            'use_scale_shift_norm': True,
        })

    secondary_model = None
    return model_config, secondary_model

model_config, secondary_model = load_all_models(
    "512x512_diffusion_uncond_finetune_008100",
    use_secondary_model=False,
    device=device)

def load_diffusion_model(model_config, diffusion_model, steps, device):
    timestep_respacing = f'ddim{steps}'
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
    model_config.update({
        'timestep_respacing': timestep_respacing,
        'diffusion_steps': diffusion_steps,
    })

    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        torch.load(f'{cache_dir}/{diffusion_model}.pt', map_location='cpu'))
    model.requires_grad_(False).eval().to(device)
    for name, param in model.named_parameters():
        if 'qkv' in name or 'norm' in name or 'proj' in name:
            param.requires_grad_()
    if model_config['use_fp16']:
        model.convert_to_fp16()
    return model, diffusion


def parse_prompt(prompt):
    if prompt.startswith('http://') or prompt.startswith('https://'):
        vals = prompt.rsplit(':', 2)
        vals = [vals[0] + ':' + vals[1], *vals[2:]]
    else:
        vals = prompt.rsplit(':', 1)
    vals = vals + ['', '1'][len(vals):]
    return vals[0], float(vals[1])
=======

# begin_create_overload

>>>>>>> parent of a1ba0ed (Update)

=======
import os
import warnings
from types import SimpleNamespace

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

__version__ = '0.0.23'

__all__ = ['create']

import sys

__resources_path__ = os.path.join(
    os.path.dirname(
        sys.modules.get(__package__).__file__ if __package__ in
        sys.modules else __file__),
    'resources',
)

import gc

# check if GPU is available
import torch

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    raise RuntimeError(
        'CUDA is not available. DiscoArt is unbearably slow on CPU. '
        'Please switch to GPU device, if you are using Google Colab, then free tier would work.'
    )

# download and load models, this will take some time on the first load

from .helper import load_all_models, load_diffusion_model, load_clip_models

model_config, secondary_model = load_all_models(
    '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model=True,
    device=device,
)

from typing import TYPE_CHECKING, overload, List, Optional

if TYPE_CHECKING:
    from docarray import DocumentArray, Document

_clip_models_cache = {}

# begin_create_overload

>>>>>>> parent of a1ba0ed (Update)

@overload
def create(
    text_prompts: Optional[List[str]] = [
<<<<<<< HEAD
<<<<<<< HEAD
        "A beautiful painting of a singular lighthouse, Trending on artstation.",
        "yellow color scheme",
=======
        'A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.',
        'yellow color scheme',
>>>>>>> parent of a1ba0ed (Update)
=======
        'A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.',
        'yellow color scheme',
>>>>>>> parent of a1ba0ed (Update)
    ],
    init_image: Optional[str] = None,
    width_height: Optional[List[int]] = [1280, 768],
    skip_steps: Optional[int] = 10,
    steps: Optional[int] = 250,
    cut_ic_pow: Optional[int] = 1,
    init_scale: Optional[int] = 1000,
    clip_guidance_scale: Optional[int] = 5000,
    tv_scale: Optional[int] = 0,
    range_scale: Optional[int] = 150,
    sat_scale: Optional[int] = 0,
    cutn_batches: Optional[int] = 4,
<<<<<<< HEAD
<<<<<<< HEAD
    diffusion_model: Optional[
        str] = "512x512_diffusion_uncond_finetune_008100",
    diffusion_sampling_mode: Optional[str] = "ddim",
=======
    diffusion_model: Optional[str] = '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model: Optional[bool] = True,
    diffusion_sampling_mode: Optional[str] = 'ddim',
>>>>>>> parent of a1ba0ed (Update)
=======
    diffusion_model: Optional[str] = '512x512_diffusion_uncond_finetune_008100',
    use_secondary_model: Optional[bool] = True,
    diffusion_sampling_mode: Optional[str] = 'ddim',
>>>>>>> parent of a1ba0ed (Update)
    perlin_init: Optional[bool] = False,
    perlin_mode: Optional[str] = 'mixed',
    seed: Optional[int] = None,
    eta: Optional[float] = 0.8,
    clamp_grad: Optional[bool] = True,
    clamp_max: Optional[float] = 0.05,
    randomize_class: Optional[bool] = True,
    clip_denoised: Optional[bool] = False,
    fuzzy_prompt: Optional[bool] = False,
    rand_mag: Optional[float] = 0.05,
    cut_overview: Optional[str] = '[12]*400+[4]*600',
    cut_innercut: Optional[str] = '[4]*400+[12]*600',
    cut_icgray_p: Optional[str] = '[0.2]*400+[0]*600',
    display_rate: Optional[int] = 10,
    n_batches: Optional[int] = 4,
    batch_size: Optional[int] = 1,
<<<<<<< HEAD
<<<<<<< HEAD
    batch_name: Optional[str] = "",
    clip_models: Optional[list] = ["ViTB32", "ViTB16", "RN50"],
):
    ...


@overload
def create(init_document: "Document"):
    ...

=======
    batch_name: Optional[str] = '',
    clip_models: Optional[list] = ['ViTB32', 'ViTB16', 'RN50'],
) -> 'DocumentArray':
    """
    Create Disco Diffusion artworks and save the result into a DocumentArray.

    :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.  The words will be analyzed by the AI and will guide the diffusion process toward the image(s) you describe. These can include commas and weights to adjust the relative importance of each element.  E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."Notice that this prompt loosely follows a structure: [subject], [prepositional details], [setting], [meta modifiers and artist]; this is a good starting point for your experiments. Developing text prompts takes practice and experience, and is not the subject of this guide.  If you are a beginner to writing text prompts, a good place to start is on a simple AI art app like Night Cafe, starry ai or WOMBO prior to using DD, to get a feel for how text gets translated into images by GAN tools.  These other apps use different technologies, but many of the same principles apply.
    :param init_image: Recall that in the image sequence above, the first image shown is just noise.  If an init_image is provided, diffusion will replace the noise with the init_image as its starting state.  To use an init_image, upload the image to the Colab instance or your Google Drive, and enter the full image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total steps to retain the character of the init. See skip_steps above for further discussion.
    :param width_height: Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so.
    :param skip_steps: Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high, so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the amount an image changes per step) declines, and image coherence from one step to the next increases.The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be skipped without affecting the final image. You can experiment with this as a way to cut render times.If you skip too many steps, however, the remaining noise may not be high enough to generate new content, and thus may not have ‘time left’ to finish an image satisfactorily.Also, depending on your other settings, you may need to skip steps to prevent CLIP from overshooting your goal, resulting in ‘blown out’ colors (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain the shapes in the original init image. However, if you’re using an init_image, you can also adjust skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by" the init_image which will retain the colors and rough layout and shapes but look quite different. With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
    :param steps: When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration) involves the AI looking at subsets of the image called ‘cuts’ and calculating the ‘direction’ the image should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser, and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image, and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps comes at the expense of longer render times.  Also, while increasing steps should generally increase image quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is directly related to the number of steps, and many other parameters have a major impact on image quality, without costing additional time.
    :param cut_ic_pow: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
    :param init_scale: This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the clip_guidance_scale (CGS) above.  Too much init scale, and the image won’t change much during diffusion. Too much CGS and the init image will be lost.
    :param clip_guidance_scale: CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the image, you’d want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale, steps and skip_steps are the most important contributors to image quality, so learn them well.
    :param tv_scale: Total variance denoising. Optional, set to zero to turn off. Controls ‘smoothness’ of final output. If used, tv_scale will try to smooth out your final image to reduce overall noise. If your image is too ‘crunchy’, increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.  See https://en.wikipedia.org/wiki/Total_variation_denoising
    :param range_scale: Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images. Higher range_scale will reduce contrast, for more muted images.
    :param sat_scale: Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation. If your image is too saturated, increase sat_scale to reduce the saturation.
    :param cutn_batches: Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however, and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts, but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will increase render times, however, as the work is being done sequentially.  DD’s default cut schedule is a good place to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.
    :param diffusion_model: Diffusion_model of choice.
    :param use_secondary_model: Option to use a secondary purpose-made diffusion model to clean up interim diffusion images for CLIP evaluation.    If this option is turned off, DD will use the regular (large) diffusion model.    Using the secondary model is faster - one user reported a 50% improvement in render speed! However, the secondary model is much smaller, and may reduce image quality and detail.  I suggest you experiment with this.
    :param diffusion_sampling_mode: Two alternate diffusion denoising algorithms. ddim has been around longer, and is more established and tested.  plms is a newly added alternate method that promises good diffusion results in fewer steps, but has not been as fully tested and may have side effects. This new plms mode is actively being researched in the #settings-and-techniques channel in the DD Discord.
    :param perlin_init: Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.  If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very interesting characteristics, distinct from random noise, so it’s worth experimenting with this for your projects. Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image you may have specified.  Further, because the 2D, 3D and video animation systems all rely on the init_image system, if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and animation modes together do make a very colorful rainbow effect, which can be used creatively.
    :param perlin_mode: sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment to see what these do in your projects.
    :param seed: Deep in the diffusion code, there is a random number ‘seed’ which is used as the basis for determining the initial state of the diffusion.  By default, this is random, but you can also specify your own seed.  This is useful if you like a particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used repeatedly, the resulting images will be quite similar but not identical.
    :param eta: eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep. 0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0, then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around 250 and up. eta has a subtle, unpredictable effect on image, so you’ll need to experiment to see how this affects your projects.
    :param clamp_grad: As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results.  Try your images with and without clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and should be reduced.
    :param clamp_max: Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting higher values (0.15-0.3) can provide interesting contrast and vibrancy.
    :param fuzzy_prompt: Controls whether to add multiple noisy prompts to the prompt losses. If True, can increase variability of image output. Experiment with this.
    :param rand_mag: Affects only the fuzzy_prompt.  Controls the magnitude of the random noise added by fuzzy_prompt.
    :param cut_overview: The schedule of overview cuts
    :param cut_innercut: The schedule of inner cuts
    :param cut_icgray_p: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
    :param display_rate: During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way to get an early peek at where your image is heading. If you don’t like the progression, just interrupt execution, change some settings, and re-run.  If you are planning a long, unmonitored batch, it’s better to set display_rate equal to steps, because displaying interim images does slow Colab down slightly.
    :param n_batches: This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details) DD will ignore n_batches and create a single set of animated frames based on the animation settings.
    :param batch_name: The name of the batch, the batch id will be named as "discoart-[batch_name]-seed". To avoid your artworks be overridden by other users, please use a unique name.
    :param clip_models: CLIP Model selectors. ViTB32, ViTB16, ViTL14, RN101, RN50, RN50x4, RN50x16, RN50x64.These various CLIP models are available for you to use during image generation.  Models have different styles or ‘flavors,’ so look around.  You can mix in multiple models as well for different results.  However, keep in mind that some models are extremely memory-hungry, and turning on additional models will take additional memory and may cause a crash.The rough order of speed/mem usage is (smallest/fastest to largest/slowest):VitB32RN50RN101VitB16RN50x4RN50x16RN50x64ViTL14For RN50x64 & ViTL14 you may need to use fewer cuts, depending on your VRAM.
    :return: a DocumentArray object that has `n_batches` Documents
    """


# end_create_overload


@overload
def create(init_document: 'Document') -> 'DocumentArray':
    """
    Create an artwork using a DocArray ``Document`` object as initial state.
    :param init_document: its ``.tags`` will be used as parameters, ``.uri`` (if present) will be used as init image.
    :return: a DocumentArray object that has `n_batches` Documents
    """
>>>>>>> parent of a1ba0ed (Update)


<<<<<<< HEAD

def load_config(user_config: Dict, ) -> Dict:
    cfg = copy.deepcopy(default_args)
    if user_config:
        cfg.update(**user_config)
    for k in user_config.keys():
        if k not in cfg: warnings.warn(f"unknown argument {k}, ignored")

    for k, v in cfg.items():
        if k in (
                "batch_size",
                "display_rate",
                "seed",
                "skip_steps",
                "steps",
                "n_batches",
                "cutn_batches",
        ) and isinstance(v, float):
            cfg[k] = int(v)
        if k == "width_height": cfg[k] = [int(vv) for vv in v]

    cfg.update(**{
        "seed": cfg["seed"] or random.randint(0, 2**32),
    })

    if cfg["batch_name"]:
        da_name = f'{__package__}-{cfg["batch_name"]}-{cfg["seed"]}'
    else:
        da_name = f'{__package__}-{cfg["seed"]}'

    cfg.update(**{"name_docarray": da_name})

    return cfg


def create(**kwargs):
#    from .runner import do_run
    _args = load_config(user_config=kwargs)
    _args = SimpleNamespace(**_args)
=======
def create(**kwargs) -> 'DocumentArray':
    from .config import load_config, print_args_table, save_config_svg
    from .runner import do_run

    if 'init_document' in kwargs:
        d = kwargs['init_document']
        _kwargs = d.tags
        if not _kwargs:
            warnings.warn(
                'init_document has no .tags, fallback to default config')
        if d.uri:
            _kwargs['init_image'] = kwargs['init_document'].uri
        else:
            warnings.warn(
                'init_document has no .uri, fallback to no init image')
        kwargs.pop('init_document')
        if kwargs:
            warnings.warn(
                'init_document has .tags and .uri, but kwargs are also present, will override .tags'
            )
            _kwargs.update(kwargs)
        _args = load_config(user_config=_kwargs)
    else:
        _args = load_config(user_config=kwargs)

    save_config_svg(_args)

    _args = SimpleNamespace(**_args)

>>>>>>> parent of a1ba0ed (Update)
=======
    batch_name: Optional[str] = '',
    clip_models: Optional[list] = ['ViTB32', 'ViTB16', 'RN50'],
) -> 'DocumentArray':
    """
    Create Disco Diffusion artworks and save the result into a DocumentArray.

    :param text_prompts: Phrase, sentence, or string of words and phrases describing what the image should look like.  The words will be analyzed by the AI and will guide the diffusion process toward the image(s) you describe. These can include commas and weights to adjust the relative importance of each element.  E.g. "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation."Notice that this prompt loosely follows a structure: [subject], [prepositional details], [setting], [meta modifiers and artist]; this is a good starting point for your experiments. Developing text prompts takes practice and experience, and is not the subject of this guide.  If you are a beginner to writing text prompts, a good place to start is on a simple AI art app like Night Cafe, starry ai or WOMBO prior to using DD, to get a feel for how text gets translated into images by GAN tools.  These other apps use different technologies, but many of the same principles apply.
    :param init_image: Recall that in the image sequence above, the first image shown is just noise.  If an init_image is provided, diffusion will replace the noise with the init_image as its starting state.  To use an init_image, upload the image to the Colab instance or your Google Drive, and enter the full image path here. If using an init_image, you may need to increase skip_steps to ~ 50% of total steps to retain the character of the init. See skip_steps above for further discussion.
    :param width_height: Desired final image size, in pixels. You can have a square, wide, or tall image, but each edge length should be set to a multiple of 64px, and a minimum of 512px on the default CLIP model setting.  If you forget to use multiples of 64px in your dimensions, DD will adjust the dimensions of your image to make it so.
    :param skip_steps: Consider the chart shown here.  Noise scheduling (denoise strength) starts very high and progressively gets lower and lower as diffusion steps progress. The noise levels in the first few steps are very high, so images change dramatically in early steps.As DD moves along the curve, noise levels (and thus the amount an image changes per step) declines, and image coherence from one step to the next increases.The first few steps of denoising are often so dramatic that some steps (maybe 10-15% of total) can be skipped without affecting the final image. You can experiment with this as a way to cut render times.If you skip too many steps, however, the remaining noise may not be high enough to generate new content, and thus may not have ‘time left’ to finish an image satisfactorily.Also, depending on your other settings, you may need to skip steps to prevent CLIP from overshooting your goal, resulting in ‘blown out’ colors (hyper saturated, solid white, or solid black regions) or otherwise poor image quality.  Consider that the denoising process is at its strongest in the early steps, so skipping steps can sometimes mitigate other problems.Lastly, if using an init_image, you will need to skip ~50% of the diffusion steps to retain the shapes in the original init image. However, if you’re using an init_image, you can also adjust skip_steps up or down for creative reasons.  With low skip_steps you can get a result "inspired by" the init_image which will retain the colors and rough layout and shapes but look quite different. With high skip_steps you can preserve most of the init_image contents and just do fine tuning of the texture.
    :param steps: When creating an image, the denoising curve is subdivided into steps for processing. Each step (or iteration) involves the AI looking at subsets of the image called ‘cuts’ and calculating the ‘direction’ the image should be guided to be more like the prompt. Then it adjusts the image with the help of the diffusion denoiser, and moves to the next step.Increasing steps will provide more opportunities for the AI to adjust the image, and each adjustment will be smaller, and thus will yield a more precise, detailed image.  Increasing steps comes at the expense of longer render times.  Also, while increasing steps should generally increase image quality, there is a diminishing return on additional steps beyond 250 - 500 steps.  However, some intricate images can take 1000, 2000, or more steps.  It is really up to the user.  Just know that the render time is directly related to the number of steps, and many other parameters have a major impact on image quality, without costing additional time.
    :param cut_ic_pow: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
    :param init_scale: This controls how strongly CLIP will try to match the init_image provided.  This is balanced against the clip_guidance_scale (CGS) above.  Too much init scale, and the image won’t change much during diffusion. Too much CGS and the init image will be lost.
    :param clip_guidance_scale: CGS is one of the most important parameters you will use. It tells DD how strongly you want CLIP to move toward your prompt each timestep.  Higher is generally better, but if CGS is too strong it will overshoot the goal and distort the image. So a happy medium is needed, and it takes experience to learn how to adjust CGS. Note that this parameter generally scales with image dimensions. In other words, if you increase your total dimensions by 50% (e.g. a change from 512 x 512 to 512 x 768), then to maintain the same effect on the image, you’d want to increase clip_guidance_scale from 5000 to 7500. Of the basic settings, clip_guidance_scale, steps and skip_steps are the most important contributors to image quality, so learn them well.
    :param tv_scale: Total variance denoising. Optional, set to zero to turn off. Controls ‘smoothness’ of final output. If used, tv_scale will try to smooth out your final image to reduce overall noise. If your image is too ‘crunchy’, increase tv_scale. TV denoising is good at preserving edges while smoothing away noise in flat regions.  See https://en.wikipedia.org/wiki/Total_variation_denoising
    :param range_scale: Optional, set to zero to turn off.  Used for adjustment of color contrast.  Lower range_scale will increase contrast. Very low numbers create a reduced color palette, resulting in more vibrant or poster-like images. Higher range_scale will reduce contrast, for more muted images.
    :param sat_scale: Saturation scale. Optional, set to zero to turn off.  If used, sat_scale will help mitigate oversaturation. If your image is too saturated, increase sat_scale to reduce the saturation.
    :param cutn_batches: Each iteration, the AI cuts the image into smaller pieces known as cuts, and compares each cut to the prompt to decide how to guide the next diffusion step.  More cuts can generally lead to better images, since DD has more chances to fine-tune the image precision in each timestep.  Additional cuts are memory intensive, however, and if DD tries to evaluate too many cuts at once, it can run out of memory.  You can use cutn_batches to increase cuts per timestep without increasing memory usage. At the default settings, DD is scheduled to do 16 cuts per timestep.  If cutn_batches is set to 1, there will indeed only be 16 cuts total per timestep. However, if cutn_batches is increased to 4, DD will do 64 cuts total in each timestep, divided into 4 sequential batches of 16 cuts each.  Because the cuts are being evaluated only 16 at a time, DD uses the memory required for only 16 cuts, but gives you the quality benefit of 64 cuts.  The tradeoff, of course, is that this will take ~4 times as long to render each image.So, (scheduled cuts) x (cutn_batches) = (total cuts per timestep). Increasing cutn_batches will increase render times, however, as the work is being done sequentially.  DD’s default cut schedule is a good place to start, but the cut schedule can be adjusted in the Cutn Scheduling section, explained below.
    :param diffusion_model: Diffusion_model of choice.
    :param use_secondary_model: Option to use a secondary purpose-made diffusion model to clean up interim diffusion images for CLIP evaluation.    If this option is turned off, DD will use the regular (large) diffusion model.    Using the secondary model is faster - one user reported a 50% improvement in render speed! However, the secondary model is much smaller, and may reduce image quality and detail.  I suggest you experiment with this.
    :param diffusion_sampling_mode: Two alternate diffusion denoising algorithms. ddim has been around longer, and is more established and tested.  plms is a newly added alternate method that promises good diffusion results in fewer steps, but has not been as fully tested and may have side effects. This new plms mode is actively being researched in the #settings-and-techniques channel in the DD Discord.
    :param perlin_init: Normally, DD will use an image filled with random noise as a starting point for the diffusion curve.  If perlin_init is selected, DD will instead use a Perlin noise model as an initial state.  Perlin has very interesting characteristics, distinct from random noise, so it’s worth experimenting with this for your projects. Beyond perlin, you can, of course, generate your own noise images (such as with GIMP, etc) and use them as an init_image (without skipping steps). Choosing perlin_init does not affect the actual diffusion process, just the starting point for the diffusion. Please note that selecting a perlin_init will replace and override any init_image you may have specified.  Further, because the 2D, 3D and video animation systems all rely on the init_image system, if you enable Perlin while using animation modes, the perlin_init will jump in front of any previous image or video input, and DD will NOT give you the expected sequence of coherent images. All of that said, using Perlin and animation modes together do make a very colorful rainbow effect, which can be used creatively.
    :param perlin_mode: sets type of Perlin noise: colored, gray, or a mix of both, giving you additional options for noise types. Experiment to see what these do in your projects.
    :param seed: Deep in the diffusion code, there is a random number ‘seed’ which is used as the basis for determining the initial state of the diffusion.  By default, this is random, but you can also specify your own seed.  This is useful if you like a particular result and would like to run more iterations that will be similar. After each run, the actual seed value used will be reported in the parameters report, and can be reused if desired by entering seed # here.  If a specific numerical seed is used repeatedly, the resulting images will be quite similar but not identical.
    :param eta: eta (greek letter η) is a diffusion model variable that mixes in a random amount of scaled noise into each timestep. 0 is no noise, 1.0 is more noise. As with most DD parameters, you can go below zero for eta, but it may give you unpredictable results. The steps parameter has a close relationship with the eta parameter. If you set eta to 0, then you can get decent output with only 50-75 steps. Setting eta to 1.0 favors higher step counts, ideally around 250 and up. eta has a subtle, unpredictable effect on image, so you’ll need to experiment to see how this affects your projects.
    :param clamp_grad: As I understand it, clamp_grad is an internal limiter that stops DD from producing extreme results.  Try your images with and without clamp_grad. If the image changes drastically with clamp_grad turned off, it probably means your clip_guidance_scale is too high and should be reduced.
    :param clamp_max: Sets the value of the clamp_grad limitation. Default is 0.05, providing for smoother, more muted coloration in images, but setting higher values (0.15-0.3) can provide interesting contrast and vibrancy.
    :param fuzzy_prompt: Controls whether to add multiple noisy prompts to the prompt losses. If True, can increase variability of image output. Experiment with this.
    :param rand_mag: Affects only the fuzzy_prompt.  Controls the magnitude of the random noise added by fuzzy_prompt.
    :param cut_overview: The schedule of overview cuts
    :param cut_innercut: The schedule of inner cuts
    :param cut_icgray_p: This sets the size of the border used for inner cuts.  High cut_ic_pow values have larger borders, and therefore the cuts themselves will be smaller and provide finer details.  If you have too many or too-small inner cuts, you may lose overall image coherency and/or it may cause an undesirable ‘mosaic’ effect.   Low cut_ic_pow values will allow the inner cuts to be larger, helping image coherency while still helping with some details.
    :param display_rate: During a diffusion run, you can monitor the progress of each image being created with this variable.  If display_rate is set to 50, DD will show you the in-progress image every 50 timesteps. Setting this to a lower value, like 5 or 10, is a good way to get an early peek at where your image is heading. If you don’t like the progression, just interrupt execution, change some settings, and re-run.  If you are planning a long, unmonitored batch, it’s better to set display_rate equal to steps, because displaying interim images does slow Colab down slightly.
    :param n_batches: This variable sets the number of still images you want DD to create.  If you are using an animation mode (see below for details) DD will ignore n_batches and create a single set of animated frames based on the animation settings.
    :param batch_name: The name of the batch, the batch id will be named as "discoart-[batch_name]-seed". To avoid your artworks be overridden by other users, please use a unique name.
    :param clip_models: CLIP Model selectors. ViTB32, ViTB16, ViTL14, RN101, RN50, RN50x4, RN50x16, RN50x64.These various CLIP models are available for you to use during image generation.  Models have different styles or ‘flavors,’ so look around.  You can mix in multiple models as well for different results.  However, keep in mind that some models are extremely memory-hungry, and turning on additional models will take additional memory and may cause a crash.The rough order of speed/mem usage is (smallest/fastest to largest/slowest):VitB32RN50RN101VitB16RN50x4RN50x16RN50x64ViTL14For RN50x64 & ViTL14 you may need to use fewer cuts, depending on your VRAM.
    :return: a DocumentArray object that has `n_batches` Documents
    """


# end_create_overload


@overload
def create(init_document: 'Document') -> 'DocumentArray':
    """
    Create an artwork using a DocArray ``Document`` object as initial state.
    :param init_document: its ``.tags`` will be used as parameters, ``.uri`` (if present) will be used as init image.
    :return: a DocumentArray object that has `n_batches` Documents
    """


def create(**kwargs) -> 'DocumentArray':
    from .config import load_config, print_args_table, save_config_svg
    from .runner import do_run

    if 'init_document' in kwargs:
        d = kwargs['init_document']
        _kwargs = d.tags
        if not _kwargs:
            warnings.warn(
                'init_document has no .tags, fallback to default config')
        if d.uri:
            _kwargs['init_image'] = kwargs['init_document'].uri
        else:
            warnings.warn(
                'init_document has no .uri, fallback to no init image')
        kwargs.pop('init_document')
        if kwargs:
            warnings.warn(
                'init_document has .tags and .uri, but kwargs are also present, will override .tags'
            )
            _kwargs.update(kwargs)
        _args = load_config(user_config=_kwargs)
    else:
        _args = load_config(user_config=kwargs)

    save_config_svg(_args)

    _args = SimpleNamespace(**_args)

>>>>>>> parent of a1ba0ed (Update)
    model, diffusion = load_diffusion_model(model_config,
                                            _args.diffusion_model,
                                            steps=_args.steps,
                                            device=device)
<<<<<<< HEAD
<<<<<<< HEAD
    clip_models = load_clip_models(device,
                                   enabled=_args.clip_models,
                                   clip_models=_clip_models_cache)
=======
=======
>>>>>>> parent of a1ba0ed (Update)

    clip_models = load_clip_models(device,
                                   enabled=_args.clip_models,
                                   clip_models=_clip_models_cache)

<<<<<<< HEAD
>>>>>>> parent of a1ba0ed (Update)
=======
>>>>>>> parent of a1ba0ed (Update)
    gc.collect()
    torch.cuda.empty_cache()
    try:
        return do_run(_args, (model, diffusion, clip_models, secondary_model),
                      device)
    except KeyboardInterrupt:
        pass
<<<<<<< HEAD
<<<<<<< HEAD


def spherical_dist_loss(x, y):
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """L2 total variation loss, as in Mahendran et al."""
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])


skip_augs = False


def sinc(x):
    return torch.where(x != 0,
                       torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    cond = torch.logical_and(-a < x, x < a)
    out = torch.where(cond, sinc(x) * sinc(x / a), x.new_zeros([]))
    return out / out.sum()


def ramp(ratio, width):
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


class MakeCutouts(nn.Module):

    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomPerspective(distortion_scale=0.4, p=0.7),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.15),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
        ])

    def forward(self, input):
        input = T.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(max_size *
                           torch.zeros(1, ).normal_(mean=0.8, std=0.3).clip(
                               float(self.cut_size / max_size), 1.0))
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety:offsety + size,
                               offsetx:offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


class MakeCutoutsDango(nn.Module):

    def __init__(self,
                 cut_size,
                 Overview=4,
                 InnerCrop=0,
                 IC_Size_Pow=0.5,
                 IC_Grey_P=0.2):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                interpolation=T.InterpolationMode.BILINEAR,
            ),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.RandomGrayscale(p=0.1),
            T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
            T.ColorJitter(brightness=0.1,
                          contrast=0.1,
                          saturation=0.1,
                          hue=0.1),
        ])

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
            **padargs,
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1: cutouts.append(cutout)
                if self.Overview >= 2: cutouts.append(gray(cutout))
                if self.Overview >= 3: cutouts.append(TF.hflip(cutout))
                if self.Overview == 4: cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([])**self.IC_Size_Pow * (max_size - min_size) +
                    min_size)
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety:offsety + size,
                               offsetx:offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)

        cutouts = torch.cat(cutouts)
        if skip_augs is not True:
            for i in range(cutouts.shape[0]):
                cutouts[i] = self.augs(cutouts[i])
        return cutouts


def resample(input, size, align_corners=True):
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), 'reflect')
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), 'reflect')
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input,
                         size,
                         mode='bicubic',
                         align_corners=align_corners)


padargs = {}


def interp(t):
    return 3 * t**2 - 2 * t**3


def perlin(width, height, scale=10, device=None):
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    wx = 1 - interp(xs)
    wy = 1 - interp(ys)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] *
                                   (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale,
                                                      height * scale)


def perlin_ms(octaves, width, height, grayscale, device):
    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    for i in range(1 if grayscale else 3):
        scale = 2**len(octaves)
        oct_width = width
        oct_height = height
        for oct in octaves:
            p = perlin(oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def create_perlin_noise(octaves, width, height, grayscale, side_y, side_x,
                        device):
    out = perlin_ms(octaves, width, height, grayscale, device)
    if grayscale:
        out = TF.resize(size=(side_y, side_x), img=out.unsqueeze(0))
        out = TF.to_pil_image(out.clamp(0, 1)).convert('RGB')
    else:
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(side_y, side_x), img=out)
        out = TF.to_pil_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def regen_perlin(perlin_mode, side_y, side_x, device, batch_size):
    if perlin_mode == 'color':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1,
                                   False, side_y, side_x, device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4,
                                    False, side_y, side_x, device)
    elif perlin_mode == 'gray':
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1,
                                   True, side_y, side_x, device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4,
                                    True, side_y, side_x, device)
    else:
        init = create_perlin_noise([1.5**-i * 0.5 for i in range(12)], 1, 1,
                                   False, side_y, side_x, device)
        init2 = create_perlin_noise([1.5**-i * 0.5 for i in range(8)], 4, 4,
                                    True, side_y, side_x, device)
    init = (TF.to_tensor(init).add(
        TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1))
    del init2
    return init.expand(batch_size, -1, -1, -1)


def do_run(args, models, device):  # -> 'DocumentArray':
    _set_seed(args.seed)
    logger.info("preparing models...")
    model, diffusion, clip_models, secondary_model = models
    normalize = T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    )
    lpips_model = lpips.LPIPS(net="vgg").to(device)

    side_x = (args.width_height[0] // 64) * 64
    side_y = (args.width_height[1] // 64) * 64
    cut_overview = eval(args.cut_overview)
    cut_innercut = eval(args.cut_innercut)
    cut_icgray_p = eval(args.cut_icgray_p)

    skip_steps = args.skip_steps

    loss_values = []
    model_stats = []

    for clip_model in clip_models:
        model_stat = {
            "clip_model": None,
            "target_embeds": [],
            "make_cutouts": None,
            "weights": [],
        }
        model_stat["clip_model"] = clip_model

        if isinstance(args.text_prompts, str):
            args.text_prompts = [args.text_prompts]

        for prompt in args.text_prompts:
            txt, weight = parse_prompt(prompt)
            txt = clip_model.encode_text(
                clip.tokenize(prompt).to(device)).float()

            model_stat["target_embeds"].append(txt)
            model_stat["weights"].append(weight)

        model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(model_stat["weights"],
                                             device=device)
        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)
    #    
    #init = None
    #if args.init_image:
    #    d = Document(uri=args.init_image).load_uri_to_image_tensor(side_x, side_y)
    #    init = TF.to_tensor(d.tensor).to(device).unsqueeze(0).mul(2).sub(1)

    #    if args.perlin_init:
    #        if args.perlin_mode == "color":
    #            init = create_perlin_noise(
    #                [1.5**-i * 0.5 for i in range(12)],1,1,False,side_y,side_x,device,)
    #            init2 = create_perlin_noise(
    #                [1.5**-i * 0.5 for i in range(8)], 4, 4, False, side_y, side_x, device)
    #        elif args.perlin_mode == "gray":
    #            init = create_perlin_noise(
    #                [1.5**-i * 0.5 for i in range(12)], 1, 1, True, side_y, side_x, device)
    #            init2 = create_perlin_noise(
    #                [1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x, device)
    #        else:
    #            init = create_perlin_noise(
    #                [1.5**-i * 0.5 for i in range(12)],1,1,False,side_y,side_x,device,)
    #            init2 = create_perlin_noise(
    #                [1.5**-i * 0.5 for i in range(8)], 4, 4, True, side_y, side_x, device)
    #        init = (TF.to_tensor(init).add(TF.to_tensor(init2)).div(2).to(device).unsqueeze(0).mul(2).sub(1))
    #        del init2

    cur_t = None

    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
            out = diffusion.p_mean_variance(model,
                                            x,
                                            my_t,
                                            clip_denoised=False,
                                            model_kwargs={"y": y})
            fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
            x_in = out["pred_xstart"] * fac + x * (1 - fac)
            x_in_grad = torch.zeros_like(x_in)
            for model_stat in model_stats:
                for i in range(args.cutn_batches):
                    t_int = (int(t.item()) + 1)
                    try:
                        input_resolution = model_stat[
                            "clip_model"].visual.input_resolution
                    except:
                        input_resolution = 224
                    cuts = MakeCutoutsDango(
                        input_resolution,
                        Overview=cut_overview[1000 - t_int],
                        InnerCrop=cut_innercut[1000 - t_int],
                        IC_Size_Pow=args.cut_ic_pow,
                        IC_Grey_P=cut_icgray_p[1000 - t_int],
                    )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = (
                        model_stat["clip_model"].encode_image(clip_in).float())
                    dists = spherical_dist_loss(
                        image_embeds.unsqueeze(1),
                        model_stat["target_embeds"].unsqueeze(0),
                    )
                    dists = dists.view([
                        cut_overview[1000 - t_int] +
                        cut_innercut[1000 - t_int],
                        n,
                        -1,
                    ])
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(losses.sum().item(
                    ))  # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += (torch.autograd.grad(
                        losses.sum() * args.clip_guidance_scale, x_in)[0] /
                                  args.cutn_batches)

            tv_losses = tv_loss(x_in)
            range_losses = range_loss(out["pred_xstart"])
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (tv_losses.sum() * args.tv_scale +
                    range_losses.sum() * args.range_scale +
                    sat_losses.sum() * args.sat_scale)
            if init is not None and args.init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * args.init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_NaN = True
                grad = torch.zeros_like(x)
        if args.clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return (grad * magnitude.clamp(max=args.clamp_max) / magnitude
                    )  # min=-0.02, min=-clamp_max,
        return grad

    if args.diffusion_sampling_mode == "ddim":
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.plms_sample_loop_progressive

    logger.info("creating artwork...")

    image_display = Output()

    org_seed = args.seed
    for _nb in range(args.n_batches):

        new_seed = org_seed + _nb
        args.seed = new_seed

        display.clear_output(wait=True)
        display.display(
            print_args_table(vars(args),
                             only_non_default=True,
                             console_print=False),
            image_display,
        )
        gc.collect()
        torch.cuda.empty_cache()

        d = Document(tags=vars(args))

        cur_t = diffusion.num_timesteps - skip_steps - 1

        if args.perlin_init:
            init = regen_perlin(args.perlin_mode, args.side_y, side_x, device,
                                args.batch_size)

        if args.diffusion_sampling_mode == "ddim":
            samples = sample_fn(
                model,
                (args.batch_size, 3, side_y, side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                eta=args.eta,
            )
        else:
            samples = sample_fn(
                model,
                (args.batch_size, 3, side_y, side_x),
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_steps,
                init_image=init,
                randomize_class=args.randomize_class,
                order=2,
            )

        threads = []
        for j, sample in enumerate(samples):
            cur_t -= 1
            with image_display:
                if j % args.display_rate == 0 or cur_t == -1:
                    for _, image in enumerate(sample["pred_xstart"]):
                        image = TF.to_pil_image(
                            image.add(1).div(2).clamp(0, 1))
                        c = Document(tags={"cur_t": cur_t})
                        c.load_pil_image_to_datauri(image)
                        image.save(f"{args.name_docarray}{_nb}.png")
                        display.clear_output(wait=True)
                        display.display(image)

        for t in threads:
            t.join()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

#from . import __resources_path__


with open(f"{__resources_path__}/default.yml") as ymlfile:
    default_args = yaml.load(ymlfile, Loader=Loader)


def print_args_table(cfg,
                     console=None,
                     only_non_default: bool = True,
                     console_print: bool = True):

    from rich.table import Table
    from rich import box
    from rich.console import Console

    if console is None:
        console = Console()

    param_str = Table(
        title=cfg["name_docarray"],
        box=box.ROUNDED,
        highlight=True,
        title_justify="left",
    )
    param_str.add_column("Argument", justify="right")
    param_str.add_column("Value", justify="left")

    for k, v in sorted(cfg.items()):
        value = str(v)
        _non_default = False
        if not default_args.get(k, None) == v:
            if not only_non_default:
                k = f"[b]{k}*[/]"
            _non_default = True

        if not only_non_default or _non_default:
            param_str.add_row(k, value)

    if console_print:
        console.print(param_str)
    return param_str



=======
    finally:
        from IPython import display
        display.clear_output(wait=True)

        _name = _args.name_docarray

        if os.path.exists(f'{_name}.protobuf.lz4'):
            from docarray import DocumentArray
            _da = DocumentArray.load_binary(f'{_name}.protobuf.lz4')
            if _da and _da[0].uri:
                _da.plot_image_sprites(skip_empty=True,
                                       show_index=True,
                                       keep_aspect_ratio=True)

        print_args_table(vars(_args))
        from IPython.display import FileLink, display

        persist_file = FileLink(
            f'{_name}.protobuf.lz4',
            result_html_prefix=
            f'▶ (in case cloud storage failed) Download the result backup: ',
        )
        config_file = FileLink(
            f'{_name}.svg',
            result_html_prefix=f'▶ Download the config as SVG image: ',
        )
        display(persist_file, config_file)

        from rich import print
        from rich.markdown import Markdown

        md = Markdown(f'''
Results are stored in a [DocumentArray](https://docarray.jina.ai/fundamentals/documentarray/) and synced to the cloud.

You can simply pull it from any machine:

```python
# pip install docarray[common]
from docarray import DocumentArray

da = DocumentArray.pull('{_name}')
```

If for some reason the cloud storage is not available, you may also download the file manually and load it from local disk:

```python
da = DocumentArray.load_binary('{_name}.protobuf.lz4')
```

More usage such as plotting, post-analysis can be found in the [README](https://github.com/jina-ai/discoart).
        ''')
        print(md)
        gc.collect()
        torch.cuda.empty_cache()
>>>>>>> parent of a1ba0ed (Update)
=======
    finally:
        from IPython import display
        display.clear_output(wait=True)

        _name = _args.name_docarray

        if os.path.exists(f'{_name}.protobuf.lz4'):
            from docarray import DocumentArray
            _da = DocumentArray.load_binary(f'{_name}.protobuf.lz4')
            if _da and _da[0].uri:
                _da.plot_image_sprites(skip_empty=True,
                                       show_index=True,
                                       keep_aspect_ratio=True)

        print_args_table(vars(_args))
        from IPython.display import FileLink, display

        persist_file = FileLink(
            f'{_name}.protobuf.lz4',
            result_html_prefix=
            f'▶ (in case cloud storage failed) Download the result backup: ',
        )
        config_file = FileLink(
            f'{_name}.svg',
            result_html_prefix=f'▶ Download the config as SVG image: ',
        )
        display(persist_file, config_file)

        from rich import print
        from rich.markdown import Markdown

        md = Markdown(f'''
Results are stored in a [DocumentArray](https://docarray.jina.ai/fundamentals/documentarray/) and synced to the cloud.

You can simply pull it from any machine:

```python
# pip install docarray[common]
from docarray import DocumentArray

da = DocumentArray.pull('{_name}')
```

If for some reason the cloud storage is not available, you may also download the file manually and load it from local disk:

```python
da = DocumentArray.load_binary('{_name}.protobuf.lz4')
```

More usage such as plotting, post-analysis can be found in the [README](https://github.com/jina-ai/discoart).
        ''')
        print(md)
        gc.collect()
        torch.cuda.empty_cache()
>>>>>>> parent of a1ba0ed (Update)
