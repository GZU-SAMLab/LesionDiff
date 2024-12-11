# sample leaf diseases inference code

import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import torchvision
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
import clip
from torchvision.transforms import Resize
import pickle
from detectron2.engine import DefaultPredictor

from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from torchvision.ops import box_convert
import random

from groundingdino.util.inference import load_model, load_image, predict, annotate, Model
from torchvision.ops import box_convert

wm = "leaf"
wm_encoder = WatermarkEncoder()
wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x

def get_tensor(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True):
    transform_list = []
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]

    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                                                (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--n_imgs",
        type=int,
        default=100,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--mask_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--reference_path",
        type=str,
        help="evaluate at this precision",
        default=""
    )
    parser.add_argument(
        "--gd_ckpt",
        type=str,
        help="evaluate at this precision",
        default="GroundingDINO/models/groundingdino_swint_ogc.pth"
    )
    parser.add_argument(
        "--gd_cfg",
        type=str,
        help="evaluate at this precision",
        default="GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    )
    parser.add_argument(
        '--test_path', 
        type=str, 
        default='./'
    )
    parser.add_argument(
        '--max_size',
        type=int,
        default='100'
    )
    return parser.parse_args()

def generate_mask(gd_ckpt, gd_cfg, gd_model, image, max_size=500) -> np:
    """
    Generate a random mask. mask part (255, 255, 255), non-mask part (0, 0, 0)
    Parameters:
        mask_rcnn_pkt: Mask R-CNN model parameters
        mask_rcnn_cfg: Mask R-CNN model configuration
        image: input image
    """
    try:
        image_source, image = load_image(image)
        if gd_model is None:
            gd_model = load_model(gd_cfg, gd_ckpt)
        image = image.float()
        gd_model = gd_model.float()
        boxes, _, _ = predict(
            model=gd_model,
            image=image,
            caption="leaf",
            box_threshold=0.35,
            text_threshold=0.25,
            device="cpu",
        )
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy().astype(int)
        if xyxy.shape[0] == 0:
            return None
        # find the max leaf
        max_leaf_area = 0
        max_leaf_box = xyxy[0].astype(int)
        for box in xyxy:
            width = box[2] - box[0]
            height = box[3] - box[1]
            area = width * height
            if area > max_leaf_area:
                max_leaf_area = area
                max_leaf_box = box
        leaf = max_leaf_box
        leaf_center_x = (leaf[0] + leaf[2]) // 2
        leaf_center_y = (leaf[1] + leaf[3]) // 2
        size = max_size
        min_center_x = leaf_center_x - (size//2)
        max_center_x = leaf_center_x + (size//2)
        # if the leaf is too close to the edge, we need to expand the mask
        if min_center_x < 0:
            size -= min_center_x
            min_center_x = 0
        if max_center_x > w:
            size -= max_center_x - w
            max_center_x = w
        min_center_y = leaf_center_y - (size//2)
        max_center_y = leaf_center_y + (size//2)
        if min_center_y < 0:
            size -= min_center_y
            min_center_y = 0
        if max_center_y > h:
            size -= max_center_y - h
            max_center_y = h
        choices_x = random.randint(leaf_center_x - (size//2) , leaf_center_x + (size//2))
        choices_y = random.randint(leaf_center_y - (size//2) , leaf_center_y + (size//2))
        choice_width = random.randint(size//2, size)
        choice_height = random.randint(size//2, size)
        left = choices_x - choice_width//2
        right = choices_x + choice_width//2
        top = choices_y - choice_height//2
        bottom = choices_y + choice_height//2
        # make sure the mask is within the image boundaries
        if left < 0:
            right -= left
            left = 0
        if top < 0:
            bottom -= top
            top = 0
        if right > w:
            left -= right - w
            right = w
        if bottom > h:
            top -= bottom - h
            bottom = h
        new_mask_img = np.zeros((h, w), dtype=np.uint8)
        new_mask_img = cv2.rectangle(new_mask_img, (left, top), (right, bottom), (255, 255, 255), -1)
        return (new_mask_img, gd_model)
    except:
        return (None, gd_model)

def change_unqualified_bbox(left, top, right, bottom, standard, size):
    if right - left < standard:
        if left - 50 > 0:
            left -= 50
        else:
            left = 0
        if right + 50 < size:
            right += 50
        else:
            right = size
    if bottom - top < standard:
        if top - 50 > 0:
            top -= 50
        else:
            top = 0
        if bottom + 50 < size:
            bottom += 50
        else:
            bottom = size
    return left, top, right, bottom

def get_ref_image(ref_image_path, gd_model):

    ### Get bbox
    image_source, image = load_image(ref_image_path)
    image = image.float()
    
    try:
        boxes, logits, phrases = predict(
            model=gd_model,
            image=image,
            caption="spot",
            box_threshold=0.35,
            text_threshold=0.25,
            device="cpu",
        )
        h, w, _ = image_source.shape
        boxes = boxes * torch.Tensor([w, h, w, h])
        xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
        bbox_disease = np.array([156, 156, 356, 356])
        type = "whole"
        if xyxy.shape[0] == 0:
            bbox_disease = np.array([156, 156, 356, 356])
        else:
            if type == "one":
                bbox_disease = xyxy[0].astype(int)
                bbox_disease = [*change_unqualified_bbox(bbox_disease[0], bbox_disease[1], bbox_disease[2], bbox_disease[3], 224, 512)]
            else:
                # find the largest bbox
                left, top, right, bottom = xyxy[0].astype(int)
                for box in xyxy:
                    box = box.astype(int)
                    if box[0] < left:
                        left = box[0]
                    if box[1] < top:
                        top = box[1]
                    if box[2] > right:
                        right = box[2]
                    if box[3] > bottom:
                        bottom = box[3]
                bbox_disease = [*change_unqualified_bbox(left=left, top=top, right=right, bottom=bottom, standard=224, size=512)]
        bbox = bbox_disease
        ref_image = image_source[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        ref_image = cv2.resize(ref_image, (224, 224))
        ref_image = Image.fromarray(ref_image).convert("RGB")
        return ref_image
    except:
        return None

def get_img_id(path):
    id = os.path.basename(path).split('.')[0]
    return id

def main():
    opt = parse()

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir
    mask_save_path = os.path.join(outpath, 'mask')
    os.makedirs(mask_save_path, exist_ok=True)

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size

    sample_path = os.path.join(outpath, "source")
    result_path = os.path.join(outpath, "results")
    grid_path=os.path.join(outpath, "grid")
    os.makedirs(sample_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)
    os.makedirs(grid_path, exist_ok=True)
  
    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    precision_scope = autocast if opt.precision=="autocast" else nullcontext
# 	这个是参考目录和健康叶子目录分开的情况
#    ref_dir = os.path.join(opt.reference_path)
#    下面是之前用的一个文件夹中有image和ref
    ref_dir = os.path.join(opt.test_path, 'ref')
    ref_img = os.listdir(ref_dir)
    imgs = [get_img_id(i) for i in ref_img]
# 	这个是参考目录和健康叶子目录分开的情况
#    img_dir = os.path.join(opt.image_path)
#    下面是之前用的一个文件夹中有image和ref
    img_dir = os.path.join(opt.test_path, 'image')
    ref_img_dir = ref_dir
    predictor = None

    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                for i, img in tqdm(enumerate(imgs), total=len(imgs), desc="Generating", leave=True, position=0):
                    
                    filename=os.path.basename(os.path.join(img_dir, img + '.jpg'))
                    img_p = Image.open(os.path.join(img_dir, img + '.jpg')).convert("RGB")
                    image_tensor = get_tensor()(img_p)
                    image_tensor = image_tensor.unsqueeze(0)
                    
                    
                    ### TODO: add mask
                    mask, predictor = generate_mask(
                        opt.gd_ckpt, 
                        opt.gd_cfg, 
                        predictor,
                        os.path.join(img_dir, img + '.jpg'),
                        max_size=opt.max_size,
                    )
                    if mask is None:
                        print(f'{img} is None')
                        continue  
                    cv2.imwrite(os.path.join(mask_save_path, img + '.jpg'), mask)

                    mask = np.array(mask)[None,None]
                    mask = 1 - mask.astype(np.float32)/255.0
                    mask[mask < 0.5] = 0
                    mask[mask >= 0.5] = 1
                    mask_tensor = torch.from_numpy(mask)
                    
                    ref_p = get_ref_image(os.path.join(ref_img_dir, img + '.jpg'), gd_model=predictor)
                    if ref_p is None:
                        continue
                    ref_tensor=get_tensor_clip()(ref_p)
                    ref_tensor = ref_tensor.unsqueeze(0)
                    ####
                    inpaint_image = image_tensor*mask_tensor
                    test_model_kwargs={}
                    test_model_kwargs['inpaint_mask']=mask_tensor.to(device)
                    test_model_kwargs['inpaint_image']=inpaint_image.to(device)
                    ref_tensor=ref_tensor.to(device)
                    uc = None
                    if opt.scale != 1.0:
                        uc = model.learnable_vector
                    c = model.get_learned_conditioning(ref_tensor.to(torch.float16))
                    c = model.proj_out(c)
                    inpaint_mask=test_model_kwargs['inpaint_mask']
                    z_inpaint = model.encode_first_stage(test_model_kwargs['inpaint_image'])
                    z_inpaint = model.get_first_stage_encoding(z_inpaint).detach()
                    test_model_kwargs['inpaint_image']=z_inpaint
                    test_model_kwargs['inpaint_mask']=Resize([z_inpaint.shape[-2],z_inpaint.shape[-1]])(test_model_kwargs['inpaint_mask'])

                    shape = [opt.C, opt.H // opt.f, opt.W // opt.f]
                    samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                        conditioning=c,
                                                        batch_size=opt.n_samples,
                                                        shape=shape,
                                                        verbose=False,
                                                        unconditional_guidance_scale=opt.scale,
                                                        unconditional_conditioning=uc,
                                                        eta=opt.ddim_eta,
                                                        x_T=start_code,
                                                        test_model_kwargs=test_model_kwargs)

                    x_samples_ddim = model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                    x_checked_image=x_samples_ddim
                    x_checked_image_torch = torch.from_numpy(x_checked_image).permute(0, 3, 1, 2)

                    def un_norm(x):
                        return (x+1.0)/2.0
                    def un_norm_clip(x):
                        x[0,:,:] = x[0,:,:] * 0.26862954 + 0.48145466
                        x[1,:,:] = x[1,:,:] * 0.26130258 + 0.4578275
                        x[2,:,:] = x[2,:,:] * 0.27577711 + 0.40821073
                        return x

                    if not opt.skip_save:
                        for i,x_sample in enumerate(x_checked_image_torch):
                            

                            all_img=[]
                            all_img.append(un_norm(image_tensor[i]).cpu())
                            all_img.append(un_norm(inpaint_image[i]).cpu())
                            ref_img=ref_tensor
                            ref_img=Resize([opt.H, opt.W])(ref_img)
                            all_img.append(un_norm_clip(ref_img[i]).cpu())
                            all_img.append(x_sample)
                            grid = torch.stack(all_img, 0)
                            grid = make_grid(grid)
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            img = Image.fromarray(grid.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(grid_path, 'grid-'+filename[:-4]+'_'+str(opt.seed)+'.png'))
                            


                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(result_path, filename[:-4]+'_'+str(opt.seed)+".png"))
                            
                            mask_save=255.*rearrange(un_norm(inpaint_mask[i]).cpu(), 'c h w -> h w c').numpy()
                            mask_save= cv2.cvtColor(mask_save,cv2.COLOR_GRAY2RGB)
                            mask_save = Image.fromarray(mask_save.astype(np.uint8))
                            mask_save.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_mask.png"))
                            GT_img=255.*rearrange(all_img[0], 'c h w -> h w c').numpy()
                            GT_img = Image.fromarray(GT_img.astype(np.uint8))
                            GT_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_GT.png"))
                            inpaint_img=255.*rearrange(all_img[1], 'c h w -> h w c').numpy()
                            inpaint_img = Image.fromarray(inpaint_img.astype(np.uint8))
                            inpaint_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_inpaint.png"))
                            ref_img=255.*rearrange(all_img[2], 'c h w -> h w c').numpy()
                            ref_img = Image.fromarray(ref_img.astype(np.uint8))
                            ref_img.save(os.path.join(sample_path, filename[:-4]+'_'+str(opt.seed)+"_ref.png"))

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
