import torch
import comfy.sample
import comfy.utils
import comfy.model_management
import latent_preview
import math
import gc
from server import PromptServer
import psutil
import ctypes
from ctypes import wintypes
import time
import platform
import subprocess
import torch.nn.functional as F

def common_ksampler(models, seed, steps, cfgs, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noises=(False, True), force_full_denoises=(False, True), boundary = 0.875):

    model_high_noise, model_low_noise = models
    steps_high_noise, steps_low_noise = steps
    cfg_high_noise, cfg_low_noise = cfgs
    disable_noise_high_noise, disable_noise_low_noise = disable_noises
    force_full_denoise_high_noise, force_full_denoise_low_noise = force_full_denoises

    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

    if (steps_high_noise is None or steps_high_noise <= 0) and (steps_low_noise is None or steps_low_noise <= 0):
        raise Exception(f"高噪步数和低噪步数不能同时为0 / high_noise_steps and low_noise_steps cannot both be 0")
    
    total_steps = steps_high_noise + steps_low_noise

    # 自动计算模型切换点，暂时先不加，后面考虑加高级采样器传入总步数做计算
    # # first, we get all sigmas
    # sampling = model_high_noise.get_model_object("model_sampling")
    # sigmas = comfy.samplers.calculate_sigmas(sampling,scheduler,steps)
    # # why are timesteps 0-1000?
    # timesteps = [sampling.timestep(sigma)/1000 for sigma in sigmas.tolist()]
    # switching_step = steps
    # for (i,t) in enumerate(timesteps[1:]):
    #     if t < boundary:
    #         switching_step = i
    #         break
    # print(f"switching model at step {switching_step}")
    # start_with_high = start_step<switching_step
    # end_wth_low = last_step>=switching_step

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    latent_image = latent["samples"]
    
    if steps_high_noise > 0:
        print("Running high noise model...")

        latent_image = comfy.sample.fix_empty_latent_channels(model_high_noise, latent_image)

        if disable_noise_high_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)

        callback = latent_preview.prepare_callback(model_high_noise, total_steps)
        
        latent_image = comfy.sample.sample(model_high_noise, noise, total_steps, cfg_high_noise, sampler_name, scheduler, positive[0], negative[0], latent_image,
                                    denoise=denoise, disable_noise=disable_noise_high_noise, start_step=0, last_step=steps_high_noise,
                                    force_full_denoise=force_full_denoise_high_noise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)


    if steps_low_noise > 0:
        print("Running low noise model...")

        latent_image = comfy.sample.fix_empty_latent_channels(model_low_noise, latent_image)

        if disable_noise_low_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = comfy.sample.prepare_noise(latent_image, seed, batch_inds)
        
        callback = latent_preview.prepare_callback(model_low_noise, total_steps)
        
        latent_image = comfy.sample.sample(model_low_noise, noise, total_steps, cfg_low_noise, sampler_name, scheduler, positive[1], negative[1], latent_image,
                                    denoise=denoise, disable_noise=disable_noise_low_noise, start_step=steps_high_noise, last_step=total_steps,
                                    force_full_denoise=force_full_denoise_low_noise, noise_mask=noise_mask, callback=callback, disable_pbar=disable_pbar, seed=seed)

    out = latent.copy()
    out["samples"] = latent_image


    return out




def calculate_middle_frame_idx(
    ratio: float,
    total_frames: int,
    downsample: int = 4,
    allow_edges: bool = False,
    edge_margin: int = 4
) -> int:
    """
    根据比例计算中间帧索引，同时对齐 latent 下采样步长，并可控制边界保护。

    Args:
        ratio (float): 中间帧比例，0~1
        total_frames (int): 视频总帧数
        downsample (int): latent 时间下采样步长，默认 WAN 2.2 是 4
        allow_edges (bool): 是否允许首尾帧重叠，默认 False
        edge_margin (int): 与首尾帧最小间距，如果 allow_edges=False，默认 4

    Returns:
        int: 对齐后的帧索引
    """
    # 1. 先按比例计算目标帧
    desired_idx = int(round(ratio * (total_frames - 1)))

    # 2. 对齐到 latent 下采样步长
    latent_idx = desired_idx // downsample
    aligned_idx = latent_idx * downsample

    # 3. 边界保护
    if allow_edges:
        # 允许首尾帧
        aligned_idx = max(0, min(aligned_idx, total_frames - 1))
    else:
        # 不允许首尾帧重叠，保持安全间距
        min_idx = edge_margin
        max_idx = total_frames - 1 - edge_margin
        # 当 total_frames 太短时，自动修正范围
        min_idx = min(min_idx, max_idx)
        max_idx = max(max_idx, min_idx)
        aligned_idx = max(min_idx, min(aligned_idx, max_idx))

    return aligned_idx



def cleanGPUUsedForce():
    gc.collect()
    comfy.model_management.unload_all_models()
    comfy.model_management.soft_empty_cache()
    PromptServer.instance.prompt_queue.set_flag("free_memory", True)



def clean_ram(clean_file_cache=True, clean_processes=True, clean_dlls=True, retry_times=3, anything=None, unique_id=None, extra_pnginfo=None):
    try:
        def get_ram_usage():
            memory = psutil.virtual_memory()
            return memory.percent, memory.available / (1024 * 1024)

        before_usage, before_available = get_ram_usage()
        system = platform.system()

        for attempt in range(retry_times):
            if clean_file_cache:
                try:
                    if system == "Windows":
                        ctypes.windll.kernel32.SetSystemFileCacheSize(-1, -1, 0)
                    elif system == "Linux":
                        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
                                      check=False, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
                except:
                    pass

            if clean_processes:
                if system == "Windows":
                    for process in psutil.process_iter(['pid', 'name']):
                        try:
                            handle = ctypes.windll.kernel32.OpenProcess(
                                wintypes.DWORD(0x001F0FFF),
                                wintypes.BOOL(False),
                                wintypes.DWORD(process.info['pid'])
                            )
                            ctypes.windll.psapi.EmptyWorkingSet(handle)
                            ctypes.windll.kernel32.CloseHandle(handle)
                        except:
                            continue

            if clean_dlls:
                try:
                    if system == "Windows":
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
                    elif system == "Linux":
                        subprocess.run(["sync"], check=True)
                except:
                    pass

            time.sleep(1)

        after_usage, after_available = get_ram_usage()
        freed_mb = after_available - before_available
        print(f"RAM清理完成 / RAM cleanup completed [{before_usage:.1f}% → {after_usage:.1f}%, 释放 / Freed: {freed_mb:.0f}MB]")

    except Exception as e:
        print(f"RAM清理失败 / RAM cleanup failed: {str(e)}")

    return anything



def image_resize(image, width, height, keep_proportion, upscale_method, divisible_by, pad_color, crop_position, unique_id, device="cpu", mask=None, per_batch=64):
        B, H, W, C = image.shape

        if device == "gpu":
            if upscale_method == "lanczos":
                raise Exception("Lanczos is not supported on the GPU")
            device = comfy.model_management.get_torch_device()
        else:
            device = torch.device("cpu")

        pillarbox_blur = keep_proportion == "pillarbox_blur"
        
        # Initialize padding variables
        pad_left = pad_right = pad_top = pad_bottom = 0

        if keep_proportion in ["resize", "total_pixels"] or keep_proportion.startswith("pad") or pillarbox_blur:
            if keep_proportion == "total_pixels":
                total_pixels = width * height
                aspect_ratio = W / H
                new_height = int(math.sqrt(total_pixels / aspect_ratio))
                new_width = int(math.sqrt(total_pixels * aspect_ratio))
                
            # If one of the dimensions is zero, calculate it to maintain the aspect ratio
            elif width == 0 and height == 0:
                new_width = W
                new_height = H
            elif width == 0 and height != 0:
                ratio = height / H
                new_width = round(W * ratio)
                new_height = height
            elif height == 0 and width != 0:
                ratio = width / W
                new_width = width
                new_height = round(H * ratio)
            elif width != 0 and height != 0:
                ratio = min(width / W, height / H)
                new_width = round(W * ratio)
                new_height = round(H * ratio)
            else:
                new_width = width
                new_height = height

            if keep_proportion.startswith("pad") or pillarbox_blur:
                # Calculate padding based on position
                if crop_position == "center":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "top":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = 0
                    pad_bottom = height - new_height
                elif crop_position == "bottom":
                    pad_left = (width - new_width) // 2
                    pad_right = width - new_width - pad_left
                    pad_top = height - new_height
                    pad_bottom = 0
                elif crop_position == "left":
                    pad_left = 0
                    pad_right = width - new_width
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top
                elif crop_position == "right":
                    pad_left = width - new_width
                    pad_right = 0
                    pad_top = (height - new_height) // 2
                    pad_bottom = height - new_height - pad_top

            width = new_width
            height = new_height
        else:
            if width == 0:
                width = W
            if height == 0:
                height = H

        if divisible_by > 1:
            width = width - (width % divisible_by)
            height = height - (height % divisible_by)

        # Preflight estimate (log-only when batching is active)
        if per_batch != 0 and B > per_batch:
            try:
                bytes_per_elem = image.element_size()  # typically 4 for float32
                est_total_bytes = B * height * width * C * bytes_per_elem
                est_mb = est_total_bytes / (1024 * 1024)
                msg = f"<tr><td>Resize</td><td>estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}</td></tr>"
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(msg, unique_id)
                    except:
                        pass
                else:
                    print(f"estimated output ~{est_mb:.2f} MB; batching {per_batch}/{B}")
            except:
                pass

        def _process_subbatch(in_image, in_mask, pad_left, pad_right, pad_top, pad_bottom):
            # Avoid unnecessary clones; only move if needed
            out_image = in_image if in_image.device == device else in_image.to(device)
            out_mask = None if in_mask is None else (in_mask if in_mask.device == device else in_mask.to(device))

            # Crop logic
            if keep_proportion == "crop":
                old_height = out_image.shape[-3]
                old_width = out_image.shape[-2]
                old_aspect = old_width / old_height
                new_aspect = width / height
                if old_aspect > new_aspect:
                    crop_w = round(old_height * new_aspect)
                    crop_h = old_height
                else:
                    crop_w = old_width
                    crop_h = round(old_width / new_aspect)
                if crop_position == "center":
                    x = (old_width - crop_w) // 2
                    y = (old_height - crop_h) // 2
                elif crop_position == "top":
                    x = (old_width - crop_w) // 2
                    y = 0
                elif crop_position == "bottom":
                    x = (old_width - crop_w) // 2
                    y = old_height - crop_h
                elif crop_position == "left":
                    x = 0
                    y = (old_height - crop_h) // 2
                elif crop_position == "right":
                    x = old_width - crop_w
                    y = (old_height - crop_h) // 2
                out_image = out_image.narrow(-2, x, crop_w).narrow(-3, y, crop_h)
                if out_mask is not None:
                    out_mask = out_mask.narrow(-1, x, crop_w).narrow(-2, y, crop_h)

            out_image = comfy.utils.common_upscale(out_image.movedim(-1,1), width, height, upscale_method, crop="disabled").movedim(1,-1)
            if out_mask is not None:
                if upscale_method == "lanczos":
                    out_mask = comfy.utils.common_upscale(out_mask.unsqueeze(1).repeat(1, 3, 1, 1), width, height, upscale_method, crop="disabled").movedim(1,-1)[:, :, :, 0]
                else:
                    out_mask = comfy.utils.common_upscale(out_mask.unsqueeze(1), width, height, upscale_method, crop="disabled").squeeze(1)

            # Pad logic
            if (keep_proportion.startswith("pad") or pillarbox_blur) and (pad_left > 0 or pad_right > 0 or pad_top > 0 or pad_bottom > 0):
                padded_width = width + pad_left + pad_right
                padded_height = height + pad_top + pad_bottom
                if divisible_by > 1:
                    width_remainder = padded_width % divisible_by
                    height_remainder = padded_height % divisible_by
                    if width_remainder > 0:
                        extra_width = divisible_by - width_remainder
                        pad_right += extra_width
                    if height_remainder > 0:
                        extra_height = divisible_by - height_remainder
                        pad_bottom += extra_height

                pad_mode = (
                    "pillarbox_blur" if pillarbox_blur else
                    "edge" if keep_proportion == "pad_edge" else
                    "edge_pixel" if keep_proportion == "pad_edge_pixel" else
                    "color"
                )
                out_image, out_mask = image_pad(out_image, pad_left, pad_right, pad_top, pad_bottom, 0, pad_color, pad_mode, mask=out_mask)

            return out_image, out_mask

        # If batching disabled (per_batch==0) or batch fits, process whole batch
        if per_batch == 0 or B <= per_batch:
            out_image, out_mask = _process_subbatch(image, mask, pad_left, pad_right, pad_top, pad_bottom)
        else:
            chunks = []
            mask_chunks = [] if mask is not None else None
            total_batches = (B + per_batch - 1) // per_batch
            current_batch = 0
            for start_idx in range(0, B, per_batch):
                current_batch += 1
                end_idx = min(start_idx + per_batch, B)
                sub_img = image[start_idx:end_idx]
                sub_mask = mask[start_idx:end_idx] if mask is not None else None
                sub_out_img, sub_out_mask = _process_subbatch(sub_img, sub_mask, pad_left, pad_right, pad_top, pad_bottom)
                chunks.append(sub_out_img.cpu())
                if mask is not None:
                    mask_chunks.append(sub_out_mask.cpu() if sub_out_mask is not None else None)
                # Per-batch progress update
                if unique_id and PromptServer is not None:
                    try:
                        PromptServer.instance.send_progress_text(
                            f"<tr><td>Resize</td><td>batch {current_batch}/{total_batches} · images {end_idx}/{B}</td></tr>",
                            unique_id
                        )
                    except:
                        pass
                else:
                    try:
                        print(f"batch {current_batch}/{total_batches} · images {end_idx}/{B}")
                    except:
                        pass
            out_image = torch.cat(chunks, dim=0)
            if mask is not None and any(m is not None for m in mask_chunks):
                out_mask = torch.cat([m for m in mask_chunks if m is not None], dim=0)
            else:
                out_mask = None

        # Progress UI
        if unique_id and PromptServer is not None:
            try:
                num_elements = out_image.numel()
                element_size = out_image.element_size()
                memory_size_mb = (num_elements * element_size) / (1024 * 1024)
                PromptServer.instance.send_progress_text(
                    f"<tr><td>Output: </td><td><b>{out_image.shape[0]}</b> x <b>{out_image.shape[2]}</b> x <b>{out_image.shape[1]} | {memory_size_mb:.2f}MB</b></td></tr>",
                    unique_id
                )
            except:
                pass

        return (out_image.cpu(), out_image.shape[2], out_image.shape[1], out_mask.cpu() if out_mask is not None else torch.zeros(64,64, device=torch.device("cpu"), dtype=torch.float32))



def image_pad(image, left, right, top, bottom, extra_padding, color, pad_mode, mask=None, target_width=None, target_height=None):
        B, H, W, C = image.shape
        # Resize masks to image dimensions if necessary
        if mask is not None:
            BM, HM, WM = mask.shape
            if HM != H or WM != W:
                mask = F.interpolate(mask.unsqueeze(1), size=(H, W), mode='nearest-exact').squeeze(1)

        # Parse background color
        bg_color = [int(x.strip())/255.0 for x in color.split(",")]
        if len(bg_color) == 1:
            bg_color = bg_color * 3  # Grayscale to RGB
        bg_color = torch.tensor(bg_color, dtype=image.dtype, device=image.device)

        # Calculate padding sizes with extra padding
        if target_width is not None and target_height is not None:
            if extra_padding > 0:
                image = comfy.utils.common_upscale(image.movedim(-1, 1), W - extra_padding, H - extra_padding, "lanczos", "disabled").movedim(1, -1)
                B, H, W, C = image.shape

            padded_width = target_width
            padded_height = target_height
            pad_left = (padded_width - W) // 2
            pad_right = padded_width - W - pad_left
            pad_top = (padded_height - H) // 2
            pad_bottom = padded_height - H - pad_top
        else:
            pad_left = left + extra_padding
            pad_right = right + extra_padding
            pad_top = top + extra_padding
            pad_bottom = bottom + extra_padding

            padded_width = W + pad_left + pad_right
            padded_height = H + pad_top + pad_bottom

        # Pillarbox blur mode
        if pad_mode == "pillarbox_blur":
            def _gaussian_blur_nchw(img_nchw, sigma_px):
                if sigma_px <= 0:
                    return img_nchw
                radius = max(1, int(3.0 * float(sigma_px)))
                k = 2 * radius + 1
                x = torch.arange(-radius, radius + 1, device=img_nchw.device, dtype=img_nchw.dtype)
                k1 = torch.exp(-(x * x) / (2.0 * float(sigma_px) * float(sigma_px)))
                k1 = k1 / k1.sum()
                kx = k1.view(1, 1, 1, k)
                ky = k1.view(1, 1, k, 1)
                c = img_nchw.shape[1]
                kx = kx.repeat(c, 1, 1, 1)
                ky = ky.repeat(c, 1, 1, 1)
                img_nchw = F.conv2d(img_nchw, kx, padding=(0, radius), groups=c)
                img_nchw = F.conv2d(img_nchw, ky, padding=(radius, 0), groups=c)
                return img_nchw

            out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
            for b in range(B):
                scale_fill = max(padded_width / float(W), padded_height / float(H)) if (W > 0 and H > 0) else 1.0
                bg_w = max(1, int(round(W * scale_fill)))
                bg_h = max(1, int(round(H * scale_fill)))
                src_b = image[b].movedim(-1, 0).unsqueeze(0)
                bg = comfy.utils.common_upscale(src_b, bg_w, bg_h, "bilinear", crop="disabled")
                y0 = max(0, (bg_h - padded_height) // 2)
                x0 = max(0, (bg_w - padded_width) // 2)
                y1 = min(bg_h, y0 + padded_height)
                x1 = min(bg_w, x0 + padded_width)
                bg = bg[:, :, y0:y1, x0:x1]
                if bg.shape[2] != padded_height or bg.shape[3] != padded_width:
                    pad_h = padded_height - bg.shape[2]
                    pad_w = padded_width - bg.shape[3]
                    pad_top_fix = max(0, pad_h // 2)
                    pad_bottom_fix = max(0, pad_h - pad_top_fix)
                    pad_left_fix = max(0, pad_w // 2)
                    pad_right_fix = max(0, pad_w - pad_left_fix)
                    bg = F.pad(bg, (pad_left_fix, pad_right_fix, pad_top_fix, pad_bottom_fix), mode="replicate")
                sigma = max(1.0, 0.006 * float(min(padded_height, padded_width)))
                bg = _gaussian_blur_nchw(bg, sigma_px=sigma)
                if C >= 3:
                    r, g, bch = bg[:, 0:1], bg[:, 1:2], bg[:, 2:3]
                    luma = 0.2126 * r + 0.7152 * g + 0.0722 * bch
                    gray = torch.cat([luma, luma, luma], dim=1)
                    desat = 0.20
                    rgb = torch.cat([r, g, bch], dim=1)
                    rgb = rgb * (1.0 - desat) + gray * desat
                    bg[:, 0:3, :, :] = rgb
                dim = 0.35
                bg = torch.clamp(bg * dim, 0.0, 1.0)
                out_image[b] = bg.squeeze(0).movedim(0, -1)
            out_image[:, pad_top:pad_top+H, pad_left:pad_left+W, :] = image
            # Mask handling for pillarbox_blur
            if mask is not None:
                fg_mask = mask
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = fg_mask
            else:
                out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
                out_masks[:, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0
            return (out_image, out_masks)

        # Standard pad logic (edge/color)
        out_image = torch.zeros((B, padded_height, padded_width, C), dtype=image.dtype, device=image.device)
        for b in range(B):
                if pad_mode == "edge":
                    # Pad with edge color (mean)
                    top_edge = image[b, 0, :, :]
                    bottom_edge = image[b, H-1, :, :]
                    left_edge = image[b, :, 0, :]
                    right_edge = image[b, :, W-1, :]
                    out_image[b, :pad_top, :, :] = top_edge.mean(dim=0)
                    out_image[b, pad_top+H:, :, :] = bottom_edge.mean(dim=0)
                    out_image[b, :, :pad_left, :] = left_edge.mean(dim=0)
                    out_image[b, :, pad_left+W:, :] = right_edge.mean(dim=0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                elif pad_mode == "edge_pixel":
                    # Pad with exact edge pixel values
                    for y in range(pad_top):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, 0, :, :]
                    for y in range(pad_top+H, padded_height):
                        out_image[b, y, pad_left:pad_left+W, :] = image[b, H-1, :, :]
                    for x in range(pad_left):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, 0, :]
                    for x in range(pad_left+W, padded_width):
                        out_image[b, pad_top:pad_top+H, x, :] = image[b, :, W-1, :]
                    out_image[b, :pad_top, :pad_left, :] = image[b, 0, 0, :]
                    out_image[b, :pad_top, pad_left+W:, :] = image[b, 0, W-1, :]
                    out_image[b, pad_top+H:, :pad_left, :] = image[b, H-1, 0, :]
                    out_image[b, pad_top+H:, pad_left+W:, :] = image[b, H-1, W-1, :]
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]
                else:
                    # Pad with specified background color
                    out_image[b, :, :, :] = bg_color.unsqueeze(0).unsqueeze(0)
                    out_image[b, pad_top:pad_top+H, pad_left:pad_left+W, :] = image[b]

        if mask is not None:
            out_masks = torch.nn.functional.pad(
                mask, 
                (pad_left, pad_right, pad_top, pad_bottom),
                mode='replicate'
            )
        else:
            out_masks = torch.ones((B, padded_height, padded_width), dtype=image.dtype, device=image.device)
            for m in range(B):
                out_masks[m, pad_top:pad_top+H, pad_left:pad_left+W] = 0.0

        return (out_image, out_masks)