import torch
import comfy.samplers
import comfy.model_management
import comfy.cldm.control_types
import node_helpers
from .cache import remove_cache
from .utils import *
from comfy.patcher_extension import CallbacksMP
from comfy.model_patcher import ModelPatcher
from comfy.model_base import WAN21
from tqdm import tqdm
from comfy.ldm.modules.attention import wrap_attn
import comfy.model_sampling
import copy


class WanVideoIntegratedKSampler:

    def __init__(self):
        self.device = comfy.model_management.intermediate_device()
    @classmethod
    def INPUT_TYPES(s):
        sageattn_modes = ["disabled", "auto", "sageattn_qk_int8_pv_fp16_cuda", "sageattn_qk_int8_pv_fp16_triton", "sageattn_qk_int8_pv_fp8_cuda", "sageattn_qk_int8_pv_fp8_cuda++", "sageattn3", "sageattn3_per_block_mean"]
        return {
            "required": {
                "model_high_noise": ("MODEL",),
                "model_low_noise": ("MODEL",),
                "clip": ("CLIP", ),
                "vae": ("VAE", {}),
                "positive_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "正向提示词 positive_prompt"}),
                "negative_prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "placeholder": "负向提示词 negative_prompt", "default": "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 10}),
                "length": ("INT", {"default": 81, "min": 17, "max": 16384, "step": 4}),
                "width": ("INT", {"default": 720, "min": 8, "max": 16384, "step": 8}),
                "height": ("INT", {"default": 1280, "min": 8, "max": 16384, "step": 8}),
                "steps_high_noise": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "cfg_high_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "steps_low_noise": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "cfg_low_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
            },
            "optional": {
                "start_image": ("IMAGE",),
                # "middle_image": ("IMAGE",),
                "end_image": ("IMAGE",),
                "clip_vision": ("CLIP_VISION",),
                "latent": ("LATENT", ),
                "torch_enable_fp16_accumulation": ("BOOLEAN", {"default": True, "tooltip": "Enable torch.backends.cuda.matmul.allow_fp16_accumulation, requires pytorch 2.7.0 nightly."}),
                "sage_attention": (sageattn_modes, {"default": "auto", "tooltip": "Global patch comfy attention to use sageattn, once patched to revert back to normal you would need to run this node again with disabled option."}),
                "wan_blocks_to_swap": ("INT", {"default": 0, "min": 0, "max": 40, "step": 1, "tooltip": "Number of transformer blocks to swap, the 14B model has 40, while the 1.3B model has 30 blocks"}),
                "sd3_shift": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step":0.01}),
                "enable_clean_gpu_memory": ("BOOLEAN", {"default": False}),
                "enable_clean_cpu_memory_after_finish": ("BOOLEAN", {"default": False}),
                "enable_sound_notification": ("BOOLEAN", {"default": False}),
                # "middle_frame_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider",}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "LATENT")
    RETURN_NAMES = ("生成图像序列FrameImages", "最后一帧LastFrameImage", "（可选）Latent")
    FUNCTION = "sample"
    CATEGORY = "sampling"
    # 注意语言文件中不能用@符号
    DESCRIPTION = "🐳 WanVideo视频集成采样器 - K采样器，视频生成采样器，高低噪集成，支持文生视频/图生视频模式，支持首尾帧生成视频，批量生成、自动显存/内存管理、sage注意力、块交换、SD3采样、声音通知等全方位功能，不需要连那么多线啦~~~~/🐳 WanVideo Integrated KSampler - K-sampler for video generation with integrated high/low noise stages, supports text-to-video/image-to-video modes, supports generating videos with start/end frames, batch generation, automatic VRAM/RAM management, sage attention, block swapping, SD3 sampling, sound notifications and more comprehensive features, no need to connect so many wires~~~~ - Github: https://github.com/luguoli - 📧Email: luguoli﹫vip.qq.com"


    def sample(self, model_high_noise, model_low_noise, clip, vae, positive_prompt, negative_prompt, batch_size, length, width, height, steps_high_noise, cfg_high_noise, steps_low_noise, cfg_low_noise, noise_seed, sampler_name, scheduler, start_image=None, middle_image=None, end_image=None, clip_vision=None, latent=None, torch_enable_fp16_accumulation=False, sage_attention="disabled", wan_blocks_to_swap=0, sd3_shift=0, enable_clean_gpu_memory=False, enable_clean_cpu_memory_after_finish=False, enable_sound_notification=False, middle_frame_ratio=0.5, unique_id=0):


        # 检查合法性
        if width <= 0 or height <= 0:
            raise Exception("宽度和高度必须大于 0 / Width and height must be greater than 0")

        if (steps_high_noise is None or steps_high_noise <= 0) and (steps_low_noise is None or steps_low_noise <= 0):
            raise Exception(f"高噪步数和低噪步数不能同时为0 / high_noise_steps and low_noise_steps cannot both be 0")

        # 自动调整到合法倍数
        multiple = 8
        width = ((width + multiple - 1) // multiple) * multiple
        height = ((height + multiple - 1) // multiple) * multiple
        print(f"⚠️ 调整尺寸为 {width}x{height} / Adjusting size to {width}x{height}")

        model_cloned = False


        if torch_enable_fp16_accumulation:
            print("✨ 启用torch fp16累加")
            
            try:
                if not model_cloned:
                    model_high_noise = model_high_noise.clone()
                    model_low_noise = model_low_noise.clone()
                    model_cloned = True

                def patch_enable_fp16_accum(model):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = True
                def patch_disable_fp16_accum(model):
                    torch.backends.cuda.matmul.allow_fp16_accumulation = False
                
                if torch_enable_fp16_accumulation:
                    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                        model_high_noise.add_callback(CallbacksMP.ON_PRE_RUN, patch_enable_fp16_accum)
                        model_high_noise.add_callback(CallbacksMP.ON_CLEANUP, patch_disable_fp16_accum)

                        model_low_noise.add_callback(CallbacksMP.ON_PRE_RUN, patch_enable_fp16_accum)
                        model_low_noise.add_callback(CallbacksMP.ON_CLEANUP, patch_disable_fp16_accum)
                    else:
                        raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.1 or higher")
                else:
                    if hasattr(torch.backends.cuda.matmul, "allow_fp16_accumulation"):
                        model_high_noise.add_callback(CallbacksMP.ON_PRE_RUN, patch_disable_fp16_accum)

                        model_low_noise.add_callback(CallbacksMP.ON_PRE_RUN, patch_disable_fp16_accum)
                    else:
                        raise RuntimeError("Failed to set fp16 accumulation, this requires pytorch 2.7.1 or higher")
            
                print("✅ 启用torch fp16累加成功")
            except Exception as e:
                print(f"⚠️ 启用torch fp16累加失败，请检查是否安装了pytorch 2.7.1或更高版本或关闭此项设置")


        if sage_attention != "disabled":
            print("✨ 启用sage注意力")
            
            try:
                if not model_cloned:
                    model_high_noise = model_high_noise.clone()
                    model_low_noise = model_low_noise.clone()
                    model_cloned = True

                def get_sage_func(sage_attention, allow_compile=False):
                    print(f"Using sage attention mode: {sage_attention}")
                    from sageattention import sageattn
                    if sage_attention == "auto":
                        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                            return sageattn(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    elif sage_attention == "sageattn_qk_int8_pv_fp16_cuda":
                        from sageattention import sageattn_qk_int8_pv_fp16_cuda
                        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                            return sageattn_qk_int8_pv_fp16_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32", tensor_layout=tensor_layout)
                    elif sage_attention == "sageattn_qk_int8_pv_fp16_triton":
                        from sageattention import sageattn_qk_int8_pv_fp16_triton
                        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                            return sageattn_qk_int8_pv_fp16_triton(q, k, v, is_causal=is_causal, attn_mask=attn_mask, tensor_layout=tensor_layout)
                    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda":
                        from sageattention import sageattn_qk_int8_pv_fp8_cuda
                        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp32", tensor_layout=tensor_layout)
                    elif sage_attention == "sageattn_qk_int8_pv_fp8_cuda++":
                        from sageattention import sageattn_qk_int8_pv_fp8_cuda
                        def sage_func(q, k, v, is_causal=False, attn_mask=None, tensor_layout="NHD"):
                            return sageattn_qk_int8_pv_fp8_cuda(q, k, v, is_causal=is_causal, attn_mask=attn_mask, pv_accum_dtype="fp32+fp16", tensor_layout=tensor_layout)
                    elif "sageattn3" in sage_attention:
                        from sageattn3 import sageattn3_blackwell
                        if sage_attention == "sageattn3_per_block_mean":
                            def sage_func(q, k, v, is_causal=False, attn_mask=None, **kwargs):
                                return sageattn3_blackwell(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=is_causal, attn_mask=attn_mask, per_block_mean=True).transpose(1, 2)
                        else:
                            def sage_func(q, k, v, is_causal=False, attn_mask=None, **kwargs):
                                return sageattn3_blackwell(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=is_causal, attn_mask=attn_mask, per_block_mean=False).transpose(1, 2)

                    if not allow_compile:
                        sage_func = torch.compiler.disable()(sage_func)

                    @wrap_attn
                    def attention_sage(q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
                        if skip_reshape:
                            b, _, _, dim_head = q.shape
                            tensor_layout="HND"
                        else:
                            b, _, dim_head = q.shape
                            dim_head //= heads
                            q, k, v = map(
                                lambda t: t.view(b, -1, heads, dim_head),
                                (q, k, v),
                            )
                            tensor_layout="NHD"
                        if mask is not None:
                            # add a batch dimension if there isn't already one
                            if mask.ndim == 2:
                                mask = mask.unsqueeze(0)
                            # add a heads dimension if there isn't already one
                            if mask.ndim == 3:
                                mask = mask.unsqueeze(1)
                        out = sage_func(q, k, v, attn_mask=mask, is_causal=False, tensor_layout=tensor_layout)
                        if tensor_layout == "HND":
                            if not skip_output_reshape:
                                out = (
                                    out.transpose(1, 2).reshape(b, -1, heads * dim_head)
                                )
                        else:
                            if skip_output_reshape:
                                out = out.transpose(1, 2)
                            else:
                                out = out.reshape(b, -1, heads * dim_head)
                        return out
                    return attention_sage

                new_attention = get_sage_func(sage_attention, allow_compile=False)

                def attention_override_sage(func, *args, **kwargs):
                    return new_attention.__wrapped__(*args, **kwargs)

                # attention override
                model_high_noise.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage
                model_low_noise.model_options["transformer_options"]["optimized_attention_override"] = attention_override_sage

                print(f"✅ 应用sage注意力成功")
            except Exception as e:
                print(f"⚠️ 启用sage注意力失败，请禁用此项设置")


        
        if wan_blocks_to_swap > 0:
            print(f"✨ 应用块交换")

            try:
                if not model_cloned:
                    model_high_noise = model_high_noise.clone()
                    model_low_noise = model_low_noise.clone()
                    model_cloned = True

                offload_img_emb = False
                offload_txt_emb = False
                use_non_blocking = False
                def swap_blocks(model_instance: ModelPatcher, device_to, lowvram_model_memory, force_patch_weights, full_load):
                    base_model = model_instance.model
                    main_device=torch.device('cuda')

                    if not isinstance(base_model, WAN21):
                        raise TypeError("swap_blocks only supports WAN21 models")
                    
                    unet = base_model.diffusion_model
                    num_blocks = len(unet.blocks)
                    swap_count = min(wan_blocks_to_swap, num_blocks)

                    if offload_txt_emb:
                        unet.text_embedding.to(model_instance.offload_device, non_blocking=use_non_blocking)
                    if offload_img_emb:
                        unet.img_emb.to(model_instance.offload_device, non_blocking=use_non_blocking)

                    with tqdm(total=num_blocks, desc="Initializing block swap", leave=True) as pbar:
                        for idx, block in enumerate(unet.blocks):
                            if idx < swap_count:
                                # 低 idx 的 block 放到 offload_device
                                block.to(model_instance.offload_device)
                            else:
                                # 其他 block 放回 GPU
                                block.to(main_device)
                            pbar.update(1)

                    comfy.model_management.soft_empty_cache()
                    gc.collect()
                
                model_high_noise.add_callback(CallbacksMP.ON_LOAD,swap_blocks)
                model_low_noise.add_callback(CallbacksMP.ON_LOAD,swap_blocks)

                print("✅ 块交换参数已应用成功")
            except Exception as e:
                print(f"⚠️ 块交换失败，请关闭此项设置")


        if sd3_shift > 0:
            print(f"✨ 应用采样算法（SD3）")

            try:
                if not model_cloned:
                    model_high_noise = model_high_noise.clone()
                    model_low_noise = model_low_noise.clone()
                    model_cloned = True

                sampling_base = comfy.model_sampling.ModelSamplingDiscreteFlow
                sampling_type = comfy.model_sampling.CONST
                class ModelSamplingAdvanced(sampling_base, sampling_type):
                    pass

                model_sampling = ModelSamplingAdvanced(model_high_noise.model.model_config)
                model_sampling.set_parameters(shift=sd3_shift, multiplier=1000)
                model_high_noise.add_object_patch("model_sampling", model_sampling)

                model_sampling = ModelSamplingAdvanced(model_low_noise.model.model_config)
                model_sampling.set_parameters(shift=sd3_shift, multiplier=1000)
                model_low_noise.add_object_patch("model_sampling", model_sampling)

                print("✅ 采样算法（SD3）已应用成功")
            except Exception as e:
                print(f"⚠️ 采样算法（SD3）应用失败，请关闭此项设置")


        models = (model_high_noise, model_low_noise)
        steps = (steps_high_noise, steps_low_noise)
        cfgs = (cfg_high_noise, cfg_low_noise)
        disable_noises = (False, True)
        force_full_denoises = (False, True)

        
        with tqdm(total=4, desc="CLIP Encoding Progress") as pbar:
            # 加载正向条件
            positive_tokens = clip.tokenize(positive_prompt)
            pbar.update(1)
            positive = clip.encode_from_tokens_scheduled(positive_tokens)
            pbar.update(1)
            # 加载负向条件
            negative_tokens = clip.tokenize(negative_prompt)
            pbar.update(1)
            negative = clip.encode_from_tokens_scheduled(negative_tokens)
            pbar.update(1)


        positive_high_noise = copy.deepcopy(positive)
        negative_high_noise = copy.deepcopy(negative)

        positive_low_noise = copy.deepcopy(positive)
        negative_low_noise = copy.deepcopy(negative)


        if latent is None:

            if all(x is None for x in [start_image, middle_image, end_image]):
                latent_image = torch.zeros([batch_size, 16, ((length - 1) // 4) + 1, height // 8, width // 8], device=self.device)
                latent = {"samples":latent_image}
                print("文生视频模式")
            else:

                # 尾帧检查
                if end_image is not None and start_image is None:
                    raise Exception("使用尾帧时必须传入首帧 / When using end_image, start_image must also be provided.")
                
                # 中间帧检查
                if middle_image is not None and (start_image is None or end_image is None):
                    raise Exception("使用中间帧时必须传入首尾帧 / When using middle_image, both start_image and end_image must be provided.")
    

                spacial_scale = vae.spacial_compression_encode()
                latent_channels = vae.latent_channels
                latent_t = ((length - 1) // 4) + 1
                latent_image = torch.zeros([batch_size, latent_channels, latent_t, height // spacial_scale, width // spacial_scale], device=self.device)

                if start_image is not None:
                    start_image, resize_width, resize_height, resize_mask = image_resize(start_image, width, height, "crop", "lanczos", 2, "0, 0, 0", "center", unique_id=unique_id, device="cpu", mask=None, per_batch=64)
                    start_image = comfy.utils.common_upscale(start_image[:length].movedim(-1, 1), resize_width, resize_height, "bilinear", "center").movedim(1, -1)
                if middle_image is not None:
                    middle_image, resize_width, resize_height, resize_mask = image_resize(middle_image, width, height, "crop", "lanczos", 2, "0, 0, 0", "center", unique_id=unique_id, device="cpu", mask=None, per_batch=64)
                    middle_image = comfy.utils.common_upscale(middle_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)
                if end_image is not None:
                    end_image, resize_width, resize_height, resize_mask = image_resize(end_image, width, height, "crop", "lanczos", 2, "0, 0, 0", "center", unique_id=unique_id, device="cpu", mask=None, per_batch=64)
                    end_image = comfy.utils.common_upscale(end_image[-length:].movedim(-1, 1), width, height, "bilinear", "center").movedim(1, -1)

                image = torch.ones((length, height, width, 3), device=self.device) * 0.5
                mask = torch.ones((1, 1, latent_image.shape[2] * 4, latent_image.shape[-2], latent_image.shape[-1]), device=self.device)

                image_high_noise = image.clone()
                image_low_noise = image.clone()

                mask_high_noise = mask.clone()
                mask_low_noise = mask.clone()


                middle_idx = calculate_middle_frame_idx(middle_frame_ratio, length)

                if start_image is not None:
                    image_high_noise[:start_image.shape[0]] = start_image
                    image_low_noise[:start_image.shape[0]] = start_image
                    mask_high_noise[:, :, :start_image.shape[0] + 3] = 0.0
                    mask_low_noise[:, :, :start_image.shape[0] + 3] = 0.0

                if middle_image is not None:
                    # TODO:中间帧闪烁问题暂未解决
                    n_middle = middle_image.shape[0]
                    image_high_noise[middle_idx:middle_idx + n_middle] = middle_image
                    mask_high_noise[:, :, middle_idx:middle_idx + n_middle + 3] = 0.05
                    
                if end_image is not None:
                    image_high_noise[-end_image.shape[0]:] = end_image
                    image_low_noise[-end_image.shape[0]:] = end_image
                    mask_high_noise[:, :, -end_image.shape[0]:] = 0.0
                    mask_low_noise[:, :, -end_image.shape[0]:] = 0.1

                concat_latent_image_high_noise = vae.encode(image_high_noise[:, :, :, :3])
                concat_latent_image_low_noise = vae.encode(image_low_noise[:, :, :, :3])

                mask_high_noise = mask_high_noise.view(1, mask_high_noise.shape[2] // 4, 4, mask_high_noise.shape[3], mask_high_noise.shape[4]).transpose(1, 2)
                mask_low_noise = mask_low_noise.view(1, mask_low_noise.shape[2] // 4, 4, mask_low_noise.shape[3], mask_low_noise.shape[4]).transpose(1, 2)

                positive_high_noise = node_helpers.conditioning_set_values(positive_high_noise, {"concat_latent_image": concat_latent_image_high_noise, "concat_mask": mask_high_noise})
                negative_high_noise = node_helpers.conditioning_set_values(negative_high_noise, {"concat_latent_image": concat_latent_image_high_noise, "concat_mask": mask_high_noise})

                positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, {"concat_latent_image": concat_latent_image_low_noise, "concat_mask": mask_low_noise})
                negative_low_noise = node_helpers.conditioning_set_values(negative_low_noise, {"concat_latent_image": concat_latent_image_low_noise, "concat_mask": mask_low_noise})

                

                clip_vision_list = []

                if clip_vision is not None:
                    if start_image is not None:
                        #clip_vision编码
                        clip_vision_encode_start_image = clip_vision.encode_image(start_image, crop=False)
                        clip_vision_list.append(clip_vision_encode_start_image)

                    if middle_image is not None:
                        #clip_vision编码
                        clip_vision_encode_middle_image = clip_vision.encode_image(middle_image, crop=False)
                        clip_vision_list.append(clip_vision_encode_middle_image)

                    if end_image is not None:
                        #clip_vision编码
                        clip_vision_encode_end_image = clip_vision.encode_image(end_image, crop=False)
                        clip_vision_list.append(clip_vision_encode_end_image)

                clip_vision_output = None
                if clip_vision_list:  # 列表非空
                    states = torch.cat([c.penultimate_hidden_states for c in clip_vision_list], dim=-2)
                    clip_vision_output = comfy.clip_vision.Output()
                    clip_vision_output.penultimate_hidden_states = states

                # 应用到正/负条件
                if clip_vision_output is not None:
                    positive_high_noise = node_helpers.conditioning_set_values(positive_high_noise, {"clip_vision_output": clip_vision_output})
                    negative_high_noise = node_helpers.conditioning_set_values(negative_high_noise, {"clip_vision_output": clip_vision_output})

                    positive_low_noise = node_helpers.conditioning_set_values(positive_low_noise, {"clip_vision_output": clip_vision_output})
                    negative_low_noise = node_helpers.conditioning_set_values(negative_low_noise, {"clip_vision_output": clip_vision_output})

                latent = {"samples":latent_image}

        positive = (positive_high_noise, positive_low_noise)
        negative = (negative_high_noise, negative_low_noise)


        print("🚀 开始采样过程/Starting Sampling...")

        if enable_clean_gpu_memory:
            print("🗑️ 预清理显存占用/Pre-cleaning GPU memory...")
            try:
                cleanGPUUsedForce()
                remove_cache('*')
            except ImportError:
                print("🔕 显存清理失败/Pre GPU memory cleaning failed")
            print("✅ 预显存清理完成/Pre GPU memory cleaning completed")


        latent_output = common_ksampler(models, noise_seed, steps, cfgs, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noises=disable_noises, force_full_denoises=force_full_denoises)

        print("🖼️ 正在解码潜空间/Decoding latent space...")
        output_images = vae.decode(latent_output["samples"])
        if len(output_images.shape) == 5: #Combine batches
            output_images = output_images.reshape(-1, output_images.shape[-3], output_images.shape[-2], output_images.shape[-1])
        print("✅ 解码完成/Decoding completed")

        # 抽取最后一帧，取帧示例：[1, 2, 3, 4, -1]
        index_list = [-1]
        # Convert list of indices to a PyTorch tensor
        indices_tensor = torch.tensor(index_list, dtype=torch.long)
        # Select the images at the specified indices
        last_image = output_images[indices_tensor]


        if enable_clean_gpu_memory:
            print("🗑️ 后清理显存占用/Post-cleaning GPU memory...")
            try:
                cleanGPUUsedForce()
                remove_cache('*')
            except ImportError:
                print("🔕 显存清理失败/Pre GPU memory cleaning failed")
            print("✅ 后显存清理完成/Post GPU memory cleaning completed")

        if enable_clean_cpu_memory_after_finish:
            print("🗑️ 完成后清理CPU内存/Post-cleaning CPU memory after finish...")
            try:
                clean_ram(clean_file_cache=True, clean_processes=True, clean_dlls=True, retry_times=3)
            except Exception as e:
                print(f"🔕 RAM清理失败/RAM cleanup failed: {str(e)}")
            else:
                print("✅ [Clean CPU Memory After Finish] RAM清理完成 / RAM cleanup completed")

        if enable_sound_notification:
            try:
                import winsound
                import time
                # 播放快速紧凑的旋律：A4, C5, E5, G5，较短间隔使旋律连贯
                frequencies = [440, 523, 659, 784]
                for freq in frequencies:
                    winsound.Beep(freq, 150)
                    time.sleep(0.005)  # 更短间隔加快节奏
                print("🎵 [Sound Notification] Completion melody played")
            except ImportError:
                print("🔕 [Sound Notification] Sound notification not supported on this system")
            except Exception as e:
                print(f"🔕 [Sound Notification] Audio playback failed: {str(e)}")

        return (output_images, last_image, latent_output)



class WanVideoIntegratedKSamplerSimple:

    def __init__(self):
        pass
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_high_noise": ("MODEL",),
                "model_low_noise": ("MODEL",),
                "steps_high_noise": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "cfg_high_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "steps_low_noise": ("INT", {"default": 4, "min": 0, "max": 10000}),
                "cfg_low_noise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "noise_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent": ("LATENT", ),
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("Latent",)
    FUNCTION = "sample"
    CATEGORY = "sampling"
    # 注意语言文件中不能用@符号
    DESCRIPTION = "🐳 WanVideo视频集成采样器(简单)——Github:@luguoli"


    def sample(self, model_high_noise, model_low_noise, steps_high_noise, cfg_high_noise, steps_low_noise, cfg_low_noise, noise_seed, sampler_name, scheduler, positive, negative, latent):
        models = (model_high_noise, model_low_noise)
        steps = (steps_high_noise, steps_low_noise)
        cfgs = (cfg_high_noise, cfg_low_noise)
        positive = (positive, positive)
        negative = (negative, negative)
        disable_noises = (False, True)
        force_full_denoises = (False, True)
        latent_output = common_ksampler(models, noise_seed, steps, cfgs, sampler_name, scheduler, positive, negative, latent, denoise=1.0, disable_noises=disable_noises, force_full_denoises=force_full_denoises)

        return (latent_output,)

NODE_CLASS_MAPPINGS = {
    "WanVideoIntegratedKSampler": WanVideoIntegratedKSampler,
    "WanVideoIntegratedKSamplerSimple": WanVideoIntegratedKSamplerSimple,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "WanVideoIntegratedKSampler": "🐳 WanVideo视频集成采样器——Github:@luguoli",
    "WanVideoIntegratedKSamplerSimple": "🐳 WanVideo视频集成采样器(简单)——Github:@luguoli",
}
