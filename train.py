import argparse
import inspect
import logging
import math
import os
from datetime import datetime
from typing import Dict, Literal, Optional, Tuple

import diffusers
import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torchvision.transforms.functional as TF
import transformers
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs as DDPK
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler
from diffusers.optimization import get_scheduler
from diffusers.utils import check_min_version, is_wandb_available
from diffusers.utils.import_utils import is_xformers_available
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import InterpolationMode
from tqdm.auto import tqdm
from transformers import (
    CLIPImageProcessor,
    CLIPTextModel,
    CLIPTokenizer,
    CLIPVisionModelWithProjection,
)

from parts2whole.data import RefTrgDataset
from parts2whole.models import (
    PoseGuider,
    ReferenceAttentionControl,
    ReferenceUNet2DConditionModel,
    UNet2DConditionModel,
)
from parts2whole.models.custom_attention_processor import (
    DecoupledCrossAttnProcessor2_0,
    set_unet_2d_condition_attn_processor,
)
from parts2whole.pipelines.pipeline_refs2image import Refs2ImagePipeline

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.25.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(
    output_dir: str,
    train_data: Dict,
    validation_data: Dict,
    pretrained_model_path: str = "runwayml/stable-diffusion-v1-5",
    pretrained_image_encoder_path: str = "lambdalabs/sd-image-variations-diffusers",
    noise_scheduler_kwargs: Dict = {},
    reference_attn_control_kwargs: Dict = {},
    pose_guider_kwargs: Dict = {},
    use_decoupled_cross_attn: bool = False,
    trainable_modules: Optional[list] = None,
    train_batch_size: int = 1,
    num_workers: int = 8,
    max_train_steps: int = 500,
    validation_steps: int = 100,
    sanity_check: bool = False,
    learning_rate: float = 3e-5,
    scale_lr: bool = False,
    lr_scheduler: str = "constant",
    lr_warmup_steps: int = 0,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_weight_decay: float = 1e-2,
    adam_epsilon: float = 1e-08,
    max_grad_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    gradient_checkpointing: bool = True,
    checkpointing_steps: int = 500,
    resume_from_checkpoint: Optional[str] = None,
    load_pretrain_ckpt: Optional[str] = None,
    load_additional_modules_from_pipeline: Optional[bool] = False,
    mixed_precision: Optional[str] = "fp16",
    use_8bit_adam: bool = False,
    ddp_kwargs: Optional[Dict] = None,
    enable_xformers_memory_efficient_attention: bool = True,
    seed: Optional[int] = None,
    classifier_free_guidance: Optional[float] = 0.05,
    reference_dropout: Optional[float] = 0.0,
    report_to: Optional[str] = None,
    tracker_project_name: str = "parts2whole",
    extra: Optional[Dict] = {},
):
    *_, config = inspect.getargvalues(inspect.currentframe())

    kwargs = DDPK(**ddp_kwargs or {})
    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        kwargs_handlers=[kwargs],
    )

    if report_to == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it for logging during training.")
        import wandb

    # Handle the output folder creation
    save_dir = os.path.join(output_dir, "samples")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    if accelerator.is_main_process:
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"))

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        filename=os.path.join(output_dir, "log.log"),
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if seed is not None:
        set_seed(seed)

    # Load scheduler, feature_extractor and models.
    noise_scheduler = DDIMScheduler.from_pretrained(
        pretrained_model_path, subfolder="scheduler", **noise_scheduler_kwargs
    )
    vae = AutoencoderKL.from_pretrained(pretrained_model_path, subfolder="vae")
    tokenizer = CLIPTokenizer.from_pretrained(pretrained_model_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(pretrained_model_path, subfolder="text_encoder")
    feature_extractor = CLIPImageProcessor.from_pretrained(
        pretrained_image_encoder_path, subfolder="feature_extractor"
    )
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(
        pretrained_image_encoder_path, subfolder="image_encoder"
    )
    unet = UNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")

    if use_decoupled_cross_attn:
        # Load decoupled cross attention weights with pretrained text cross attention layer
        def set_cross_attn_proc_func(
            name: str,
            hidden_size: int,
            cross_attention_dim: Optional[int],
            ori_attn_proc: object,
        ):
            assert cross_attention_dim is not None
            return DecoupledCrossAttnProcessor2_0(
                hidden_size=hidden_size,
                cross_attention_dim=cross_attention_dim,
                max_image_length=6,  # FIXME: hard-code
            )

        set_unet_2d_condition_attn_processor(unet, set_cross_attn_proc_func=set_cross_attn_proc_func)

        # copy decoupled attention weights from original unet
        state_dict = unet.state_dict()
        for key in state_dict.keys():
            if "_dc" in key:
                state_dict[key] = state_dict[key.replace("_dc", "").replace(".processor", "")].clone()
        unet.load_state_dict(state_dict)

    # Load additional pose_guider and reference_unet from scratch or from pretrained pipeline
    if not load_additional_modules_from_pipeline:
        pose_guider = PoseGuider(**OmegaConf.to_container(pose_guider_kwargs))
        ref_unet = ReferenceUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="unet")
    else:
        pose_guider = PoseGuider.from_pretrained(pretrained_model_path, subfolder="pose_guider")
        ref_unet = ReferenceUNet2DConditionModel.from_pretrained(pretrained_model_path, subfolder="reference_unet")
        accelerator.print("Loaded pose_guider and reference_unet from pipeline")

    reference_control_writer = ReferenceAttentionControl(
        ref_unet,
        do_classifier_free_guidance=False,
        mode="write",
        batch_size=train_batch_size,
        **reference_attn_control_kwargs,
    )
    reference_control_reader = ReferenceAttentionControl(
        unet,
        do_classifier_free_guidance=False,
        mode="read",
        batch_size=train_batch_size,
        **reference_attn_control_kwargs,
    )

    # Set trainable parameters
    # Freeze vae and image_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    image_encoder.requires_grad_(False)

    # Set unet trainable parameters
    if trainable_modules is not None:
        unet.requires_grad_(False)
        for name, module in unet.named_modules():
            for trainable_module in trainable_modules:
                if trainable_module in name:
                    module.requires_grad_(True)
    else:
        unet.requires_grad_(True)

    pose_guider.requires_grad_(True)
    ref_unet.requires_grad_(True)

    _trainable_params = lambda m: [p for p in m.parameters() if p.requires_grad]
    trainable_params = _trainable_params(unet) + _trainable_params(pose_guider) + _trainable_params(ref_unet)
    trainable_mb = sum(p.numel() for p in trainable_params) / 1e6
    logger.info(f"Trainable parameters: {trainable_mb:.2f}M")

    # Enable memory efficient attention for xformers
    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Enable gradient checkpointing
    if gradient_checkpointing:
        unet.enable_gradient_checkpointing()
        ref_unet.enable_gradient_checkpointing()

    if scale_lr:
        learning_rate = learning_rate * gradient_accumulation_steps * train_batch_size * accelerator.num_processes

    # Initialize the optimizer
    if use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "Please install bitsandbytes to use 8-bit Adam. You can do so by running `pip install bitsandbytes`"
            )

        optimizer_cls = bnb.optim.AdamW8bit
    else:
        optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        trainable_params,
        lr=learning_rate,
        betas=(adam_beta1, adam_beta2),
        weight_decay=adam_weight_decay,
        eps=adam_epsilon,
    )

    # Get the training dataset
    train_dataset = RefTrgDataset(**train_data)

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    validation_scheduler = DDIMScheduler.from_pretrained(pretrained_model_path, subfolder="scheduler")
    validation_pipeline = Refs2ImagePipeline(
        vae=vae,
        image_encoder=image_encoder,
        reference_unet=ref_unet,
        unet=unet,
        pose_guider=pose_guider,
        scheduler=validation_scheduler,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
    )
    validation_pipeline.enable_vae_slicing()

    # Scheduler
    lr_scheduler = get_scheduler(
        lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=lr_warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    # Prepare everything with our `accelerator`.
    (
        unet,
        pose_guider,
        ref_unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    ) = accelerator.prepare(
        unet,
        pose_guider,
        ref_unet,
        optimizer,
        train_dataloader,
        lr_scheduler,
    )

    # For mixed precision training we cast the image_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move image_encoder and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / gradient_accumulation_steps)
    # Afterwards we recalculate our number of training epochs
    num_train_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers(tracker_project_name)

    # Train!
    total_batch_size = train_batch_size * accelerator.num_processes * gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Potentially load in the weights and states from a previous save
    if resume_from_checkpoint:
        if resume_from_checkpoint != "latest":
            path = os.path.basename(resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            accelerator.print(f"Checkpoint '{resume_from_checkpoint}' does not exist. Starting a new training run.")
            resume_from_checkpoint = None
            global_step = 0
        else:
            accelerator.print(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(output_dir, path))
            global_step = int(path.split("-")[1])

            first_epoch = global_step // num_update_steps_per_epoch
            resume_step = global_step % num_update_steps_per_epoch
    else:
        global_step = 0

    if load_pretrain_ckpt:
        path = os.path.basename(load_pretrain_ckpt)
        accelerator.print(f"loading from checkpoint {path}")
        accelerator.load_state(os.path.join(output_dir, path))

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(
        range(global_step, max_train_steps),
        disable=not accelerator.is_local_main_process,
    )
    progress_bar.set_description("Steps")

    def clip_encode_text(text):
        text_token = tokenizer(
            text,
            max_length=tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).input_ids.to(accelerator.device)
        encoder_hidden_states = text_encoder(text_token)[0]

        return encoder_hidden_states

    def train_step(batch):
        # Prepare batch data for training
        pixel_values = batch["pixel_values"]
        appearance_dict = batch["appearance"]
        structure_dict = batch["structure"]
        mask_dict = batch["mask"]

        with accelerator.accumulate(unet, pose_guider, ref_unet):
            # Convert training objective images to latent space
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = latents * 0.18215

            # Sample noise that we'll add to the latents
            noise = torch.randn_like(latents)
            bsz = latents.shape[0]

            # Sample a random timestep for each video
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz,),
                device=latents.device,
            )
            timesteps = timesteps.long()

            # Add noise to the latents according to the noise magnitude at each timestep
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Convert pose images to latent space
            # TODO: may expand to multiple formats like openpose, densepose, etc.
            pose_type = extra.get("pose_type", list(structure_dict.keys())[0])
            pixel_values_pose = structure_dict[pose_type]

            if reference_dropout and torch.rand(1).item() < reference_dropout:
                pose_fea = None
            else:
                pose_fea = pose_guider(pixel_values_pose)

            # Update the reference attention features to the bank
            if reference_dropout and torch.rand(1).item() < reference_dropout:
                reference_control_writer.clear()
            else:
                # Process reference text for conditioning ref unet -> [bs, 77, 768]
                ref_timesteps = torch.zeros_like(timesteps)  # Use t=0 in ref_unet
                for key, value in appearance_dict.items():
                    if reference_dropout and torch.rand(1).item() < reference_dropout:
                        continue

                    vmask = mask_dict.get(
                        key,
                        torch.ones((bsz, 1) + value.shape[-2:], device=latents.device),
                    )

                    # Convert reference image to latent space
                    latents_ref_img = vae.encode(value).latent_dist.sample()
                    latents_ref_img = latents_ref_img * vae.config.scaling_factor

                    # Get text encode feature
                    text_encoder_hidden_states = clip_encode_text(key).repeat(bsz, 1, 1)

                    # Pass the reference image through the reference unet
                    ref_unet(
                        latents_ref_img,
                        ref_timesteps,
                        text_encoder_hidden_states,
                        cross_attention_kwargs={"hidden_states_mask": vmask},
                    )

                reference_control_reader.update(reference_control_writer)

            # Prepare clip conditioning embedding
            encoder_hidden_states = []

            # Process caption for clip conditioning -> [bs, 77, 768]
            text_encoder_hidden_states_dunet = clip_encode_text(batch["caption"])
            encoder_hidden_states.append(text_encoder_hidden_states_dunet)

            # Process reference appearance images for clip conditioning -> [bs,1,768]
            image_encoder_hidden_states_dunet = []
            for key, value in appearance_dict.items():
                value = (value / 2 + 0.5).to(dtype=torch.float32)  # Normalize to [0, 1]
                # Get the image embedding for conditioning -> [bs,1,768]
                clip_ref_image = feature_extractor(value, do_rescale=False, return_tensors="pt").pixel_values.to(
                    device=accelerator.device, dtype=weight_dtype
                )
                _encoder_hidden_states = image_encoder(clip_ref_image).image_embeds.unsqueeze(1)
                image_encoder_hidden_states_dunet.append(_encoder_hidden_states)

            # Padding image_encoder_hidden_states_dunet to length 6 # FIXME: hard-code 6
            if use_decoupled_cross_attn and len(image_encoder_hidden_states_dunet) < 6:
                image_encoder_hidden_states_dunet += [
                    torch.zeros_like(image_encoder_hidden_states_dunet[0])
                    for _ in range(6 - len(image_encoder_hidden_states_dunet))
                ]

            # Concatenate image_encoder_hidden_states_dunet
            encoder_hidden_states += image_encoder_hidden_states_dunet

            # Get the target for loss depending on the prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(latents, noise, timesteps)
            else:
                raise ValueError(f"Unknown prediction type {noise_scheduler.prediction_type}")

            # Predict the noise residual and compute loss
            encoder_hidden_states = torch.cat(encoder_hidden_states, dim=1)

            if classifier_free_guidance is not None:
                drop_mask = torch.rand(bsz, device=latents.device) < classifier_free_guidance
                encoder_hidden_states[drop_mask] = 0.0

            model_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                pose_cond_fea=pose_fea,
                return_dict=False,
            )[0]

            # Compute loss
            loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

            return loss, latents

    def load_items_from_dict(data_dict):
        images = {}
        image_list = []

        def recursive_load(path, key, container, image_list):
            if isinstance(path, dict):
                container[key] = {}
                for k, v in path.items():
                    recursive_load(v, k, container[key], image_list)
            else:
                if path.endswith("npy"):
                    arr = np.load(path)
                    container[key] = arr
                    # image_list.append(arr)
                else:
                    # Load and store the image
                    image = Image.open(path)
                    container[key] = image
                    image_list.append(image)

        recursive_load(data_dict, "root", images, image_list)
        return images["root"], image_list

    def prepare_batched_data(batch, device, dtype):
        """Recursively move tensors in a nested dict to the device and cast to weight_dtype."""
        for key, value in batch.items():
            if isinstance(value, dict):
                batch[key] = prepare_batched_data(value, device, dtype)
            elif isinstance(value, torch.Tensor):
                batch[key] = value.to(device).to(dtype)
        return batch

    def validation_step(device, keyword="validation"):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

        # collate validation data for each process
        num_samples_per_process = math.ceil(len(validation_data.inputs) / accelerator.num_processes)
        start_idx = accelerator.process_index * num_samples_per_process
        end_idx = start_idx + num_samples_per_process

        log_images = []  # for wandb
        for i, sample in enumerate(validation_data.inputs):
            if i < start_idx or i >= end_idx:
                continue

            sample_dict = OmegaConf.to_container(sample)
            prompt = sample_dict.pop("prompt")
            ref_dict, ref_list = load_items_from_dict(sample_dict)
            height, width = validation_data.height, validation_data.width
            images = validation_pipeline(
                prompt=prompt,
                generator=generator,
                **ref_dict,
                **validation_data,
            ).images  # List[PIL.Image]

            # concat ref image, pose images and generated images
            images = [image.resize((width, height)).convert("RGB") for image in ref_list] + images
            widths, heights = zip(*(i.size for i in images))
            total_width = sum(widths)
            max_height = max(heights)
            total_image = Image.new("RGB", (total_width, max_height))
            x_offset = 0
            for im in images:
                total_image.paste(im, (x_offset, 0))
                x_offset += im.size[0]

            save_path = os.path.join(save_dir, f"sample-{global_step}-{keyword}-{i}.png")
            total_image.save(save_path)

            # append to log_images for wandb
            log_images.append((total_image, f"sample_{i}"))

        if accelerator.is_main_process:  # FIXME: only main process can save images
            # save to wandb
            for tracker in accelerator.trackers:
                if tracker.name == "wandb":
                    tracker.log({keyword: [wandb.Image(image, caption=caption) for image, caption in log_images]})

    # sanity check
    if sanity_check:
        validation_step(accelerator.device)

    for epoch in range(first_epoch, num_train_epochs):
        unet.train()
        pose_guider.train()
        ref_unet.train()
        train_loss = 0.0
        for step, batch in enumerate(train_dataloader):
            # Move the batch to the correct device
            prepare_batched_data(batch, accelerator.device, weight_dtype)

            # Skip steps until we reach the resumed step
            if resume_from_checkpoint and epoch == first_epoch and step < resume_step:
                if step % gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue

            loss, _ = train_step(batch)

            # Gather the losses across all processes for logging (if we use distributed training).
            avg_loss = accelerator.gather(loss.repeat(train_batch_size)).mean()
            train_loss += avg_loss.item() / gradient_accumulation_steps

            # Backpropagate
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_params, max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            optimizer.zero_grad()
            reference_control_reader.clear()
            reference_control_writer.clear()
            logs = {
                "step_loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
            }
            progress_bar.set_postfix(**logs)

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1
                accelerator.log({"train_loss": train_loss}, step=global_step)
                train_loss = 0.0

                if global_step % checkpointing_steps == 0:
                    if accelerator.is_main_process:
                        # Save accelerator state
                        save_path = os.path.join(output_dir, f"checkpoint-{global_step}")
                        accelerator.save_state(save_path)
                        logger.info(f"Saved state to {save_path}")

                if global_step % validation_steps == 0:
                    validation_step(accelerator.device)

            if global_step >= max_train_steps:
                break

    # Create the pipeline using the trained modules and save it.
    accelerator.wait_for_everyone()

    # Save pipeline
    if accelerator.is_main_process:
        unet = accelerator.unwrap_model(unet)
        ref_unet = accelerator.unwrap_model(ref_unet)
        pose_guider = accelerator.unwrap_model(pose_guider)
        pipeline = Refs2ImagePipeline(
            vae=vae,
            feature_extractor=feature_extractor,
            image_encoder=image_encoder,
            reference_unet=ref_unet,
            unet=unet,
            pose_guider=pose_guider,
            scheduler=noise_scheduler,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
        )
        pipeline.save_pretrained(output_dir)

        # Save decoupled attn weights if using decoupled cross attn
        if use_decoupled_cross_attn:
            from safetensors.torch import save_file

            dc_state_dict = {}
            state_dict = unet.state_dict()
            for key in state_dict.keys():
                if "_dc" in key:
                    dc_state_dict[key] = state_dict[key]
            save_file(dc_state_dict, os.path.join(output_dir, "decoupled_attn.pth"))

    logger.info("Running inference for collecting generated images...")
    validation_step(accelerator.device)

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/train.yaml")
    args = parser.parse_args()

    main(**OmegaConf.load(args.config))
