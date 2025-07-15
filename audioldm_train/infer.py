import os
import sys
import argparse
import yaml
import torch
import numpy  # Import numpy for allowlisting
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

# Add project root to system path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from audioldm_train.utilities.data.dataset import AudioDataset
from audioldm_train.utilities.tools import get_restore_step, build_dataset_json_from_list
from audioldm_train.utilities.model_util import instantiate_from_config

def infer(dataset_json, configs, config_yaml_path, exp_group_name, exp_name):
    """
    Perform inference using the specified configuration and dataset.

    Args:
        dataset_json (str): Path to the dataset JSON file.
        configs (dict): Configuration dictionary loaded from YAML.
        config_yaml_path (str): Path to the configuration YAML file.
        exp_group_name (str): Name of the experiment group.
        exp_name (str): Name of the experiment.
    """
    # Set random seed for reproducibility
    if "seed" in configs:
        seed_everything(configs["seed"])
    else:
        print("Setting random seed to 0")
        seed_everything(0)

    # Set precision for matrix multiplication if specified
    if "precision" in configs:
        torch.set_float32_matmul_precision(configs["precision"])

    log_path = configs["log_directory"]

    # Initialize dataloader add-ons
    dataloader_add_ons = configs["data"].get("dataloader_add_ons", [])

    # Initialize validation dataset and dataloader
    val_dataset = AudioDataset(
        configs, split="test", add_ons=dataloader_add_ons, dataset_json=dataset_json
    )
    val_loader = DataLoader(val_dataset, batch_size=1)

    # Determine checkpoint path
    checkpoint_path = os.path.join(log_path, exp_group_name, exp_name, "checkpoints")
    os.makedirs(checkpoint_path, exist_ok=True)

    # Copy configuration file to experiment directory
    wandb_path = os.path.join(log_path, exp_group_name, exp_name)
    os.makedirs(wandb_path, exist_ok=True)
    os.system(f"cp {config_yaml_path} {wandb_path}")

    # Load checkpoint
    resume_from_checkpoint = configs.get("reload_from_ckpt")
    if os.listdir(checkpoint_path):
        print(f"Loading checkpoint from path: {checkpoint_path}")
        restore_step, n_step = get_restore_step(checkpoint_path)
        resume_from_checkpoint = os.path.join(checkpoint_path, restore_step)
        print(f"Resuming from checkpoint: {resume_from_checkpoint}")
    elif resume_from_checkpoint:
        print(f"Reloading checkpoint specified in config: {resume_from_checkpoint}")
    else:
        raise ValueError("No checkpoint found and no reload_from_ckpt specified in config.")

    # Initialize model
    latent_diffusion = instantiate_from_config(configs["model"])
    latent_diffusion.set_log_dir(log_path, exp_group_name, exp_name)

    # Retrieve evaluation parameters
    eval_params = configs["model"]["params"]["evaluation_params"]
    guidance_scale = eval_params["unconditional_guidance_scale"]
    ddim_sampling_steps = eval_params["ddim_sampling_steps"]
    n_candidates_per_samples = eval_params["n_candidates_per_samples"]

    # Allowlist numpy globals for safe deserialization
    torch.serialization.add_safe_globals([numpy.core.multiarray.scalar, numpy.dtype])

    try:
        # Load checkpoint with weights_only=True for security
        checkpoint = torch.load(resume_from_checkpoint, weights_only=True, map_location="cpu")
        latent_diffusion.load_state_dict(checkpoint["state_dict"])
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        raise

    # Move model to GPU and set to evaluation mode
    latent_diffusion.eval()
    latent_diffusion = latent_diffusion.cuda()

    # Generate samples
    latent_diffusion.generate_sample(
        val_loader,
        unconditional_guidance_scale=guidance_scale,
        ddim_steps=ddim_sampling_steps,
        n_gen=n_candidates_per_samples,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference script for AudioLDM model.")
    parser.add_argument(
        "--config_yaml",
        type=str,
        required=True,
        help="Path to the configuration YAML file."
    )
    parser.add_argument(
        "--list_inference",
        type=str,
        required=True,
        help="Path to the file containing captions for inference."
    )
    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=True,
        help="Path to the model checkpoint for reloading."
    )

    args = parser.parse_args()

    # Verify CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    # Load configuration
    config_yaml_path = args.config_yaml
    if not os.path.exists(config_yaml_path):
        raise FileNotFoundError(f"Configuration file not found: {config_yaml_path}")

    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)
    config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    # Build dataset JSON from inference list
    dataset_json = build_dataset_json_from_list(args.list_inference)

    # Extract experiment names
    exp_name = os.path.basename(config_yaml_path).split(".")[0]
    exp_group_name = os.path.basename(os.path.dirname(config_yaml_path))

    # Run inference
    infer(dataset_json, config_yaml, config_yaml_path, exp_group_name, exp_name)
