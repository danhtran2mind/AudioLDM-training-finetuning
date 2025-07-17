# ... [Previous imports and code remain unchanged]

def main(configs, config_yaml_path, exp_group_name, exp_name, perform_validation, accelerator, wandb_off):
    # ... [Previous code in main function until logger initialization remains unchanged]

    # Initialize logger based on wandb_off flag
    if not wandb_off:
        wandb_logger = WandbLogger(
            save_dir=wandb_path,
            project=configs["project"],
            config=configs,
            name="%s/%s" % (exp_group_name, exp_name),
        )
    else:
        wandb_logger = None  # No logger when Wandb is disabled
        print("Wandb logging is disabled.")

    latent_diffusion.test_data_subset_path = test_data_subset_folder

    print("==> Save checkpoint every %s steps" % save_checkpoint_every_n_steps)
    print("==> Perform validation every %s epochs" % validation_every_n_epochs)

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=wandb_logger,  # Use the conditional logger
        max_steps=max_steps,
        num_sanity_val_steps=1,
        limit_val_batches=limit_val_batches,
        check_val_every_n_epoch=validation_every_n_epochs,
        strategy=DDPStrategy(find_unused_parameters=True) if accelerator == "gpu" else 'auto',
        callbacks=[checkpoint_callback],
    )
    # ... [Rest of the main function remains unchanged]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config_yaml",
        type=str,
        required=True,
        help="path to config .yaml file",
    )
    parser.add_argument(
        "--reload_from_ckpt",
        type=str,
        required=False,
        default=None,
        help="path to pretrained checkpoint",
    )
    parser.add_argument(
        "--val",
        action="store_true",
        help="perform validation",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="accelerator type: gpu or cpu",
    )
    parser.add_argument(
        "--wandb_off",
        action="store_true",
        help="disable Wandb logging",
    )

    args = parser.parse_args()

    perform_validation = args.val
    accelerator = args.accelerator
    wandb_off = args.wandb_off  # New argument

    if accelerator == "gpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available, use --accelerator cpu instead")

    config_yaml = args.config_yaml
    exp_name = os.path.basename(config_yaml.split(".")[0])
    exp_group_name = os.path.basename(os.path.dirname(config_yaml))

    config_yaml_path = os.path.join(config_yaml)
    config_yaml = yaml.load(open(config_yaml_path, "r"), Loader=yaml.FullLoader)

    if args.reload_from_ckpt is not None:
        config_yaml["reload_from_ckpt"] = args.reload_from_ckpt

    if perform_validation:
        config_yaml["model"]["params"]["cond_stage_config"][
            "crossattn_audiomae_generated"
        ]["params"]["use_gt_mae_output"] = False
        config_yaml["step"]["limit_val_batches"] = None

    main(config_yaml, config_yaml_path, exp_group_name, exp_name, perform_validation, accelerator, wandb_off)
