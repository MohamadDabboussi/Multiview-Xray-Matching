import os
import yaml
import wandb
import argparse
from src.train.train import train_model, load_config

def parse_args():
    parser = argparse.ArgumentParser(description="Run training with a given config file")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Initialize WandB logging
    if config['training'].get('wandb_project') is not None:
        wandb.init(project=config['training']['wandb_project'], config=config)

    # Train the model and get metrics
    trainer = train_model(config)

    # Save configuration and files
    save_path = f"output/{config['training']['name']}"
    os.makedirs(save_path, exist_ok=True)
    with open(f"{save_path}/{config['training']['name']}.yml", "w") as file:
        yaml.dump(config, file)

if __name__ == "__main__":
    main()
