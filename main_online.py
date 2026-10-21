import argparse

from src.train.train_online import load_config, train_model_online


def main():
    parser = argparse.ArgumentParser(description="End-to-end correspondence pretraining with on-the-fly DRRs")
    parser.add_argument("--config", required=True, help="Path to the YAML configuration file")
    args = parser.parse_args()
    train_model_online(load_config(args.config))


if __name__ == "__main__":
    main()
