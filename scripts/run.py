from dotenv import load_dotenv
from pathlib import Path
import argparse
import json
import os
import sys
import uuid
import base64
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from evolution.genetic_algorithm import GeneticAlgorithm


def short_id():
    # Generate a random UUID
    u = uuid.uuid4()
    # Encode bytes to base64 and remove padding/newlines
    return base64.urlsafe_b64encode(u.bytes).decode('utf-8').rstrip('=\n')

if __name__ == "__main__":
    load_dotenv(PROJECT_ROOT / ".env")
    parser = argparse.ArgumentParser(description="Run CA Evolution")
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    ga = GeneticAlgorithm(config)
    best_chromosome = ga.run()
