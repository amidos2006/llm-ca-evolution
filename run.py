from genetic_algorithm import GeneticAlgorithm
import yaml
import os
import argparse
import json
import uuid
import base64

def short_id():
    # Generate a random UUID
    u = uuid.uuid4()
    # Encode bytes to base64 and remove padding/newlines
    return base64.urlsafe_b64encode(u.bytes).decode('utf-8').rstrip('=\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CA Evolution")
    parser.add_argument("-c", "--config", type=str, help="Path to the configuration YAML file")
    args = parser.parse_args()

    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    ga = GeneticAlgorithm(config)
    best_chromosome = ga.run()