import yaml
import argparse  # Import the argparse library
from llm_handler import get_translation  # Import the new generic function
from utils import load_yaml, save_yaml

def main():
    # 1. Set up the command-line argument parser
    parser = argparse.ArgumentParser(description="Translate cloud configuration files using an LLM.")
    
    parser.add_argument("--source", required=True, choices=['aws', 'azure', 'gcp'], help="Source cloud provider.")
    parser.add_argument("--target", required=True, choices=['aws', 'azure', 'gcp'], help="Target cloud provider.")
    parser.add_argument("--input", required=True, help="Path to the source input YAML file.")
    parser.add_argument("--output", required=True, help="Path for the translated output YAML file.")
    
    args = parser.parse_args()

    # 2. Load the source YAML file using the provided input path
    source_yaml = load_yaml(args.input)
    translated_resources = []

    print(f"Starting conversion from {args.source.upper()} to {args.target.upper()}...")

    for resource in source_yaml.get("resources", []):
        # 3. Pass the source and target clouds to the handler function
        converted_dict = get_translation(resource, args.source, args.target)
        
        if "error" not in converted_dict:
            translated_resources.append(converted_dict)
        else:
            print(f"Skipping resource {resource.get('id')} due to error: {converted_dict.get('error')}")

    # 4. Prepare and save the final output file
    output = {
        "version": 1,
        "provider": args.target,
        "resources": translated_resources
    }

    save_yaml(args.output, output)
    print(f"✅ Conversion complete. Output saved to {args.output}")

if __name__ == "__main__":
    main()