import os
import json
from hashlib import sha256
import boto3
from botocore.exceptions import ClientError
from utils import load_cache, save_cache, get_cache_key

# --- Load Configuration from JSON File ---
with open('config.json', 'r') as f:
    config = json.load(f)

# --- Use Loaded Configuration ---
AWS_REGION = os.getenv("AWS_DEFAULT_REGION", config.get("aws_region"))
MODEL_ID = config.get("model_id")
CACHE_DIR = config.get("cache_dir")
PROMPT_TEMPLATE = config.get("prompt_template")

try:
    bedrock_client = boto3.client(service_name="bedrock-runtime", region_name=AWS_REGION)
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception as e:
    print(f"❌ Error creating Bedrock client: {e}")
    bedrock_client = None

# --- Generic LLM Query Function ---
def query_llm(service_block, source_cloud, target_cloud):
    # Format the prompt template with dynamic values
    prompt = PROMPT_TEMPLATE.format(
        source_cloud=source_cloud.upper(),
        target_cloud=target_cloud.upper(),
        service_block_json=json.dumps(service_block, indent=2)
    )

    native_request = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    }
    request_body = json.dumps(native_request)

    try:
        response = bedrock_client.invoke_model(
            modelId=MODEL_ID,
            body=request_body,
            contentType="application/json",
            accept="application/json"
        )
        response_body = json.loads(response.get("body").read())
        json_string = response_body["content"][0]["text"]
        
        try:
            return json.loads(json_string)
        except json.JSONDecodeError as e:
            print("❌ LLM Error: AI did not return a valid JSON object.")
            print("\n--- PROBLEMATIC STRING FROM LLM ---")
            print(json_string)
            print("--- END OF PROBLEMATIC STRING ---\n")
            return {"error": f"Invalid JSON from AI: {e}"}

    except ClientError as e:
        print(f"❌ Bedrock Error: {e}")
        return {"error": "Bedrock failed to respond."}

# --- Generic Main Translator Function ---
def get_translation(service_block, source_cloud, target_cloud):
    if not bedrock_client:
        return {"error": "Bedrock client not initialized."}

    cache_path = os.path.join(CACHE_DIR, f"{source_cloud}_to_{target_cloud}_cache.json")
    cache = load_cache(cache_path)
    key = get_cache_key(service_block)

    if key in cache:
        print(f"✅ Cache hit for: {service_block.get('id')}")
        return cache[key]

    print(f"⚠️  Cache miss → Querying Bedrock for: {service_block.get('id')}")
    result_dict = query_llm(service_block, source_cloud, target_cloud)
    
    if "error" not in result_dict:
        cache[key] = result_dict
        save_cache(cache_path, cache)
        
    return result_dict