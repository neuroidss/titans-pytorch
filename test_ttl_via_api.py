# FILE: test_ttl_via_api.py

import requests
import json
import time
import argparse
import uuid
import sys # Added import

BASE_URL = "http://localhost:8000"
ACTIVE_TTA_MODEL_NAME = None
API_KEY = None # Will be created

def create_and_get_api_key():
    global API_KEY
    print("\n--- Creating a new API key ---")
    try:
        response = requests.post(f"{BASE_URL}/admin/create_api_key", params={"key_description": "test_ttl_key"})
        response.raise_for_status()
        key_data = response.json()
        API_KEY = key_data.get("api_key")
        if API_KEY:
            print(f"API Key created: {API_KEY}")
            return True
        else:
            print("Failed to create API key from response:", key_data)
            return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to create API key: {e}")
        return False


def get_available_models():
    # ... (get_available_models from previous step, no changes needed here for API key, as /v1/models is usually public)
    global ACTIVE_TTA_MODEL_NAME
    print("\n--- Fetching available models from server ---")
    try:
        response = requests.get(f"{BASE_URL}/v1/models")
        response.raise_for_status()
        models_data = response.json()
        print(json.dumps(models_data, indent=2))
        if models_data and models_data.get("data"):
            available_ids = [m['id'] for m in models_data['data']]
            print(f"Server potentially lists: {available_ids}")
            if ACTIVE_TTA_MODEL_NAME is None:
                print("Error: ACTIVE_TTA_MODEL_NAME not set. Please specify target model via --target_model_id.")
                return False
            if ACTIVE_TTA_MODEL_NAME not in available_ids:
                print(f"Warning: Target model '{ACTIVE_TTA_MODEL_NAME}' not in server's list: {available_ids}.")
            print(f"Will use '{ACTIVE_TTA_MODEL_NAME}' for chat completion tests.")
            return True
        else:
            print("No model data returned from /v1/models"); return False
    except requests.exceptions.RequestException as e:
        print(f"Failed to get models: {e}"); return False


def call_chat_completions_api(messages: list, max_tokens=200, temperature=0.1, stream=False):
    if ACTIVE_TTA_MODEL_NAME is None or API_KEY is None:
        print("Cannot call chat completions: Active TTA model name or API Key not set.")
        return None

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}" # Add API Key
    }
    payload = {
        "model": ACTIVE_TTA_MODEL_NAME,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": stream,
    }
    print(f"\n--- Sending to API ({'stream' if stream else 'non-stream'}): ---")
    print(f"Model: {ACTIVE_TTA_MODEL_NAME}, API Key: {API_KEY[:8]}...")
    # ... (rest of call_chat_completions_api from previous step remains the same for request/response handling)
    try:
        response = requests.post(f"{BASE_URL}/v1/chat/completions", headers=headers, json=payload, stream=stream)
        response.raise_for_status()
        print("\n--- API Response: ---")
        if stream:
            full_content = ""; print("Streamed content: ", end="")
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith("data: "):
                        json_data_str = decoded_line[len("data: "):]
                        if json_data_str.strip() == "[DONE]": print("\n[STREAM DONE]"); break
                        try:
                            json_data = json.loads(json_data_str)
                            delta = json_data.get("choices", [{}])[0].get("delta", {})
                            content_chunk = delta.get("content")
                            if content_chunk: print(content_chunk, end="", flush=True); full_content += content_chunk
                            finish_reason = json_data.get("choices", [{}])[0].get("finish_reason")
                            if finish_reason: print(f"\nFinish Reason: {finish_reason}")
                        except json.JSONDecodeError: print(f"Error decoding JSON: {json_data_str}")
            print("\n--------------------")
            return full_content.strip()
        else:
            response_data = response.json(); print(json.dumps(response_data, indent=2))
            content = response_data.get("choices", [{}])[0].get("message", {}).get("content", "")
            print("--------------------"); return content.strip()
    except requests.exceptions.RequestException as e:
        print(f"API request failed: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try: print(f"Response content: {e.response.json()}")
            except json.JSONDecodeError: print(f"Response content (not JSON): {e.response.text}")
        print("--------------------"); return None


def test_stateful_ttl_across_calls():
    print("\n=== Test: Stateful TTL Demonstration Across API Calls ===")

    # Call 1: Provide a piece of information
    fact_to_remember = f"The secret ingredient for the space muffin is stardust powder code {str(uuid.uuid4())[:6]}."
    messages1 = [
        {"role": "user", "content": f"Please remember this: {fact_to_remember}. Just acknowledge that you've noted it."}
    ]
    print("\n--- Call 1: Stating a fact ---")
    response1 = call_chat_completions_api(messages1, max_tokens=200, temperature=0.01)
    if response1:
        print(f"Model's acknowledgement: {response1}")

    print("\nWaiting a moment before the next call...")
    time.sleep(2)

    # Call 2: Ask to recall the information
    messages2 = [
        {"role": "user", "content": "A few moments ago, I told you a secret ingredient for space muffins. What was that ingredient?"}
    ]
    print("\n--- Call 2: Recalling the fact ---")
    response2 = call_chat_completions_api(messages2, max_tokens=200, temperature=0.01)
    if response2:
        print(f"Model's recall response: {response2}")
        if fact_to_remember.lower().split("is ")[-1] in response2.lower(): # Check for the "stardust powder code..." part
            print("Stateful TTL Test (Recall Across Calls): PASSED - Model recalled the specific fact.")
        else:
            print(f"Stateful TTL Test (Recall Across Calls): FAILED - Model did not recall the specific fact. Expected part of: '{fact_to_remember}'")

    print("\n---------------------------------------------\n")
    time.sleep(1)

    # Call 3: Another instruction based task, continuing session
    instruction = "From now on, whenever I say 'Alpha', you say 'Beta'."
    messages3_setup = [
        {"role": "user", "content": f"{instruction} Understood?"}
    ]
    print("\n--- Call 3: Setting up an instruction ---")
    response3_setup = call_chat_completions_api(messages3_setup, max_tokens=200, temperature=0.01)
    if response3_setup: print(f"Model's acknowledgement: {response3_setup}")

    time.sleep(2)
    messages3_test = [
        {"role": "user", "content": "Alpha"}
    ]
    print("\n--- Call 4: Testing the instruction ---")
    response3_test = call_chat_completions_api(messages3_test, max_tokens=200, temperature=0.01)
    if response3_test:
        print(f"Model's response to 'Alpha': {response3_test}")
        if "beta" in response3_test.lower():
            print("Stateful TTL Test (Instruction Across Calls): PASSED")
        else:
            print("Stateful TTL Test (Instruction Across Calls): FAILED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test TTA server API with stateful TTL.")
    parser.add_argument(
        "--target_model_id",
        type=str,
        default="Qwen3-0.6B-tta-ttl",
        help="The TTA model ID the server is expected to be running."
    )
    args = parser.parse_args()

    ACTIVE_TTA_MODEL_NAME = args.target_model_id
    print(f"Test script configured to target model ID: {ACTIVE_TTA_MODEL_NAME}")

    print("Waiting for server to start...")
    time.sleep(3)

    if not create_and_get_api_key():
        print("Failed to create API key. Aborting tests.")
        sys.exit(1)

    if not get_available_models(): # This will also set ACTIVE_TTA_MODEL_NAME if --target_model_id used
        print("Could not verify models with server. Aborting tests.")
    else:
        # First run the single-call tests (from previous step, slightly adapted)
        # test_single_call_ttl() # You can uncomment this if you keep that function

        # Then run the new stateful tests
        test_stateful_ttl_across_calls()

    print("\n\nReminder on Stateful TTL Testing:")
    print("This script now attempts to test stateful TTL *across multiple API calls* using the same API key.")
    print("The server is designed to persist Neural Memory states for each API key.")

