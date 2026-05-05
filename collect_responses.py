# =============================================================
# collect_responses.py
# Main Script — Sends 30 prompts to 5 LLMs, 3 times each
# Saves all 450 responses to Excel (.xlsx) and CSV (.csv)
#
# PhD Research — AI Bias in Large Language Models:
# A Study on Gen Z Perceptions of Indian Political Narratives
# =============================================================

import os
import time
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
from prompts import PROMPTS

# Load API keys from .env file
load_dotenv()

# =============================================================
# SETTINGS — You can change these numbers if needed
# =============================================================

# How many times to ask each prompt to each model
GENERATIONS_PER_PROMPT = 3

# Maximum length of each response (in tokens, roughly 1 token = 1 word)
MAX_TOKENS = 2000

# Seconds to wait between each API call (to avoid being blocked)
DELAY_BETWEEN_CALLS = 3

# Folder where results will be saved
OUTPUT_FOLDER = "results"


# =============================================================
# MODEL CONFIGURATIONS
# Each model needs: a display name, the provider type,
# the exact model string, and which API key to use
# =============================================================

MODELS = {
    "ChatGPT": {
        "provider": "openai",
        "model": "gpt-4o",
        "api_key_env": "OPENAI_API_KEY"
    },
    "Claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-6",
        "api_key_env": "ANTHROPIC_API_KEY"
    },
    "Gemini": {
        "provider": "google",
        "model": "gemini-2.5-flash",
        "api_key_env": "GOOGLE_API_KEY"
    },
    "DeepSeek": {
        "provider": "deepseek",
        "model": "deepseek-chat",
        "api_key_env": "DEEPSEEK_API_KEY"
    },
    "Grok": {
        "provider": "xai",
        "model": "grok-3",
        "api_key_env": "XAI_API_KEY"
    }
}


# =============================================================
# FUNCTIONS TO TALK TO EACH LLM
# Each function sends a prompt and returns the response text
# and the exact model version used
# =============================================================

def call_openai(prompt, model, api_key):
    """Send a prompt to OpenAI (ChatGPT) and get the response."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.7
    )
    text = response.choices[0].message.content
    model_version = response.model
    return text, model_version


def call_anthropic(prompt, model, api_key):
    """Send a prompt to Anthropic (Claude) and get the response."""
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text
    model_version = response.model
    return text, model_version


def call_google(prompt, model, api_key):
    """Send a prompt to Google (Gemini) and get the response."""
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gen_model = genai.GenerativeModel(model)
    response = gen_model.generate_content(prompt)
    text = response.text
    model_version = model  # Gemini doesn't return exact version
    return text, model_version


def call_deepseek(prompt, model, api_key):
    """Send a prompt to DeepSeek and get the response."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.7
    )
    text = response.choices[0].message.content
    model_version = response.model
    return text, model_version


def call_xai(prompt, model, api_key):
    """Send a prompt to xAI (Grok) and get the response."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
        temperature=0.7
    )
    text = response.choices[0].message.content
    model_version = response.model
    return text, model_version


def call_model(provider, prompt, model, api_key):
    """
    Routes the prompt to the correct LLM based on provider type.
    This is like a traffic controller — it decides which function to use.
    """
    if provider == "openai":
        return call_openai(prompt, model, api_key)
    elif provider == "anthropic":
        return call_anthropic(prompt, model, api_key)
    elif provider == "google":
        return call_google(prompt, model, api_key)
    elif provider == "deepseek":
        return call_deepseek(prompt, model, api_key)
    elif provider == "xai":
        return call_xai(prompt, model, api_key)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# =============================================================
# MAIN DATA COLLECTION LOGIC
# This is the heart of the script — it loops through every
# prompt, every model, and every generation, collecting responses
# =============================================================

def main():
    # Create the results folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # This list will store all collected data
    all_results = []

    # Calculate total number of API calls for progress tracking
    total_calls = len(PROMPTS) * len(MODELS) * GENERATIONS_PER_PROMPT
    completed_calls = 0

    # Record when data collection started
    collection_start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print("=" * 65)
    print("  LLM RESPONSE COLLECTION SCRIPT")
    print("  PhD Research — AI Bias in Large Language Models")
    print("=" * 65)
    print(f"  Prompts:              {len(PROMPTS)}")
    print(f"  Models:               {len(MODELS)}")
    print(f"  Generations per pair: {GENERATIONS_PER_PROMPT}")
    print(f"  Total API calls:      {total_calls}")
    print(f"  Started at:           {collection_start_time}")
    print("=" * 65)
    print()

    # ---------------------------------------------------------
    # LOOP 1: Go through each of the 30 prompts
    # ---------------------------------------------------------
    for prompt_data in PROMPTS:

        prompt_id = prompt_data["id"]
        category = prompt_data["category"]
        title = prompt_data["title"]
        prompt_text = prompt_data["prompt"]

        print(f"--- Prompt {prompt_id}/30: {title} ---")

        # ---------------------------------------------------------
        # LOOP 2: For each prompt, go through each of the 5 models
        # ---------------------------------------------------------
        for model_name, model_config in MODELS.items():

            provider = model_config["provider"]
            model_string = model_config["model"]
            api_key = os.getenv(model_config["api_key_env"])

            # Check if API key exists
            if not api_key or api_key.startswith("paste_your"):
                print(f"  ⚠️  Skipping {model_name} — API key not set")
                completed_calls += GENERATIONS_PER_PROMPT
                continue

            # ---------------------------------------------------------
            # LOOP 3: Ask the same question 3 times to capture variability
            # ---------------------------------------------------------
            for gen_num in range(1, GENERATIONS_PER_PROMPT + 1):

                completed_calls += 1
                progress = (completed_calls / total_calls) * 100

                print(f"  [{completed_calls}/{total_calls}] ({progress:.1f}%) "
                      f"{model_name} — Generation {gen_num}/3 ... ",
                      end="", flush=True)

                try:
                    # Record the exact time of this API call
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Send the prompt and get the response
                    response_text, model_version = call_model(
                        provider, prompt_text, model_string, api_key
                    )

                    # Count the number of words in the response
                    word_count = len(response_text.split())

                    # Save all the data for this response
                    result = {
                        "prompt_id": prompt_id,
                        "category": category,
                        "title": title,
                        "prompt_text": prompt_text,
                        "model_name": model_name,
                        "model_version": model_version,
                        "generation_number": gen_num,
                        "response_text": response_text,
                        "word_count": word_count,
                        "timestamp": timestamp,
                        "collection_date": collection_start_time
                    }
                    all_results.append(result)
                    print(f"✅ ({word_count} words)")

                except Exception as e:
                    # If something goes wrong, record the error but keep going
                    result = {
                        "prompt_id": prompt_id,
                        "category": category,
                        "title": title,
                        "prompt_text": prompt_text,
                        "model_name": model_name,
                        "model_version": model_string,
                        "generation_number": gen_num,
                        "response_text": f"ERROR: {str(e)}",
                        "word_count": 0,
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "collection_date": collection_start_time
                    }
                    all_results.append(result)
                    print(f"❌ Error: {str(e)[:80]}")

                # Wait between calls to avoid rate limiting
                time.sleep(DELAY_BETWEEN_CALLS)

        print()  # Blank line between prompts for readability

    # =============================================================
    # SAVE RESULTS TO FILES
    # =============================================================

    print("=" * 65)
    print("  SAVING RESULTS...")
    print("=" * 65)

    # Create a pandas DataFrame (like a spreadsheet in memory)
    df = pd.DataFrame(all_results)

    # Generate filename with today's date
    date_string = datetime.now().strftime("%Y-%m-%d_%H%M")

    # Save as CSV
    csv_filename = f"{OUTPUT_FOLDER}/llm_responses_{date_string}.csv"
    df.to_csv(csv_filename, index=False, encoding="utf-8")
    print(f"  ✅ CSV saved:   {csv_filename}")

    # Save as Excel
    xlsx_filename = f"{OUTPUT_FOLDER}/llm_responses_{date_string}.xlsx"
    df.to_excel(xlsx_filename, index=False, engine="openpyxl")
    print(f"  ✅ Excel saved: {xlsx_filename}")

    # =============================================================
    # PRINT SUMMARY
    # =============================================================

    print()
    print("=" * 65)
    print("  COLLECTION COMPLETE — SUMMARY")
    print("=" * 65)
    print(f"  Total responses collected: {len(all_results)}")
    print(f"  Successful responses:      {len([r for r in all_results if not r['response_text'].startswith('ERROR')])}")
    print(f"  Failed responses:          {len([r for r in all_results if r['response_text'].startswith('ERROR')])}")
    print()

    # Show count per model
    print("  Responses per model:")
    for model_name in MODELS:
        count = len([r for r in all_results if r["model_name"] == model_name and not r["response_text"].startswith("ERROR")])
        print(f"    {model_name:12s}: {count} responses")

    print()
    print(f"  Files saved in: {OUTPUT_FOLDER}/")
    print(f"  CSV file:       {csv_filename}")
    print(f"  Excel file:     {xlsx_filename}")
    print("=" * 65)
    print()
    print("  NEXT STEPS:")
    print("  1. Download the Excel/CSV files from the 'results' folder")
    print("  2. Use these for sentiment analysis, framing analysis,")
    print("     and factual accuracy benchmarking in your thesis")
    print("=" * 65)


# =============================================================
# This line tells Python: "Run the main function when I
# execute this file"
# =============================================================
if __name__ == "__main__":
    main()