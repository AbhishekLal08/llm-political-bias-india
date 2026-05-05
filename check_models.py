# =============================================================
# check_models.py
# Script to verify which LLM models are accessible via your API keys
# PhD Research — AI Bias in Large Language Models
# =============================================================

import os
from dotenv import load_dotenv

# Load API keys from .env file
load_dotenv()

print("=" * 60)
print("LLM MODEL VERIFICATION SCRIPT")
print("PhD Research — AI Bias in LLMs")
print("=" * 60)

# ---- 1. CHECK OPENAI (ChatGPT) ----
print("\n[1/5] Checking OpenAI (ChatGPT)...")
try:
    from openai import OpenAI
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10
    )
    model_used = response.model
    print(f"  ✅ SUCCESS — Model: {model_used}")
except Exception as e:
    print(f"  ❌ FAILED — {e}")

# ---- 2. CHECK ANTHROPIC (Claude) ----
print("\n[2/5] Checking Anthropic (Claude)...")
try:
    import anthropic
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=10,
        messages=[{"role": "user", "content": "Say hello in one word."}]
    )
    print(f"  ✅ SUCCESS — Model: {response.model}")
except Exception as e:
    print(f"  ❌ FAILED — {e}")

# ---- 3. CHECK GOOGLE (Gemini) ----
print("\n[3/5] Checking Google (Gemini)...")
try:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel("gemini-2.5-flash")
    response = model.generate_content("Say hello in one word.")
    print(f"  ✅ SUCCESS — Model: gemini-2.5-flash")
except Exception as e:
    print(f"  ❌ FAILED — {e}")

# ---- 4. CHECK DEEPSEEK ----
print("\n[4/5] Checking DeepSeek...")
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com"
    )
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10
    )
    model_used = response.model
    print(f"  ✅ SUCCESS — Model: {model_used}")
except Exception as e:
    print(f"  ❌ FAILED — {e}")

# ---- 5. CHECK XAI (Grok) ----
print("\n[5/5] Checking xAI (Grok)...")
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("XAI_API_KEY"),
        base_url="https://api.x.ai/v1"
    )
    response = client.chat.completions.create(
        model="grok-3",
        messages=[{"role": "user", "content": "Say hello in one word."}],
        max_tokens=10
    )
    model_used = response.model
    print(f"  ✅ SUCCESS — Model: {model_used}")
except Exception as e:
    print(f"  ❌ FAILED — {e}")

print("\n" + "=" * 60)
print("VERIFICATION COMPLETE")
print("Record the model names above in your thesis methodology.")
print("=" * 60)