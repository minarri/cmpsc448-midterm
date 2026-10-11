"""Make one small, independent test request to each configured LLM API."""

import os
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEST_PROMPT = "Explain machine learning in one sentence."
MAX_OUTPUT_TOKENS = 128

DEFAULT_MODELS = {
    "GPT": "gpt-6-luna",
    "Claude": "claude-haiku-5-5",
    "Gemini": "gemini-3.5-flash-lite",
}


def classifyApiError(error):
    """Return a safe error category without displaying the provider message."""
    className = error.__class__.__name__.lower()
    statusCode = getattr(error, "status_code", getattr(error, "code", None))
    errorText = str(error).lower()

    if statusCode == 401 or "authentication" in className:
        return "authentication failed (check the API key)"
    if statusCode == 403 or "permissiondenied" in className:
        return "permission denied for this key or model"
    if statusCode == 404 or "notfound" in className:
        return "model or API endpoint is unavailable"
    if statusCode == 429 or "ratelimit" in className or "resourceexhausted" in className:
        if any(word in errorText for word in ("credit balance", "billing", "insufficient_quota")):
            return "quota, credits, or billing limit reached"
        return "rate limit reached"
    if statusCode in (408, 504) or "timeout" in className:
        return "request timed out"
    if "connect" in className or "network" in className:
        return "network connection failed"
    if isinstance(statusCode, int) and statusCode >= 500:
        return "provider service is temporarily unavailable"
    if statusCode == 400:
        return "request rejected (check the model ID and account access)"
    return f"unexpected {error.__class__.__name__}"


def callOpenAi(apiKey, modelId):
    client = OpenAI(api_key=apiKey, timeout=30.0, max_retries=0)
    response = client.responses.create(
        model=modelId,
        input=TEST_PROMPT,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    return response.output_text


def callAnthropic(apiKey, modelId):
    client = Anthropic(api_key=apiKey, timeout=30.0, max_retries=0)
    response = client.messages.create(
        model=modelId,
        max_tokens=MAX_OUTPUT_TOKENS,
        
        messages=[{"role": "user", "content": TEST_PROMPT}],
    )
    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def callGemini(apiKey, modelId):
    client = genai.Client(api_key=apiKey)
    response = client.models.generate_content(
        model=modelId,
        contents=TEST_PROMPT,
        config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
    )
    return response.text or ""


def main():
    load_dotenv(PROJECT_ROOT / ".env")

    providers = [
        ("GPT", "OPENAI_API_KEY", "OPENAI_MODEL", callOpenAi),
        ("Claude", "ANTHROPIC_API_KEY", "ANTHROPIC_MODEL", callAnthropic),
        ("Gemini", "GEMINI_API_KEY", "GEMINI_MODEL", callGemini),
    ]

    print(f'Test prompt: "{TEST_PROMPT}"')
    print("Each configured provider will receive one potentially billable request.\n")

    successCount = 0
    for providerName, keyVariable, modelVariable, callProvider in providers:
        apiKey = os.getenv(keyVariable, "").strip()
        modelId = os.getenv(modelVariable, DEFAULT_MODELS[providerName]).strip()

        print(f"Provider: {providerName}")
        print(f"Model:    {modelId or '(not configured)'}")

        if not apiKey:
            print(f"Status:   FAILURE - missing {keyVariable}\n")
            continue
        if not modelId:
            print(f"Status:   FAILURE - missing {modelVariable}\n")
            continue

        try:
            outputText = callProvider(apiKey, modelId)
            if not outputText.strip():
                raise ValueError("Provider returned an empty response")
            print("Status:   SUCCESS")
            print(f"Response: {outputText}\n")
            successCount += 1
        except Exception as error:  # Provider SDKs use different exception classes.
            print(f"Status:   FAILURE - {classifyApiError(error)}\n")

    print(f"Summary: {successCount}/{len(providers)} providers succeeded.")
    return 0 if successCount == len(providers) else 1


if __name__ == "__main__":
    raise SystemExit(main())
