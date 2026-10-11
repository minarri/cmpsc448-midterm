"""Collect matching prompt responses from GPT, Claude, and Gemini."""

import argparse
import csv
import fcntl
import os
import time
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from google import genai
from google.genai import types
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPTS_PATH = PROJECT_ROOT / "data" / "prompts.csv"
NEW_PROMPTS_PATH = PROJECT_ROOT / "data" / "new_prompts.csv"
OUTPUT_PATH = PROJECT_ROOT / "data" / "raw_responses.csv"
FAILURES_PATH = PROJECT_ROOT / "data" / "collection_failures.csv"

PROMPT_COLUMNS = ["prompt_id", "category", "prompt"]
OUTPUT_COLUMNS = [
    "LLM_name",
    "LLM_Input",
    "LLM_output",
    "prompt_id",
    "category",
    "model_id",
    "collected_at",
]
FAILURE_COLUMNS = [
    "prompt_id",
    "category",
    "LLM_name",
    "model_id",
    "failure_type",
    "attempts",
    "attempted_at",
]

DEFAULT_MODELS = {
    "GPT": "gpt-6-luna",
    "Claude": "claude-haiku-5-5",
    "Gemini": "gemini-3.5-flash-lite",
}
MAX_OUTPUT_TOKENS = 10000
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2


class TruncatedResponseError(Exception):
    """Raised when a provider explicitly reports an incomplete response."""


def positiveInt(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parseArguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--prompts-path",
        type=Path,
        default=PROMPTS_PATH,
        help="Prompt CSV path (default: data/prompts.csv).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=OUTPUT_PATH,
        help="Response CSV path (default: data/raw_responses.csv).",
    )
    parser.add_argument(
        "--max-new-prompts",
        type=positiveInt,
        help="Process at most this many prompts that still need at least one response.",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate inputs and report pending requests without making API calls.",
    )
    return parser.parse_args()


def resolveProjectPath(path):
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def validateCollectionPaths(promptPath, outputPath):
    if promptPath == outputPath:
        raise ValueError("Prompt input and response output paths must be different.")
    if promptPath != PROMPTS_PATH.resolve() and outputPath == OUTPUT_PATH.resolve():
        raise ValueError(
            "A non-default prompt file cannot write to data/raw_responses.csv. "
            "Choose a separate --output-path."
        )
    if outputPath.exists() and not outputPath.is_file():
        raise ValueError(f"Response output path is not a file: {outputPath}")
    if not outputPath.parent.is_dir():
        raise ValueError(f"Response output directory does not exist: {outputPath.parent}")


def completionValueName(value):
    if value is None:
        return ""
    valueName = getattr(value, "name", value)
    return str(valueName).rsplit(".", maxsplit=1)[-1].upper()


def utcTimestamp():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def classifyApiError(error):
    """Return a safe category and whether collection may retry the error."""
    if isinstance(error, TruncatedResponseError):
        return "truncated_response", False

    className = error.__class__.__name__.lower()
    statusCode = getattr(error, "status_code", getattr(error, "code", None))
    errorText = str(error).lower()

    if statusCode == 401 or "authentication" in className:
        return "authentication_failed", False
    if statusCode == 403 or "permissiondenied" in className:
        return "permission_denied", False
    if statusCode == 404 or "notfound" in className:
        return "model_unavailable", False
    if statusCode == 429 or "ratelimit" in className or "resourceexhausted" in className:
        if any(word in errorText for word in ("credit balance", "billing", "insufficient_quota")):
            return "quota_or_billing_limit", False
        return "rate_limited", True
    if statusCode in (408, 409, 504) or "timeout" in className:
        return "temporary_timeout", True
    if "connect" in className or "network" in className:
        return "temporary_network_error", True
    if isinstance(statusCode, int) and statusCode >= 500:
        return "temporary_provider_error", True
    if statusCode == 400:
        return "invalid_request_or_model", False
    return f"unexpected_{error.__class__.__name__}", False


def loadPrompts(promptPath):
    if not promptPath.exists():
        raise FileNotFoundError(f"Prompt file not found: {promptPath}")

    with promptPath.open("r", encoding="utf-8", newline="") as promptFile:
        reader = csv.DictReader(promptFile)
        missingColumns = [column for column in PROMPT_COLUMNS if column not in (reader.fieldnames or [])]
        if missingColumns:
            raise ValueError(f"{promptPath.name} is missing columns: {', '.join(missingColumns)}")
        prompts = list(reader)

    if not prompts:
        raise ValueError(f"{promptPath.name} does not contain any prompts")

    promptIds = set()
    promptTexts = set()
    for rowNumber, prompt in enumerate(prompts, start=2):
        if None in prompt:
            raise ValueError(f"{promptPath.name} row {rowNumber} has extra fields")
        for column in PROMPT_COLUMNS:
            if prompt[column] is None or not prompt[column].strip():
                raise ValueError(f"{promptPath.name} row {rowNumber} has an empty {column}")
        if prompt["prompt_id"] in promptIds:
            raise ValueError(f"Duplicate prompt_id in {promptPath.name}: {prompt['prompt_id']}")
        if prompt["prompt"] in promptTexts:
            raise ValueError(f"Duplicate prompt text in {promptPath.name}: row {rowNumber}")
        promptIds.add(prompt["prompt_id"])
        promptTexts.add(prompt["prompt"])

    return prompts


def verifyExpectedNewPrompts(promptPath, prompts):
    if promptPath != NEW_PROMPTS_PATH.resolve():
        return

    expectedIds = {f"p{number}" for number in range(101, 1101)}
    actualIds = {prompt["prompt_id"] for prompt in prompts}
    if len(prompts) != 1000 or actualIds != expectedIds:
        raise ValueError(
            "new_prompts.csv must contain exactly the prompt IDs p101 through p1100"
        )

    expectedCategories = {
        "General knowledge",
        "Creative writing",
        "Programming",
        "Mathematical/logical reasoning",
        "Text transformation",
    }
    categoryCounts = Counter(prompt["category"] for prompt in prompts)
    if set(categoryCounts) != expectedCategories or any(
        categoryCounts[category] != 200 for category in expectedCategories
    ):
        raise ValueError("new_prompts.csv must contain exactly 200 prompts per category")


def loadExistingRecords(outputPath, prompts, providerNames):
    if not outputPath.exists() or outputPath.stat().st_size == 0:
        return [], set()

    with outputPath.open("r", encoding="utf-8", newline="") as outputFile:
        reader = csv.DictReader(outputFile)
        if reader.fieldnames != OUTPUT_COLUMNS:
            raise ValueError(
                f"{outputPath.name} has an invalid schema. Expected: {', '.join(OUTPUT_COLUMNS)}"
            )
        records = list(reader)

    promptsById = {prompt["prompt_id"]: prompt for prompt in prompts}
    completedPairs = set()
    for rowNumber, record in enumerate(records, start=2):
        if None in record:
            raise ValueError(f"{outputPath.name} row {rowNumber} has extra fields")
        for column in OUTPUT_COLUMNS:
            if record[column] is None or not record[column].strip():
                raise ValueError(f"{outputPath.name} row {rowNumber} has an empty {column}")

        if record["LLM_name"] not in providerNames:
            raise ValueError(
                f"{outputPath.name} row {rowNumber} has unknown LLM_name: "
                f"{record['LLM_name']}"
            )
        if record["prompt_id"] not in promptsById:
            raise ValueError(
                f"{outputPath.name} row {rowNumber} has unexpected prompt_id: "
                f"{record['prompt_id']}"
            )

        prompt = promptsById[record["prompt_id"]]
        if record["LLM_Input"] != prompt["prompt"] or record["category"] != prompt["category"]:
            raise ValueError(
                f"{outputPath.name} row {rowNumber} does not match its source prompt"
            )

        pair = (record["prompt_id"], record["LLM_name"])
        if pair in completedPairs:
            raise ValueError(
                f"{outputPath.name} already contains duplicate "
                f"(prompt_id, LLM_name) entry: {pair}"
            )
        completedPairs.add(pair)

    return records, completedPairs


def verifyModelConsistency(existingRecords, modelIds):
    for providerName, configuredModel in modelIds.items():
        existingModels = {
            record["model_id"]
            for record in existingRecords
            if record["LLM_name"] == providerName and record["model_id"].strip()
        }
        if len(existingModels) > 1:
            raise ValueError(f"Existing {providerName} records already mix model IDs: {existingModels}")
        if existingModels and configuredModel not in existingModels:
            existingModel = next(iter(existingModels))
            raise ValueError(
                f"Configured {providerName} model '{configuredModel}' differs from existing "
                f"dataset model '{existingModel}'. Match the model or start a new dataset."
            )


@contextmanager
def holdOutputLock(outputPath):
    with outputPath.open("a+", encoding="utf-8", newline="") as lockFile:
        try:
            fcntl.flock(lockFile.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                f"Another collector is already writing to {outputPath}"
            ) from error
        try:
            yield
        finally:
            fcntl.flock(lockFile.fileno(), fcntl.LOCK_UN)


def appendCsvRow(path, fieldNames, row):
    isNewFile = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
        if isNewFile:
            writer.writeheader()
        writer.writerow(row)
        csvFile.flush()
        os.fsync(csvFile.fileno())


def callOpenAi(apiKey, modelId, promptText):
    client = OpenAI(api_key=apiKey, timeout=60.0, max_retries=0)
    response = client.responses.create(
        model=modelId,
        input=promptText,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        store=False,
    )
    if completionValueName(getattr(response, "status", None)) == "INCOMPLETE":
        incompleteDetails = getattr(response, "incomplete_details", None)
        incompleteReason = getattr(incompleteDetails, "reason", None)
        reasonText = f" (reason: {incompleteReason})" if incompleteReason else ""
        raise TruncatedResponseError(f"OpenAI reported an incomplete response{reasonText}.")
    return response.output_text


def callAnthropic(apiKey, modelId, promptText):
    client = Anthropic(api_key=apiKey, timeout=60.0, max_retries=0)
    response = client.messages.create(
        model=modelId,
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[{"role": "user", "content": promptText}],
        thinking={"type": "disabled"},
        output_config={"effort": "medium"},
    )
    print("Claude stop reason:", response.stop_reason)
    print("Claude content types:", [
        block.type for block in response.content
    ])

    if completionValueName(getattr(response, "stop_reason", None)) == "MAX_TOKENS":
        raise TruncatedResponseError("Anthropic reported stop_reason=max_tokens.")

    return "".join(
        block.text for block in response.content if getattr(block, "type", "") == "text"
    )


def callGemini(apiKey, modelId, promptText):
    client = genai.Client(api_key=apiKey)
    response = client.models.generate_content(
        model=modelId,
        contents=promptText,
        config=types.GenerateContentConfig(max_output_tokens=MAX_OUTPUT_TOKENS),
    )
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        finishReason = getattr(candidates[0], "finish_reason", None)
        if completionValueName(finishReason) == "MAX_TOKENS":
            raise TruncatedResponseError("Gemini reported finish_reason=MAX_TOKENS.")
    return response.text or ""


def collectWithRetries(callProvider, apiKey, modelId, promptText):
    lastFailure = "unknown_failure"

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            outputText = callProvider(apiKey, modelId, promptText)
            if not outputText.strip():
                return None, "empty_response", attempt
            return outputText, None, attempt
        except Exception as error:  # Provider SDKs use different exception classes.
            if isinstance(error, TruncatedResponseError):
                print(f"  WARNING: {error} The response will not be saved.")
            else:
                print(f"API ERROR TYPE: {type(error).__name__}")
                print(f"API ERROR DETAILS: {error}")
            lastFailure, isRetryable = classifyApiError(error)
            if not isRetryable or attempt == MAX_ATTEMPTS:
                return None, lastFailure, attempt

            waitSeconds = BACKOFF_SECONDS * (2 ** (attempt - 1))
            print(f"  Temporary error ({lastFailure}); retrying in {waitSeconds} seconds...")
            time.sleep(waitSeconds)

    return None, lastFailure, MAX_ATTEMPTS


def selectPromptsForRun(prompts, completedPairs, providerNames, maxNewPrompts):
    pendingPrompts = [
        prompt
        for prompt in prompts
        if any(
            (prompt["prompt_id"], providerName) not in completedPairs
            for providerName in providerNames
        )
    ]
    if maxNewPrompts is not None:
        return pendingPrompts[:maxNewPrompts]
    return pendingPrompts


def printProgress(prompts, selectedPrompts, completedPairs, providerNames):
    print("Progress by model:")
    for providerName in providerNames:
        completed = sum(
            (prompt["prompt_id"], providerName) in completedPairs for prompt in prompts
        )
        print(
            f"  {providerName}: completed {completed}/{len(prompts)}, "
            f"remaining {len(prompts) - completed}"
        )

    print(f"Prompts selected for this run: {len(selectedPrompts)}")
    print("Planned API requests by model:")
    totalPlanned = 0
    for providerName in providerNames:
        planned = sum(
            (prompt["prompt_id"], providerName) not in completedPairs
            for prompt in selectedPrompts
        )
        totalPlanned += planned
        print(f"  {providerName}: {planned}")
    print(f"  Total: {totalPlanned}")


def main():
    args = parseArguments()
    promptPath = resolveProjectPath(args.prompts_path)
    outputPath = resolveProjectPath(args.output_path)

    load_dotenv(PROJECT_ROOT / ".env")

    modelIds = {
        "GPT": os.getenv("OPENAI_MODEL", DEFAULT_MODELS["GPT"]).strip(),
        "Claude": os.getenv("ANTHROPIC_MODEL", DEFAULT_MODELS["Claude"]).strip(),
        "Gemini": os.getenv("GEMINI_MODEL", DEFAULT_MODELS["Gemini"]).strip(),
    }
    providerSettings = [
        ("GPT", os.getenv("OPENAI_API_KEY", "").strip(), modelIds["GPT"], callOpenAi),
        (
            "Claude",
            os.getenv("ANTHROPIC_API_KEY", "").strip(),
            modelIds["Claude"],
            callAnthropic,
        ),
        ("Gemini", os.getenv("GEMINI_API_KEY", "").strip(), modelIds["Gemini"], callGemini),
    ]
    providerNames = [providerName for providerName, _, _, _ in providerSettings]

    try:
        validateCollectionPaths(promptPath, outputPath)
        prompts = loadPrompts(promptPath)
        verifyExpectedNewPrompts(promptPath, prompts)
    except (OSError, ValueError) as error:
        print(f"Collection stopped: {error}")
        return 1

    if args.preflight_only:
        try:
            existingRecords, completedPairs = loadExistingRecords(
                outputPath, prompts, providerNames
            )
            verifyModelConsistency(existingRecords, modelIds)
        except (OSError, ValueError) as error:
            print(f"Collection stopped: {error}")
            return 1

        selectedPrompts = selectPromptsForRun(
            prompts, completedPairs, providerNames, args.max_new_prompts
        )
        print("Collection configuration:")
        for providerName, _, modelId, _ in providerSettings:
            print(f"  {providerName}: {modelId}")
        print(f"Prompt input: {promptPath}")
        print(f"Response output: {outputPath}")
        printProgress(prompts, selectedPrompts, completedPairs, providerNames)
        print("Preflight complete. No API requests were made and no files were changed.")
        return 0

    missingSettings = []
    keyVariables = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GEMINI_API_KEY"]
    modelVariables = ["OPENAI_MODEL", "ANTHROPIC_MODEL", "GEMINI_MODEL"]
    for index, (_, apiKey, modelId, _) in enumerate(providerSettings):
        if not apiKey:
            missingSettings.append(keyVariables[index])
        if not modelId:
            missingSettings.append(modelVariables[index])
    if missingSettings:
        print(f"Collection stopped. Missing configuration: {', '.join(missingSettings)}")
        return 1

    try:
        with holdOutputLock(outputPath):
            existingRecords, completedPairs = loadExistingRecords(
                outputPath, prompts, providerNames
            )
            verifyModelConsistency(existingRecords, modelIds)
            selectedPrompts = selectPromptsForRun(
                prompts, completedPairs, providerNames, args.max_new_prompts
            )
            totalRunPairs = len(selectedPrompts) * len(providerSettings)
            successfulCount = 0
            failureCount = 0

            print("Collection configuration (loaded once for this run):")
            for providerName, _, modelId, _ in providerSettings:
                print(f"  {providerName}: {modelId}")
            print(f"Prompt input: {promptPath}")
            print(f"Response output: {outputPath}")
            printProgress(prompts, selectedPrompts, completedPairs, providerNames)
            print()

            requestNumber = 0
            for prompt in selectedPrompts:
                for providerName, apiKey, modelId, callProvider in providerSettings:
                    requestNumber += 1
                    pair = (prompt["prompt_id"], providerName)
                    if pair in completedPairs:
                        print(
                            f"[{requestNumber}/{totalRunPairs}] Skipping saved "
                            f"{providerName} response for {prompt['prompt_id']}"
                        )
                        continue
                    """
                    print(
                        f"[{requestNumber}/{totalRunPairs}] Collecting "
                        f"{providerName} response for {prompt['prompt_id']}"
                    )
                    """
                
                    outputText, failureType, attempts = collectWithRetries(
                        callProvider,
                        apiKey,
                        modelId,
                        prompt["prompt"],
                    )
                

                    if outputText is not None:
                        appendCsvRow(
                            outputPath,
                            OUTPUT_COLUMNS,
                            {
                                "LLM_name": providerName,
                                "LLM_Input": prompt["prompt"],
                                "LLM_output": outputText,
                                "prompt_id": prompt["prompt_id"],
                                "category": prompt["category"],
                                "model_id": modelId,
                                "collected_at": utcTimestamp(),
                            },
                        )
                        completedPairs.add(pair)
                        successfulCount += 1
                        #print("  Saved successfully.")
                    else:
                        appendCsvRow(
                            FAILURES_PATH,
                            FAILURE_COLUMNS,
                            {
                                "prompt_id": prompt["prompt_id"],
                                "category": prompt["category"],
                                "LLM_name": providerName,
                                "model_id": modelId,
                                "failure_type": failureType,
                                "attempts": attempts,
                                "attempted_at": utcTimestamp(),
                            },
                        )
                        failureCount += 1
                        print(f"  Failed safely: {failureType}. No dataset row was written.")

            print("\nCollection run complete.")
            print(f"New responses saved: {successfulCount}")
            print(f"Failures recorded:   {failureCount}")
            printProgress(prompts, [], completedPairs, providerNames)
            if failureCount:
                print(f"Failure log:         {FAILURES_PATH}")
            return 0 if failureCount == 0 else 1
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Collection stopped: {error}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
