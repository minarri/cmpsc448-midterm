r"""Analyze descriptive linguistic fingerprints in collected LLM responses.

Feature definitions:
- Words are lowercased ``\w+`` tokens from the training tokenizer. This means
  contractions are split at apostrophes, and numbers/underscores count as words.
- ``token_count`` uses the same regex as ``src/train_models.py`` and therefore
  includes punctuation and Markdown markers as separate tokens.
- Sentences are approximated by splitting at sentence-ending punctuation or
  blank lines. This heuristic can miscount abbreviations, lists, and source code,
  especially in programming responses.
- Markdown headings, bullet items, and bold spans are counted only outside
  fenced code blocks. A fenced block is counted when an opening fence is found,
  including an unclosed final fence.
- Standard deviations are sample standard deviations (pandas ``ddof=1``).
- Percentages in the summary CSV use a 0-to-100 scale.

The analysis is descriptive. Observed differences do not establish which
features the CNN or LSTM learned.
"""

import math
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw_responses.csv"
NEW_DATA_PATH = PROJECT_ROOT / "data" / "new_responses.csv"
RESULTS_PATH = PROJECT_ROOT / "results"
FEATURES_PATH = RESULTS_PATH / "rq4_response_features.csv"
SUMMARY_PATH = RESULTS_PATH / "rq4_fingerprint_summary.csv"

MODEL_ORDER = ["GPT", "Claude", "Gemini"]
CATEGORY_ORDER = [
    "General knowledge",
    "Creative writing",
    "Programming",
    "Mathematical/logical reasoning",
    "Text transformation",
]
IDENTIFIER_COLUMNS = ["prompt_id", "category", "LLM_name"]
FEATURE_COLUMNS = [
    "word_count",
    "token_count",
    "unique_word_count",
    "type_token_ratio",
    "sentence_count",
    "avg_sentence_length",
    "markdown_heading_count",
    "bullet_list_count",
    "bold_marker_count",
    "code_block_count",
]
FORMATTING_FEATURES = [
    "markdown_heading_count",
    "bullet_list_count",
    "bold_marker_count",
    "code_block_count",
]
COUNT_FEATURES = [
    "word_count",
    "token_count",
    "unique_word_count",
    "sentence_count",
] + FORMATTING_FEATURES
MODEL_PAIRS = [
    ("GPT", "Claude"),
    ("GPT", "Gemini"),
    ("Claude", "Gemini"),
]

# This is exactly the tokenizer pattern used by src/train_models.py.
TOKEN_PATTERN = re.compile(r"\[sep\]|\w+|[^\w\s]", flags=re.UNICODE)
WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)
SENTENCE_BOUNDARY_PATTERN = re.compile(
    r"[.!?]+(?:[\"')\]}]+)?(?:\s+|$)|\n\s*\n",
    flags=re.UNICODE,
)
FENCE_OPEN_PATTERN = re.compile(r"^[ \t]{0,3}(`{3,}|~{3,})(.*)$")
MARKDOWN_HEADING_PATTERN = re.compile(r"^[ \t]{0,3}#{1,6}(?:[ \t]+|$)", re.MULTILINE)
BULLET_ITEM_PATTERN = re.compile(r"^[ \t]{0,3}[-+*][ \t]+\S", re.MULTILINE)
BOLD_SPAN_PATTERN = re.compile(r"(\*\*|__)(?=\S)[^\n]+?(?<=\S)\1")


def maskFencedCodeBlocks(text):
    """Return text with fenced blocks masked by newlines and the block count."""
    outsideLines = []
    insideFence = False
    fenceCharacter = ""
    fenceLength = 0
    codeBlockCount = 0

    for line in text.splitlines(keepends=True):
        lineWithoutEnding = line.rstrip("\r\n")
        if not insideFence:
            openingMatch = FENCE_OPEN_PATTERN.match(lineWithoutEnding)
            if openingMatch:
                fence = openingMatch.group(1)
                insideFence = True
                fenceCharacter = fence[0]
                fenceLength = len(fence)
                codeBlockCount += 1
                outsideLines.append("\n")
            else:
                outsideLines.append(line)
            continue

        closingPattern = rf"^[ \t]{{0,3}}{re.escape(fenceCharacter)}{{{fenceLength},}}[ \t]*$"
        if re.match(closingPattern, lineWithoutEnding):
            insideFence = False
        outsideLines.append("\n")

    return "".join(outsideLines), codeBlockCount


def calculateResponseFeatures(text):
    """Calculate all requested features for one response string."""
    safeText = "" if text is None else str(text)
    tokens = TOKEN_PATTERN.findall(safeText.lower())
    words = [token for token in tokens if WORD_PATTERN.fullmatch(token)]
    wordCount = len(words)
    uniqueWordCount = len(set(words))

    sentenceParts = [
        part
        for part in SENTENCE_BOUNDARY_PATTERN.split(safeText.strip())
        if WORD_PATTERN.search(part)
    ]
    sentenceCount = len(sentenceParts)
    if wordCount and sentenceCount == 0:
        sentenceCount = 1

    outsideCodeText, codeBlockCount = maskFencedCodeBlocks(safeText)
    return {
        "word_count": wordCount,
        "token_count": len(tokens),
        "unique_word_count": uniqueWordCount,
        "type_token_ratio": uniqueWordCount / wordCount if wordCount else 0.0,
        "sentence_count": sentenceCount,
        "avg_sentence_length": wordCount / sentenceCount if sentenceCount else 0.0,
        "markdown_heading_count": len(
            MARKDOWN_HEADING_PATTERN.findall(outsideCodeText)
        ),
        "bullet_list_count": len(BULLET_ITEM_PATTERN.findall(outsideCodeText)),
        "bold_marker_count": len(BOLD_SPAN_PATTERN.findall(outsideCodeText)),
        "code_block_count": codeBlockCount,
    }


def loadAndValidateDataset():
    requiredColumns = set(IDENTIFIER_COLUMNS + ["LLM_Input", "LLM_output"])
    dataFrame = pd.concat(
        [
            pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False),
            pd.read_csv(NEW_DATA_PATH, dtype=str, keep_default_na=False),
        ],
        ignore_index=True,
    )
    missingColumns = sorted(requiredColumns - set(dataFrame.columns))
    if missingColumns:
        raise ValueError(
            f"raw_responses.csv is missing columns: {', '.join(missingColumns)}"
        )
    if len(dataFrame) != 3300:
        raise ValueError(f"Expected 3300 responses, found {len(dataFrame)}.")

    modelCounts = dataFrame["LLM_name"].value_counts().to_dict()
    if set(modelCounts) != set(MODEL_ORDER) or any(
        modelCounts.get(modelName) != 1100 for modelName in MODEL_ORDER
    ):
        raise ValueError(f"Expected 1100 responses per LLM, found {modelCounts}.")

    promptGroups = dataFrame.groupby("prompt_id", sort=False)
    if dataFrame["prompt_id"].nunique() != 1100:
        raise ValueError(
            f"Expected 1100 prompt IDs, found {dataFrame['prompt_id'].nunique()}."
        )
    for promptId, promptRows in promptGroups:
        if len(promptRows) != 3 or set(promptRows["LLM_name"]) != set(MODEL_ORDER):
            raise ValueError(
                f"Prompt {promptId} does not have one response from every LLM."
            )
        if promptRows["category"].nunique() != 1:
            raise ValueError(f"Prompt {promptId} has inconsistent categories.")

    if set(dataFrame["category"]) != set(CATEGORY_ORDER):
        raise ValueError(
            "Dataset categories do not match the five expected project categories."
        )
    categoryModelCounts = pd.crosstab(
        dataFrame["category"], dataFrame["LLM_name"]
    ).reindex(index=CATEGORY_ORDER, columns=MODEL_ORDER, fill_value=0)
    if not categoryModelCounts.eq(220).all().all():
        raise ValueError(
            "Expected 220 responses per model in each category, found:\n"
            f"{categoryModelCounts.to_string()}"
        )
    return dataFrame


def buildFeatureTable(dataFrame):
    featureRows = []
    for record in dataFrame.to_dict(orient="records"):
        featureRows.append(
            {
                "prompt_id": record["prompt_id"],
                "category": record["category"],
                "LLM_name": record["LLM_name"],
                **calculateResponseFeatures(record["LLM_output"]),
            }
        )
    return pd.DataFrame(featureRows, columns=IDENTIFIER_COLUMNS + FEATURE_COLUMNS)


def validateFeatureTable(featureTable):
    if len(featureTable) != 3300:
        raise ValueError(f"Expected 3300 feature rows, found {len(featureTable)}.")

    numericValues = featureTable[FEATURE_COLUMNS]
    if numericValues.isna().any().any():
        raise ValueError("At least one calculated feature is NaN.")
    if not numericValues.map(lambda value: math.isfinite(float(value))).all().all():
        raise ValueError("At least one calculated feature is infinite.")
    if numericValues.lt(0).any().any():
        raise ValueError("Calculated feature values must be nonnegative.")
    if not featureTable[COUNT_FEATURES].map(
        lambda value: float(value).is_integer()
    ).all().all():
        raise ValueError("Count features must contain integer values.")
    if featureTable["type_token_ratio"].gt(1).any():
        raise ValueError("type_token_ratio cannot exceed 1.")


def appendDescriptiveRows(summaryRows, groupFrame, scope, category):
    for modelName in MODEL_ORDER:
        modelFrame = groupFrame[groupFrame["LLM_name"] == modelName]
        for featureName in FEATURE_COLUMNS:
            featureValues = modelFrame[featureName]
            statistics = {
                "mean": featureValues.mean(),
                "median": featureValues.median(),
                "sample_standard_deviation": featureValues.std(ddof=1),
            }
            if featureName in FORMATTING_FEATURES:
                statistics["responses_with_feature_percent"] = (
                    featureValues.gt(0).mean() * 100
                )
            for statisticName, value in statistics.items():
                summaryRows.append(
                    {
                        "row_type": "descriptive_statistic",
                        "scope": scope,
                        "category": category,
                        "model": modelName,
                        "first_model": "",
                        "second_model": "",
                        "feature": featureName,
                        "statistic": statisticName,
                        "value": value,
                        "n": len(featureValues),
                    }
                )


def appendMatchedPromptRows(summaryRows, featureTable, scope, category):
    promptWordCounts = featureTable.pivot(
        index=["prompt_id", "category"],
        columns="LLM_name",
        values="word_count",
    )
    if category != "All":
        promptWordCounts = promptWordCounts[
            promptWordCounts.index.get_level_values("category") == category
        ]

    for firstModel, secondModel in MODEL_PAIRS:
        differences = promptWordCounts[firstModel] - promptWordCounts[secondModel]
        statistics = {
            "mean_paired_difference": differences.mean(),
            "median_paired_difference": differences.median(),
            "first_model_longer_percent": differences.gt(0).mean() * 100,
        }
        for statisticName, value in statistics.items():
            summaryRows.append(
                {
                    "row_type": "matched_prompt_comparison",
                    "scope": scope,
                    "category": category,
                    "model": "",
                    "first_model": firstModel,
                    "second_model": secondModel,
                    "feature": "word_count_difference",
                    "statistic": statisticName,
                    "value": value,
                    "n": len(differences),
                }
            )


def buildSummaryTable(featureTable):
    summaryRows = []
    appendDescriptiveRows(summaryRows, featureTable, "overall", "All")
    appendMatchedPromptRows(summaryRows, featureTable, "overall", "All")

    for category in CATEGORY_ORDER:
        categoryFrame = featureTable[featureTable["category"] == category]
        appendDescriptiveRows(summaryRows, categoryFrame, "category", category)
        appendMatchedPromptRows(
            summaryRows, featureTable, "category", category
        )

    summaryTable = pd.DataFrame(summaryRows)
    summaryTable["value"] = summaryTable["value"].round(6)
    if summaryTable[["value", "n"]].isna().any().any():
        raise ValueError("The summary contains a missing numeric value.")
    if not summaryTable["value"].map(
        lambda value: math.isfinite(float(value))
    ).all():
        raise ValueError("The summary contains an infinite value.")
    return summaryTable


def printCompactSummary(featureTable):
    print("\nRQ4 linguistic fingerprint summary")
    print(
        "Values: mean/median words, mean type-token ratio, mean words per "
        "sentence, and formatting presence rates."
    )
    groups = [("Overall", featureTable)] + [
        (category, featureTable[featureTable["category"] == category])
        for category in CATEGORY_ORDER
    ]
    for groupName, groupFrame in groups:
        print(f"\n{groupName}")
        print(
            f"{'Model':<8}{'Words mean/median':>20}{'TTR mean':>11}"
            f"{'Words/sent':>12}{'Heading%':>11}{'Bullet%':>10}"
            f"{'Bold%':>9}{'Code%':>9}"
        )
        for modelName in MODEL_ORDER:
            modelFrame = groupFrame[groupFrame["LLM_name"] == modelName]
            print(
                f"{modelName:<8}"
                f"{modelFrame['word_count'].mean():>9.1f}/"
                f"{modelFrame['word_count'].median():<10.1f}"
                f"{modelFrame['type_token_ratio'].mean():>11.3f}"
                f"{modelFrame['avg_sentence_length'].mean():>12.1f}"
                f"{modelFrame['markdown_heading_count'].gt(0).mean() * 100:>10.1f}%"
                f"{modelFrame['bullet_list_count'].gt(0).mean() * 100:>9.1f}%"
                f"{modelFrame['bold_marker_count'].gt(0).mean() * 100:>8.1f}%"
                f"{modelFrame['code_block_count'].gt(0).mean() * 100:>8.1f}%"
            )


def printMatchedPromptSummary(featureTable):
    print("\nMatched-prompt word-count comparisons")
    groups = [("Overall", "All")] + [
        (category, category) for category in CATEGORY_ORDER
    ]
    promptWordCounts = featureTable.pivot(
        index=["prompt_id", "category"],
        columns="LLM_name",
        values="word_count",
    )
    for groupName, category in groups:
        groupCounts = promptWordCounts
        if category != "All":
            groupCounts = promptWordCounts[
                promptWordCounts.index.get_level_values("category") == category
            ]
        print(f"\n{groupName}")
        for firstModel, secondModel in MODEL_PAIRS:
            differences = groupCounts[firstModel] - groupCounts[secondModel]
            print(
                f"  {firstModel} vs {secondModel}: "
                f"mean difference={differences.mean():.1f}, "
                f"median difference={differences.median():.1f}, "
                f"first model longer={differences.gt(0).mean() * 100:.1f}%"
            )


def main():
    if not RESULTS_PATH.is_dir():
        raise FileNotFoundError(f"Existing results directory not found: {RESULTS_PATH}")

    dataFrame = loadAndValidateDataset()
    featureTable = buildFeatureTable(dataFrame)
    validateFeatureTable(featureTable)
    summaryTable = buildSummaryTable(featureTable)

    # All validation completes before either requested result file is written.
    featureTable.to_csv(FEATURES_PATH, index=False)
    summaryTable.to_csv(SUMMARY_PATH, index=False)

    print("Validated 3300 responses, 1100 per LLM, with complete prompt matching.")
    printCompactSummary(featureTable)
    printMatchedPromptSummary(featureTable)
    print("\nDescriptive differences do not establish which features the classifiers learned.")
    print(f"\nSaved: {FEATURES_PATH}")
    print(f"Saved: {SUMMARY_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
