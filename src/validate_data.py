"""Validate the collected LLM response dataset without making API calls."""

import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATASET_PATH = PROJECT_ROOT / "data" / "new_responses.csv"
DEFAULT_PROMPTS_PATH = PROJECT_ROOT / "data" / "new_prompts.csv"

EXPECTED_FAMILIES = {"GPT", "Claude", "Gemini"}

EXPECTED_CATEGORIES = {
    "General knowledge",
    "Creative writing",
    "Programming",
    "Mathematical/logical reasoning",
    "Text transformation",
}
EXPECTED_CONSTRAINT_LEVELS = {"open", "moderate", "high"}

REQUIRED_COLUMNS = {
    "LLM_name",
    "LLM_Input",
    "LLM_output",
    "prompt_id",
    "category",
    "model_id",
    "collected_at",
}
PROMPT_COLUMNS = {"prompt_id", "category", "prompt", "constraint_level"}


def printReport(checks, datasetPath):
    print(f"Dataset validation: {datasetPath}")
    for checkName, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {checkName}: {detail}")
    overallPass = all(passed for _, passed, _ in checks)
    print(f"\nOverall result: {'PASS' if overallPass else 'FAIL'}")
    return overallPass


def validateDataset(datasetPath=DEFAULT_DATASET_PATH, promptsPath=DEFAULT_PROMPTS_PATH):
    datasetPath = Path(datasetPath)
    promptsPath = Path(promptsPath)
    checks = []

    if not datasetPath.exists():
        checks.append(("Dataset file", False, "file does not exist"))
        return printReport(checks, datasetPath)
    if not promptsPath.exists():
        checks.append(("Prompt source", False, "prompts.csv does not exist"))
        return printReport(checks, datasetPath)

    try:
        dataFrame = pd.read_csv(datasetPath, dtype=str, keep_default_na=False)
        promptsFrame = pd.read_csv(promptsPath, dtype=str, keep_default_na=False)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        checks.append(("Readable CSV files", False, error.__class__.__name__))
        return printReport(checks, datasetPath)

    missingColumns = sorted(REQUIRED_COLUMNS - set(dataFrame.columns))
    missingPromptColumns = sorted(PROMPT_COLUMNS - set(promptsFrame.columns))
    columnsPass = not missingColumns and not missingPromptColumns
    columnDetail = "all required columns are present"
    if missingColumns:
        columnDetail = f"dataset missing {', '.join(missingColumns)}"
    if missingPromptColumns:
        columnDetail += f"; prompts missing {', '.join(missingPromptColumns)}"
    checks.append(("Required columns", columnsPass, columnDetail))
    if not columnsPass:
        return printReport(checks, datasetPath)

    checks.append(
        (
            "Dataset rows",
            len(dataFrame) > 0,
            f"{len(dataFrame)} response rows found",
        )
    )

    emptyCounts = {
        column: int(dataFrame[column].str.strip().eq("").sum())
        for column in REQUIRED_COLUMNS
    }
    fieldsWithEmptyValues = {
        column: count for column, count in emptyCounts.items() if count > 0
    }
    checks.append(
        (
            "Non-empty values",
            not fieldsWithEmptyValues,
            "no required values are empty"
            if not fieldsWithEmptyValues
            else f"empty value counts: {fieldsWithEmptyValues}",
        )
    )

    duplicateMask = dataFrame.duplicated(subset=["prompt_id", "LLM_name"], keep=False)
    duplicatePairs = sorted(
        set(map(tuple, dataFrame.loc[duplicateMask, ["prompt_id", "LLM_name"]].values.tolist()))
    )
    checks.append(
        (
            "Duplicate records",
            not duplicatePairs,
            "no duplicate (prompt_id, LLM_name) pairs"
            if not duplicatePairs
            else f"duplicate pairs: {duplicatePairs}",
        )
    )

    observedFamilies = set(dataFrame["LLM_name"])
    unknownFamilies = sorted(observedFamilies - EXPECTED_FAMILIES)
    checks.append(
        (
            "LLM labels",
            not unknownFamilies,
            "labels are GPT, Claude, or Gemini"
            if not unknownFamilies
            else f"unknown labels: {unknownFamilies}",
        )
    )

    familyCounts = {
        family: int((dataFrame["LLM_name"] == family).sum())
        for family in sorted(EXPECTED_FAMILIES)
    }
    countsAreBalanced = len(set(familyCounts.values())) == 1
    checks.append(("Balanced family counts", countsAreBalanced, str(familyCounts)))

    promptSourceHasDuplicates = promptsFrame["prompt_id"].duplicated(keep=False).any()
    promptSourceHasEmptyValues = promptsFrame[list(PROMPT_COLUMNS)].apply(
        lambda column: column.str.strip().eq("").any()
    ).any()
    invalidCategories = sorted(set(promptsFrame["category"]) - EXPECTED_CATEGORIES)
    invalidConstraintLevels = sorted(
        set(promptsFrame["constraint_level"]) - EXPECTED_CONSTRAINT_LEVELS
    )
    promptSourcePass = (
        not promptSourceHasDuplicates
        and not promptSourceHasEmptyValues
        and not invalidCategories
        and not invalidConstraintLevels
    )
    promptSourceProblems = []
    if promptSourceHasDuplicates:
        promptSourceProblems.append("duplicate prompt IDs")
    if promptSourceHasEmptyValues:
        promptSourceProblems.append("empty prompt fields")
    if invalidCategories:
        promptSourceProblems.append(f"unexpected categories {invalidCategories}")
    if invalidConstraintLevels:
        promptSourceProblems.append(f"invalid constraint levels {invalidConstraintLevels}")
    checks.append(
        (
            "Prompt source",
            promptSourcePass,
            "prompt IDs and categories are valid"
            if promptSourcePass
            else "; ".join(promptSourceProblems),
        )
    )

    categoryCounts = {
        category: int((promptsFrame["category"] == category).sum())
        for category in sorted(EXPECTED_CATEGORIES)
    }
    checks.append(("Prompt category counts", not invalidCategories, str(categoryCounts)))

    constraintCounts = {
        level: int((promptsFrame["constraint_level"] == level).sum())
        for level in sorted(EXPECTED_CONSTRAINT_LEVELS)
    }
    checks.append(
        (
            "Prompt constraint levels",
            not invalidConstraintLevels and sum(constraintCounts.values()) == len(promptsFrame),
            str(constraintCounts),
        )
    )

    sourceById = promptsFrame.drop_duplicates("prompt_id").set_index("prompt_id")
    sourcePromptIds = set(sourceById.index)
    datasetPromptIds = set(dataFrame["prompt_id"])
    unknownPromptIds = sorted(datasetPromptIds - sourcePromptIds)
    missingPromptIds = sorted(sourcePromptIds - datasetPromptIds)

    inconsistentRows = []
    for rowIndex, row in dataFrame.iterrows():
        promptId = row["prompt_id"]
        if promptId not in sourceById.index:
            continue
        sourceRow = sourceById.loc[promptId]
        if row["category"] != sourceRow["category"] or row["LLM_Input"] != sourceRow["prompt"]:
            inconsistentRows.append(int(rowIndex) + 2)
    consistencyPass = not unknownPromptIds and not inconsistentRows
    consistencyDetail = "all response prompt IDs, categories, and inputs match prompts.csv"
    if not consistencyPass:
        consistencyDetail = (
            f"unknown prompt IDs: {unknownPromptIds}; inconsistent CSV rows: {inconsistentRows}"
        )
    checks.append(("Prompt consistency", consistencyPass, consistencyDetail))

    incompletePrompts = {}
    for promptId in sorted(sourcePromptIds):
        promptRows = dataFrame[dataFrame["prompt_id"] == promptId]
        promptFamilies = set(promptRows["LLM_name"])
        if promptFamilies != EXPECTED_FAMILIES or len(promptRows) != len(EXPECTED_FAMILIES):
            incompletePrompts[promptId] = {
                "present": sorted(promptFamilies & EXPECTED_FAMILIES),
                "row_count": len(promptRows),
            }
    completenessPass = not incompletePrompts and not missingPromptIds
    checks.append(
        (
            "Three responses per prompt",
            completenessPass,
            f"all {len(sourcePromptIds)} prompts have one response from each family"
            if completenessPass
            else f"incomplete prompt IDs: {sorted(incompletePrompts)}",
        )
    )

    modelCounts = {
        family: int(
            dataFrame.loc[dataFrame["LLM_name"] == family, "model_id"]
            .loc[lambda values: values.str.strip().ne("")]
            .nunique()
        )
        for family in sorted(EXPECTED_FAMILIES)
    }
    modelsAreConsistent = all(count == 1 for count in modelCounts.values())
    checks.append(
        (
            "One model per family",
            modelsAreConsistent,
            f"distinct model counts: {modelCounts}",
        )
    )

    parsedTimestamps = pd.to_datetime(
        dataFrame["collected_at"],
        format="ISO8601",
        utc=True,
        errors="coerce",
    )
    invalidTimestampCount = int(parsedTimestamps.isna().sum())
    checks.append(
        (
            "Collection timestamps",
            invalidTimestampCount == 0,
            "all timestamps are valid ISO 8601 values"
            if invalidTimestampCount == 0
            else f"{invalidTimestampCount} invalid timestamps",
        )
    )

    return printReport(checks, datasetPath)


def main():
    parser = argparse.ArgumentParser(description="Validate collected LLM response data.")
    parser.add_argument(
        "dataset",
        nargs="?",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="response CSV to validate (default: data/raw_responses.csv)",
    )
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="source prompts CSV (default: data/prompts.csv)",
    )
    arguments = parser.parse_args()
    return 0 if validateDataset(arguments.dataset, arguments.prompts) else 1


if __name__ == "__main__":
    raise SystemExit(main())
