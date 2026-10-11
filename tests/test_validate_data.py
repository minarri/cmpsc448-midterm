"""Local tests for dataset validation; no API requests are made."""

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from src.validate_data import REQUIRED_COLUMNS, validateDataset


class ValidateDataTests(unittest.TestCase):
    def writeCsv(self, path, fieldNames, rows):
        with path.open("w", encoding="utf-8", newline="") as csvFile:
            writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
            writer.writeheader()
            writer.writerows(rows)

    def createMockFiles(self, directory):
        promptsPath = directory / "prompts.csv"
        datasetPath = directory / "raw_responses.csv"
        promptText = "Explain why leaves are green."
        self.writeCsv(
            promptsPath,
            ["prompt_id", "category", "prompt"],
            [
                {
                    "prompt_id": "mock-001",
                    "category": "General knowledge",
                    "prompt": promptText,
                }
            ],
        )

        modelIds = {
            "GPT": "gpt-mock",
            "Claude": "claude-mock",
            "Gemini": "gemini-mock",
        }
        rows = [
            {
                "LLM_name": family,
                "LLM_Input": promptText,
                "LLM_output": f"Mock response from {family}.",
                "prompt_id": "mock-001",
                "category": "General knowledge",
                "model_id": modelId,
                "collected_at": "2026-10-08T12:00:00+00:00",
            }
            for family, modelId in modelIds.items()
        ]
        self.writeCsv(datasetPath, list(REQUIRED_COLUMNS), rows)
        return datasetPath, promptsPath, rows

    def testCompleteBalancedDatasetPasses(self):
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            directory = Path(temporaryDirectory)
            datasetPath, promptsPath, _ = self.createMockFiles(directory)
            with redirect_stdout(io.StringIO()):
                isValid = validateDataset(datasetPath, promptsPath)
            self.assertTrue(isValid)

    def testDuplicateAndMixedModelDatasetFails(self):
        with tempfile.TemporaryDirectory() as temporaryDirectory:
            directory = Path(temporaryDirectory)
            datasetPath, promptsPath, rows = self.createMockFiles(directory)
            duplicateRow = dict(rows[0])
            duplicateRow["model_id"] = "another-gpt-model"
            self.writeCsv(datasetPath, list(REQUIRED_COLUMNS), rows + [duplicateRow])
            with redirect_stdout(io.StringIO()):
                isValid = validateDataset(datasetPath, promptsPath)
            self.assertFalse(isValid)


if __name__ == "__main__":
    unittest.main()
