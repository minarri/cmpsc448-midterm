"""Tests for retry and safe error-classification helpers; no APIs are called."""

import unittest
from unittest.mock import patch

from src import collect_data, test_apis


class FakeHttpError(Exception):
    def __init__(self, statusCode, message="provider error"):
        super().__init__(message)
        self.status_code = statusCode


class FakeGoogleError(Exception):
    def __init__(self, code, message="provider error"):
        super().__init__(message)
        self.code = code


class PipelineHelperTests(unittest.TestCase):
    def testTestScriptClassifiesCommonFailures(self):
        self.assertIn("authentication", test_apis.classifyApiError(FakeHttpError(401)))
        self.assertIn("rate limit", test_apis.classifyApiError(FakeGoogleError(429)))
        self.assertIn("billing", test_apis.classifyApiError(FakeHttpError(429, "billing limit")))
        self.assertIn("unavailable", test_apis.classifyApiError(FakeGoogleError(404)))
        self.assertIn("network", test_apis.classifyApiError(ConnectionError()))

    def testCollectorRetriesTemporaryFailureThenSucceeds(self):
        callCount = 0

        def temporaryFailureThenSuccess(apiKey, modelId, promptText):
            nonlocal callCount
            callCount += 1
            if callCount == 1:
                raise FakeHttpError(503)
            return "Mock successful response"

        with patch("src.collect_data.time.sleep") as sleepMock:
            outputText, failureType, attempts = collect_data.collectWithRetries(
                temporaryFailureThenSuccess,
                "fake-key",
                "fake-model",
                "fake prompt",
            )

        self.assertEqual(outputText, "Mock successful response")
        self.assertIsNone(failureType)
        self.assertEqual(attempts, 2)
        sleepMock.assert_called_once_with(2)

    def testCollectorRejectsModelMixing(self):
        existingRecords = [
            {"LLM_name": "GPT", "model_id": "first-model"},
            {"LLM_name": "GPT", "model_id": "second-model"},
        ]
        with self.assertRaises(ValueError):
            collect_data.verifyModelConsistency(
                existingRecords,
                {"GPT": "first-model", "Claude": "c", "Gemini": "g"},
            )


if __name__ == "__main__":
    unittest.main()
