"""Train TextCNN and TextLSTM classifiers for LLM authorship attribution."""

import argparse
import copy
import csv
import random
import re
import statistics
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "raw_responses.csv"
NEW_DATA_PATH = PROJECT_ROOT / "data" / "new_responses.csv"
RESULTS_PATH = PROJECT_ROOT / "results"

LABEL_MAPPING = {"GPT": 0, "Claude": 1, "Gemini": 2}
LABEL_NAMES = ["GPT", "Claude", "Gemini"]
INPUT_TYPES = ["output_only", "input_only", "input_output"]

PAD_TOKEN = "<pad>"
UNKNOWN_TOKEN = "<unk>"
SEPARATOR_TOKEN = "[sep]"
PAD_ID = 0
UNKNOWN_ID = 1
MINIMUM_PADDED_LENGTH = 5
TOKEN_PATTERN = re.compile(r"\[sep\]|\w+|[^\w\s]", flags=re.UNICODE)

SEED = 42
EMBEDDING_DIM = 64
DROPOUT = 0.3
DEFAULT_LEARNING_RATE = 0.001
DEFAULT_MAX_LENGTH = 512
CNN_FILTERS = 64
CNN_KERNEL_SIZES = (3, 4, 5)
LSTM_HIDDEN_DIM = 96
EARLY_STOPPING_PATIENCE = 4


def setRandomSeeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def resolveDevice(requestedDevice):
    if requestedDevice == "cpu":
        return torch.device("cpu")
    if requestedDevice == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available in this PyTorch installation.")
        return torch.device("mps")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loadAndVerifyDataset():
    requiredColumns = {
        "LLM_name",
        "LLM_Input",
        "LLM_output",
        "prompt_id",
        "category",
    }
    dataFrame = pd.concat(
        [
            pd.read_csv(DATA_PATH, dtype=str, keep_default_na=False),
            pd.read_csv(NEW_DATA_PATH, dtype=str, keep_default_na=False),
        ],
        ignore_index=True,
    )
    missingColumns = sorted(requiredColumns - set(dataFrame.columns))
    if missingColumns:
        raise ValueError(f"raw_responses.csv is missing columns: {', '.join(missingColumns)}")
    if len(dataFrame) != 3300:
        raise ValueError(f"Expected 3300 responses, found {len(dataFrame)}.")

    observedLabels = set(dataFrame["LLM_name"])
    if observedLabels != set(LABEL_MAPPING):
        raise ValueError(f"Expected labels {set(LABEL_MAPPING)}, found {observedLabels}.")
    if dataFrame["LLM_output"].str.strip().eq("").any():
        raise ValueError("At least one LLM_output value is empty.")

    labelCounts = Counter(dataFrame["LLM_name"])
    if any(labelCounts[labelName] != 1100 for labelName in LABEL_NAMES):
        raise ValueError(f"Expected 1100 responses per label, found {dict(labelCounts)}.")

    for promptId, promptRows in dataFrame.groupby("prompt_id", sort=False):
        if len(promptRows) != 3 or set(promptRows["LLM_name"]) != set(LABEL_MAPPING):
            raise ValueError(f"Prompt {promptId} does not have one response from each LLM.")
        if promptRows["category"].nunique() != 1:
            raise ValueError(f"Prompt {promptId} has inconsistent categories.")
        if promptRows["LLM_Input"].nunique() != 1:
            raise ValueError(f"Prompt {promptId} has inconsistent LLM_Input values.")

    promptTable = dataFrame[["prompt_id", "category"]].drop_duplicates()
    if len(promptTable) != 1100:
        raise ValueError(f"Expected 1100 unique prompts, found {len(promptTable)}.")
    categoryCounts = Counter(promptTable["category"])
    if len(categoryCounts) != 5 or any(count != 220 for count in categoryCounts.values()):
        raise ValueError(f"Expected five categories with 220 prompts each, found {dict(categoryCounts)}.")

    print("Dataset verification: 3300 responses, 1100 prompts, and three balanced labels.")
    print(f"Prompt categories: {dict(sorted(categoryCounts.items()))}")
    return dataFrame


def createPromptSplits(dataFrame, seed):
    promptTable = dataFrame[["prompt_id", "category"]].drop_duplicates()
    randomGenerator = random.Random(seed)
    trainIds = set()
    validationIds = set()
    testIds = set()

    for category in sorted(promptTable["category"].unique()):
        categoryIds = sorted(
            promptTable.loc[promptTable["category"] == category, "prompt_id"].tolist()
        )
        if len(categoryIds) != 220:
            raise ValueError(f"Category {category} has {len(categoryIds)} prompts instead of 220.")
        randomGenerator.shuffle(categoryIds)
        trainIds.update(categoryIds[:154])
        validationIds.update(categoryIds[154:187])
        testIds.update(categoryIds[187:220])

    if trainIds & validationIds or trainIds & testIds or validationIds & testIds:
        raise AssertionError("A prompt_id appears in more than one data split.")
    allPromptIds = set(promptTable["prompt_id"])
    if trainIds | validationIds | testIds != allPromptIds:
        raise AssertionError("The prompt split does not cover every prompt_id.")
    if (len(trainIds), len(validationIds), len(testIds)) != (770, 165, 165):
        raise AssertionError("Expected 770/165/165 prompt IDs in the data split.")

    splitFrames = {
        "train": dataFrame[dataFrame["prompt_id"].isin(trainIds)].reset_index(drop=True),
        "validation": dataFrame[dataFrame["prompt_id"].isin(validationIds)].reset_index(drop=True),
        "test": dataFrame[dataFrame["prompt_id"].isin(testIds)].reset_index(drop=True),
    }
    expectedRows = {"train": 2310, "validation": 495, "test": 495}
    for splitName, splitFrame in splitFrames.items():
        if len(splitFrame) != expectedRows[splitName]:
            raise AssertionError(
                f"Expected {expectedRows[splitName]} {splitName} responses, found {len(splitFrame)}."
            )

    print("Prompt split: 770 train / 165 validation / 165 test (no prompt overlap).")
    print("Each category contributes 154/33/33 prompts to train/validation/test.")
    return splitFrames


def createCrossCategorySplits(dataFrame, heldOutCategory, seed):
    promptTable = dataFrame[["prompt_id", "category"]].drop_duplicates()
    categories = set(promptTable["category"])
    if heldOutCategory not in categories:
        raise ValueError(
            f"Unknown held-out category: {heldOutCategory}. "
            f"Choose from: {', '.join(sorted(categories))}."
        )

    randomGenerator = random.Random(seed)
    trainIds = set()
    validationIds = set()
    testIds = set(
        promptTable.loc[
            promptTable["category"] == heldOutCategory, "prompt_id"
        ].tolist()
    )
    trainingCategories = sorted(categories - {heldOutCategory})
    if len(trainingCategories) != 4 or len(testIds) != 220:
        raise AssertionError("RQ3 requires four training categories and 220 held-out prompts.")

    for category in trainingCategories:
        categoryIds = sorted(
            promptTable.loc[promptTable["category"] == category, "prompt_id"].tolist()
        )
        if len(categoryIds) != 220:
            raise ValueError(f"Category {category} has {len(categoryIds)} prompts instead of 220.")
        randomGenerator.shuffle(categoryIds)
        trainIds.update(categoryIds[:176])
        validationIds.update(categoryIds[176:])

    if trainIds & validationIds or trainIds & testIds or validationIds & testIds:
        raise AssertionError("A prompt_id appears in more than one RQ3 data split.")
    if trainIds | validationIds | testIds != set(promptTable["prompt_id"]):
        raise AssertionError("The RQ3 split does not cover every prompt_id.")

    splitFrames = {
        "train": dataFrame[dataFrame["prompt_id"].isin(trainIds)].reset_index(drop=True),
        "validation": dataFrame[
            dataFrame["prompt_id"].isin(validationIds)
        ].reset_index(drop=True),
        "test": dataFrame[dataFrame["prompt_id"].isin(testIds)].reset_index(drop=True),
    }
    expectedPrompts = {"train": 704, "validation": 176, "test": 220}
    expectedRows = {"train": 2112, "validation": 528, "test": 660}
    for splitName, splitFrame in splitFrames.items():
        if splitFrame["prompt_id"].nunique() != expectedPrompts[splitName]:
            raise AssertionError(
                f"Expected {expectedPrompts[splitName]} RQ3 {splitName} prompts."
            )
        if len(splitFrame) != expectedRows[splitName]:
            raise AssertionError(
                f"Expected {expectedRows[splitName]} RQ3 {splitName} responses."
            )

    if heldOutCategory in set(splitFrames["train"]["category"]):
        raise AssertionError("The held-out category entered the RQ3 training split.")
    if heldOutCategory in set(splitFrames["validation"]["category"]):
        raise AssertionError("The held-out category entered the RQ3 validation split.")
    if set(splitFrames["test"]["category"]) != {heldOutCategory}:
        raise AssertionError("The RQ3 test split contains a non-held-out category.")

    print(
        "RQ3 prompt split: 704 train / 176 validation / 220 held-out test "
        "(no prompt overlap)."
    )
    print(f"RQ3 held-out category: {heldOutCategory}")
    print(f"RQ3 training/validation categories: {', '.join(trainingCategories)}")
    print("Verified: held-out category is absent from training and validation.")
    return splitFrames


def tokenizeText(text):
    return TOKEN_PATTERN.findall(text.lower())


def selectInputText(record, inputType):
    if inputType == "output_only":
        return record["LLM_output"]
    if inputType == "input_only":
        return record["LLM_Input"]
    if inputType == "input_output":
        return f"{record['LLM_Input']} {SEPARATOR_TOKEN} {record['LLM_output']}"
    raise ValueError(f"Unknown input type: {inputType}")


def buildVocabulary(trainingTexts):
    tokenCounts = Counter()
    for text in trainingTexts:
        tokenCounts.update(tokenizeText(text))

    vocabulary = {PAD_TOKEN: PAD_ID, UNKNOWN_TOKEN: UNKNOWN_ID, SEPARATOR_TOKEN: 2}
    sortedTokens = sorted(tokenCounts.items(), key=lambda item: (-item[1], item[0]))
    for token, _ in sortedTokens:
        if token not in vocabulary:
            vocabulary[token] = len(vocabulary)
    return vocabulary


def encodeText(text, vocabulary, maxLength):
    tokenIds = [vocabulary.get(token, UNKNOWN_ID) for token in tokenizeText(text)[:maxLength]]
    return tokenIds or [UNKNOWN_ID]


class LlmTextDataset(Dataset):
    def __init__(self, dataFrame, inputType, vocabulary, maxLength):
        self.examples = []
        for record in dataFrame.to_dict(orient="records"):
            inputText = selectInputText(record, inputType)
            tokenIds = encodeText(inputText, vocabulary, maxLength)
            labelId = LABEL_MAPPING[record["LLM_name"]]
            self.examples.append((tokenIds, labelId))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collateBatch(batch):
    tokenLists, labelIds = zip(*batch)
    lengths = torch.tensor([len(tokenIds) for tokenIds in tokenLists], dtype=torch.long)
    paddedLength = max(int(lengths.max()), MINIMUM_PADDED_LENGTH)
    inputIds = torch.full((len(batch), paddedLength), PAD_ID, dtype=torch.long)
    for rowIndex, tokenIds in enumerate(tokenLists):
        inputIds[rowIndex, : len(tokenIds)] = torch.tensor(tokenIds, dtype=torch.long)
    labels = torch.tensor(labelIds, dtype=torch.long)
    return inputIds, lengths, labels


def createDataLoaders(splitFrames, inputType, batchSize, maxLength, seed):
    trainingRecords = splitFrames["train"].to_dict(orient="records")
    trainingTexts = [selectInputText(record, inputType) for record in trainingRecords]

    # Only training text is used to fit this experiment's vocabulary.
    vocabulary = buildVocabulary(trainingTexts)
    datasets = {
        splitName: LlmTextDataset(splitFrame, inputType, vocabulary, maxLength)
        for splitName, splitFrame in splitFrames.items()
    }
    shuffleGenerator = torch.Generator().manual_seed(seed)
    loaders = {
        "train": DataLoader(
            datasets["train"],
            batch_size=batchSize,
            shuffle=True,
            collate_fn=collateBatch,
            generator=shuffleGenerator,
            num_workers=0,
        ),
        "train_evaluation": DataLoader(
            datasets["train"],
            batch_size=batchSize,
            shuffle=False,
            collate_fn=collateBatch,
            num_workers=0,
        ),
        "validation": DataLoader(
            datasets["validation"],
            batch_size=batchSize,
            shuffle=False,
            collate_fn=collateBatch,
            num_workers=0,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batchSize,
            shuffle=False,
            collate_fn=collateBatch,
            num_workers=0,
        ),
    }
    return loaders, vocabulary


def calculatePercentile(values, percentile):
    sortedValues = sorted(values)
    position = (len(sortedValues) - 1) * percentile / 100
    lowerIndex = int(position)
    upperIndex = min(lowerIndex + 1, len(sortedValues) - 1)
    fraction = position - lowerIndex
    return sortedValues[lowerIndex] + fraction * (
        sortedValues[upperIndex] - sortedValues[lowerIndex]
    )


def printLengthDiagnostics(splitFrames, maxLength):
    print(f"\nToken lengths before truncation (current maximum: {maxLength})")
    for inputType in INPUT_TYPES:
        print(f"\nInput type: {inputType}")
        for splitName in ("train", "validation", "test"):
            records = splitFrames[splitName].to_dict(orient="records")
            lengths = [
                len(tokenizeText(selectInputText(record, inputType)))
                for record in records
            ]
            exceedingCount = sum(length > maxLength for length in lengths)
            exceedingPercent = 100 * exceedingCount / len(lengths)
            print(
                f"  {splitName:<10} examples={len(lengths):>3}  "
                f"mean={statistics.mean(lengths):>7.2f}  "
                f"median={statistics.median(lengths):>6.1f}  "
                f"p90={calculatePercentile(lengths, 90):>6.1f}  "
                f"p95={calculatePercentile(lengths, 95):>6.1f}  "
                f"max={max(lengths):>4}  "
                f">{maxLength}={exceedingCount:>3} ({exceedingPercent:>5.1f}%)"
            )


class TextCNN(nn.Module):
    def __init__(
        self,
        vocabularySize,
        embeddingDim=EMBEDDING_DIM,
        numberOfFilters=CNN_FILTERS,
        kernelSizes=CNN_KERNEL_SIZES,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabularySize, embeddingDim, padding_idx=PAD_ID)
        self.convolutions = nn.ModuleList(
            nn.Conv1d(embeddingDim, numberOfFilters, kernelSize)
            for kernelSize in kernelSizes
        )
        self.dropout = nn.Dropout(dropout)
        self.outputLayer = nn.Linear(numberOfFilters * len(kernelSizes), len(LABEL_NAMES))

    def forward(self, inputIds, lengths):
        del lengths  # The CNN uses global pooling over the padded batch width.
        embedded = self.embedding(inputIds).transpose(1, 2)
        pooledOutputs = []
        for convolution in self.convolutions:
            activated = torch.relu(convolution(embedded))
            pooledOutputs.append(torch.max(activated, dim=2).values)
        combined = torch.cat(pooledOutputs, dim=1)
        return self.outputLayer(self.dropout(combined))


class TextLSTM(nn.Module):
    def __init__(
        self,
        vocabularySize,
        embeddingDim=EMBEDDING_DIM,
        hiddenDim=LSTM_HIDDEN_DIM,
        dropout=DROPOUT,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocabularySize, embeddingDim, padding_idx=PAD_ID)
        self.lstm = nn.LSTM(embeddingDim, hiddenDim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.outputLayer = nn.Linear(hiddenDim, len(LABEL_NAMES))

    def forward(self, inputIds, lengths):
        embedded = self.embedding(inputIds)
        packedInput = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hiddenState, _) = self.lstm(packedInput)
        finalHiddenState = hiddenState[-1]
        return self.outputLayer(self.dropout(finalHiddenState))


def runTrainingEpoch(model, dataLoader, lossFunction, optimizer, device):
    model.train()
    totalLoss = 0.0
    correctPredictions = 0
    exampleCount = 0

    for inputIds, lengths, labels in dataLoader:
        inputIds = inputIds.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputIds, lengths)
        loss = lossFunction(logits, labels)
        loss.backward()
        optimizer.step()

        batchSize = labels.size(0)
        totalLoss += loss.item() * batchSize
        correctPredictions += (logits.argmax(dim=1) == labels).sum().item()
        exampleCount += batchSize

    return totalLoss / exampleCount, correctPredictions / exampleCount


def evaluateModel(model, dataLoader, lossFunction, device):
    model.eval()
    totalLoss = 0.0
    correctPredictions = 0
    exampleCount = 0
    trueLabels = []
    predictedLabels = []

    with torch.no_grad():
        for inputIds, lengths, labels in dataLoader:
            inputIds = inputIds.to(device)
            labels = labels.to(device)
            logits = model(inputIds, lengths)
            loss = lossFunction(logits, labels)

            predictions = logits.argmax(dim=1)
            batchSize = labels.size(0)
            totalLoss += loss.item() * batchSize
            correctPredictions += (predictions == labels).sum().item()
            exampleCount += batchSize
            trueLabels.extend(labels.cpu().tolist())
            predictedLabels.extend(predictions.cpu().tolist())

    return {
        "loss": totalLoss / exampleCount,
        "accuracy": correctPredictions / exampleCount,
        "true_labels": trueLabels,
        "predicted_labels": predictedLabels,
    }


def verifyForwardShape(model, dataLoader, device):
    inputIds, lengths, _ = next(iter(dataLoader))
    model.eval()
    with torch.no_grad():
        logits = model(inputIds.to(device), lengths)
    expectedShape = (inputIds.size(0), len(LABEL_NAMES))
    if tuple(logits.shape) != expectedShape:
        raise AssertionError(f"Expected forward shape {expectedShape}, found {tuple(logits.shape)}.")


def trainWithEarlyStopping(model, loaders, epochs, learningRate, patience, device):
    lossFunction = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)
    bestValidationLoss = float("inf")
    bestModelState = None
    bestEpoch = 0
    epochsWithoutImprovement = 0
    history = []

    for epoch in range(1, epochs + 1):
        trainLoss, trainAccuracy = runTrainingEpoch(
            model, loaders["train"], lossFunction, optimizer, device
        )
        validationResults = evaluateModel(model, loaders["validation"], lossFunction, device)
        history.append(
            {
                "epoch": epoch,
                "train_loss": trainLoss,
                "train_accuracy": trainAccuracy,
                "validation_loss": validationResults["loss"],
                "validation_accuracy": validationResults["accuracy"],
            }
        )
        print(
            f"  Epoch {epoch:02d}: train loss={trainLoss:.4f}, "
            f"validation loss={validationResults['loss']:.4f}, "
            f"validation accuracy={validationResults['accuracy']:.3f}"
        )

        if validationResults["loss"] < bestValidationLoss - 1e-4:
            bestValidationLoss = validationResults["loss"]
            bestModelState = copy.deepcopy(model.state_dict())
            bestEpoch = epoch
            epochsWithoutImprovement = 0
        else:
            epochsWithoutImprovement += 1
            if epochsWithoutImprovement >= patience:
                print(f"  Early stopping after epoch {epoch}.")
                break

    if bestModelState is None:
        raise RuntimeError("Training did not produce a model checkpoint.")
    model.load_state_dict(bestModelState)
    return history, bestEpoch, lossFunction


def calculateClassificationMetrics(trueLabels, predictedLabels):
    confusionMatrix = [[0 for _ in LABEL_NAMES] for _ in LABEL_NAMES]
    for trueLabel, predictedLabel in zip(trueLabels, predictedLabels):
        confusionMatrix[trueLabel][predictedLabel] += 1

    perClassMetrics = {}
    for classIndex, labelName in enumerate(LABEL_NAMES):
        truePositive = confusionMatrix[classIndex][classIndex]
        falsePositive = sum(
            confusionMatrix[rowIndex][classIndex]
            for rowIndex in range(len(LABEL_NAMES))
            if rowIndex != classIndex
        )
        falseNegative = sum(
            confusionMatrix[classIndex][columnIndex]
            for columnIndex in range(len(LABEL_NAMES))
            if columnIndex != classIndex
        )
        precision = truePositive / (truePositive + falsePositive) if truePositive + falsePositive else 0.0
        recall = truePositive / (truePositive + falseNegative) if truePositive + falseNegative else 0.0
        f1Score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        perClassMetrics[labelName] = {
            "precision": precision,
            "recall": recall,
            "f1": f1Score,
        }

    macroMetrics = {
        metricName: sum(perClassMetrics[label][metricName] for label in LABEL_NAMES)
        / len(LABEL_NAMES)
        for metricName in ("precision", "recall", "f1")
    }
    return confusionMatrix, perClassMetrics, macroMetrics


def writeCsv(path, fieldNames, rows):
    with path.open("w", encoding="utf-8", newline="") as csvFile:
        writer = csv.DictWriter(csvFile, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(rows)


def saveHistory(architecture, inputType, history, resultPrefix=None):
    filePrefix = resultPrefix or f"{architecture.lower()}_{inputType}"
    path = RESULTS_PATH / f"{filePrefix}_history.csv"
    formattedRows = []
    for row in history:
        formattedRows.append(
            {
                "epoch": row["epoch"],
                "train_loss": f"{row['train_loss']:.6f}",
                "train_accuracy": f"{row['train_accuracy']:.6f}",
                "validation_loss": f"{row['validation_loss']:.6f}",
                "validation_accuracy": f"{row['validation_accuracy']:.6f}",
            }
        )
    writeCsv(
        path,
        ["epoch", "train_loss", "train_accuracy", "validation_loss", "validation_accuracy"],
        formattedRows,
    )


def saveConfusionMatrix(
    architecture, inputType, confusionMatrix, resultPrefix=None
):
    filePrefix = resultPrefix or f"{architecture.lower()}_{inputType}"
    path = RESULTS_PATH / f"{filePrefix}_confusion.csv"
    rows = []
    for rowIndex, labelName in enumerate(LABEL_NAMES):
        rows.append(
            {
                "actual_label": labelName,
                **{
                    f"predicted_{predictedName}": confusionMatrix[rowIndex][columnIndex]
                    for columnIndex, predictedName in enumerate(LABEL_NAMES)
                },
            }
        )
    writeCsv(
        path,
        ["actual_label"] + [f"predicted_{labelName}" for labelName in LABEL_NAMES],
        rows,
    )


def createModel(architecture, vocabularySize, lstmDropout=DROPOUT):
    if architecture == "CNN":
        return TextCNN(vocabularySize)
    if architecture == "LSTM":
        return TextLSTM(vocabularySize, dropout=lstmDropout)
    raise ValueError(f"Unknown architecture: {architecture}")


def runExperiment(
    architecture,
    inputType,
    splitFrames,
    device,
    epochs,
    batchSize,
    learningRate,
    maxLength,
    seed,
    validationOnly=False,
    lstmDropout=DROPOUT,
    resultPrefix=None,
):
    print(f"\nTraining {architecture} with {inputType} input")
    setRandomSeeds(seed)
    loaders, vocabulary = createDataLoaders(
        splitFrames, inputType, batchSize, maxLength, seed
    )
    print(f"  Training-only vocabulary size: {len(vocabulary)}")

    model = createModel(
        architecture, len(vocabulary), lstmDropout=lstmDropout
    ).to(device)
    verifyForwardShape(model, loaders["train"], device)
    print(f"  Forward shape check: [batch_size, {len(LABEL_NAMES)}]")

    history, bestEpoch, lossFunction = trainWithEarlyStopping(
        model,
        loaders,
        epochs,
        learningRate,
        EARLY_STOPPING_PATIENCE,
        device,
    )

    if validationOnly:
        bestHistory = next(row for row in history if row["epoch"] == bestEpoch)
        bestAccuracyHistory = max(history, key=lambda row: row["validation_accuracy"])
        print("\nBest validation checkpoint")
        print(f"  Best epoch by validation loss: {bestEpoch}")
        print(f"  Best validation loss: {bestHistory['validation_loss']:.6f}")
        print(
            "  Validation accuracy at best-loss checkpoint: "
            f"{bestHistory['validation_accuracy']:.6f}"
        )
        print(
            f"  Best validation accuracy: {bestAccuracyHistory['validation_accuracy']:.6f} "
            f"(epoch {bestAccuracyHistory['epoch']})"
        )
        print("  Held-out test set was not evaluated; no result files were written.")
        return None

    # The test set is first evaluated here, after validation-based model selection.
    trainResults = evaluateModel(model, loaders["train_evaluation"], lossFunction, device)
    validationResults = evaluateModel(model, loaders["validation"], lossFunction, device)
    testResults = evaluateModel(model, loaders["test"], lossFunction, device)
    confusionMatrix, perClassMetrics, macroMetrics = calculateClassificationMetrics(
        testResults["true_labels"], testResults["predicted_labels"]
    )

    saveHistory(architecture, inputType, history, resultPrefix=resultPrefix)
    saveConfusionMatrix(
        architecture,
        inputType,
        confusionMatrix,
        resultPrefix=resultPrefix,
    )

    metricRow = {
        "architecture": architecture,
        "input_type": inputType,
        "best_epoch": bestEpoch,
        "train_accuracy": trainResults["accuracy"],
        "validation_accuracy": validationResults["accuracy"],
        "test_accuracy": testResults["accuracy"],
        "test_precision_macro": macroMetrics["precision"],
        "test_recall_macro": macroMetrics["recall"],
        "test_f1_macro": macroMetrics["f1"],
    }
    for labelName in LABEL_NAMES:
        labelKey = labelName.lower()
        metricRow[f"test_precision_{labelKey}"] = perClassMetrics[labelName]["precision"]
        metricRow[f"test_recall_{labelKey}"] = perClassMetrics[labelName]["recall"]
        metricRow[f"test_f1_{labelKey}"] = perClassMetrics[labelName]["f1"]

    if resultPrefix is not None:
        metricRow["validation_loss"] = validationResults["loss"]
        metricRow["confusion_matrix"] = confusionMatrix

    print(
        f"  Best epoch={bestEpoch}; test accuracy={testResults['accuracy']:.3f}; "
        f"test macro F1={macroMetrics['f1']:.3f}"
    )
    return metricRow


def saveMetrics(metricRows):
    metricColumns = [
        "architecture",
        "input_type",
        "best_epoch",
        "train_accuracy",
        "validation_accuracy",
        "test_accuracy",
        "test_precision_macro",
        "test_recall_macro",
        "test_f1_macro",
    ]
    for labelName in LABEL_NAMES:
        labelKey = labelName.lower()
        metricColumns.extend(
            [
                f"test_precision_{labelKey}",
                f"test_recall_{labelKey}",
                f"test_f1_{labelKey}",
            ]
        )

    formattedRows = []
    for row in metricRows:
        formattedRow = dict(row)
        for column in metricColumns:
            if column not in {"architecture", "input_type", "best_epoch"}:
                formattedRow[column] = f"{row[column]:.6f}"
        formattedRows.append(formattedRow)
    writeCsv(RESULTS_PATH / "metrics.csv", metricColumns, formattedRows)


def categorySlug(category):
    return re.sub(r"[^a-z0-9]+", "_", category.lower()).strip("_")


def saveRq3Metrics(metricRows, heldOutCategory):
    metricColumns = [
        "held_out_category",
        "architecture",
        "input_type",
        "best_epoch",
        "validation_loss",
        "validation_accuracy",
        "held_out_test_accuracy",
        "test_f1_macro",
    ] + [f"test_recall_{labelName.lower()}" for labelName in LABEL_NAMES]
    formattedRows = []
    for row in metricRows:
        formattedRows.append(
            {
                "held_out_category": heldOutCategory,
                "architecture": row["architecture"],
                "input_type": row["input_type"],
                "best_epoch": row["best_epoch"],
                "validation_loss": f"{row['validation_loss']:.6f}",
                "validation_accuracy": f"{row['validation_accuracy']:.6f}",
                "held_out_test_accuracy": f"{row['test_accuracy']:.6f}",
                "test_f1_macro": f"{row['test_f1_macro']:.6f}",
                **{
                    f"test_recall_{labelName.lower()}": (
                        f"{row[f'test_recall_{labelName.lower()}']:.6f}"
                    )
                    for labelName in LABEL_NAMES
                },
            }
        )
    path = RESULTS_PATH / f"rq3_{categorySlug(heldOutCategory)}_metrics.csv"
    writeCsv(path, metricColumns, formattedRows)


def printRq3Results(metricRows, heldOutCategory):
    chanceAccuracy = 1 / len(LABEL_NAMES)
    print(f"\nRQ3 results: held-out category = {heldOutCategory}")
    print(f"Three-class chance baseline: {chanceAccuracy:.3%}")
    for row in metricRows:
        difference = row["test_accuracy"] - chanceAccuracy
        relation = "above" if difference >= 0 else "below"
        print(f"\n{row['architecture']} (output_only)")
        print(f"  Best validation epoch: {row['best_epoch']}")
        print(f"  Validation loss: {row['validation_loss']:.6f}")
        print(
            "  Validation accuracy at best-loss checkpoint: "
            f"{row['validation_accuracy']:.3%}"
        )
        print(f"  Held-out category test accuracy: {row['test_accuracy']:.3%}")
        print(f"  Test macro F1: {row['test_f1_macro']:.6f}")
        print(
            f"  Test accuracy is {abs(difference):.2%} {relation} "
            "the chance baseline."
        )
        print("  Per-class recall:")
        for labelName in LABEL_NAMES:
            recall = row[f"test_recall_{labelName.lower()}"]
            print(f"    {labelName}: {recall:.3%}")
        print("  Confusion matrix (rows=actual, columns=predicted):")
        print(f"    {'':<8}" + "".join(f"{label:>9}" for label in LABEL_NAMES))
        for labelName, values in zip(LABEL_NAMES, row["confusion_matrix"]):
            print(f"    {labelName:<8}" + "".join(f"{value:>9}" for value in values))

    print(
        "\nThe original 73.3% CNN result is not a directly equivalent comparison "
        "because it used a different test distribution."
    )


def printComparisonTable(metricRows):
    print("\nFinal held-out test comparison (495 responses per experiment)")
    print(f"{'Architecture':<14}{'Input type':<18}{'Test accuracy':>14}")
    print("-" * 46)
    for row in metricRows:
        print(
            f"{row['architecture']:<14}{row['input_type']:<18}"
            f"{row['test_accuracy']:>14.3f}"
        )
    print("\nThese results are preliminary; high training accuracy alone is not evidence of authorship identification.")


def parseArguments():
    parser = argparse.ArgumentParser(
        description="Train CNN and LSTM classifiers on the collected LLM responses."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument("--lstm-dropout", type=float, default=DROPOUT)
    parser.add_argument("--device", choices=["auto", "cpu", "mps"], default="auto")
    parser.add_argument(
        "--rq3",
        action="store_true",
        help="run output-only cross-category generalization for CNN and LSTM",
    )
    parser.add_argument(
        "--held-out-category",
        default="Creative writing",
        help="category used only as the RQ3 held-out test set",
    )
    parser.add_argument(
        "--length-stats-only",
        "--length-diagnostics",
        dest="length_stats_only",
        action="store_true",
        help="print pre-truncation token-length statistics and exit without training",
    )
    parser.add_argument(
        "--lstm-output-validation-only",
        action="store_true",
        help="train only the output-only LSTM and report validation results without testing or saving",
    )
    arguments = parser.parse_args()
    if arguments.epochs < 1:
        parser.error("--epochs must be at least 1")
    if arguments.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if arguments.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if arguments.max_length < MINIMUM_PADDED_LENGTH:
        parser.error(f"--max-length must be at least {MINIMUM_PADDED_LENGTH}")
    if not 0 <= arguments.lstm_dropout < 1:
        parser.error("--lstm-dropout must be at least 0 and less than 1")
    if arguments.rq3 and (
        arguments.length_stats_only or arguments.lstm_output_validation_only
    ):
        parser.error(
            "--rq3 cannot be combined with --length-stats-only or "
            "--lstm-output-validation-only"
        )
    if arguments.rq3 and arguments.max_length != DEFAULT_MAX_LENGTH:
        parser.error(f"--rq3 requires --max-length {DEFAULT_MAX_LENGTH}")
    if arguments.rq3 and arguments.learning_rate != DEFAULT_LEARNING_RATE:
        parser.error(f"--rq3 requires --learning-rate {DEFAULT_LEARNING_RATE}")
    if arguments.rq3 and arguments.lstm_dropout != DROPOUT:
        parser.error(f"--rq3 requires --lstm-dropout {DROPOUT}")
    return arguments


def main():
    arguments = parseArguments()
    if not RESULTS_PATH.is_dir():
        raise FileNotFoundError(f"Existing results directory not found: {RESULTS_PATH}")

    device = resolveDevice(arguments.device)
    print(f"Using device: {device}")
    dataFrame = loadAndVerifyDataset()

    if arguments.rq3:
        splitFrames = createCrossCategorySplits(
            dataFrame, arguments.held_out_category, SEED
        )
        categoryFileSlug = categorySlug(arguments.held_out_category)
        metricRows = []
        for architecture in ("CNN", "LSTM"):
            metricRows.append(
                runExperiment(
                    architecture=architecture,
                    inputType="output_only",
                    splitFrames=splitFrames,
                    device=device,
                    epochs=arguments.epochs,
                    batchSize=arguments.batch_size,
                    learningRate=DEFAULT_LEARNING_RATE,
                    maxLength=DEFAULT_MAX_LENGTH,
                    seed=SEED,
                    lstmDropout=DROPOUT,
                    resultPrefix=(
                        f"rq3_{categoryFileSlug}_{architecture.lower()}_output_only"
                    ),
                )
            )
        saveRq3Metrics(metricRows, arguments.held_out_category)
        printRq3Results(metricRows, arguments.held_out_category)
        print(f"\nRQ3 CSV files saved in: {RESULTS_PATH}")
        return 0

    splitFrames = createPromptSplits(dataFrame, SEED)
    if arguments.length_stats_only:
        printLengthDiagnostics(splitFrames, arguments.max_length)
        return 0
    if arguments.lstm_output_validation_only:
        runExperiment(
            architecture="LSTM",
            inputType="output_only",
            splitFrames=splitFrames,
            device=device,
            epochs=arguments.epochs,
            batchSize=arguments.batch_size,
            learningRate=arguments.learning_rate,
            maxLength=arguments.max_length,
            seed=SEED,
            validationOnly=True,
            lstmDropout=arguments.lstm_dropout,
        )
        return 0

    metricRows = []
    for inputType in INPUT_TYPES:
        for architecture in ("CNN", "LSTM"):
            metricRows.append(
                runExperiment(
                    architecture=architecture,
                    inputType=inputType,
                    splitFrames=splitFrames,
                    device=device,
                    epochs=arguments.epochs,
                    batchSize=arguments.batch_size,
                    learningRate=arguments.learning_rate,
                    maxLength=arguments.max_length,
                    seed=SEED,
                    lstmDropout=arguments.lstm_dropout,
                )
            )

    saveMetrics(metricRows)
    printComparisonTable(metricRows)
    print(f"\nEvaluation CSV files saved in: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
