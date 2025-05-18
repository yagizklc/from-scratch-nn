from pathlib import Path
import sys
from typing import Any, Callable
from datasets import Dataset, DatasetDict, load_dataset
import matplotlib.pyplot as plt
import io
from PIL import Image
import numpy as np
import evaluate
import torch
from transformers.image_processing_base import BatchFeature
from transformers.models.vit.image_processing_vit import ViTImageProcessor
from transformers.models.auto.image_processing_auto import AutoImageProcessor
from transformers.models.vit.modeling_vit import ViTForImageClassification
from transformers.training_args import TrainingArguments
from transformers.trainer import Trainer

DATASET_NAME = "pcuenq/oxford-pets"
MODEL_NAME = "google/vit-base-patch16-224"
MODEL_PATH = Path(__file__).parent.parent.parent / "vit-base-oxford-iiit-pets"
ACCURACY = evaluate.load("accuracy")


def load_dataset_from_hf(dataset_name: str) -> Dataset:
    return load_dataset(dataset_name)["train"]  # type: ignore


def show_samples(ds: Dataset, rows: int, cols: int):
    samples = ds.shuffle().select(np.arange(rows * cols))  # selecting random images
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    # plotting
    for i in range(rows * cols):
        img_bytes = samples[i]["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes))
        label = samples[i]["label"]
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.show()


def _split_dataset(dataset: Dataset, test_size: float = 0.2) -> DatasetDict:
    train_dataset = dataset.train_test_split(test_size=test_size)
    ds_train = train_dataset["train"]
    eval_dataset = train_dataset["test"].train_test_split(test_size=0.5)
    ds_valid, ds_test = eval_dataset["train"], eval_dataset["test"]
    return DatasetDict({"train": ds_train, "eval": ds_valid, "test": ds_test})


def _create_label_mapping(dataset: Dataset) -> tuple[dict[str, int], dict[int, str]]:
    labels = dataset.unique(column="label")
    label2id = {c: idx for idx, c in enumerate(labels)}
    id2label = {idx: c for idx, c in enumerate(labels)}
    return label2id, id2label


def get_processor(model_name: str = MODEL_NAME) -> ViTImageProcessor:
    return AutoImageProcessor.from_pretrained(model_name)


def collate_fn(batch: list[BatchFeature]) -> dict[str, torch.Tensor]:
    return {
        "pixel_values": torch.stack([x["pixel_values"] for x in batch]),
        "labels": torch.tensor([x["labels"] for x in batch]),
    }


def compute_metrics(eval_preds: tuple[torch.Tensor, torch.Tensor]):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=1)
    return ACCURACY.compute(predictions=predictions, references=labels)


def create_model(num_labels: int, id2label: dict[int, str], label2id: dict[str, int]):
    return ViTForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def _freeze_weights(model: ViTForImageClassification) -> None:
    for name, p in model.named_parameters():
        if not name.startswith("classifier"):
            p.requires_grad = False


def train_model(
    model: ViTForImageClassification,
    train_dataset: Dataset,
    valid_dataset: Dataset,
    processor: ViTImageProcessor,
) -> Trainer:
    training_args = TrainingArguments(
        output_dir=str(MODEL_PATH),
        per_device_train_batch_size=16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        num_train_epochs=5,
        learning_rate=3e-4,
        save_total_limit=2,
        remove_unused_columns=False,
        push_to_hub=True,
        report_to="tensorboard",
        load_best_model_at_end=True,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collate_fn,
        compute_metrics=compute_metrics,  # type: ignore
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        processing_class=processor,
    )
    trainer.train()
    return trainer


def create_transform_function(processor: ViTImageProcessor, label2id: dict[str, int]):
    def _transform_dataset(batch: dict[str, Any]) -> BatchFeature:
        # convert each image to PIL Image for RGB conversion
        images = [
            Image.open(io.BytesIO(x["bytes"])).convert("RGB") for x in batch["image"]
        ]
        # convert to pytorch tensor - returns a dict with keys 'pixel_values' and 'labels'
        inputs: BatchFeature = processor(images=images, return_tensors="pt")
        # convert labels to ids
        inputs["labels"] = [label2id[x] for x in batch["label"]]
        return inputs

    return _transform_dataset


def show_predictions(
    rows: int,
    cols: int,
    test_dataset: Dataset,
    trainer: Trainer,
    id2label: dict[int, str],
) -> None:
    samples = test_dataset.shuffle().select(np.arange(rows * cols))
    predictions = trainer.predict(samples).predictions.argmax(axis=1)  # type: ignore
    fig = plt.figure(figsize=(cols * 4, rows * 4))
    for i in range(rows * cols):
        img_bytes = samples[i]["image"]["bytes"]
        img = Image.open(io.BytesIO(img_bytes))
        prediction = predictions[i]
        label = f"label: {samples[i]['label']}\npredicted: {id2label[prediction]}"
        fig.add_subplot(rows, cols, i + 1)
        plt.imshow(img)
        plt.title(label)
        plt.axis("off")
    plt.show()


def load_pretrained_model():
    # Load the processor and model
    processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
    model = ViTForImageClassification.from_pretrained(MODEL_PATH)
    return processor, model


def main() -> None:
    _dataset = load_dataset_from_hf(DATASET_NAME)
    print(f"Dataset loaded, {len(_dataset)=}")

    datasets = _split_dataset(_dataset)
    print(
        f"Dataset split, {len(datasets['train'])=}, {len(datasets['eval'])=}, {len(datasets['test'])=}"
    )

    label2id, id2label = _create_label_mapping(datasets["train"])
    print(f"Label mapping created:{len(label2id)=}, {id2label[0]=}")

    processor = get_processor()
    print(f"Processor created, {type(processor)=}")

    transform_fn = create_transform_function(processor, label2id)
    datasets = datasets.with_transform(transform=transform_fn)

    print(f"Transformed dataset, {type(datasets['train'])=}")
    print(f"Items: {len(datasets['train'][0])=}")
    print(
        f"Pixel values shape: {datasets['train'][0]['pixel_values'].shape} (batch_size, num_channels, height, width)"
    )

    if len(sys.argv) > 1 and sys.argv[1] == "train":
        model = create_model(len(label2id), id2label, label2id)
        print(f"Model created, {type(model)=}")
        _freeze_weights(model)
        num_params = sum([p.numel() for p in model.parameters()])
        trainable_params = sum(
            [p.numel() for p in model.parameters() if p.requires_grad]
        )
        print(f"Model parameters: {num_params = :,} | {trainable_params = :,}")

        trainer = train_model(
            model=model,
            train_dataset=datasets["train"],
            valid_dataset=datasets["eval"],
            processor=processor,
        )
        print(f"Trainer created, {type(trainer)=}")
        trainer.evaluate(eval_dataset=datasets["test"])  # type: ignore

    else:
        print("Loading pretrained model")
        processor, model = load_pretrained_model()
        trainer = Trainer(
            model=model,
            args=TrainingArguments(output_dir=str(MODEL_PATH)),
            data_collator=collate_fn,
            compute_metrics=compute_metrics,  # type: ignore
            eval_dataset=datasets["test"],
        )
        print(f"Trainer created, {type(trainer)=}")

    show_predictions(
        rows=5,
        cols=5,
        test_dataset=datasets["test"],
        trainer=trainer,
        id2label=id2label,
    )


if __name__ == "__main__":
    main()
