import json
from pathlib import Path

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer


class Inferencer(BaseTrainer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        model,
        config,
        device,
        dataloaders,
        text_encoder,
        save_path,
        metrics=None,
        batch_transforms=None,
        skip_model_load=False,
    ):
        """
        Initialize the Inferencer.

        Args:
            model (nn.Module): PyTorch model.
            config (DictConfig): run config containing inferencer config.
            device (str): device for tensors and model.
            dataloaders (dict[DataLoader]): dataloaders for different
                sets of data.
            text_encoder (CTCTextEncoder): text encoder.
            save_path (str): path to save model predictions and other
                information.
            metrics (dict): dict with the definition of metrics for
                inference (metrics[inference]). Each metric is an instance
                of src.metrics.BaseMetric.
            batch_transforms (dict[nn.Module] | None): transforms that
                should be applied on the whole batch. Depend on the
                tensor name.
            skip_model_load (bool): if False, require the user to set
                pre-trained checkpoint path. Set this argument to True if
                the model desirable weights are defined outside of the
                Inferencer Class.
        """
        assert (
            skip_model_load or config.inferencer.get("from_pretrained") is not None
        ), "Provide checkpoint or set skip_model_load=True"

        self.config = config
        self.cfg_trainer = self.config.inferencer

        self.device = device

        self.model = model
        self.batch_transforms = batch_transforms

        self.text_encoder = text_encoder

        # define dataloaders
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items()}

        # path definition

        self.save_path = save_path

        # define metrics
        self.metrics = metrics
        if self.metrics is not None:
            self.evaluation_metrics = MetricTracker(
                *[m.name for m in self.metrics["inference"]],
                writer=None,
            )
        else:
            self.evaluation_metrics = None

        if not skip_model_load:
            # init model
            self._from_pretrained(config.inferencer.get("from_pretrained"))

    def run_inference(self):
        """
        Run inference on each partition.

        Returns:
            part_logs (dict): part_logs[part_name] contains logs
                for the part_name partition.
        """
        part_logs = {}
        for part, dataloader in self.evaluation_dataloaders.items():
            logs = self._inference_part(part, dataloader)
            part_logs[part] = logs
        return part_logs

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        outputs = self.model(**batch)
        batch.update(outputs)

        predictions, raw_predictions, beam_predictions = self._decode_batch(batch)

        batch["predictions"] = predictions
        batch["beam_predictions"] = beam_predictions
        batch["raw_predictions"] = raw_predictions

        if metrics is not None and "text" in batch:
            batch_with_predictions = batch | {
                "predictions": predictions,
                "beam_predictions": beam_predictions,
            }
            for met in self.metrics["inference"]:
                metrics.update(met.name, met(**batch_with_predictions))

        if self.save_path is not None:
            self._save_predictions(
                batch, predictions, raw_predictions, beam_predictions, part
            )

        return batch

    def _decode_batch(self, batch):
        log_probs = batch["log_probs"].detach().cpu()
        log_probs_length = batch["log_probs_length"].detach().cpu()

        predictions = []
        raw_predictions = []
        beam_predictions = []
        beam_size = self.config.inferencer.get("beam_size")

        for log_prob_vec, length in zip(log_probs, log_probs_length):
            decoded, raw, beams = self.text_encoder.decode_logits(
                log_prob_vec,
                int(length),
                beam_size=beam_size,
            )
            predictions.append(decoded)
            raw_predictions.append(raw)
            beam_predictions.append(beams[0].text if beams else decoded)
        return predictions, raw_predictions, beam_predictions

    def _save_predictions(
        self, batch, predictions, raw_predictions, beam_predictions, part
    ):
        texts = batch.get("text")
        normalized_targets = None
        if texts is not None:
            normalized_targets = [self.text_encoder.normalize_text(t) for t in texts]

        save_dir = self.save_path / part
        save_dir.mkdir(exist_ok=True, parents=True)

        metadata = []
        for prediction_argmax, raw_prediction, beam_prediction, audio_path in zip(
            predictions,
            raw_predictions,
            beam_predictions,
            batch["audio_path"],
        ):
            prediction_path = Path(audio_path)
            save_path = save_dir / f"{prediction_path.stem}.txt"

            with save_path.open("w", encoding="utf-8") as f:
                f.write(beam_prediction)

            metadata_entry = {
                "utt_id": prediction_path.stem,
                "prediction_argmax": prediction_argmax,
                "prediction_beam": beam_prediction,
                "raw_prediction": raw_prediction,
            }
            metadata.append(metadata_entry)

        if normalized_targets is not None:
            for meta, target in zip(metadata, normalized_targets):
                meta["target"] = target

        metadata_path = save_dir / "metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

    def _inference_part(self, part, dataloader):
        """
        Run inference on a given partition and save predictions

        Args:
            part (str): name of the partition.
            dataloader (DataLoader): dataloader for the given partition.
        Returns:
            logs (dict): metrics, calculated on the partition.
        """

        self.is_train = False
        self.model.eval()

        self.evaluation_metrics.reset()

        # create Save dir
        if self.save_path is not None:
            (self.save_path / part).mkdir(exist_ok=True, parents=True)

        with torch.no_grad():
            for batch_idx, batch in tqdm(
                enumerate(dataloader),
                desc=part,
                total=len(dataloader),
            ):
                batch = self.process_batch(
                    batch_idx=batch_idx,
                    batch=batch,
                    part=part,
                    metrics=self.evaluation_metrics,
                )

        return self.evaluation_metrics.result()
