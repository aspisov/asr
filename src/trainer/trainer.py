from pathlib import Path

import pandas as pd

from src.logger.utils import plot_spectrogram
from src.metrics.tracker import MetricTracker
from src.metrics.utils import calc_cer, calc_wer
from src.trainer.base_trainer import BaseTrainer


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.optimizer.zero_grad()

        outputs = self.model(**batch)
        batch.update(outputs)

        self._attach_predictions(batch)

        all_losses = self.criterion(**batch)
        batch.update(all_losses)

        if self.is_train:
            batch["loss"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(**batch)
        else:
            # Log Stuff
            self.log_spectrogram(**batch)
            self.log_predictions(**batch)

    def log_spectrogram(self, spectrogram, **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image("spectrogram", image)

    def log_predictions(
        self,
        text,
        log_probs,
        log_probs_length,
        audio_path,
        predictions=None,
        raw_predictions=None,
        beam_predictions=None,
        examples_to_log=10,
        **batch,
    ):
        predictions = (
            predictions if predictions is not None else batch.get("predictions", [])
        )
        raw_predictions = (
            raw_predictions
            if raw_predictions is not None
            else batch.get("raw_predictions", [])
        )
        beam_predictions = (
            beam_predictions
            if beam_predictions is not None
            else batch.get("beam_predictions", [])
        )
        tuples = list(
            zip(
                predictions,
                beam_predictions,
                text,
                raw_predictions,
                audio_path,
            )
        )

        rows = {}
        for argmax_pred, beam_pred, target, raw_pred, audio_path in tuples[
            :examples_to_log
        ]:
            target = self.text_encoder.normalize_text(target)
            chosen_prediction = beam_pred if beam_pred else argmax_pred
            wer = calc_wer(target, chosen_prediction) * 100
            cer = calc_cer(target, chosen_prediction) * 100

            rows[Path(audio_path).name] = {
                "target": target,
                "raw prediction": raw_pred,
                "argmax": argmax_pred,
                "beam": beam_pred,
                "wer": wer,
                "cer": cer,
            }
        self.writer.add_table(
            "predictions", pd.DataFrame.from_dict(rows, orient="index")
        )

    def _attach_predictions(self, batch):
        if "log_probs" not in batch:
            return

        log_probs = batch["log_probs"].detach().cpu()
        lengths = batch["log_probs_length"].detach().cpu()

        predictions = []
        raw_predictions = []
        beam_predictions = []
        beam_size = self.config.trainer.get("beam_size")

        for sample_log_prob, length in zip(log_probs, lengths):
            decoded, raw, beams = self.text_encoder.decode_logits(
                sample_log_prob,
                int(length),
                beam_size=beam_size,
            )
            predictions.append(decoded)
            raw_predictions.append(raw)
            beam_predictions.append(beams[0].text if beams else decoded)

        batch["predictions"] = predictions
        batch["raw_predictions"] = raw_predictions
        batch["beam_predictions"] = beam_predictions
