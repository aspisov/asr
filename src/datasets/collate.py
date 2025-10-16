import torch
import torch.nn.functional as F
from einops import rearrange
from loguru import logger


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor | list]): padded batch.
    """

    spectrograms: list[torch.Tensor] = []
    spectrogram_lengths: list[int] = []
    audios: list[torch.Tensor] = []
    audio_lengths: list[int] = []
    text_encoded_list: list[torch.Tensor] = []
    text_encoded_lengths: list[int] = []
    texts: list[str] = []
    audio_paths: list[str] = []

    dropped = 0

    for item in dataset_items:
        spectrogram = item["spectrogram"].squeeze(0)
        audio = item["audio"].squeeze(0)
        text_encoded = item["text_encoded"].squeeze(0).long()

        spec_len = spectrogram.shape[-1]
        target_len = text_encoded.shape[0]

        spectrograms.append(spectrogram)
        spectrogram_lengths.append(spec_len)

        audios.append(audio)
        audio_lengths.append(audio.shape[-1])

        text_encoded_list.append(text_encoded)
        text_encoded_lengths.append(target_len)

        texts.append(item["text"])
        audio_paths.append(item["audio_path"])

    if not spectrograms:
        raise ValueError("All samples in the batch were dropped due to target length")

    if dropped:
        logger.info("Dropped %s samples from batch because of target length", dropped)

    max_spec_len = max(spectrogram_lengths)
    max_audio_len = max(audio_lengths) if audio_lengths else 0

    padded_spectrograms = [
        F.pad(spec, (0, max_spec_len - spec.shape[-1])) for spec in spectrograms
    ]
    padded_audios = [
        F.pad(audio, (0, max_audio_len - audio.shape[-1])) for audio in audios
    ]

    batch = {
        "spectrogram": rearrange(
            torch.stack(padded_spectrograms),
            "batch_size n_feat seq_len -> batch_size seq_len n_feat",
        ),
        "spectrogram_length": torch.tensor(spectrogram_lengths, dtype=torch.long),
        "audio": torch.stack(padded_audios),
        "audio_length": torch.tensor(audio_lengths, dtype=torch.long),
        "text": texts,
        "text_encoded": torch.cat(text_encoded_list),
        "text_encoded_length": torch.tensor(text_encoded_lengths, dtype=torch.long),
        "audio_path": audio_paths,
    }

    return batch
