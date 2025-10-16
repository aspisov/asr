from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class CustomDirAudioDataset(BaseDataset):
    _SUPPORTED_EXTS = {".mp3", ".wav", ".flac", ".m4a"}

    def __init__(self, audio_dir, transcription_dir=None, *args, **kwargs):
        audio_dir = Path(audio_dir)
        if transcription_dir is not None:
            transcription_dir = Path(transcription_dir)

        data = []
        for path in sorted(audio_dir.iterdir()):
            if not path.is_file() or path.suffix.lower() not in self._SUPPORTED_EXTS:
                continue

            entry = {"path": str(path)}
            if transcription_dir is not None and transcription_dir.exists():
                transcription_path = transcription_dir / f"{path.stem}.txt"
                if transcription_path.exists():
                    with transcription_path.open() as file:
                        entry["text"] = file.read().strip()

            if "text" not in entry:
                entry["text"] = ""

            info = torchaudio.info(str(path))
            entry["audio_len"] = info.num_frames / info.sample_rate

            data.append(entry)

        super().__init__(data, *args, **kwargs)
