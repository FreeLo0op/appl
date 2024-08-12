import os
from pathlib import Path
from typing import Optional, Union, Dict, List

import torch
from demucs.apply import BagOfModels, apply_model
from demucs.audio import save_audio as save_audio
from demucs.pretrained import get_model
from demucs.separate import load_track as load_track
from loguru import logger


class Seperate:
    def __init__(self, args) -> None:
        self.args = args
        self.method = "demucs"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.init_model("htdemucs", device=self.device)

        self.audio_channels = self.model.audio_channels
        self.samplerate = self.model.samplerate
        print(f"INFO: demucs - sample rate {self.samplerate}, channel {self.audio_channels}")

    def init_model(self,
        name: str = "htdemucs",
        device: Optional[Union[str, torch.device]] = None,
        segment: Optional[int] = None,
    ) -> torch.nn.Module:
        """
        Initialize the model

        Args:
            name: Name of the model
            device: Device to use
            segment: Set split size of each chunk. This can help save memory of graphic card.

        Returns:
            The model
        """

        model = get_model(name)
        model.eval()

        if device is not None:
            model.to(device)

        logger.info(f"Model {name} loaded on {device}")

        if isinstance(model, BagOfModels) and len(model.models) > 1:
            logger.info(
                f"Selected model is a bag of {len(model.models)} models. "
                f"You will see {len(model.models)} progress bars per track."
            )

        if segment is not None:
            if isinstance(model, BagOfModels):
                for m in model.models:
                    m.segment = segment
            else:
                model.segment = segment

        return model
    
    def load_track(self,
        path: Union[str, Path],
    ) -> torch.Tensor:
        """
        Load audio track

        Args:
            path: Path to the audio file

        Returns:
            The audio
        """

        return load_track(path, self.audio_channels, self.samplerate)
    
    def separate_audio(self,
        audio: torch.Tensor,
        shifts: int = 1,
        num_workers: int = 0,
        progress: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Separate audio into sources

        Args:
            audio: The audio
            shifts: Run the model N times, larger values will increase the quality but also the time
            num_workers: Number of workers to use
            progress: Show progress bar

        Returns:
            The separated tracks
        """

        device = next(self.model.parameters()).device

        ref = audio.mean(0)
        audio = (audio - ref.mean()) / audio.std()

        sources = apply_model(
            self.model,
            audio[None],
            device=device,
            shifts=shifts,
            split=True,
            overlap=0.25,
            progress=progress,
            num_workers=num_workers,
        )[0]

        sources = sources * ref.std() + ref.mean()

        return dict(zip(self.model.sources, sources))

    def save_audio(self,
        path: Union[str, Path],
        track: torch.Tensor,
    ) -> None:
        """
        Save audio track

        Args:
            model: The model
            path: Path to save the audio file
            track: The audio tracks
        """

        save_audio(
            track,
            path,
            self.samplerate,
            clip="rescale",
            as_float=False,
            bits_per_sample=16,
        )


if __name__ == "__main__":
    from types import SimpleNamespace
    args = SimpleNamespace(output='./align_output/')

    seperator = Seperate(args)
    audio = seperator.load_track("./resources/peiqi/peiqi.mp3")
    print("audio = ", audio.shape)
    tracks = seperator.separate_audio(audio, shifts=1, num_workers=0, progress=True)

    # save only "vocals"
    seperator.save_audio(os.path.join(args.output, "vocals_only.wav"), tracks["vocals"])
    print("Seperate: Success!")

