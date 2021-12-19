from dataclasses import dataclass, field


@dataclass
class VCTKDatasetParams:

    text_dir: str
    mels_dir: str
    text_ext: str = field(default=".TextGrid")
    mels_ext: str = field(default=".pkl")
