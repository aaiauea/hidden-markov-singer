"""HMS - Hidden Markov Singer.

A small, data-efficient, non-neural singing synthesizer:

    score (phonemes + notes + durations)
        -> HMM state sequence          hms.core.duration
        -> acoustic parameters (MLPG)  hms.core.generation
        -> WORLD waveform              hms.vocoder

Train with `hms train`, synthesize with `hms synth`.  See docs/architecture.md.
"""

__version__ = "0.1.0"

__all__ = ["HMSModel", "__version__"]


def __getattr__(name):  # lazy: keeps `import hms` free of numpy/yaml imports
    if name == "HMSModel":
        from hms.core.model import HMSModel
        return HMSModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
