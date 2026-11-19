"""Exception hierarchy. Every failure the CLI reports maps to one of these."""


class QuietWhisperError(Exception):
    """Base class for expected, user-reportable failures."""

    exit_code = 1


class AudioError(QuietWhisperError):
    """The input could not be opened or decoded, or contains no audio."""

    exit_code = 2


class ModelNotAvailableError(QuietWhisperError):
    """A required model is not present in the local cache (run `hearken setup`)."""

    exit_code = 3


class EngineError(QuietWhisperError):
    """An ASR engine failed while transcribing."""

    exit_code = 4


class ConfigError(QuietWhisperError):
    """Invalid preset, vocabulary file, or option combination."""

    exit_code = 5
