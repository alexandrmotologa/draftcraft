"""Domain exception hierarchy for DraftCraft."""

class DraftCraftError(Exception):
    """Base exception for all DraftCraft errors."""
    pass


class TokenizerAlignmentError(DraftCraftError):
    """Raised when tokenizers fail alignment or have incompatible vocabularies."""
    pass


class ModelDimensionMismatchError(DraftCraftError):
    """Raised when hidden dimensions or vocabulary projections do not match."""
    pass


class ConfigurationError(DraftCraftError):
    """Raised when configuration parameters are invalid or contradictory."""
    pass


class TrainingDivergenceError(DraftCraftError):
    """Raised when training loss diverges or produces numerical instabilities."""
    pass


class CheckpointError(DraftCraftError):
    """Raised when checkpoint loading or serialization fails."""
    pass
