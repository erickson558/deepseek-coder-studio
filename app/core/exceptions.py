class ProjectError(Exception):
    """Base exception for the project."""


class DependencyUnavailableError(ProjectError):
    """Raised when an optional runtime dependency is missing."""


class ConfigurationError(ProjectError):
    """Raised when configuration is invalid or incomplete."""


class DatasetValidationError(ProjectError):
    """Raised when a dataset sample cannot be normalised or validated."""
