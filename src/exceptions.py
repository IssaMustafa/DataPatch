class NotEnoughRows(Exception):
    """
    Exception raised when too few rows remain in the dataset after filtering for missing data.

    Args:
        message (str): The exception message.

    Attributes:
        message (str): The exception message.
    """
    def __init__(self, message):
        super().__init__(message)
