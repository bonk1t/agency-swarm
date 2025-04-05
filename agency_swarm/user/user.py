class User:
    """Represents a human user in the system."""

    def __init__(self, name: str = None):
        self.name = name or "user"
        self.description = "Human user of the system"
