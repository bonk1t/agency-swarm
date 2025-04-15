class User:
    """Represents a human user in the system."""

    name = "user"
    description = "Human user of the system"

    def __init__(self):
        self.id = f"user_{id(self)}"

    def get_role(self) -> str:
        """Return the role in conversations."""
        return "user"
