"""Common types used across Agency Swarm."""

from typing import Any, Callable, Dict, TypedDict


class ThreadsCallbacks(TypedDict):
    """Callbacks for thread persistence.

    Attributes:
        load: Function that loads thread data from storage. Should return a dict mapping
            chat_id to thread data.
        save: Function that saves thread data to storage. Takes a dict of thread data
            and persists it.
    """

    load: Callable[[], Dict]
    save: Callable[[Dict], Any]
