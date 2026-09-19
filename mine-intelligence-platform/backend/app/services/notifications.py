from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

from .user_context import get_current_username

# Per-user, in-memory notification feed. Kept separate from AnalysisSession
# deliberately - a new dataset upload replaces the session object entirely,
# but notifications should persist and accumulate across that, not vanish.
_NOTIFICATIONS: dict[str, list[dict[str, Any]]] = {}
_MAX_NOTIFICATIONS = 30


def add_notification(kind: str, message: str) -> None:
    username = get_current_username()
    bucket = _NOTIFICATIONS.setdefault(username, [])
    bucket.insert(
        0,
        {
            "id": uuid.uuid4().hex,
            "kind": kind,
            "message": message,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "read": False,
        },
    )
    del bucket[_MAX_NOTIFICATIONS:]


def list_notifications() -> list[dict[str, Any]]:
    username = get_current_username()
    return _NOTIFICATIONS.get(username, [])


def mark_all_read() -> None:
    username = get_current_username()
    for item in _NOTIFICATIONS.get(username, []):
        item["read"] = True
