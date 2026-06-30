"""Wire schemas for the SHL recommender API.

The schema is non-negotiable (the evaluator rejects deviations), so the philosophy is
asymmetric:

* **Input** is parsed *leniently* -- unknown fields are ignored and roles are not hard
  rejected -- so a slightly-off request never yields an automatic 422 (which would be a
  failed turn we could not intercept).
* **Output** is *strict* (``extra="forbid"``) and additionally sanitized against the
  catalog before sending, guaranteeing every item is real and the shape is exact.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

VALID_ROLES = {"user", "assistant", "system"}


# --- Request (lenient) ---------------------------------------------------
class Message(BaseModel):
    model_config = ConfigDict(extra="ignore")  # tolerate extra metadata fields
    role: str = "user"
    content: str = ""


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    messages: list[Message] = Field(default_factory=list)


# --- Response (strict) ---------------------------------------------------
class Recommendation(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reply: str
    recommendations: list[Recommendation] = Field(default_factory=list)
    end_of_conversation: bool = False
