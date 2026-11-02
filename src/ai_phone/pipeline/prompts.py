"""What the phone model is told. The fixed part is the same on every call and is cached by the model worker;
the call section changes per call (the calling AI's --prompt, or the inbound instructions)."""
from __future__ import annotations

END_MARKER = "[END]"
DISCLOSURE = "Hi, this is an AI assistant calling."
FILLER = "One moment."
STILL_HERE = "Still with you, one moment."
RELAY_GOODBYE = "Sorry, I have to go. Goodbye."

PHONE_INSTRUCTIONS = (
    "You are an AI assistant speaking on a live phone call. Speak naturally and keep replies brief by default. "
    "When the call instructions or the other person ask for specific content or a longer response, provide it. "
    "Carry out the call instructions directly instead of merely announcing or summarizing them. "
    "On outbound calls the AI disclosure is played separately; do not repeat it. "
    "Never use lists, markdown, emoji or stage directions. Do not invent facts you "
    "were not given; if you are asked something you do not know, say so plainly. When the purpose of the call is done, "
    f'or the other person says goodbye, finish with "Goodbye." followed by {END_MARKER} exactly once. '
    f"Never use {END_MARKER} in an ordinary answer, after a question, or while waiting for the other person to reply."
)
CALL_SECTION_HEADER = "\n\nThis call:\n"
DEFAULT_CALL_SECTION = (
    "No specific task was provided for this outbound call. Have a brief, natural conversation. "
    "Open with a simple greeting, then respond to what the person says and answer their questions. "
    "If asked why you called, say you were not given a specific purpose. "
    "Do not invent a reason, business, message from the owner, or a test. "
    "Do not keep reintroducing the call or asking to discuss an unspecified matter."
)
# A factual event keeps the history valid for chat templates and reusable CLI sessions.
# It must not impose an opening-only instruction that remains in subsequent turns.
OPENING_CUE = f'(The call is connected. The other person has answered and has already heard: "{DISCLOSURE}")'


def fixed_instructions() -> str:
    """The cached phone instructions. Host identity is never added to the model's context."""
    return PHONE_INSTRUCTIONS


def system_prompt(fixed: str, call_section: str | None) -> str:
    section = (call_section or "").strip() or DEFAULT_CALL_SECTION
    return fixed + CALL_SECTION_HEADER + section


def cache_probe_sections() -> tuple[str, str]:
    """Two call sections that differ from their first character: the model worker tokenises a prompt with each and
    caches the common prefix, which is then exactly the fixed part plus the header."""
    return "Ask whether they are open on Sunday.", "Book a table for two at eight."
