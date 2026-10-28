"""What the phone model is told. The fixed part is the same on every call and is cached by the model worker;
the call section changes per call (the calling AI's --prompt, or the inbound instructions)."""
from __future__ import annotations

END_MARKER = "[END]"
DISCLOSURE = "Hi, this is an AI assistant calling."
FILLER = "One moment."
STILL_HERE = "Still with you, one moment."
RELAY_GOODBYE = "Sorry, I have to go. Goodbye."

PHONE_INSTRUCTIONS = (
    "You are an AI assistant speaking on a live phone call. Speak in short, natural sentences, one or two at a time, "
    "the way people talk on the phone. Never use lists, markdown, emoji or stage directions. Do not invent facts you "
    "were not given; if you are asked something you do not know, say so plainly. When the purpose of the call is done, "
    f'or the other person says goodbye, finish with "Goodbye." followed by {END_MARKER} exactly once. '
    f"Never use {END_MARKER} in an ordinary answer, after a question, or while waiting for the other person to reply."
)
CALL_SECTION_HEADER = "\n\nThis call:\n"
DEFAULT_CALL_SECTION = "You are calling on behalf of the owner of this phone number. Be polite and brief."
OPENING_CUE = (f'(The other person has just answered. You have already said: "{DISCLOSURE}" '
               "Now say, in one sentence, why you are calling.)")


def fixed_instructions(host_line: str | None) -> str:
    """The cached part: the phone instructions, plus the host line for a local brain (never sent to a remote one)."""
    return PHONE_INSTRUCTIONS + (f" {host_line}" if host_line else "")


def system_prompt(fixed: str, call_section: str | None) -> str:
    section = (call_section or "").strip() or DEFAULT_CALL_SECTION
    return fixed + CALL_SECTION_HEADER + section


def cache_probe_sections() -> tuple[str, str]:
    """Two call sections that differ from their first character: the model worker tokenises a prompt with each and
    caches the common prefix, which is then exactly the fixed part plus the header."""
    return "Ask whether they are open on Sunday.", "Book a table for two at eight."
