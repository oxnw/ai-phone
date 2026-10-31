"""Caller-ID changes shared by setup and the line CLI. Carrier code stays on this side of the daemon."""
from __future__ import annotations

import time
from pathlib import Path

from ai_phone import paths
from ai_phone.carriers import managed_caller_id
from ai_phone.config import E164_RE, ConfigError, load_config
from ai_phone.lines.config_writer import read_raw, update_line, write_raw
from ai_phone.lines.journal import Journal, line_lock


def caller_id_line(config_path: Path, line_id: str) -> dict:
    line = (read_raw(config_path).get("lines") or {}).get(line_id)
    if not isinstance(line, dict):
        raise ValueError(f"no line {line_id}; add a phone line before choosing its caller ID")
    if line.get("kind", "trunk") != "trunk":
        raise ValueError(f"line {line_id} uses a SIP address; choose a phone line to change its caller ID")
    provider = line.get("provider")
    if provider is not None and (not isinstance(provider, dict) or provider.get("state", "pending") != "ready"):
        raise ValueError(f"line {line_id} is not ready; finish its setup before choosing its caller ID")
    if line.get("from_user", "number") == "account" and not managed_caller_id(line):
        raise ValueError(f"line {line_id} uses its SIP account in From; configure the caller ID with your provider")
    return line


def eligible_caller_id(line: dict, number: str, *, verified: bool = False) -> None:
    if not isinstance(number, str) or not E164_RE.fullmatch(number):
        raise ValueError("choose a phone number in E.164 format, like +15555550100")
    carrier = (line.get("provider") or {}).get("name", "")
    allowed = any(isinstance(v, dict) and v.get("number") == number and v.get("provider", "") == carrier
                  for v in line.get("verified_caller_ids") or [])
    if number != line.get("number") and not allowed and not verified:
        raise ValueError("that number is not verified for this line; verify it first or choose the line's own number")


def caller_id_journal(line_id: str, line: dict) -> Journal:
    path = paths.lines_dir() / f"{line_id}.journal.json"
    if not path.exists():
        raise ValueError(f"line {line_id}'s provisioning journal is missing; restore it before changing the carrier caller ID")
    journal = Journal.open(paths.lines_dir(), line_id, line["provider"]["name"])
    resources = journal.resources()
    recorded = line["provider"].get("resources") or {}
    if not recorded or any(resources.get(key) != value for key, value in recorded.items()):
        raise ValueError(f"line {line_id}'s provisioning journal does not match its carrier resources")
    return journal


def apply_caller_id(config_path: Path, line_id: str, number: str, *, provider=None, verified: bool = False,
                    resume: bool = False) -> None:
    """Apply a remote change before saving locally; a failed local save leaves a resumable carrier change."""
    with line_lock(paths.lines_dir()):
        load_config(config_path)
        line = caller_id_line(config_path, line_id)
        journal = None
        if managed_caller_id(line):
            if provider is None:
                raise ValueError("this caller-ID change needs the line's carrier credentials")
            journal = caller_id_journal(line_id, line)
            provider.validate_context(journal)
            pending = provider.pending_caller_id(journal)
            if pending and pending != number:
                raise ValueError(f"line {line_id} has an unfinished caller-ID change; run ai-phone lines resume {line_id} first")
            if resume and pending != number:
                raise ValueError("there is no matching caller-ID change to resume")
        eligible_caller_id(line, number, verified=verified or bool(resume and journal))
        if journal:
            confirmed = provider.apply_caller_id(journal, number)
            if confirmed != number:
                raise ValueError("the carrier did not confirm the selected caller ID; the local setting was kept")
        verification = list(line.get("verified_caller_ids") or [])
        if (verified or resume) and number != line.get("number"):
            verification = [v for v in verification if v.get("number") != number]
            verification.append({"number": number, "provider": (line.get("provider") or {}).get("name", ""),
                                 "verified_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())})
        before = read_raw(config_path)
        fields = {"caller_id": number}
        if "verified_caller_ids" in line or verified or resume:
            fields["verified_caller_ids"] = verification
        update_line(config_path, line_id, **fields)
        try:
            load_config(config_path)
        except ConfigError:
            write_raw(config_path, before)
            raise
        if journal:
            provider.finish_caller_id(journal)
