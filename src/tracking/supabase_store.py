"""
Persistenza opzionale del prediction log su Supabase (REST).

Usa la service role key solo lato server (Streamlit Secrets / env), mai nel browser.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

TABLE_NAME = "exchange_predictions"
ROW_ID = "main"


def _credentials() -> tuple[str | None, str | None]:
    url = os.environ.get("SUPABASE_URL", "").strip()
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY", "").strip()
    if url and key:
        return url, key
    try:
        import streamlit as st

        if hasattr(st, "secrets"):
            url = str(st.secrets.get("SUPABASE_URL", "") or "").strip() or None
            key = str(st.secrets.get("SUPABASE_SERVICE_ROLE_KEY", "") or "").strip() or None
            if url and key:
                return url, key
    except Exception:
        pass
    return None, None


def is_enabled() -> bool:
    u, k = _credentials()
    return bool(u and k)


def _headers(key: str) -> dict[str, str]:
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=representation",
    }


def fetch_payload() -> dict[str, Any] | None:
    """
    Legge il JSON dal database. Ritorna None se errore di rete/API.
    Ritorna dict vuoto strutturato se riga assente (primo utilizzo).
    """
    url_base, key = _credentials()
    if not url_base or not key:
        return None
    api = f"{url_base.rstrip('/')}/rest/v1/{TABLE_NAME}?id=eq.{ROW_ID}&select=payload"
    req = urllib.request.Request(api, headers=_headers(key), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
        rows = json.loads(raw)
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, TimeoutError):
        return None
    if not rows:
        return {"version": 1, "last_updated": "", "predictions": []}
    payload = rows[0].get("payload")
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        return {"version": 1, "last_updated": "", "predictions": []}
    return payload


def save_payload(data: dict[str, Any]) -> bool:
    """Aggiorna la riga `main`; se non esiste, la crea (POST)."""
    url_base, key = _credentials()
    if not url_base or not key:
        return False
    patch_url = f"{url_base.rstrip('/')}/rest/v1/{TABLE_NAME}?id=eq.{ROW_ID}"
    body = json.dumps({"payload": data}, ensure_ascii=False).encode("utf-8")
    headers = _headers(key)
    headers["Prefer"] = "return=representation"
    req = urllib.request.Request(patch_url, data=body, headers=headers, method="PATCH")
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            out = resp.read().decode("utf-8").strip()
        if out in ("[]", ""):
            return _post_new_row(url_base, key, data)
        return True
    except urllib.error.HTTPError as e:
        if e.code in (404, 405):
            return _post_new_row(url_base, key, data)
        return False
    except (urllib.error.URLError, TimeoutError):
        return False


def _post_new_row(url_base: str, key: str, data: dict[str, Any]) -> bool:
    api = f"{url_base.rstrip('/')}/rest/v1/{TABLE_NAME}"
    body = json.dumps({"id": ROW_ID, "payload": data}, ensure_ascii=False).encode("utf-8")
    headers = _headers(key)
    headers["Prefer"] = "return=minimal"
    req = urllib.request.Request(api, data=body, headers=headers, method="POST")
    try:
        urllib.request.urlopen(req, timeout=45)
        return True
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return False


__all__ = ["TABLE_NAME", "ROW_ID", "fetch_payload", "is_enabled", "save_payload"]
