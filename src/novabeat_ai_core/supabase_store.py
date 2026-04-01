from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str
    table: str = "generations"


class SupabaseGenerationStore:
    def __init__(self, config: SupabaseConfig):
        if not config.url or not config.key:
            raise ValueError("Supabase URL y key son obligatorios")
        self.config = config

    def save_generation(self, provider: str, prompt: str, output_file: str | Path, extra: Dict[str, Any] | None = None) -> None:
        payload = {
            "provider": provider,
            "prompt": prompt,
            "output_file": str(output_file),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "extra": extra or {},
        }
        body = json.dumps(payload).encode("utf-8")
        endpoint = f"{self.config.url.rstrip('/')}/rest/v1/{self.config.table}"
        req = urllib.request.Request(
            endpoint,
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "apikey": self.config.key,
                "Authorization": f"Bearer {self.config.key}",
                "Prefer": "return=minimal",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=15):
                return
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Error guardando en Supabase ({exc.code}): {detail}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"No se pudo conectar a Supabase: {exc}") from exc
