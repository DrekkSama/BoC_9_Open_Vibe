# Purpose: JSONL telemetry recorder — captures army state transitions and game summaries.
# Key Decisions: Inline append to data/telemetry.jsonl (events are rare, no batching needed).
#   Always-on (per user choice; deviates from AGENTS.md competition-safety — revisit before competition).
#   Bot version hardcoded here — keep in sync with pyproject.toml.
#   All fields are flat scalars or arrays — DuckDB read_json_auto() ingests into a clean table.
# Limitations: No async I/O (safe because events fire a handful of times per game, not per-unit).
#   DuckDB load: CREATE TABLE telem AS SELECT * FROM read_json_auto('data/telemetry.jsonl');

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from sc2.data import Result

# Keep in sync with pyproject.toml [tool.poetry] version
BOT_VERSION: str = "0.1.0"
SCHEMA_VERSION: int = 1
TELEMETRY_FILE: str = "telemetry.jsonl"
DATA_DIR: str = "data"


class TelemetryRecorder:
    """Appends telemetry events as JSONL to data/telemetry.jsonl.

    Events are written inline (one open-append-write per event). This is safe
    because state transitions fire at most a few times per game and the game
    summary fires once — never in a per-unit hot path.
    """

    def __init__(self, ai: Any) -> None:
        self._ai = ai
        self._match_id: str = self._build_match_id()
        self._file_path: Path = Path(DATA_DIR) / TELEMETRY_FILE

    # ── Public API ──────────────────────────────────────────────────────────

    def record_state_transition(
        self,
        from_state: str,
        to_state: str,
        game_time: float,
        army_supply: float,
        combat_sim_state: dict[str, Any],
    ) -> None:
        """Record a defend↔attack state transition.

        combat_sim_state is flattened to top-level columns for DuckDB:
        any_squad_engaged (bool), engaged_squad_ids (list[str]), engaged_count (int).
        """
        payload = {
            "event": "state_transition",
            "from": from_state,
            "to": to_state,
            "game_time": round(game_time, 2),
            "army_supply": round(army_supply, 1),
            "any_squad_engaged": combat_sim_state.get("any_squad_engaged", False),
            "engaged_squad_ids": combat_sim_state.get("engaged_squad_ids", []),
            "engaged_count": combat_sim_state.get("engaged_count", 0),
        }
        self._write(payload)

    def record_game_summary(
        self,
        result: Result,
        game_length: float,
        winner: str,
        map_name: str,
    ) -> None:
        """Record the end-of-game summary."""
        result_name = result.name if isinstance(result, Result) else str(result)
        payload = {
            "event": "game_summary",
            "result": result_name,
            "game_length_seconds": round(game_length, 2),
            "winner": winner,
            "map_name": map_name,
        }
        self._write(payload)

    def flush(self) -> None:
        """No-op — events are written inline. Kept for future batching."""
        pass

    # ── Internals ───────────────────────────────────────────────────────────

    def _build_match_id(self) -> str:
        """Generate a deterministic-per-run match ID."""
        ai = self._ai
        opponent_id: str = ai.opponent_id if ai.opponent_id else "local"
        map_name: str = ai.game_info.map_name.replace(" ", "_")
        startup_ts: str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{map_name}_{opponent_id}_{startup_ts}"

    def _common_fields(self) -> dict[str, Any]:
        """Fields included on every telemetry record."""
        ai = self._ai
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "schema_version": SCHEMA_VERSION,
            "bot_version": BOT_VERSION,
            "environment": "ladder" if "--LadderServer" in sys.argv else "local",
            "match_id": self._match_id,
            "matchup": f"{ai.race.name}_v_{ai.enemy_race.name}",
            "map_name": ai.game_info.map_name,
        }

    def _write(self, payload: dict[str, Any]) -> None:
        """Append one JSON line to the telemetry file. Never crashes the bot."""
        record: dict[str, Any] = {**self._common_fields(), **payload}
        try:
            os.makedirs(DATA_DIR, exist_ok=True)
            with open(self._file_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, default=str) + "\n")
        except OSError:
            # Telemetry is best-effort — never crash the bot over a log write.
            pass