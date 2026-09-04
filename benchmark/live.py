"""Live streaming display.

TTY (stderr): a rich `Live` panel — per-request table (status / TTFT /
reasoning tokens / answer tokens / tokens-per-sec) plus a scrolling trace of
the model's streamed reasoning and tool calls.

Non-TTY (captured runs): one concise `[done]` line per request to stderr, so
stdout stays clean for `METRIC` lines.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass, field
from typing import Any

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table


@dataclass
class ReqState:
    name: str
    phase: str = ""
    status: str = "pending"
    started_at: float = 0.0
    ttft_s: float | None = None
    elapsed_s: float = 0.0
    reasoning_chars: int = 0
    content_chars: int = 0
    reasoning_tokens: int = 0
    completion_tokens: int = 0
    tps: float = 0.0
    tool_calls: list = field(default_factory=list)
    finish_reason: str = ""
    error: str = ""
    trace: str = ""
    reasoning_starved: bool = False

class Tracker:
    def __init__(self) -> None:
        self.states: list[ReqState] = []
        self.active: ReqState | None = None

    def add(self, name: str, phase: str) -> ReqState:
        s = ReqState(name=name, phase=phase)
        self.states.append(s)
        return s

    def touch(self, s: ReqState) -> None:
        self.active = s


def _fmt_dur(v: float | None) -> str:
    return f"{v * 1000:.0f}ms" if v is not None else "—"


_FINAL = ("done", "ok", "fail", "error")


def _fmt_tok(tokens: int, chars: int, done: bool) -> str:
    return str(tokens) if done else f"{chars}c"


def _fmt_tps(s: ReqState) -> str:
    if s.status in _FINAL and s.tps > 0:
        return f"{s.tps:.1f}"
    if s.elapsed_s > 0.01:
        return f"~{(s.reasoning_chars + s.content_chars) / 4 / s.elapsed_s:.1f}"
    return "—"


class _LiveView:
    """Renderable that rebuilds from live tracker state on every render.

    rich's Live refresh thread calls `__rich_console__` repeatedly, so the panel
    always shows current state even if no `update()` calls arrive.
    """

    def __init__(self, ui: LiveUI) -> None:
        self.ui = ui

    def __rich_console__(self, console: Console, options) -> Any:
        yield self.ui._render()


class LiveUI:
    def __init__(self, tracker: Tracker, header: str, enabled: bool) -> None:
        self.tracker = tracker
        self.header = header
        self.enabled = enabled
        self.console = Console(stderr=True)
        self.live: Live | None = None
        self._last = 0.0
        if enabled:
            # auto_refresh=True (default) spawns a thread that re-renders the
            # live view periodically, so the panel always updates even if no
            # chunks arrive; update() adds immediate redraws during streaming.
            self.live = Live(_LiveView(self), console=self.console, refresh_per_second=4)

    def start(self) -> None:
        if self.live:
            self.live.start()

    def stop(self) -> None:
        if self.live:
            self.live.stop()

    def update(self, force: bool = False) -> None:
        if not self.live:
            return
        now = time.monotonic()
        if not force and now - self._last < 0.1:
            return
        self._last = now
        self.live.refresh()

    def note_done(self, s: ReqState) -> None:
        if self.live:
            self.update(force=True)
            return
        tps = f"{s.tps:.1f} t/s" if s.tps else ""
        tok = f"R={s.reasoning_tokens}t A={max(s.completion_tokens - s.reasoning_tokens, 0)}t"
        warn = " \033[1;33m⚠️ [BURNING MAX TOKENS IN REASONING]\033[0m" if s.reasoning_starved else ""
        print(
            f"[done] {s.name:>6} {s.phase:<14} ttft={_fmt_dur(s.ttft_s)} {tok:<16} {tps} {s.finish_reason or s.error}{warn}",
            file=sys.stderr,
        )

    def _render(self):
        table = Table(box=box.SIMPLE_HEAD, expand=True, pad_edge=False)
        table.add_column("req", no_wrap=True, style="bold")
        table.add_column("phase", no_wrap=True)
        table.add_column("status", no_wrap=True)
        table.add_column("ttft", justify="right", no_wrap=True)
        table.add_column("R", justify="right", no_wrap=True)
        table.add_column("A", justify="right", no_wrap=True)
        table.add_column("t/s", justify="right", no_wrap=True)
        table.add_column("result", no_wrap=True)
        max_rows = 10
        total_states = len(self.tracker.states)
        if total_states > max_rows:
            hidden_count = total_states - max_rows
            table.add_row(
                f"[dim]+{hidden_count} done[/dim]",
                "[dim]earlier[/dim]",
                "[dim]completed[/dim]",
                "—",
                "—",
                "—",
                "—",
                "[dim]…[/dim]",
            )
            visible_states = self.tracker.states[-max_rows:]
        else:
            visible_states = self.tracker.states

        for s in visible_states:
            final = s.status in _FINAL
            table.add_row(
                s.name,
                s.phase,
                s.status,
                _fmt_dur(s.ttft_s),
                _fmt_tok(s.reasoning_tokens, s.reasoning_chars, final),
                _fmt_tok(max(s.completion_tokens - s.reasoning_tokens, 0), s.content_chars, final),
                _fmt_tps(s),
                f"[bold yellow]⚠️ STARVED[/bold yellow] {s.finish_reason or 'length'}"[:44] if s.reasoning_starved else (", ".join(s.tool_calls) or s.finish_reason or s.error or "")[:44],
            )
        active = self.tracker.active
        trace = active.trace if active else ""
        if not trace:
            trace = "(waiting for first token…)"
        trace_panel = Panel(
            trace[-2000:],
            title=f"live trace — {active.name if active else '—'}{' [bold yellow][THINKING... burning tokens][/bold yellow]' if active and active.reasoning_chars > 2000 and active.content_chars == 0 else (' [dim][thinking][/dim]' if active and active.reasoning_chars > 0 and active.content_chars == 0 else '')}",
            border_style="dim",
            style="dim",
            height=8,
        )
        return Group(Panel(self.header, style="bold cyan"), table, trace_panel)
