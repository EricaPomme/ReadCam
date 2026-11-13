"""Results list panel with per-row copy buttons."""

import logging

import customtkinter as ctk

from ..results import Result
from ..util import copy_to_clipboard, format_relative_time, truncate
from .theme import COLOURS, FONTS

logger = logging.getLogger(__name__)

# Hover shades for the badge copy buttons (keyed like theme.COLOURS)
_BADGE_HOVER = {
    "badge_qr": "#27ae60",
    "badge_barcode": "#2980b9",
    "badge_text": "#d35400",
}


def _badge_key(det_type: str) -> str:
    """Map a detection type to its colour key."""
    if det_type == "QR":
        return "badge_qr"
    elif det_type == "TXT":
        return "badge_text"
    return "badge_barcode"


class ToolTip:
    """Minimal tooltip that appears on hover, below a widget."""

    def __init__(self, widget, text: str, delay_ms: int = 600):
        self._widget = widget
        self._text = text
        self._delay_ms = delay_ms
        self._tip = None
        self._after_id = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self._widget.after(self._delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self._widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        x = self._widget.winfo_rootx() + 10
        y = self._widget.winfo_rooty() + self._widget.winfo_height() + 4
        self._tip = tw = ctk.CTkToplevel(self._widget)
        tw.overrideredirect(True)
        tw.attributes("-topmost", True)
        label = ctk.CTkLabel(
            tw, text=self._text, font=FONTS["small"],
            fg_color=COLOURS["bg"], corner_radius=4,
            padx=6, pady=4, wraplength=420, justify="left",
        )
        label.pack()
        tw.wm_geometry(f"+{x}+{y}")

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class ResultsPanel(ctk.CTkScrollableFrame):
    """Scrollable list of scanned results with per-row Copy buttons."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLOURS["card_bg"])

        self.header = ctk.CTkLabel(
            self,
            text="Results",
            font=FONTS["heading"],
            text_color=COLOURS["fg"],
        )
        self.header.pack(fill="x", padx=15, pady=(10, 5))

        self._result_widgets: dict[tuple[str, str], ctk.CTkFrame] = {}
        # Per-row child widgets, so updates don't rely on tree-walking.
        self._row_meta_labels: dict[tuple[str, str], ctk.CTkLabel] = {}
        self._row_bars: dict[tuple[str, str], ctk.CTkProgressBar] = {}
        # Last displayed key order, so re-packing only happens when the
        # sorted order actually changes (pack() on an already-managed
        # widget moves it to the bottom, so packing in sorted order
        # reorders the list).
        self._displayed_order: list[tuple[str, str]] = []

    def update_results(self, results: list[Result]) -> None:
        """Refresh the list of results, adding, updating, and removing rows."""
        current_keys = {(r.type, r.value) for r in results}

        # Bars are normalised dynamically against the strongest current
        # result, so the bar ranks results within the live set rather than
        # saturating at CONFIDENCE_MAX for everything long-lived.
        max_conf = max((r.confidence for r in results), default=1.0)

        # Remove stale widgets
        for key in list(self._result_widgets.keys()):
            if key not in current_keys:
                widget = self._result_widgets.pop(key)
                self._row_meta_labels.pop(key, None)
                self._row_bars.pop(key, None)
                widget.destroy()

        # Add or update rows
        for result in results:
            key = (result.type, result.value)
            if key not in self._result_widgets:
                widget = self._build_result_row(result, max_conf)
                self._result_widgets[key] = widget
                widget.pack(fill="x", padx=10, pady=(0, 5))
            else:
                self._update_result_row(key, result, max_conf)

        # Re-pack rows so the displayed order matches the sorted order,
        # but only when the order actually changed. pack() on an
        # already-managed widget moves it to the bottom, so packing in
        # sorted order reorders the list; skipping the loop when the
        # order is unchanged avoids the per-refresh geometry churn.
        new_order = [(r.type, r.value) for r in results]
        if new_order != self._displayed_order:
            for result in results:
                widget = self._result_widgets[(result.type, result.value)]
                widget.pack(fill="x", padx=10, pady=(0, 5))
            self._displayed_order = new_order

        # Update header count
        self.header.configure(text=f"Results ({len(results)})")

    def _build_result_row(self, result: Result, max_conf: float) -> ctk.CTkFrame:
        """Create a new row widget for a single result."""
        row = ctk.CTkFrame(self, fg_color=COLOURS["card_bg"], corner_radius=8)

        # Layout: badge (click to copy) | content
        row.columnconfigure(0, weight=0, minsize=64)  # Badge / copy button
        row.columnconfigure(1, weight=1)               # Content

        # --- Badge doubles as the copy button ---
        badge_colour = self._badge_colour(result.type)
        badge = ctk.CTkButton(
            row,
            text=result.type,
            font=FONTS["badge"],
            text_color="white",
            fg_color=badge_colour,
            hover_color=self._badge_hover(result.type),
            corner_radius=6,
            width=60,
            height=44,
            command=lambda v=result.value: self._on_copy(v),
        )
        badge.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        ToolTip(badge, f"Copy: {result.value}")

        # --- Value and metadata ---
        content_frame = ctk.CTkFrame(row, fg_color="transparent")
        content_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        value_text = truncate(result.value, 80)
        value_label = ctk.CTkLabel(
            content_frame,
            text=value_text,
            font=FONTS["mono"],
            text_color=COLOURS["fg"],
            wraplength=480,
        )
        value_label.pack(fill="x", anchor="w")

        meta_label = ctk.CTkLabel(
            content_frame,
            text=ResultRowMeta.format_meta(result),
            font=FONTS["small"],
            text_color="#888888",
        )
        meta_label.pack(fill="x", anchor="w", pady=(2, 0))
        self._row_meta_labels[(result.type, result.value)] = meta_label

        confidence_pct = self._confidence_pct(result, max_conf)
        progress = ctk.CTkProgressBar(
            content_frame,
            height=8,
            border_width=0,
        )
        progress.set(confidence_pct)
        progress.pack(fill="x", pady=(2, 0))
        self._row_bars[(result.type, result.value)] = progress

        return row

    def _update_result_row(self, key: tuple[str, str], result: Result, max_conf: float) -> None:
        """Update metadata in an existing row without rebuilding it."""
        self._row_bars[key].set(self._confidence_pct(result, max_conf))
        self._row_meta_labels[key].configure(text=ResultRowMeta.format_meta(result))

    @staticmethod
    def _confidence_pct(result: Result, max_conf: float) -> float:
        """Normalise a result's confidence relative to the strongest current result."""
        return min(result.confidence / max(max_conf, 1.0), 1.0)

    @staticmethod
    def _on_copy(value: str) -> None:
        """Copy the result value to clipboard."""
        ok = copy_to_clipboard(value)
        if ok:
            logger.info("Copied to clipboard: %r", value)
        else:
            logger.warning("Clipboard copy failed for %r", value)

    @staticmethod
    def _badge_colour(det_type: str) -> str:
        """Return the hex colour for a detection type badge."""
        return COLOURS[_badge_key(det_type)]

    @staticmethod
    def _badge_hover(det_type: str) -> str:
        """Return the hover hex colour for a detection type badge."""
        return _BADGE_HOVER[_badge_key(det_type)]


class ResultRowMeta:
    """Formats the per-row metadata line."""

    @staticmethod
    def format_meta(result: Result) -> str:
        """Build 'seen 12x · visible 45s · 2s ago' style metadata."""
        return (
            f"seen {result.hits}x \u00b7 "
            f"visible {ResultRowMeta._fmt_duration(result.visible_seconds)} \u00b7 "
            f"{format_relative_time(result.last_seen)}"
        )

    @staticmethod
    def _fmt_duration(seconds: float) -> str:
        """Format a duration compactly ('45s', '2m05s', '1h02m')."""
        seconds = int(seconds)
        if seconds < 60:
            return f"{seconds}s"
        minutes, secs = divmod(seconds, 60)
        if minutes < 60:
            return f"{minutes}m{secs:02d}s"
        hours, minutes = divmod(minutes, 60)
        return f"{hours}h{minutes:02d}m"