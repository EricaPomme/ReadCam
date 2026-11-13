"""Panel of live filter-tuning sliders below the preview."""

import logging
from typing import Callable, Optional

import customtkinter as ctk

from ..config import (
    CONFIDENCE_DECAY_DELAY,
    CONFIDENCE_REMOVE_DELAY,
    TEXT_MIN_ALNUM_RATIO,
    TEXT_MIN_CONFIDENCE,
    TEXT_MIN_LETTERS,
    TEXT_MIN_LENGTH,
    TEXT_REQUIRED_HITS,
)
from .theme import COLOURS, FONTS

logger = logging.getLogger(__name__)

# Slider specs: (holder key, label, min, max, steps, default, formatter)
_SLIDERS = [
    ("min_confidence", "Text confidence", 0, 100, 20,
     TEXT_MIN_CONFIDENCE, lambda v: f"{v:.0f}"),
    ("min_length", "Min length", 1, 20, 19,
     TEXT_MIN_LENGTH, lambda v: f"{v:.0f}"),
    ("required_hits", "Required hits", 1, 10, 9,
     TEXT_REQUIRED_HITS, lambda v: f"{v:.0f}"),
    ("min_alnum_ratio", "Alnum ratio", 0.0, 1.0, 20,
     TEXT_MIN_ALNUM_RATIO, lambda v: f"{v:.2f}"),
    ("min_letters", "Min letters", 0, 10, 10,
     TEXT_MIN_LETTERS, lambda v: f"{v:.0f}"),
    ("decay_delay", "Decay delay (s)", 0, 30, 30,
     CONFIDENCE_DECAY_DELAY, lambda v: f"{v:.0f}"),
    ("remove_delay", "Remove delay (s)", 10, 120, 22,
     CONFIDENCE_REMOVE_DELAY, lambda v: f"{v:.0f}"),
]


class FilterPanel(ctk.CTkFrame):
    """Panel of sliders bound to a shared filter holder.

    Each slider writes directly into *filters* (a dict shared with
    ``ResultsManager`` and ``OCRDecoder``), so changes take effect on
    the next OCR pass without restarting anything. *on_change_end*
    (if given) is called on slider release, e.g. to request an
    immediate OCR pass for fast feedback.
    """

    def __init__(
        self,
        master,
        filters: dict,
        on_change_end: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLOURS["card_bg"])

        self._filters = filters
        self.on_change_end = on_change_end
        self._sliders: dict[str, ctk.CTkSlider] = {}
        self._labels: dict[str, ctk.CTkLabel] = {}

        # Seed any missing holder keys with the config defaults so the
        # consumers' fallbacks and the slider positions always agree.
        for key, _label, _lo, _hi, _steps, default, _fmt in _SLIDERS:
            self._filters.setdefault(key, default)

        # Header row: title + reset button
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", padx=8, pady=(4, 0))

        ctk.CTkLabel(
            header,
            text="Filters",
            font=FONTS["body"],
            anchor="w",
        ).pack(side="left")

        ctk.CTkButton(
            header,
            text="Reset",
            font=FONTS["small"],
            width=60,
            height=24,
            command=self._reset,
        ).pack(side="right")

        # Body: one label + slider row per tunable
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="x", padx=0, pady=(2, 6))

        for spec in _SLIDERS:
            self._build_row(spec, body)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_row(self, spec: tuple, body: ctk.CTkFrame) -> None:
        """Add a label + slider + value row for one tunable."""
        key, label, lo, hi, steps, default, fmt = spec

        row = ctk.CTkFrame(body, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=2)

        name = ctk.CTkLabel(
            row, text=label, font=FONTS["small"], anchor="w", width=120,
        )
        name.grid(row=0, column=0, sticky="w")

        slider = ctk.CTkSlider(
            row,
            from_=lo, to=hi,
            number_of_steps=steps,
            command=lambda v, k=key, f=fmt: self._on_slide(k, v, f),
        )
        slider.set(default)
        slider.grid(row=0, column=1, padx=6, sticky="ew")
        slider.bind("<ButtonRelease-1>", self._on_release)
        row.columnconfigure(1, weight=1)

        value = ctk.CTkLabel(
            row, text=fmt(default), font=FONTS["small"], width=40,
        )
        value.grid(row=0, column=2, sticky="e")

        self._sliders[key] = slider
        self._labels[key] = value

    # ------------------------------------------------------------------
    # Slider callbacks
    # ------------------------------------------------------------------

    def _on_slide(self, key: str, value, fmt: Callable) -> None:
        """Write the new value into the holder and refresh the label."""
        value = float(value)
        self._filters[key] = value
        self._labels[key].configure(text=fmt(value))

        # Guardrail: decay delay must stay below remove delay
        if key == "decay_delay":
            remove = self._filters.get("remove_delay", CONFIDENCE_REMOVE_DELAY)
            if value >= remove:
                new_remove = min(120.0, value + 5.0)
                self._filters["remove_delay"] = new_remove
                self._sliders["remove_delay"].set(new_remove)
                self._labels["remove_delay"].configure(text=f"{new_remove:.0f}")
        elif key == "remove_delay":
            decay = self._filters.get("decay_delay", CONFIDENCE_DECAY_DELAY)
            if value <= decay:
                new_decay = max(0.0, value - 5.0)
                self._filters["decay_delay"] = new_decay
                self._sliders["decay_delay"].set(new_decay)
                self._labels["decay_delay"].configure(text=f"{new_decay:.0f}")

    def _on_release(self, _event) -> None:
        """Slider released: optionally trigger an immediate OCR pass."""
        if self.on_change_end:
            self.on_change_end()

    def _reset(self) -> None:
        """Restore every slider to its config default."""
        for key, _label, _lo, _hi, _steps, default, fmt in _SLIDERS:
            self._filters[key] = float(default)
            self._sliders[key].set(default)
            self._labels[key].configure(text=fmt(default))
        logger.info("Filter thresholds reset to defaults")
