"""Panel of live filter tuning below the preview, split into two cells.

Left cell: OCR thresholds (text confidence, length, hit counts,
result decay/removal). Right cell: image adjustments (resolution,
colour mode, contrast, saturation, sharpness) consumed by the
capture thread's ``ImageEnhancer``.
"""

import logging
from typing import Callable, Optional

import customtkinter as ctk

from ..config import (
    COLOUR_MODES,
    CONFIDENCE_DECAY_DELAY,
    CONFIDENCE_REMOVE_DELAY,
    IMAGE_CONTRAST,
    IMAGE_MEDIAN_BLUR,
    IMAGE_SATURATION,
    IMAGE_SHARPNESS,
    OCR_DENOISE,
    RESOLUTIONS,
    TEXT_MIN_ALNUM_RATIO,
    TEXT_MIN_CONFIDENCE,
    TEXT_MIN_LETTERS,
    TEXT_MIN_LENGTH,
    TEXT_REQUIRED_HITS,
)
from .theme import COLOURS, FONTS

logger = logging.getLogger(__name__)

# OCR slider specs: (holder key, label, min, max, steps, default, formatter)
_OCR_SLIDERS = [
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

# Image adjustment slider specs (same shape as _OCR_SLIDERS)
_IMAGE_SLIDERS = [
    ("contrast", "Contrast", 0.5, 2.0, 15,
     IMAGE_CONTRAST, lambda v: f"{v:.2f}"),
    ("saturation", "Saturation", 0.0, 2.0, 20,
     IMAGE_SATURATION, lambda v: f"{v:.2f}"),
    ("median_blur", "Median blur", 0, 5, 5,
     IMAGE_MEDIAN_BLUR, lambda v: "off" if v < 1 else f"{2 * int(v) + 1}"),
    ("sharpness", "Sharpness", 0.0, 2.0, 20,
     IMAGE_SHARPNESS, lambda v: f"{v:.2f}"),
]

# Checkbox specs: (holder key, label, default)
_CHECKBOXES = [
    ("ocr_denoise", "OCR denoise", OCR_DENOISE),
]

# Dropdown specs: (holder key, label, options, default)
_DROPDOWNS = [
    ("resolution", "Resolution", RESOLUTIONS, RESOLUTIONS[0]),
    ("colour_mode", "Colour mode", COLOUR_MODES, COLOUR_MODES[0]),
]


class FilterPanel(ctk.CTkFrame):
    """Panel of OCR and image controls bound to a shared filter holder.

    Controls write directly into *filters* (a dict shared with
    ``ResultsManager``, ``OCRDecoder`` and the capture thread), so
    changes take effect on the next pass without restarting anything.
    *on_change_end* (if given) is called after a slider release,
    e.g. to request an immediate OCR pass for fast feedback.
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
        self._menus: dict[str, ctk.CTkOptionMenu] = {}
        self._checkboxes: dict[str, ctk.CTkCheckBox] = {}

        # Seed any missing holder keys with the config defaults so the
        # consumers' fallbacks and the control positions always agree.
        for key, _label, _lo, _hi, _steps, default, _fmt in _OCR_SLIDERS + _IMAGE_SLIDERS:
            self._filters.setdefault(key, default)
        for key, _label, _options, default in _DROPDOWNS:
            self._filters.setdefault(key, default)
        for key, _label, default in _CHECKBOXES:
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

        # Body: two side-by-side cells (OCR left, Image right)
        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="x", padx=0, pady=(2, 6))
        body.columnconfigure(0, weight=1, uniform="cells")
        body.columnconfigure(1, weight=1, uniform="cells")

        ocr_cell = ctk.CTkFrame(body, fg_color="transparent")
        ocr_cell.grid(row=0, column=0, sticky="nsew", padx=(8, 4))
        image_cell = ctk.CTkFrame(body, fg_color="transparent")
        image_cell.grid(row=0, column=1, sticky="nsew", padx=(4, 8))

        ctk.CTkLabel(
            ocr_cell, text="OCR", font=FONTS["small"],
            anchor="w",
        ).pack(fill="x", pady=(0, 2))
        for spec in _OCR_SLIDERS:
            self._build_slider(spec, ocr_cell)

        ctk.CTkLabel(
            image_cell, text="Image", font=FONTS["small"],
            anchor="w",
        ).pack(fill="x", pady=(0, 2))
        for spec in _DROPDOWNS:
            self._build_dropdown(spec, image_cell)
        for spec in _IMAGE_SLIDERS:
            self._build_slider(spec, image_cell)
        for spec in _CHECKBOXES:
            self._build_checkbox(spec, image_cell)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_slider(self, spec: tuple, cell: ctk.CTkFrame) -> None:
        """Add a label + slider + value row for one tunable."""
        key, label, lo, hi, steps, default, fmt = spec

        row = ctk.CTkFrame(cell, fg_color="transparent")
        row.pack(fill="x", pady=2)

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

    def _build_dropdown(self, spec: tuple, cell: ctk.CTkFrame) -> None:
        """Add a label + option-menu row for one discrete setting."""
        key, label, options, default = spec

        row = ctk.CTkFrame(cell, fg_color="transparent")
        row.pack(fill="x", pady=2)

        name = ctk.CTkLabel(
            row, text=label, font=FONTS["small"], anchor="w", width=120,
        )
        name.grid(row=0, column=0, sticky="w")

        menu = ctk.CTkOptionMenu(
            row,
            values=options,
            font=FONTS["small"],
            command=lambda v, k=key: self._on_dropdown(k, v),
        )
        menu.set(default)
        menu.grid(row=0, column=1, padx=6, sticky="ew")
        row.columnconfigure(1, weight=1)

        self._menus[key] = menu

    def _build_checkbox(self, spec: tuple, cell: ctk.CTkFrame) -> None:
        """Add a checkbox row for one boolean toggle."""
        key, label, default = spec

        box = ctk.CTkCheckBox(
            cell,
            text=label,
            font=FONTS["small"],
            command=lambda k=key: self._on_checkbox(k),
        )
        if default:
            box.select()
        else:
            box.deselect()
        box.pack(fill="x", pady=2)

        self._checkboxes[key] = box

    # ------------------------------------------------------------------
    # Control callbacks
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

    def _on_dropdown(self, key: str, value: str) -> None:
        """Write the new selection into the holder."""
        self._filters[key] = value
        logger.info("%s changed to %s", key, value)

    def _on_checkbox(self, key: str) -> None:
        """Write the checkbox state into the holder."""
        self._filters[key] = bool(self._checkboxes[key].get())
        logger.info("%s changed to %s", key, self._filters[key])

    def _on_release(self, _event=None) -> None:
        """Slider released (or programmatic): optionally trigger an
        immediate OCR pass."""
        if self.on_change_end:
            self.on_change_end()

    def _reset(self) -> None:
        """Restore every control to its config default."""
        for key, _label, _lo, _hi, _steps, default, fmt in _OCR_SLIDERS + _IMAGE_SLIDERS:
            self._filters[key] = float(default)
            self._sliders[key].set(default)
            self._labels[key].configure(text=fmt(default))
        for key, _label, _options, default in _DROPDOWNS:
            self._filters[key] = default
            self._menus[key].set(default)
        for key, _label, default in _CHECKBOXES:
            self._filters[key] = default
            if default:
                self._checkboxes[key].select()
            else:
                self._checkboxes[key].deselect()
        logger.info("Filter thresholds reset to defaults")
        # Reset changes thresholds without a slider release, so fire the
        # change hook manually (drops invalidated results, re-runs OCR).
        self._on_release()
