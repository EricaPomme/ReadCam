"""Capture saving panel: output path, disk save, clipboard, bounding boxes."""

import logging
import sys
from pathlib import Path
from typing import Callable

import customtkinter as ctk
from tkinter import filedialog

from .theme import COLOURS, FONTS

logger = logging.getLogger(__name__)


class ToolTip:
    """Minimal hover tooltip for a widget.

    Shows a small yellow-on-dark tip after a short delay; avoid the
    heavyweight third-party dependency for a handful of labels.
    """

    def __init__(self, widget, text: str, delay_ms: int = 500):
        self.widget = widget
        self.text = text
        self.delay_ms = delay_ms
        self._after_id = None
        self._tip = None
        widget.bind("<Enter>", self._schedule, add="+")
        widget.bind("<Leave>", self._hide, add="+")
        widget.bind("<ButtonPress>", self._hide, add="+")

    def _schedule(self, _event=None) -> None:
        self._cancel()
        self._after_id = self.widget.after(self.delay_ms, self._show)

    def _cancel(self) -> None:
        if self._after_id is not None:
            self.widget.after_cancel(self._after_id)
            self._after_id = None

    def _show(self) -> None:
        if self._tip is not None:
            return
        x = self.widget.winfo_rootx() + self.widget.winfo_width() // 2
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 4
        self._tip = ctk.CTkToplevel(self.widget)
        self._tip.overrideredirect(True)
        self._tip.attributes("-topmost", True)
        self._tip.geometry(f"+{x}+{y}")
        ctk.CTkLabel(
            self._tip, text=self.text, font=FONTS["small"],
            fg_color="#1a1a2e", text_color="#ffffc8",
            corner_radius=4, padx=6, pady=3,
        ).pack()

    def _hide(self, _event=None) -> None:
        self._cancel()
        if self._tip is not None:
            self._tip.destroy()
            self._tip = None


class CapturePanel(ctk.CTkFrame):
    """Panel below the preview for saving the current frame.

    Elements: the output path (clicking it opens the directory chooser,
    it absorbs all spare space), Disk (save to the output directory),
    Clip (copy the image to the clipboard), and a BB checkbox controlling
    whether bounding boxes and labels are included in the capture.
    The panel only manages the UI; the app supplies the save/copy
    callbacks and the frame.
    """

    def __init__(
        self,
        master,
        on_disk: Callable,
        on_clip: Callable,
        **kwargs,
    ):
        super().__init__(master, **kwargs)
        self.configure(fg_color=COLOURS["card_bg"])

        self.on_disk = on_disk
        self.on_clip = on_clip

        self._output_dir = Path.home()
        self.include_boxes = ctk.BooleanVar(value=True)

        # Column 1 (the clickable path label) absorbs all spare space
        self.columnconfigure(1, weight=1)

        ctk.CTkLabel(
            self, text="\U0001F4BE", font=("Segoe UI Emoji", 20),  # Floppy disk
        ).grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")

        # The path label doubles as the Select button: clicking it opens
        # the directory chooser.
        self.path_label = ctk.CTkButton(
            self, text="", font=FONTS["small"], anchor="w",
            fg_color=COLOURS["bg"], hover_color=COLOURS["hover"],
            text_color=COLOURS["fg"], corner_radius=4,
            height=24, cursor="hand2",
            command=self._on_select,
        )
        self.path_label.grid(row=0, column=1, padx=4, pady=6, sticky="ew")
        self._update_path_label()

        self.disk_btn = ctk.CTkButton(
            self, text="Disk", font=FONTS["small"],
            width=70, height=24,
            command=self._on_disk,
        )
        self.disk_btn.grid(row=0, column=2, padx=4, pady=6)

        self.clip_btn = ctk.CTkButton(
            self, text="Clip", font=FONTS["small"],
            width=70, height=24,
            command=self._on_clip,
        )
        self.clip_btn.grid(row=0, column=3, padx=4, pady=6)

        if sys.platform != "win32":
            # Clipboard image capture is Windows-only (CF_DIB)
            self.clip_btn.configure(state="disabled")

        self.bb_check = ctk.CTkCheckBox(
            self, text="\U0001F5D7", font=("Segoe UI Emoji", 17),
            variable=self.include_boxes, width=50,
        )
        self.bb_check.grid(row=0, column=4, padx=(4, 8), pady=6)

        ToolTip(self.disk_btn, "Save to Disk")
        ToolTip(self.clip_btn, "Save to Clipboard")
        ToolTip(self.bb_check, "Overlay bounding box on output")

    @property
    def output_dir(self) -> Path:
        """The directory captures are written to."""
        return self._output_dir

    def _update_path_label(self) -> None:
        """Show the output path, shortened when it is under the home dir."""
        try:
            text = str(self._output_dir).replace(
                str(Path.home()), "~", 1,
            )
        except Exception:
            text = str(self._output_dir)
        self.path_label.configure(text=text)

    def _on_select(self) -> None:
        """Choose a new output directory."""
        chosen = filedialog.askdirectory(
            initialdir=str(self._output_dir),
            title="Select output directory",
            parent=self,
        )
        if not chosen:
            return
        self._output_dir = Path(chosen)
        self._update_path_label()
        logger.info("Capture output directory set to %s", self._output_dir)

    def _on_disk(self) -> None:
        """Save the current frame to the output directory."""
        self.on_disk()

    def _on_clip(self) -> None:
        """Copy the current frame to the clipboard."""
        self.on_clip()
