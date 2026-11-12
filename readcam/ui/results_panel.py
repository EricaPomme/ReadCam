"""Results list panel with per-row copy buttons."""

import customtkinter as ctk

from ..results import Result
from ..util import copy_to_clipboard, format_relative_time, truncate
from .theme import COLOURS, FONTS


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

    def update_results(self, results: list[Result]) -> None:
        """Refresh the list of results, adding, updating, and removing rows."""
        current_keys = {(r.type, r.value) for r in results}

        # Remove stale widgets
        for key in list(self._result_widgets.keys()):
            if key not in current_keys:
                widget = self._result_widgets.pop(key)
                widget.destroy()

        # Add or update rows
        for result in results:
            key = (result.type, result.value)
            if key not in self._result_widgets:
                widget = self._build_result_row(result)
                self._result_widgets[key] = widget
                widget.pack(fill="x", padx=10, pady=(0, 5))
            else:
                self._update_result_row(self._result_widgets[key], result)

        # Update header count
        self.header.configure(text=f"Results ({len(results)})")

    def _build_result_row(self, result: Result) -> ctk.CTkFrame:
        """Create a new row widget for a single result."""
        row = ctk.CTkFrame(self, fg_color=COLOURS["card_bg"], corner_radius=8)

        # Layout: badge | content (value + meta + bar) | copy button
        row.columnconfigure(0, weight=0, minsize=50)  # Badge
        row.columnconfigure(1, weight=1)               # Content
        row.columnconfigure(2, weight=0, minsize=65)   # Copy

        # --- Badge ---
        badge_colour = self._badge_colour(result.type)
        badge = ctk.CTkLabel(
            row,
            text=result.type,
            font=FONTS["badge"],
            text_color="white",
            fg_color=badge_colour,
            corner_radius=4,
            width=50,
        )
        badge.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        # --- Value and metadata ---
        content_frame = ctk.CTkFrame(row, fg_color="transparent")
        content_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        value_text = truncate(result.value, 60)
        value_label = ctk.CTkLabel(
            content_frame,
            text=value_text,
            font=FONTS["mono"],
            text_color=COLOURS["fg"],
            wraplength=300,
        )
        value_label.pack(fill="x", anchor="w")

        meta_label = ctk.CTkLabel(
            content_frame,
            text=f"{result.type} \u00b7 {format_relative_time(result.last_seen)}",
            font=FONTS["small"],
            text_color="#888888",
        )
        meta_label.pack(fill="x", anchor="w", pady=(2, 0))

        confidence_pct = min(result.confidence / 10.0, 1.0)
        progress = ctk.CTkProgressBar(
            content_frame,
            width=100,
            height=8,
            border_width=0,
        )
        progress.set(confidence_pct)
        progress.pack(anchor="w", pady=(2, 0))

        # --- Copy button ---
        copy_btn = ctk.CTkButton(
            row,
            text="Copy",
            font=FONTS["small"],
            fg_color=COLOURS["success"],
            hover_color="#27ae60",
            width=60,
            height=28,
            command=lambda v=result.value: self._on_copy(v),
        )
        copy_btn.grid(row=0, column=2, padx=5, pady=5, sticky="nsew")

        return row

    def _update_result_row(self, row: ctk.CTkFrame, result: Result) -> None:
        """Update metadata in an existing row without rebuilding it."""
        confidence_pct = min(result.confidence / 10.0, 1.0)
        for child in row.winfo_children():
            if isinstance(child, ctk.CTkProgressBar):
                child.set(confidence_pct)
            elif isinstance(child, ctk.CTkFrame) and child.grid_info().get("column") in (None, 1):
                # content frame -- find the meta label
                for subchild in child.winfo_children():
                    if isinstance(subchild, ctk.CTkLabel):
                        txt = subchild.cget("text")
                        if "\u00b7" in txt:  # contains the middle-dot separator
                            subchild.configure(
                                text=f"{result.type} \u00b7 {format_relative_time(result.last_seen)}"
                            )
                            break

    @staticmethod
    def _on_copy(value: str) -> None:
        """Copy the result value to clipboard."""
        copy_to_clipboard(value)

    @staticmethod
    def _badge_colour(det_type: str) -> str:
        """Return the hex colour for a detection type badge."""
        if det_type == "QR":
            return COLOURS["badge_qr"]
        elif det_type == "TXT":
            return COLOURS["badge_text"]
        else:
            return COLOURS["badge_barcode"]