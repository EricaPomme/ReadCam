"""UI theme configuration for CustomTkinter."""

import customtkinter as ctk


def setup_theme() -> None:
    """Configure the default CTk appearance mode and colour theme."""
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("green")


# Colour palette (hex for CTk)
COLOURS = {
    "bg": "#1a1a2e",
    "fg": "#e0e0e0",
    "accent": "#2ecc71",
    "card_bg": "#252540",
    "hover": "#2a2a45",
    "border": "#3a3a55",

    # Detection type badges
    "badge_qr": "#2ecc71",
    "badge_barcode": "#3498db",
    "badge_text": "#e67e22",

    # Status semantics
    "success": "#2ecc71",
    "warning": "#f1c40f",
    "error": "#e74c3c",
    "info": "#3498db",
}

# Font shortcuts
FONTS = {
    "heading": ("Segoe UI", 16, "bold"),
    "body": ("Segoe UI", 13),
    "small": ("Segoe UI", 11),
    "mono": ("Consolas", 13),
    "badge": ("Segoe UI", 10, "bold"),
}