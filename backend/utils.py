"""
utils.py
========
Terminal display utilities for the Indian Railways Intelligence System.
Uses ANSI escape codes only – no external dependencies beyond stdlib.
"""

import os
import sys
import math
import shutil
import datetime


# ─── ANSI colour palette ──────────────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    ITALIC  = "\033[3m"

    BLACK   = "\033[30m"
    RED     = "\033[31m"
    GREEN   = "\033[32m"
    YELLOW  = "\033[33m"
    BLUE    = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN    = "\033[36m"
    WHITE   = "\033[37m"

    BG_BLACK   = "\033[40m"
    BG_RED     = "\033[41m"
    BG_GREEN   = "\033[42m"
    BG_YELLOW  = "\033[43m"
    BG_BLUE    = "\033[44m"
    BG_MAGENTA = "\033[45m"
    BG_CYAN    = "\033[46m"
    BG_WHITE   = "\033[47m"

    # Bright variants
    B_RED     = "\033[91m"
    B_GREEN   = "\033[92m"
    B_YELLOW  = "\033[93m"
    B_BLUE    = "\033[94m"
    B_MAGENTA = "\033[95m"
    B_CYAN    = "\033[96m"
    B_WHITE   = "\033[97m"

    # 256-colour orange
    ORANGE    = "\033[38;5;208m"
    DARK_GRAY = "\033[38;5;240m"
    LIGHT_GRAY= "\033[38;5;250m"


def term_width() -> int:
    return shutil.get_terminal_size((120, 40)).columns


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def now_str() -> str:
    return datetime.datetime.now().strftime("%d %b %Y  %H:%M:%S")


# ─── Box drawing ─────────────────────────────────────────────────────────────
def box(title: str, content_lines: list[str], color: str = C.CYAN,
        width: int | None = None, title_color: str | None = None) -> str:
    """Draw a Unicode box around content_lines. Returns multi-line string."""
    w = (width or term_width()) - 2
    tc = title_color or color

    def strip_ansi(s: str) -> str:
        import re
        return re.sub(r"\033\[[0-9;]*m", "", s)

    top    = f"{color}╔{'═' * (w)}╗{C.RESET}"
    title_pad = w - len(strip_ansi(title)) - 2
    title_row = f"{color}║ {tc}{C.BOLD}{title}{C.RESET}{color}{' ' * title_pad} ║{C.RESET}"
    sep    = f"{color}╠{'═' * (w)}╣{C.RESET}"
    bottom = f"{color}╚{'═' * (w)}╝{C.RESET}"

    rows = [top, title_row, sep]
    for line in content_lines:
        visible = len(strip_ansi(line))
        pad = w - visible - 2
        pad = max(pad, 0)
        rows.append(f"{color}║ {C.RESET}{line}{' ' * pad}{color} ║{C.RESET}")
    rows.append(bottom)
    return "\n".join(rows)


def thin_box(title: str, content_lines: list[str], color: str = C.BLUE,
             width: int | None = None) -> str:
    """Light-border version of box()."""
    w = (width or term_width()) - 2

    def strip_ansi(s: str) -> str:
        import re
        return re.sub(r"\033\[[0-9;]*m", "", s)

    top    = f"{color}┌{'─' * (w)}┐{C.RESET}"
    t_pad  = w - len(strip_ansi(title)) - 2
    t_row  = f"{color}│ {C.BOLD}{title}{C.RESET}{' ' * t_pad}{color} │{C.RESET}"
    sep    = f"{color}├{'─' * (w)}┤{C.RESET}"
    bottom = f"{color}└{'─' * (w)}┘{C.RESET}"

    rows = [top, t_row, sep]
    for line in content_lines:
        visible = len(strip_ansi(line))
        pad = max(w - visible - 2, 0)
        rows.append(f"{color}│ {C.RESET}{line}{' ' * pad}{color} │{C.RESET}")
    rows.append(bottom)
    return "\n".join(rows)


# ─── Table rendering ─────────────────────────────────────────────────────────
def render_table(headers: list[str], rows: list[list[str]],
                 col_colors: list[str] | None = None,
                 header_color: str = C.B_CYAN,
                 row_color: str = C.WHITE,
                 alt_color: str = C.LIGHT_GRAY,
                 max_col_width: int = 30,
                 width: int | None = None) -> str:
    """
    Render an aligned table.
    col_colors: per-column colours applied to each cell value.
    """
    def strip_ansi(s: str) -> str:
        import re
        return re.sub(r"\033\[[0-9;]*m", "", s)

    def trunc(s: str, n: int) -> str:
        return s if len(s) <= n else s[:n - 1] + "…"

    if col_colors is None:
        col_colors = [row_color] * len(headers)

    # Compute column widths
    col_w = [max(len(strip_ansi(h)), 6) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            if i < len(col_w):
                col_w[i] = min(max(col_w[i], len(strip_ansi(str(cell)))), max_col_width)

    sep_char = "─"
    sep = "  " + "  ".join(sep_char * w for w in col_w)

    def fmt_row(cells, colors, bold=False):
        parts = []
        for i, (cell, w) in enumerate(zip(cells, col_w)):
            cell_str = trunc(str(cell), w)
            pad = w - len(strip_ansi(cell_str))
            color = colors[i] if i < len(colors) else row_color
            prefix = C.BOLD if bold else ""
            parts.append(f"{prefix}{color}{cell_str}{C.RESET}{' ' * pad}")
        return "  " + "  ".join(parts)

    lines = []
    lines.append(f"{C.DIM}{sep}{C.RESET}")
    lines.append(fmt_row(headers, [header_color] * len(headers), bold=True))
    lines.append(f"{C.DIM}{sep}{C.RESET}")
    for idx, row in enumerate(rows):
        c = col_colors if idx % 2 == 0 else [alt_color] * len(col_colors)
        lines.append(fmt_row(row, c))
    lines.append(f"{C.DIM}{sep}{C.RESET}")
    return "\n".join(lines)


# ─── Progress / gauge bars ───────────────────────────────────────────────────
def gauge_bar(value: float, max_value: float = 100, width: int = 20,
              fill_char: str = "█", empty_char: str = "░") -> str:
    """Return a coloured progress bar string."""
    ratio = min(value / max_value, 1.0) if max_value > 0 else 0
    filled = int(ratio * width)
    empty  = width - filled

    if ratio < 0.4:
        color = C.B_GREEN
    elif ratio < 0.7:
        color = C.B_YELLOW
    else:
        color = C.B_RED

    return f"{color}{fill_char * filled}{C.DARK_GRAY}{empty_char * empty}{C.RESET}"


def congestion_badge(level: str) -> str:
    """Coloured badge for congestion level."""
    badges = {
        "Low":      f"{C.BG_GREEN}{C.BLACK} LOW    {C.RESET}",
        "Medium":   f"{C.BG_YELLOW}{C.BLACK} MEDIUM {C.RESET}",
        "High":     f"{C.BG_RED}{C.WHITE} HIGH   {C.RESET}",
        "Critical": f"{C.BG_RED}{C.WHITE}{C.BOLD} CRIT!  {C.RESET}",
        "Extreme":  f"{C.BG_MAGENTA}{C.WHITE}{C.BOLD} EXTREME{C.RESET}",
    }
    return badges.get(level, f"{C.DIM}{level}{C.RESET}")


def risk_badge(level: str) -> str:
    badges = {
        "Low":    f"{C.B_GREEN}● LOW{C.RESET}",
        "Medium": f"{C.B_YELLOW}● MEDIUM{C.RESET}",
        "High":   f"{C.B_RED}● HIGH{C.RESET}",
    }
    return badges.get(level, f"{C.DIM}{level}{C.RESET}")


def delay_color(minutes: float) -> str:
    if minutes < 5:
        return f"{C.B_GREEN}{minutes:.0f} min{C.RESET}"
    elif minutes < 20:
        return f"{C.B_YELLOW}{minutes:.0f} min{C.RESET}"
    else:
        return f"{C.B_RED}{minutes:.0f} min{C.RESET}"


def reliability_bar(score: float) -> str:
    return gauge_bar(score, 100, 15) + f"  {score:.0f}/100"


# ─── Banners ─────────────────────────────────────────────────────────────────
RAILWAYS_LOGO = r"""
  ██╗██████╗     ██████╗  █████╗ ██╗██╗     ██╗    ██╗ █████╗ ██╗   ██╗███████╗
  ██║██╔══██╗    ██╔══██╗██╔══██╗██║██║     ██║    ██║██╔══██╗╚██╗ ██╔╝██╔════╝
  ██║██████╔╝    ██████╔╝███████║██║██║     ██║ █╗ ██║███████║ ╚████╔╝ ███████╗
  ██║██╔══██╗    ██╔══██╗██╔══██║██║██║     ██║███╗██║██╔══██║  ╚██╔╝  ╚════██║
  ██║██║  ██║    ██║  ██║██║  ██║██║███████╗╚███╔███╔╝██║  ██║   ██║   ███████║
  ╚═╝╚═╝  ╚═╝    ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝ ╚══╝╚══╝ ╚═╝  ╚═╝   ╚═╝   ╚══════╝
"""

def print_banner(title: str, subtitle: str = "", color: str = C.B_BLUE):
    w = term_width()
    print()
    print(f"{color}{RAILWAYS_LOGO}{C.RESET}")
    print(f"{C.BOLD}{C.B_CYAN}{'─' * w}{C.RESET}")
    print(f"{C.BOLD}{C.B_WHITE}  {title}{C.RESET}")
    if subtitle:
        print(f"{C.DIM}  {subtitle}{C.RESET}")
    print(f"{C.BOLD}{C.B_CYAN}{'─' * w}{C.RESET}")
    print()


def print_section(title: str, color: str = C.B_MAGENTA):
    w = term_width()
    bar = "═" * (w - 4)
    print()
    print(f"{color}{C.BOLD}  ╔{bar}╗{C.RESET}")
    print(f"{color}{C.BOLD}  ║  {title.upper():<{w-8}} ║{C.RESET}")
    print(f"{color}{C.BOLD}  ╚{bar}╝{C.RESET}")
    print()


def spinner_wait(message: str, seconds: float = 0.6):
    """Simple fake spinner while something loads."""
    import time
    frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
    n = max(1, int(seconds / 0.08))
    for i in range(n):
        sys.stdout.write(f"\r  {C.B_CYAN}{frames[i % len(frames)]}{C.RESET}  {message}   ")
        sys.stdout.flush()
        import time as _t
        _t.sleep(0.08)
    sys.stdout.write(f"\r  {C.B_GREEN}✓{C.RESET}  {message}   \n")
    sys.stdout.flush()


def prompt(message: str, color: str = C.B_YELLOW) -> str:
    return input(f"\n{color}  ▶  {message}{C.RESET}  ")


def error(msg: str):
    print(f"\n  {C.B_RED}✗  ERROR:{C.RESET}  {msg}\n")


def success(msg: str):
    print(f"\n  {C.B_GREEN}✓{C.RESET}  {msg}\n")


def info(msg: str):
    print(f"  {C.B_BLUE}ℹ{C.RESET}  {msg}")
