"""Interactive English-to-Gujarati translator for the trained Transformer model.

Run from the project directory:
    python main.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import textwrap
import threading
from pathlib import Path

import torch
from tokenizers import Tokenizer

from config import get_config
from model import build_transformer


def configure_unicode_output() -> None:
    """Make Gujarati text and box drawing work in Windows terminals and pipes."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def configure_windows_console() -> str | None:
    """Use a Gujarati-capable font in the classic Windows Console when possible.

    Windows Terminal and VS Code own their font settings, so they intentionally do
    not allow a child process to change the font.  The Windows Console Host does,
    and Nirmala UI is included with supported Windows releases and has Gujarati
    OpenType shaping support.
    """
    if os.name != "nt":
        return None

    try:
        import ctypes
        from ctypes import wintypes

        class Coord(ctypes.Structure):
            _fields_ = [("X", wintypes.SHORT), ("Y", wintypes.SHORT)]

        class ConsoleFontInfoEx(ctypes.Structure):
            _fields_ = [
                ("cbSize", wintypes.ULONG),
                ("nFont", wintypes.DWORD),
                ("dwFontSize", Coord),
                ("FontFamily", wintypes.UINT),
                ("FontWeight", wintypes.UINT),
                ("FaceName", wintypes.WCHAR * 32),
            ]

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetStdHandle.argtypes = [wintypes.DWORD]
        kernel32.GetStdHandle.restype = wintypes.HANDLE
        kernel32.SetConsoleCP.argtypes = [wintypes.UINT]
        kernel32.SetConsoleOutputCP.argtypes = [wintypes.UINT]
        kernel32.GetCurrentConsoleFontEx.argtypes = [
            wintypes.HANDLE,
            wintypes.BOOL,
            ctypes.POINTER(ConsoleFontInfoEx),
        ]
        kernel32.GetCurrentConsoleFontEx.restype = wintypes.BOOL
        kernel32.SetCurrentConsoleFontEx.argtypes = [
            wintypes.HANDLE,
            wintypes.BOOL,
            ctypes.POINTER(ConsoleFontInfoEx),
        ]
        kernel32.SetCurrentConsoleFontEx.restype = wintypes.BOOL
        # CP_UTF8 keeps other console writers and user input on the same encoding.
        kernel32.SetConsoleCP(65001)
        kernel32.SetConsoleOutputCP(65001)

        standard_output = kernel32.GetStdHandle(wintypes.DWORD(-11).value)  # STD_OUTPUT_HANDLE
        invalid_handle = ctypes.c_void_p(-1).value
        if standard_output in (None, 0, invalid_handle):
            return None

        info = ConsoleFontInfoEx()
        info.cbSize = ctypes.sizeof(info)
        if not kernel32.GetCurrentConsoleFontEx(standard_output, False, ctypes.byref(info)):
            return None

        # Keep the user's current size and only replace the face.  This setting is
        # scoped to the current classic-console window; it does not change Windows
        # or terminal profile preferences.
        info.FaceName = "Nirmala UI"
        info.FontWeight = 400
        if kernel32.SetCurrentConsoleFontEx(standard_output, False, ctypes.byref(info)):
            return "Nirmala UI"
    except (AttributeError, OSError):
        pass
    return None


configure_unicode_output()
WINDOWS_CONSOLE_FONT = configure_windows_console()


PROJECT_DIR = Path(__file__).resolve().parent
RESET = "\033[0m"
STYLES = {
    "title": "\033[1;36m",
    "label": "\033[1;33m",
    "success": "\033[1;32m",
    "muted": "\033[2;37m",
    "error": "\033[1;31m",
}


def styled(text: str, style: str) -> str:
    """Apply ANSI styling only when stdout is an interactive terminal."""
    return f"{STYLES[style]}{text}{RESET}" if sys.stdout.isatty() else text


def terminal_width() -> int:
    return max(64, min(shutil.get_terminal_size(fallback=(88, 24)).columns, 100))


def print_rule(character: str = "─") -> None:
    print(styled(character * terminal_width(), "muted"))


def print_banner() -> None:
    width = terminal_width()
    print()
    print(styled("╭" + "─" * (width - 2) + "╮", "title"))
    print(styled("│" + " ENGLISH  →  GUJARATI TRANSLATOR ".center(width - 2) + "│", "title"))
    print(styled("│" + " Local Transformer Model ".center(width - 2) + "│", "muted"))
    print(styled("╰" + "─" * (width - 2) + "╯", "title"))
    print()


class LoadingIndicator:
    """A dependency-free spinner for work that can take several seconds."""

    def __init__(self, message: str) -> None:
        self.message = message
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> "LoadingIndicator":
        if not sys.stdout.isatty():
            print(f"{self.message}...", flush=True)
            return self

        self._thread = threading.Thread(target=self._animate, daemon=True)
        self._thread.start()
        return self

    def _animate(self) -> None:
        frames = ("|", "/", "-", "\\")
        index = 0
        while not self._stop.is_set():
            frame = frames[index % len(frames)]
            status = styled(f"{self.message}  {frame}", "title")
            sys.stdout.write(f"\r{status}")
            sys.stdout.flush()
            index += 1
            self._stop.wait(0.12)

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join()
            # Erase the spinner line before the regular interface is printed.
            sys.stdout.write("\r" + " " * terminal_width() + "\r")
            sys.stdout.flush()


def find_checkpoint(weights_dir: Path, requested: str | None) -> Path:
    """Return a selected checkpoint, defaulting to the most recently updated one."""
    if requested:
        candidate = Path(requested)
        if not candidate.is_absolute():
            candidate = PROJECT_DIR / candidate
        if not candidate.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {candidate}")
        return candidate

    checkpoints = list(weights_dir.glob("*.pt"))
    if not checkpoints:
        raise FileNotFoundError(
            f"No model checkpoint was found in '{weights_dir}'. "
            "Train the model first, then run this program again."
        )
    return max(checkpoints, key=lambda path: path.stat().st_mtime)


def load_checkpoint(path: Path, device: torch.device) -> dict:
    """Load both older PyTorch checkpoints and current safe-loading checkpoints."""
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=device)


class Translator:
    """Loads the trained artifacts once and translates individual sentences."""

    def __init__(self, checkpoint_path: Path, force_cpu: bool = False) -> None:
        self.config = get_config()
        self.device = torch.device(
            "cpu" if force_cpu or not torch.cuda.is_available() else "cuda"
        )
        self.source_tokenizer = Tokenizer.from_file(
            str(PROJECT_DIR / self.config["tokenizer_file"].format(self.config["lang_src"]))
        )
        self.target_tokenizer = Tokenizer.from_file(
            str(PROJECT_DIR / self.config["tokenizer_file"].format(self.config["lang_tgt"]))
        )

        self.model = build_transformer(
            self.source_tokenizer.get_vocab_size(),
            self.target_tokenizer.get_vocab_size(),
            self.config["seq_len"],
            self.config["seq_len"],
            self.config["d_model"],
        ).to(self.device)

        checkpoint = load_checkpoint(checkpoint_path, self.device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        self.max_source_tokens = self.config["seq_len"] - 2
        self.source_sos = self.source_tokenizer.token_to_id("[SOS]")
        self.source_eos = self.source_tokenizer.token_to_id("[EOS]")
        self.source_pad = self.source_tokenizer.token_to_id("[PAD]")
        self.target_sos = self.target_tokenizer.token_to_id("[SOS]")
        self.target_eos = self.target_tokenizer.token_to_id("[EOS]")

        required_ids = (
            self.source_sos,
            self.source_eos,
            self.source_pad,
            self.target_sos,
            self.target_eos,
        )
        if any(token_id is None for token_id in required_ids):
            raise ValueError("Tokenizer is missing one or more required special tokens.")

    @torch.inference_mode()
    def translate(self, text: str) -> tuple[str, int]:
        """Translate text and return the result plus the number of truncated tokens."""
        source_ids = self.source_tokenizer.encode(text).ids
        truncated_tokens = max(0, len(source_ids) - self.max_source_tokens)
        source_ids = source_ids[: self.max_source_tokens]

        encoder_tokens = [self.source_sos, *source_ids, self.source_eos]
        encoder_tokens.extend(
            [self.source_pad] * (self.config["seq_len"] - len(encoder_tokens))
        )
        source = torch.tensor([encoder_tokens], dtype=torch.long, device=self.device)
        source_mask = (source != self.source_pad).unsqueeze(1).unsqueeze(1).int()
        encoder_output = self.model.encode(source, source_mask)

        generated = [self.target_sos]
        for _ in range(self.config["seq_len"] - 1):
            decoder_input = torch.tensor([generated], dtype=torch.long, device=self.device)
            size = decoder_input.size(1)
            decoder_mask = torch.tril(
                torch.ones((size, size), dtype=torch.bool, device=self.device)
            ).unsqueeze(0)
            decoder_output = self.model.decode(
                encoder_output, source_mask, decoder_input, decoder_mask
            )
            next_token = int(torch.argmax(self.model.project(decoder_output[:, -1]), dim=1).item())
            generated.append(next_token)
            if next_token == self.target_eos:
                break

        return self.target_tokenizer.decode(generated, skip_special_tokens=True).strip(), truncated_tokens


def print_help() -> None:
    print(styled("Commands", "label"))
    print("  /help       Show these commands")
    print("  /clear      Clear the terminal")
    print("  /quit       Exit the translator")
    print("  Ctrl+C      Exit the translator")


def print_translation(source: str, translation: str, truncated_tokens: int) -> None:
    width = terminal_width()
    print()
    print_rule()
    print(styled("ENGLISH", "label"))
    print(textwrap.fill(source, width=width))
    print()
    print(styled("ગુજરાતી", "success"))
    print(textwrap.fill(translation or "[No translation produced]", width=width))
    if truncated_tokens:
        print()
        print(
            styled(
                f"Note: {truncated_tokens} source token(s) exceeded the model limit and were omitted.",
                "muted",
            )
        )
    print_rule()


def run_interactive(translator: Translator, checkpoint_path: Path) -> None:
    print_banner()
    device_name = "GPU (CUDA)" if translator.device.type == "cuda" else "CPU"
    print(
        styled("Model ready", "success")
        + f"  •  {checkpoint_path.name}  •  {device_name}"
    )
    print(styled("Type English text to translate. Use /help for commands.", "muted"))

    while True:
        try:
            print()
            text = input(styled("English  › ", "label")).strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n{styled('Goodbye!', 'success')}")
            return

        command = text.lower()
        if command in {"/quit", "/exit", "quit", "exit"}:
            print(styled("Goodbye!", "success"))
            return
        if command == "/help":
            print_help()
            continue
        if command == "/clear":
            os.system("cls" if os.name == "nt" else "clear")
            print_banner()
            continue
        if not text:
            print(styled("Please enter an English sentence, or type /help.", "muted"))
            continue

        try:
            translation, truncated_tokens = translator.translate(text)
            print_translation(text, translation, truncated_tokens)
        except RuntimeError as error:
            print(styled(f"Translation failed: {error}", "error"))


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Translate English text to Gujarati locally.")
    parser.add_argument(
        "--checkpoint",
        help="Path to a specific .pt checkpoint (defaults to the newest checkpoint in weights/).",
    )
    parser.add_argument("--cpu", action="store_true", help="Run on CPU even when CUDA is available.")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = get_config()

    try:
        checkpoint_path = find_checkpoint(PROJECT_DIR / config["model_folder"], arguments.checkpoint)
        with LoadingIndicator("Loading translation model"):
            translator = Translator(checkpoint_path, force_cpu=arguments.cpu)
    except (FileNotFoundError, OSError, RuntimeError, ValueError, KeyError) as error:
        print(styled(f"Unable to start translator: {error}", "error"), file=sys.stderr)
        raise SystemExit(1) from error

    run_interactive(translator, checkpoint_path)


if __name__ == "__main__":
    main()
