from __future__ import annotations

import re

NOTE_NAMES_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_NAMES_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

NOTE_TO_PC: dict[str, int] = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}

_NO_CHORD = {"N", "NO_CHORD", "NOCHORD", "N.C.", "NC", "X", "NONE"}
_ROOT_RE = re.compile(r"^([A-Ga-g])([#b]?)(.*)$")

_QUALITY_MAP = {
    "": "maj",
    "maj": "maj",
    "major": "maj",
    "min": "min",
    "minor": "min",
    "m": "min",
    "7": "7",
    "maj7": "maj7",
    "min7": "min7",
    "m7": "min7",
    "dim": "dim",
    "dim7": "dim7",
    "hdim7": "min7b5",
    "min7b5": "min7b5",
    "m7b5": "min7b5",
    "aug": "aug",
    "sus2": "sus2",
    "sus4": "sus4",
    "sus": "sus4",
    "6": "6",
    "maj6": "6",
    "min6": "min6",
    "m6": "min6",
    "9": "9",
    "maj9": "maj9",
    "min9": "min9",
    "m9": "min9",
    "7b9": "7b9",
    "7#9": "7#9",
    "add9": "add9",
}


def _normalize_note_name(name: str) -> str | None:
    if not name:
        return None
    name = name.strip()
    if len(name) == 0:
        return None
    root = name[0].upper() + name[1:]
    if root in NOTE_TO_PC:
        return root
    return None


def _pc_to_note(pc: int) -> str:
    return NOTE_NAMES_SHARP[int(pc) % 12]


def _split_bass(label: str) -> tuple[str, str | None]:
    if "/" not in label:
        return label, None
    main, bass = label.split("/", 1)
    bass = bass.strip() if bass is not None else None
    if bass == "":
        bass = None
    return main, bass


def _parse_root_quality(main: str) -> tuple[str | None, str]:
    if ":" in main:
        root, qual = main.split(":", 1)
        root = root.strip()
        qual = qual.strip()
        return root, qual
    match = _ROOT_RE.match(main.strip())
    if match:
        root = f"{match.group(1).upper()}{match.group(2)}"
        qual = match.group(3) or ""
        return root, qual
    return None, ""


def _normalize_quality(raw: str) -> str:
    qual = raw.strip().lower().replace("(", "").replace(")", "").replace(" ", "")
    if qual in _QUALITY_MAP:
        return _QUALITY_MAP[qual]

    if "sus2" in qual:
        return "sus2"
    if "sus4" in qual or "sus" in qual:
        return "sus4"
    if "hdim" in qual or "m7b5" in qual:
        return "min7b5"
    if "dim7" in qual:
        return "dim7"
    if "dim" in qual:
        return "dim"
    if "aug" in qual:
        return "aug"
    if "maj" in qual and "9" in qual:
        return "maj9"
    if "min" in qual and "9" in qual:
        return "min9"
    if "7b9" in qual or "b9" in qual:
        return "7b9"
    if "7#9" in qual or "#9" in qual:
        return "7#9"
    if "maj" in qual and "7" in qual:
        return "maj7"
    if "min" in qual and "7" in qual:
        return "min7"
    if qual.startswith("m") and "7" in qual:
        return "min7"
    if "9" in qual:
        return "9"
    if "7" in qual:
        return "7"
    if "min" in qual or qual.startswith("m"):
        return "min"

    return "maj"


def _degree_to_interval(quality: str, token: str) -> int | None:
    token = token.strip().lower()
    accidental = 0
    if token.startswith("b"):
        accidental = -1
        token = token[1:]
    elif token.startswith("#"):
        accidental = 1
        token = token[1:]

    if token == "3":
        base = 3 if quality in ("min", "min7", "min9", "min6", "min7b5", "dim", "dim7") else 4
        return base + accidental
    if token == "5":
        if quality in ("dim", "dim7", "min7b5"):
            base = 6
        elif quality == "aug":
            base = 8
        else:
            base = 7
        return base + accidental
    if token == "7":
        if quality in ("maj7", "maj9"):
            base = 11
        elif quality == "dim7":
            base = 9
        else:
            base = 10
        return base + accidental
    if token == "6":
        return 9 + accidental
    if token == "9":
        return 14 + accidental
    if token == "11":
        return 17 + accidental
    if token == "13":
        return 21 + accidental
    return None


def _normalize_bass(root: str, quality: str, bass: str | None) -> str | None:
    if not bass:
        return None
    bass = bass.strip()
    if bass == "":
        return None
    note = _normalize_note_name(bass)
    if note is not None:
        return note

    root_pc = NOTE_TO_PC.get(root)
    if root_pc is None:
        return None
    interval = _degree_to_interval(quality, bass)
    if interval is None:
        return None
    return _pc_to_note(int(root_pc + interval))


def split_chord_label(label: str) -> tuple[str | None, str | None, str | None]:
    """
    Parse a chord label into (root, quality, bass_note). Returns (None, None, None)
    for no-chord labels. Quality is normalized to internal tokens.
    """
    if not label:
        return None, None, None
    raw = label.strip()
    if raw.upper() in _NO_CHORD:
        return None, None, None

    main, bass = _split_bass(raw)
    root_raw, qual_raw = _parse_root_quality(main)
    root = _normalize_note_name(root_raw or "")
    if root is None:
        return None, None, None

    quality = _normalize_quality(qual_raw)
    bass_note = _normalize_bass(root, quality, bass)
    return root, quality, bass_note


def format_chord_label(root: str, quality: str, bass: str | None = None) -> str:
    label = f"{root}:{quality}" if quality else root
    if bass:
        label = f"{label}/{bass}"
    return label


def normalize_madmom_chord_label(label: str) -> str:
    """
    Normalize madmom chord labels to the internal format root:quality[/bass].
    Unrecognized labels fall back to a major/minor triad when possible.
    """
    root, quality, bass = split_chord_label(label)
    if root is None or quality is None:
        return "N"
    return format_chord_label(root, quality, bass)
