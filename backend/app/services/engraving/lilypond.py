from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import subprocess
import math
from typing import List, Tuple

from app.schemas import ChordSegment

NOTE_TO_LILYPOND = {
    "C": "c", "C#": "cis", "D": "d", "D#": "dis", "E": "e", "F": "f",
    "F#": "fis", "G": "g", "G#": "gis", "A": "a", "A#": "ais", "B": "b",
    # si en el futuro usas bemoles:
    "Db": "des", "Eb": "ees", "Gb": "ges", "Ab": "aes", "Bb": "bes",
}

QUAL_TO_SUFFIX = {
    "maj": "",     # mayor por defecto
    "min": ":m",
    "7": ":7",
    "maj7": ":maj7",
    "min7": ":m7",
}

@dataclass
class QuantCfg:
    # grid en negras (quarterLength). Real Book típico: corchea => 0.5
    grid_q: float = 0.5

def _parse_label_to_chordmode(label: str) -> str:
    """
    label backend: "G#:7", "A:min", "C:maj"
    lilypond chordmode: "gis:7", "a:m", "c"
    """
    if not label or label == "N":
        return "s"  # skip/rest en chordmode

    if ":" in label:
        root, qual = label.split(":", 1)
        qual = qual.strip()
    else:
        root, qual = label, "maj"

    root = root.strip()
    lp_root = NOTE_TO_LILYPOND.get(root)
    if not lp_root:
        # fallback: intenta normalizar
        lp_root = NOTE_TO_LILYPOND.get(root.replace("♯", "#").replace("♭", "b"), "c")

    suffix = QUAL_TO_SUFFIX.get(qual, f":{qual}" if qual else "")
    return f"{lp_root}{suffix}"

def _duration_token_from_quarters(q: float) -> str:
    """
    Convierte duración en negras a token LilyPond.
    LilyPond: 1=redonda,2=blanca,4=negra,8=corchea,16=semicorchea
    soporta puntos con '.' (ej: 2. = blanca con puntillo => 3 negras)
    """
    # trabajamos con tolerancia por cuantización
    eps = 1e-6

    # diccionario de duraciones comunes en negras
    # (token -> quarters)
    common = [
        ("1", 4.0),
        ("2.", 3.0),
        ("2", 2.0),
        ("4.", 1.5),
        ("4", 1.0),
        ("8.", 0.75),
        ("8", 0.5),
        ("16.", 0.375),
        ("16", 0.25),
    ]
    for tok, qq in common:
        if abs(q - qq) < 0.01 + eps:
            return tok

    # fallback: aproxima a potencia de 2
    # token = 4 / q  => q=1 =>4, q=0.5 =>8, q=2=>2 ...
    denom = int(round(4.0 / max(q, 0.25)))
    denom = max(1, min(64, denom))
    return str(denom)

def quantize_chords_to_grid(
    chords: List[ChordSegment],
    tempo_bpm: float,
    cfg: QuantCfg,
) -> List[Tuple[str, float]]:
    """
    Devuelve lista de (chordmode_token, duration_in_quarters)
    cuantizado a cfg.grid_q.
    """
    tempo = max(30.0, float(tempo_bpm) if tempo_bpm else 120.0)
    sec_per_q = 60.0 / tempo
    grid = cfg.grid_q

    # convierte a quarters y cuantiza
    items = []
    for c in chords:
        if c.end <= c.start:
            continue
        s_q = c.start / sec_per_q
        e_q = c.end / sec_per_q
        qs = round(s_q / grid) * grid
        qe = round(e_q / grid) * grid
        if qe <= qs:
            qe = qs + grid
        items.append((float(qs), float(qe), c.label))

    if not items:
        return []

    items.sort(key=lambda x: x[0])

    # rellena huecos con silencio (skip)
    merged = []
    cur_s, cur_e, cur_l = items[0]
    for s, e, lab in items[1:]:
        if s <= cur_e + 1e-6:
            # solapa: cierra anterior en s
            if s > cur_s + 1e-6:
                merged.append((cur_s, s, cur_l))
            cur_s, cur_e, cur_l = s, e, lab
        else:
            merged.append((cur_s, cur_e, cur_l))
            merged.append((cur_e, s, "N"))  # gap
            cur_s, cur_e, cur_l = s, e, lab
    merged.append((cur_s, cur_e, cur_l))

    # comprime por etiqueta igual contigua
    out = []
    prev_tok = None
    prev_len = 0.0
    for s, e, lab in merged:
        tok = _parse_label_to_chordmode(lab)
        ln = max(grid, e - s)
        if prev_tok is None:
            prev_tok, prev_len = tok, ln
        elif tok == prev_tok:
            prev_len += ln
        else:
            out.append((prev_tok, prev_len))
            prev_tok, prev_len = tok, ln
    if prev_tok is not None:
        out.append((prev_tok, prev_len))

    return out

def build_lilypond_score(
    chords: List[ChordSegment],
    tempo_bpm: float,
    time_signature: str = "4/4",
    key_tonic: str | None = None,
    key_mode: str = "major",
    title: str = "Lead Sheet",
    composer: str = "",
    cfg: QuantCfg = QuantCfg(grid_q=0.5),
    # markers editoriales automáticos simples
    rehearsal_every_measures: int = 8,
) -> str:
    """
    Genera un .ly con:
    - ChordNames arriba
    - Staff con improvisationOn (slashes por pulso)
    - BarNumbers visibles
    - Rehearsal marks cada N compases (A,B,C...)
    """
    num, den = 4, 4
    try:
        a, b = time_signature.split("/")
        num, den = int(a), int(b)
    except Exception:
        pass

    # slashes: 1 por negra (beat) en 4/4 típico.
    # En general, definimos "beat" como negra.
    beats_per_bar = num * (4.0 / den)  # en negras
    # para la parte rítmica, usamos negras
    slash_tokens = " ".join(["c4"] * int(round(beats_per_bar)))

    # cuantiza acordes a grid
    qseq = quantize_chords_to_grid(chords, tempo_bpm, cfg=cfg)
    if not qseq:
        qseq = [("s", beats_per_bar)]  # al menos un compás

    # convierte a chordmode con duraciones lilypond
    chordmode_parts = []
    for tok, qlen in qseq:
        # puede requerir partir en varios tokens si no encaja en una duración simple
        # estrategia robusta: trocear en grid
        grid = cfg.grid_q
        n = int(math.ceil(qlen / grid - 1e-6))
        remaining = n * grid
        # comprimimos con duraciones comunes (4,2,1,0.5,0.25) donde sea posible
        # para mantenerlo estable: emitimos en múltiplos de grid con token 'dur' por bloque
        # (LilyPond chordmode acepta: c8 c8 c8 ... perfectamente)
        dur_tok = _duration_token_from_quarters(grid)
        for _ in range(n):
            chordmode_parts.append(f"{tok}{dur_tok}")

    chordmode = " ".join(chordmode_parts)

    # rehearsal marks automáticos: A en compás 1, luego cada N compases
    # (editorialmente aceptable como “secciones” base; se puede mejorar luego con detección de forma)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    marks = []
    if rehearsal_every_measures > 0:
        # insertamos \mark \markup \box "A" etc.
        # en lilypond, marcas se insertan en la voz musical (slashes staff)
        # generamos una secuencia por compases.
        for m in range(1, 2000):
            if (m - 1) % rehearsal_every_measures == 0:
                idx = (m - 1) // rehearsal_every_measures
                if idx < len(letters):
                    marks.append((m, letters[idx]))
                else:
                    break

    # construye slashes por compases hasta cubrir duración total aproximada en compases
    # estimación: longitud total en negras / beats_per_bar
    total_quarters = sum(q for _, q in qseq)
    total_bars = max(1, int(math.ceil(total_quarters / beats_per_bar - 1e-6)))

    slash_bars = []
    for bar in range(1, total_bars + 1):
        # mark si toca
        mk = next((l for (m, l) in marks if m == bar), None)
        if mk:
            slash_bars.append(f'\\mark \\markup \\box "{mk}"')
        slash_bars.append(slash_tokens)
        slash_bars.append("|")
    slashes = " ".join(slash_bars)

    key_clause = ""
    if key_tonic:
        lp_key = NOTE_TO_LILYPOND.get(key_tonic)
        if lp_key and key_mode in ("major", "minor"):
            key_clause = f"  \\\\key {lp_key} \\\\{key_mode}\n"

    ly = f"""
\\version "2.24.0"

\\paper {{
  #(set-paper-size "letter")
  top-margin = 12\\mm
  bottom-margin = 12\\mm
  left-margin = 14\\mm
  right-margin = 14\\mm
  ragged-last = ##f
}}

\\header {{
  title = "{title}"
  composer = "{composer}"
  tagline = ##f
}}

global = {{
  \\numericTimeSignature
{key_clause}  \\time {num}/{den}
  \\tempo 4 = {int(round(tempo_bpm or 120))}
}}

chords = \\chordmode {{
  \\set chordChanges = ##t
  {chordmode}
}}

rhythm = {{
  \\global
  \\override Score.BarNumber.break-visibility = ##(#t #t #t)
  \\override Score.BarNumber.font-size = #1
  \\override Score.RehearsalMark.self-alignment-X = #LEFT
  \\override Score.RehearsalMark.font-size = #2
  \\improvisationOn
  {slashes}
}}

\\layout {{
  \\context {{
    \\ChordNames
    \\override ChordName.font-size = #2
  }}
  \\context {{
    \\Score
    \\override SpacingSpanner.base-shortest-duration = #(ly:make-moment 1/16)
  }}
}}

\\score {{
  <<
    \\new ChordNames \\with {{ alignAboveContext = "staff" }} \\chords
    \\new Staff = "staff" \\with {{
      instrumentName = "Gtr."
    }} \\rhythm
  >>
}}
"""
    return ly.strip() + "\n"

def render_lilypond_pdf(ly_text: str, out_dir: Path, basename: str = "score") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    ly_path = out_dir / f"{basename}.ly"
    ly_path.write_text(ly_text, encoding="utf-8")

    # LilyPond genera basename.pdf en out_dir
    cmd = [
        "lilypond",
        "-dno-point-and-click",
        "--pdf",
        "-o",
        str(out_dir / basename),
        str(ly_path),
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    pdf_path = out_dir / f"{basename}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError("LilyPond no generó el PDF esperado.")
    return pdf_path
