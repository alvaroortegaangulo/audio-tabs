"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Barline,
  ChordSymbol,
  Formatter,
  Renderer,
  Stave,
  StaveModifier,
  StaveNote,
  Voice,
} from "vexflow/bravura";

import { BACKEND } from "@/lib/api";

type ChordSegment = {
  start: number;
  end: number;
  label: string;
  confidence: number;
};

type KeySignature = {
  tonic: string;
  mode: "major" | "minor";
  fifths: number;
  name: string;
  vexflow: string;
  use_flats: boolean;
  score: number;
};

export type JobResult = {
  tempo_bpm: number;
  time_signature: string;
  chords: ChordSegment[];
  key_signature?: KeySignature | null;
};

function parseTimeSignature(timeSignature: string): { num: number; den: number } {
  const [n, d] = (timeSignature ?? "4/4").split("/");
  const num = Number.parseInt(n ?? "4", 10);
  const den = Number.parseInt(d ?? "4", 10);
  if (!Number.isFinite(num) || !Number.isFinite(den) || num <= 0 || den <= 0) return { num: 4, den: 4 };
  return { num, den };
}

function durationForDenominator(den: number): string {
  switch (den) {
    case 1:
      return "w";
    case 2:
      return "h";
    case 4:
      return "q";
    case 8:
      return "8";
    case 16:
      return "16";
    case 32:
      return "32";
    default:
      return "q";
  }
}

function parseChordLabel(label: string): { root: string; quality: string } | null {
  if (!label || label === "N") return null;
  const [rootRaw, qualityRaw] = label.split(":");
  const root = (rootRaw ?? "").trim();
  const quality = (qualityRaw ?? "maj").trim();
  if (!root) return null;
  return { root, quality };
}

function buildChordSymbol(label: string) {
  const parsed = parseChordLabel(label);
  if (!parsed) return null;

  const cs = new ChordSymbol();
  cs.setVertical(ChordSymbol.VerticalJustify.TOP);
  cs.setHorizontal(ChordSymbol.HorizontalJustify.CENTER);
  cs.setFont("system-ui", 14, 600);

  cs.addGlyphOrText(parsed.root);

  const quality = parsed.quality;
  const suffix =
    quality === "min"
      ? "m"
      : quality === "7"
        ? "7"
        : quality === "maj7"
          ? "maj7"
          : quality === "min7"
            ? "m7"
            : "";
  if (suffix) cs.addTextSuperscript(suffix);

  return cs;
}

export default function LeadSheet({ result, jobId }: { result: JobResult; jobId: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  const tempo = useMemo(() => Math.max(30, Math.min(300, Number(result.tempo_bpm) || 120)), [result.tempo_bpm]);
  const timeSig = useMemo(() => parseTimeSignature(result.time_signature), [result.time_signature]);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const obs = new ResizeObserver((entries) => {
      const next = Math.floor(entries[0]?.contentRect?.width ?? 0);
      if (next > 0) setWidth(next);
    });

    obs.observe(el);
    return () => obs.disconnect();
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || width < 320) return;

    const chords = [...(result.chords ?? [])].sort((a, b) => a.start - b.start);
    const totalSec = chords.reduce((acc, c) => Math.max(acc, c.end), 0);

    const secPerQuarter = 60 / tempo;
    const beatQuarters = 4 / timeSig.den;
    const measureQuarters = timeSig.num * beatQuarters;
    const totalQuarters = totalSec / secPerQuarter;
    const measures = Math.max(1, Math.ceil(totalQuarters / measureQuarters));

    const measuresPerLine = width < 560 ? 1 : width < 900 ? 2 : 4;

    const paddingX = 18;
    const paddingY = 16;
    const lineHeight = 120;
    const lineCount = Math.ceil(measures / measuresPerLine);
    const staveWidth = Math.floor((width - paddingX * 2) / measuresPerLine);
    const height = paddingY * 2 + lineCount * lineHeight;

    el.innerHTML = "";
    const renderer = new Renderer(el, Renderer.Backends.SVG);
    renderer.resize(width, height);
    const ctx = renderer.getContext();
    ctx.setFillStyle("#111827");
    ctx.setStrokeStyle("#111827");

    let segIdx = 0;
    let lastLabel = "";

    const labelAt = (tSec: number) => {
      while (segIdx < chords.length && chords[segIdx]!.end <= tSec) segIdx += 1;
      const seg = chords[segIdx];
      if (!seg) return "N";
      if (seg.start > tSec || seg.end <= tSec) return "N";
      return seg.label || "N";
    };

    for (let m = 0; m < measures; m += 1) {
      const line = Math.floor(m / measuresPerLine);
      const col = m % measuresPerLine;
      const x = paddingX + col * staveWidth;
      const y = paddingY + line * lineHeight;

      const stave = new Stave(x, y, staveWidth, { space_above_staff_ln: 6, space_below_staff_ln: 6 });

      if (col !== 0) stave.setBegBarType(Barline.type.NONE);
      stave.setEndBarType(m === measures - 1 ? Barline.type.END : Barline.type.SINGLE);

      const isSystemStart = col === 0;
      if (isSystemStart) {
        stave.addClef("treble");
        if (result.key_signature?.vexflow) stave.setKeySignature(result.key_signature.vexflow);
        stave.setTimeSignature(`${timeSig.num}/${timeSig.den}`);
      }

      stave.setText(String(m + 1), StaveModifier.Position.ABOVE, { shift_y: -10, shift_x: 0 });

      stave.setContext(ctx).draw();

      const dur = durationForDenominator(timeSig.den);
      const notes: StaveNote[] = [];
      for (let b = 0; b < timeSig.num; b += 1) {
        const note = new StaveNote({ keys: ["b/4"], duration: dur, type: "s" });

        const tBeatQ = m * measureQuarters + b * beatQuarters;
        const tBeatSec = tBeatQ * secPerQuarter;
        const lbl = labelAt(tBeatSec);

        if (lbl === "N") {
          lastLabel = "";
        } else {
          const shouldShow = b === 0 || lbl !== lastLabel;
          if (shouldShow) {
            const cs = buildChordSymbol(lbl);
            if (cs) note.addModifier(cs, 0);
          }
          lastLabel = lbl;
        }

        notes.push(note);
      }

      const voice = new Voice({ num_beats: timeSig.num, beat_value: timeSig.den }).setStrict(false);
      voice.addTickables(notes);

      new Formatter().joinVoices([voice]).format([voice], staveWidth - 20);
      voice.draw(ctx, stave);
    }
  }, [result, tempo, timeSig.den, timeSig.num, width]);

  const pdfUrl = `${BACKEND}/v1/jobs/${jobId}/score.pdf`;
  const xmlUrl = `${BACKEND}/v1/jobs/${jobId}/musicxml`;
  const midiUrl = `${BACKEND}/v1/jobs/${jobId}/transcription.mid`;

  return (
    <section className="scoreShell">
      <div className="scoreToolbar">
        <div className="scoreMeta">
          <div className="scoreTitle">Lead sheet</div>
          <div className="scoreSub">
            {result.key_signature?.name ? `${result.key_signature.name} · ` : ""}
            {Math.round(tempo)} bpm · {result.time_signature}
          </div>
        </div>
        <div className="scoreActions">
          <a className="btn btnSecondary" href={xmlUrl} target="_blank" rel="noreferrer">
            MusicXML
          </a>
          <a className="btn btnSecondary" href={midiUrl} target="_blank" rel="noreferrer">
            MIDI
          </a>
          <a className="btn" href={pdfUrl} target="_blank" rel="noreferrer">
            PDF
          </a>
        </div>
      </div>

      <div className="scoreViewport">
        <div ref={containerRef} />
      </div>
    </section>
  );
}
