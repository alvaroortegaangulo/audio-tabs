"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import {
  Accidental,
  Barline,
  Beam,
  Dot,
  Formatter,
  Renderer,
  Stave,
  StaveModifier,
  StaveNote,
  StaveTie,
  Tuplet,
  Voice,
} from "vexflow/bravura";

import { BACKEND } from "@/lib/api";

type KeySignature = {
  tonic: string;
  mode: "major" | "minor";
  fifths: number;
  name: string;
  vexflow: string;
  use_flats: boolean;
  score: number;
};

type TupletSpec = {
  num_notes: number;
  notes_occupied: number;
};

type ScoreItem = {
  rest?: boolean;
  keys?: string[];
  duration: string;
  dots?: number;
  tuplet?: TupletSpec | null;
  tie?: "start" | "stop" | "continue" | null;
};

type ScoreMeasure = {
  number: number;
  items: ScoreItem[];
};

type ScoreData = {
  grid_q: number;
  grid_kind: "straight" | "triplet";
  measures: ScoreMeasure[];
};

export type JobResult = {
  tempo_bpm: number;
  time_signature: string;
  key_signature?: KeySignature | null;
  transcription_backend?: string | null;
  transcription_error?: string | null;
  score?: ScoreData | null;
};

function parseTimeSignature(timeSignature: string): { num: number; den: number } {
  const [n, d] = (timeSignature ?? "4/4").split("/");
  const num = Number.parseInt(n ?? "4", 10);
  const den = Number.parseInt(d ?? "4", 10);
  if (!Number.isFinite(num) || !Number.isFinite(den) || num <= 0 || den <= 0) return { num: 4, den: 4 };
  return { num, den };
}

function ensureKeys(keys: string[] | undefined): string[] {
  return (keys ?? []).filter(Boolean);
}

function addDots(note: StaveNote, dots: number) {
  for (let i = 0; i < dots; i += 1) {
    Dot.buildAndAttach([note], { all: true });
  }
}

export default function ScoreViewer({ result, jobId }: { result: JobResult; jobId: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(0);

  const tempo = useMemo(() => Math.max(30, Math.min(300, Number(result.tempo_bpm) || 120)), [result.tempo_bpm]);
  const timeSig = useMemo(() => parseTimeSignature(result.time_signature), [result.time_signature]);
  const score = result.score;

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

    el.innerHTML = "";

    if (!score?.measures?.length) {
      el.innerHTML = `<div style="color:#6b7280;font-size:14px;padding:8px 2px">No hay partitura disponible.</div>`;
      return;
    }

    const measures = score.measures;
    const measuresPerLine = width < 640 ? 1 : width < 980 ? 2 : 4;

    const paddingX = 18;
    const paddingY = 16;
    const lineHeight = 170;
    const lineCount = Math.ceil(measures.length / measuresPerLine);
    const staveWidth = Math.floor((width - paddingX * 2) / measuresPerLine);
    const height = paddingY * 2 + lineCount * lineHeight;

    const renderer = new Renderer(el, Renderer.Backends.SVG);
    renderer.resize(width, height);
    const ctx = renderer.getContext();
    ctx.setFillStyle("#111827");
    ctx.setStrokeStyle("#111827");

    const timeSigString = `${timeSig.num}/${timeSig.den}`;
    const beamGroups = Beam.getDefaultBeamGroups(timeSigString);
    const keySig = result.key_signature?.vexflow ?? undefined;

    for (let mi = 0; mi < measures.length; mi += 1) {
      const line = Math.floor(mi / measuresPerLine);
      const col = mi % measuresPerLine;
      const x = paddingX + col * staveWidth;
      const y = paddingY + line * lineHeight;

      const stave = new Stave(x, y, staveWidth, { space_above_staff_ln: 10, space_below_staff_ln: 10 });
      const isLineStart = col === 0;

      if (!isLineStart) stave.setBegBarType(Barline.type.NONE);
      stave.setEndBarType(mi === measures.length - 1 ? Barline.type.END : Barline.type.SINGLE);

      if (isLineStart) {
        stave.addClef("treble");
        if (keySig) stave.setKeySignature(keySig);
        stave.setTimeSignature(timeSigString);
      }

      stave.setText(String(measures[mi]!.number), StaveModifier.Position.ABOVE, { shift_y: -12, shift_x: 0 });
      stave.setContext(ctx).draw();

      const notes: StaveNote[] = [];
      const tuplets: Tuplet[] = [];
      const ties: StaveTie[] = [];

      let activeTuplet: { spec: TupletSpec; notes: StaveNote[] } | null = null;

      const items = measures[mi]!.items ?? [];
      for (let ii = 0; ii < items.length; ii += 1) {
        const it = items[ii]!;
        const isRest = Boolean(it.rest) || ensureKeys(it.keys).length === 0;
        const dur = it.duration;
        const dots = Number(it.dots ?? 0) || 0;

        const note = isRest
          ? new StaveNote({ keys: ["b/4"], duration: dur, type: "r" })
          : new StaveNote({ keys: ensureKeys(it.keys), duration: dur });

        if (dots > 0) addDots(note, dots);
        notes.push(note);

        if (it.tuplet) {
          if (!activeTuplet) {
            activeTuplet = { spec: it.tuplet, notes: [note] };
          } else if (
            activeTuplet.spec.num_notes === it.tuplet.num_notes &&
            activeTuplet.spec.notes_occupied === it.tuplet.notes_occupied
          ) {
            activeTuplet.notes.push(note);
          } else {
            if (activeTuplet.notes.length > 0) {
              tuplets.push(
                new Tuplet(activeTuplet.notes, {
                  num_notes: activeTuplet.spec.num_notes,
                  notes_occupied: activeTuplet.spec.notes_occupied,
                })
              );
            }
            activeTuplet = { spec: it.tuplet, notes: [note] };
          }

          if (activeTuplet && activeTuplet.notes.length === it.tuplet.num_notes) {
            tuplets.push(
              new Tuplet(activeTuplet.notes, {
                num_notes: activeTuplet.spec.num_notes,
                notes_occupied: activeTuplet.spec.notes_occupied,
              })
            );
            activeTuplet = null;
          }
        } else if (activeTuplet) {
          if (activeTuplet.notes.length > 0) {
            tuplets.push(
              new Tuplet(activeTuplet.notes, {
                num_notes: activeTuplet.spec.num_notes,
                notes_occupied: activeTuplet.spec.notes_occupied,
              })
            );
          }
          activeTuplet = null;
        }
      }

      if (activeTuplet?.notes.length) {
        tuplets.push(
          new Tuplet(activeTuplet.notes, {
            num_notes: activeTuplet.spec.num_notes,
            notes_occupied: activeTuplet.spec.notes_occupied,
          })
        );
      }

      // Build simple ties based on the backend-emitted tie flags.
      for (let i = 0; i < items.length - 1; i += 1) {
        const a = items[i]!;
        const b = items[i + 1]!;
        const aTie = a.tie ?? null;
        const bTie = b.tie ?? null;
        if (!aTie || !bTie) continue;
        if (!(aTie === "start" || aTie === "continue")) continue;
        if (!(bTie === "stop" || bTie === "continue")) continue;

        const aKeys = ensureKeys(a.keys);
        const bKeys = ensureKeys(b.keys);
        if (aKeys.length === 0 || bKeys.length === 0) continue;
        if (aKeys.join(",") !== bKeys.join(",")) continue;

        const firstNote = notes[i]!;
        const lastNote = notes[i + 1]!;
        const idx = aKeys.map((_k, ix) => ix);
        ties.push(new StaveTie({ first_note: firstNote, last_note: lastNote, first_indices: idx, last_indices: idx }));
      }

      const voice = new Voice({ num_beats: timeSig.num, beat_value: timeSig.den }).setStrict(false);
      voice.addTickables(notes);

      if (keySig) Accidental.applyAccidentals([voice], keySig);

      const beams = Beam.generateBeams(notes, { groups: beamGroups, beam_rests: false });

      new Formatter().joinVoices([voice]).format([voice], staveWidth - 26);
      voice.draw(ctx, stave);

      for (const beam of beams) beam.setContext(ctx).draw();
      for (const tuplet of tuplets) tuplet.setContext(ctx).draw();
      for (const tie of ties) tie.setContext(ctx).draw();
    }
  }, [result, score, tempo, timeSig.den, timeSig.num, width]);

  const xmlUrl = `${BACKEND}/v1/jobs/${jobId}/musicxml`;
  const midiUrl = `${BACKEND}/v1/jobs/${jobId}/transcription.mid`;

  return (
    <section className="scoreShell">
      <div className="scoreToolbar">
        <div className="scoreMeta">
          <div className="scoreTitle">Partitura</div>
          <div className="scoreSub">
            {result.key_signature?.name ? `${result.key_signature.name} · ` : ""}
            {Math.round(tempo)} bpm · {result.time_signature} · {result.transcription_backend ?? "basic_pitch"}
          </div>
          {result.transcription_error && (
            <div className="scoreSub" style={{ color: "crimson" }}>
              {result.transcription_error}
            </div>
          )}
        </div>
        <div className="scoreActions">
          <a className="btn btnSecondary" href={xmlUrl} target="_blank" rel="noreferrer">
            MusicXML
          </a>
          <a className="btn" href={midiUrl} target="_blank" rel="noreferrer">
            MIDI
          </a>
        </div>
      </div>

      <div className="scoreViewport">
        <div ref={containerRef} />
      </div>
    </section>
  );
}
