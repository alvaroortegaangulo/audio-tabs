"use client";

import { useState } from "react";
import { createJob } from "@/lib/api";
import { useRouter } from "next/navigation";

export default function UploadForm() {
  const [file, setFile] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const router = useRouter();

  async function onSubmit() {
    if (!file) return;
    setBusy(true);
    setErr(null);
    try {
      const { job_id } = await createJob(file);
      router.push(`/jobs/${job_id}`);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <h2>Sube un audio (wav/mp3)</h2>
      <input
        type="file"
        accept=".wav,.mp3,.m4a,.flac,.ogg"
        onChange={(e) => setFile(e.target.files?.[0] ?? null)}
      />
      <div style={{ height: 12 }} />
      <button className="btn" disabled={!file || busy} onClick={onSubmit}>
        {busy ? "Procesando..." : "Extraer acordes"}
      </button>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      <p style={{ opacity: 0.7 }}>
        MVP: detecta maj/min/7, segmenta y genera MusicXML renderizable.
      </p>
    </div>
  );
}
