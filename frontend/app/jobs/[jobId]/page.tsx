"use client";

import { useEffect, useState } from "react";
import { getJob, getResult } from "@/lib/api";
import ScorePdf from "@/components/ScorePdf";

export default function JobPage({ params }: { params: { jobId?: string } }) {
  const jobId = params.jobId ?? "";

  const [status, setStatus] = useState<string>("queued");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    if (!jobId) return;

    let timer: any;

    async function poll() {
      try {
        const j = await getJob(jobId);
        setStatus(j.status);
        setError(j.error ?? null);

        if (j.status === "done") {
          const res = await getResult(jobId);
          setResult(res);
          return;
        }
        if (j.status === "error") return;
      } catch (e: any) {
        setError(e?.message ?? String(e));
      }

      timer = setTimeout(poll, 1000);
    }

    poll();
    return () => timer && clearTimeout(timer);
  }, [jobId]);

  return (
    <>
      <h1>Job {jobId || "(sin id)"}</h1>

      <div className="card">
        <p>
          Estado: <span className="mono">{status}</span>
        </p>
        {error && <p style={{ color: "crimson" }}>{error}</p>}
      </div>

      {status === "done" && jobId && <ScorePdf jobId={jobId} />}

      {/* Debug opcional (puedes borrar este bloque cuando ya no lo necesites) */}
      {result && (
        <div className="card">
          <h2>Chords (debug)</h2>
          <p>
            Tempo: {result.tempo_bpm?.toFixed?.(1)} bpm · Compás: {result.time_signature}
          </p>
          <div className="mono" style={{ whiteSpace: "pre-wrap" }}>
            {(result.chords ?? [])
              .map(
                (c: any) =>
                  `${c.start.toFixed(2)}-${c.end.toFixed(2)}  ${c.label}  (p=${c.confidence.toFixed(2)})`
              )
              .join("\n")}
          </div>
        </div>
      )}
    </>
  );
}
