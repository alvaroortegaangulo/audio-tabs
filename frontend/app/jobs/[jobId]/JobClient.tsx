"use client";

import { useEffect, useState } from "react";
import { getJob, getResult } from "@/lib/api";
import LeadSheet from "@/components/LeadSheet";
import ScoreViewer from "@/components/ScoreViewer";

type JobClientProps = {
  jobId: string;
};

export default function JobClient({ jobId }: JobClientProps) {
  const [status, setStatus] = useState<string>("queued");
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<any>(null);

  useEffect(() => {
    if (!jobId) return;

    let timer: ReturnType<typeof setTimeout> | undefined;

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
    return () => {
      if (timer) {
        clearTimeout(timer);
      }
    };
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

      {status === "done" && jobId && result && (
        <>
          {Array.isArray(result?.chords) ? (
            <LeadSheet jobId={jobId} result={result} />
          ) : (
            <ScoreViewer jobId={jobId} result={result} />
          )}
        </>
      )}
    </>
  );
}
