"use client";

import { BACKEND } from "@/lib/api";

export default function ScorePdf({ jobId }: { jobId: string }) {
  const url = `${BACKEND}/v1/jobs/${jobId}/score.pdf`;

  return (
    <div className="card">
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: 12 }}>
        <h2 style={{ margin: 0 }}>Partitura (Real Book PDF)</h2>
        <a className="btn" href={url} target="_blank" rel="noreferrer">
          Descargar PDF
        </a>
      </div>

      <div style={{ height: 12 }} />

      <iframe
        src={url}
        style={{ width: "100%", height: "80vh", border: "1px solid #e5e7eb", borderRadius: 12 }}
        title="Score PDF"
      />
    </div>
  );
}
