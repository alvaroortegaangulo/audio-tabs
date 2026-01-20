"use client";

import { useEffect, useRef, useState } from "react";
import { getJob, getMusicXML } from "@/lib/api";
import OSMDViewer from "@/components/OSMDViewer";

type JobClientProps = {
  jobId: string;
};

export default function JobClient({ jobId }: JobClientProps) {
  const [status, setStatus] = useState<string>("queued");
  const [error, setError] = useState<string | null>(null);
  const [xmlContent, setXmlContent] = useState<string | null>(null);
  const [xmlLoading, setXmlLoading] = useState(false);
  const xmlContentRef = useRef<string | null>(null);
  const xmlLoadingRef = useRef(false);

  useEffect(() => {
    xmlContentRef.current = xmlContent;
  }, [xmlContent]);

  useEffect(() => {
    xmlLoadingRef.current = xmlLoading;
  }, [xmlLoading]);

  useEffect(() => {
    if (!jobId) return;

    setStatus("queued");
    setError(null);
    setXmlContent(null);
    setXmlLoading(false);
    xmlContentRef.current = null;
    xmlLoadingRef.current = false;

    let timer: ReturnType<typeof setTimeout> | undefined;
    let cancelled = false;

    async function poll() {
      try {
        const j = await getJob(jobId);
        if (cancelled) return;
        setStatus(j.status);
        setError(j.error ?? null);

        if (j.status === "done") {
          if (!xmlContentRef.current && !xmlLoadingRef.current) {
            setXmlLoading(true);
            try {
              const xmlText = await getMusicXML(jobId);
              if (!cancelled) {
                setXmlContent(xmlText);
              }
            } catch (e: any) {
              if (!cancelled) {
                setError(e?.message ?? String(e));
              }
            } finally {
              if (!cancelled) {
                setXmlLoading(false);
              }
            }
          }
          if (!xmlContentRef.current) {
            timer = setTimeout(poll, 1000);
          }
          return;
        }
        if (j.status === "error") return;
      } catch (e: any) {
        if (!cancelled) {
          setError(e?.message ?? String(e));
          setXmlLoading(false);
        }
      }

      timer = setTimeout(poll, 1000);
    }

    poll();
    return () => {
      cancelled = true;
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

      {status === "done" && jobId && xmlContent && <OSMDViewer musicXML={xmlContent} />}
      {status === "done" && jobId && !xmlContent && (
        <div className="card">
          <p>{xmlLoading ? "Cargando partitura..." : "Partitura no disponible aun."}</p>
        </div>
      )}
    </>
  );
}
