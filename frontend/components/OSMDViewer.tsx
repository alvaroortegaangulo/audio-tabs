"use client";

import { useEffect, useRef, useState } from "react";

export default function OSMDViewer({ musicXML }: { musicXML: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;

    async function run() {
      setErr(null);
      if (!ref.current) return;

      const mod = await import("opensheetmusicdisplay");
      const { OpenSheetMusicDisplay } = mod as any;

      ref.current.innerHTML = "";
      const osmd = new OpenSheetMusicDisplay(ref.current, {
        drawingParameters: "default",
        drawPartNames: true,
        drawPartAbbreviations: true,
        drawTitle: true,
        drawSubtitle: true,
        drawFingerings: true,
        autoResize: true,
        renderSingleHorizontalStaffline: false,
      });

      try {
        await osmd.load(musicXML);
        if (!disposed) osmd.render();
      } catch (e: any) {
        setErr(e?.message ?? String(e));
      }
    }

    run();
    return () => { disposed = true; };
  }, [musicXML]);

  return (
    <div className="card">
      <h2>Partitura (MusicXML)</h2>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      <div ref={ref} style={{ width: "100%" }} />
    </div>
  );
}
