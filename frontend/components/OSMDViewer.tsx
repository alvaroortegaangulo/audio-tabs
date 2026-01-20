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
      // Configure OSMD for Guitar Lead Sheet
      const osmd = new OpenSheetMusicDisplay(ref.current, {
        drawingParameters: "compacttight", // or "default"
        drawPartNames: false,
        drawTitle: true,
        drawSubtitle: true,
        drawFingerings: true, // Crucial for Tab
        autoResize: true,
        // Ensure that we show both staves if they are in the XML
        // OSMD usually auto-detects multiple staves.
      });

      // Additional options usually passed via Engines
      // osmd.setOptions({ ... }) could be used if strictly typed but constructor opts work.

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
      <div ref={ref} />
    </div>
  );
}
