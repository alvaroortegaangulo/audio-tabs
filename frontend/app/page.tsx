import UploadForm from "@/components/UploadForm";

export default function Page() {
  return (
    <>
      <h1>Audio → Lead Sheet</h1>
      <div className="row">
        <div style={{ flex: 1, minWidth: 320 }}>
          <UploadForm />
        </div>
        <div style={{ flex: 1, minWidth: 320 }} className="card">
          <h2>Qué hace esta versión</h2>
          <ul>
            <li>Convierte el audio a mono 44.1kHz</li>
            <li>Extrae la progresión de acordes</li>
            <li>Genera MusicXML y PDF (LilyPond)</li>
            <li>Renderiza un lead sheet en SVG (VexFlow)</li>
          </ul>
        </div>
      </div>
    </>
  );
}

