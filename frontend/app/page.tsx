import UploadForm from "@/components/UploadForm";

export default function Page() {
  return (
    <>
      <h1>Chord Extractor (v1)</h1>
      <div className="row">
        <div style={{ flex: 1, minWidth: 320 }}>
          <UploadForm />
        </div>
        <div style={{ flex: 1, minWidth: 320 }} className="card">
          <h2>Qué hace esta versión</h2>
          <ul>
            <li>Convierte audio a mono 44.1k</li>
            <li>Extrae cromagrama + suavizado por Viterbi</li>
            <li>Genera MusicXML con acordes</li>
            <li>Renderiza en navegador con OSMD</li>
          </ul>
        </div>
      </div>
    </>
  );
}
