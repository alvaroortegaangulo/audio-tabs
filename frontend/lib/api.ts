export const BACKEND = process.env.NEXT_PUBLIC_BACKEND_URL!;

export async function createJob(file: File): Promise<{ job_id: string; status: string }> {
  const fd = new FormData();
  fd.append("file", file);

  const res = await fetch(`${BACKEND}/v1/jobs`, { method: "POST", body: fd });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function getJob(jobId: string): Promise<{ job_id: string; status: string; error?: string | null }> {
  const res = await fetch(`${BACKEND}/v1/jobs/${jobId}`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function getResult(jobId: string): Promise<any> {
  const res = await fetch(`${BACKEND}/v1/jobs/${jobId}/result.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.json();
}

export async function getMusicXML(jobId: string): Promise<string> {
  const res = await fetch(`${BACKEND}/v1/jobs/${jobId}/musicxml`, { cache: "no-store" });
  if (!res.ok) throw new Error(await res.text());
  return await res.text();
}
