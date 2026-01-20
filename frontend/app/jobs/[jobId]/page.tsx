import JobClient from "./JobClient";

type PageProps = {
  params?: Promise<{
    jobId?: string | string[];
  }>;
};

export default async function JobPage({ params }: PageProps) {
  const resolvedParams = (await params) ?? {};
  const jobIdParam = resolvedParams.jobId;
  const jobId = Array.isArray(jobIdParam) ? jobIdParam[0] : jobIdParam ?? "";

  return <JobClient jobId={jobId} />;
}
