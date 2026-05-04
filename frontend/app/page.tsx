"use client";

import { Download, FileSpreadsheet, Loader2, Play, RotateCcw } from "lucide-react";
import { useMemo, useState } from "react";
import dynamic from "next/dynamic";

const TopicGraph = dynamic(() => import("./TopicGraph"), { ssr: false });

const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_BASE ??
  "http://127.0.0.1:8010"
).replace(/\/$/, "");
const API_BASE_IS_LOCAL = API_BASE.includes("127.0.0.1") || API_BASE.includes("localhost");
const DEFAULT_LDA_STOPWORDS = [
  "mohs", "mohs surgery", "surgery", "skin", "cancer",
  "dermatologist", "doctor", "basal cell carcinoma", "squamous cell carcinoma",
];
const CATEGORIES = [
  "Management/recovery", "Clinical presentation", "Procedure/reconstruction",
  "Emotion/anxiety", "Cost/access", "Information appraisal",
];
const PROGRESS_STEPS = [
  "Collecting Reddit posts",
  "Collecting Reddit comments",
  "Cleaning text",
  "Running LDA",
  "Running sentiment",
  "Generating figures",
  "Exporting results",
];

type Topic = {
  topic: number;
  keywords: string[];
  doc_count: number;
  percentage: number;
  distinctiveness: number;
  representative_document: string;
  example_documents: Array<{ id: string; type: string; date: string; score: number; permalink: string; text: string }>;
  label: string;
  category: string;
  llm_topic_title?: string;
  llm_summary?: string;
  llm_explanation?: string;
  notable_recommendations?: string;
  cautions_or_uncertainties?: string;
  official_practice_area?: string;
  comparison_guidance?: string;
  llm_error?: string;
};

type AnalysisResult = {
  corpus_stats: Record<string, string | number>;
  topics: Topic[];
  export_links: Array<{ name: string; url: string }>;
};

type AnalysisJob = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  step_index: number;
  step: string;
  steps: string[];
  collection_progress?: {
    posts_collected: number;
    comments_collected: number;
    max_posts: number;
    eta_seconds: number | null;
  };
  result: AnalysisResult | null;
  error: string | null;
};

export default function Home() {
  const [form, setForm] = useState({
    subreddit: "r/MohsSurgery",
    keywords: DEFAULT_LDA_STOPWORDS.join(", "),
    start_date: "2024-01-01",
    end_date: new Date().toISOString().slice(0, 10),
    k: 5,
    auto_k: true,
    max_results: 100,
  });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [running, setRunning] = useState(false);
  const [progressIndex, setProgressIndex] = useState(-1);
  const [progressStatus, setProgressStatus] = useState<AnalysisJob["status"] | "idle">("idle");
  const [collectionProgress, setCollectionProgress] = useState<AnalysisJob["collection_progress"] | null>(null);
  const [error, setError] = useState("");

  async function parseJsonResponse<T>(response: Response): Promise<T> {
    const text = await response.text();
    try {
      return text ? (JSON.parse(text) as T) : ({} as T);
    } catch {
      throw new Error(`Backend returned a non-JSON response from ${response.url}: ${text.slice(0, 180)}`);
    }
  }

  function networkErrorMessage(err: unknown) {
    const message = err instanceof Error ? err.message : "Analysis failed";
    if (message === "Failed to fetch" || message.includes("fetch"))
      return `Could not reach the backend at ${API_BASE}.`;
    return message;
  }

  const payload = useMemo(() => ({
    ...form,
    keywords: form.keywords.split(",").map((s) => s.trim().replace(/^"|"$/g, "")).filter(Boolean),
    k: Number(form.k),
    max_results: Number(form.max_results),
  }), [form]);

  async function runAnalysis() {
    setRunning(true);
    setProgressStatus("queued");
    setError("");
    setResult(null);
    setProgressIndex(-1);
    setCollectionProgress(null);
    try {
      const response = await fetch(`${API_BASE}/analysis-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await parseJsonResponse<{ job_id: string; detail?: string }>(response);
      if (!response.ok) throw new Error(data.detail ?? "Analysis failed");

      let finished = false;
      while (!finished) {
        await new Promise((r) => window.setTimeout(r, 700));
        const statusResponse = await fetch(`${API_BASE}/analysis-jobs/${data.job_id}`);
        const job = await parseJsonResponse<AnalysisJob & { detail?: string }>(statusResponse);
        if (!statusResponse.ok) throw new Error(job.detail ?? "Could not read analysis status");
        setProgressStatus(job.status);
        setProgressIndex(job.step_index);
        setCollectionProgress(job.collection_progress ?? null);
        if (job.status === "completed") {
          if (!job.result) throw new Error("Analysis completed without results");
          setResult(job.result);
          setTopics(job.result.topics);
          finished = true;
        }
        if (job.status === "failed") throw new Error(job.error ?? "Analysis failed");
      }
    } catch (err) {
      setError(networkErrorMessage(err));
    } finally {
      setRunning(false);
    }
  }

  function reset() {
    setResult(null);
    setTopics([]);
    setProgressStatus("idle");
    setProgressIndex(-1);
    setError("");
  }

  const primaryExport = result?.export_links.find((l) => l.name === "final_topics_comparison.xlsx");
  const hasResult = !!result;

  // ── Start page: centered form ─────────────────────────────────────────────
  if (!hasResult && !running) {
    return (
      <main className="flex min-h-screen flex-col bg-ink text-slate-100">
        <div className="mx-auto flex w-full max-w-lg flex-col px-6 pt-24 pb-20">
          <div className="mb-8">
            <h1 className="text-2xl font-semibold text-white">Post-Mohs Recovery Analysis</h1>
            <p className="mt-2 text-sm leading-6 text-slate-400">
              Mine r/MohsSurgery for recovery-advice themes and read LLM-summarized topic interpretations.
            </p>
          </div>

          <div className="rounded-xl border border-white/[0.06] bg-slate-900/70 p-6">
            <div className="grid gap-5">
              <Field label="Subreddit">
                <input className="field" value={form.subreddit} onChange={(e) => setForm({ ...form, subreddit: e.target.value })} />
              </Field>
              <Field label="LDA filter words">
                <textarea className="field" rows={4} value={form.keywords} onChange={(e) => setForm({ ...form, keywords: e.target.value })} />
                <p className="text-[11px] text-slate-600">Comma-separated words to exclude from topic modelling</p>
              </Field>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Start date">
                  <input type="date" className="field" value={form.start_date} onChange={(e) => setForm({ ...form, start_date: e.target.value })} />
                </Field>
                <Field label="End date">
                  <input type="date" className="field" value={form.end_date} onChange={(e) => setForm({ ...form, end_date: e.target.value })} />
                </Field>
              </div>
              <div className="grid grid-cols-2 gap-4">
                <Field label="Number of topics">
                  <div className="flex gap-2">
                    <button
                      type="button"
                      onClick={() => setForm((f) => ({ ...f, auto_k: !f.auto_k }))}
                      className={`shrink-0 rounded-md border px-3 py-2 text-sm font-medium transition ${form.auto_k ? "border-accent/60 bg-teal-950/40 text-accent" : "border-line bg-slate-900/80 text-slate-400 hover:border-slate-500"}`}
                    >
                      Auto
                    </button>
                    <input
                      type="number" min={2} max={50}
                      className={`field ${form.auto_k ? "opacity-40 cursor-not-allowed" : ""}`}
                      value={form.k}
                      disabled={form.auto_k}
                      onChange={(e) => setForm({ ...form, k: Number(e.target.value) })}
                    />
                  </div>
                  {form.auto_k && <p className="mt-1 text-[11px] text-slate-600">Optimal k selected via coherence scoring</p>}
                </Field>
                <Field label="Max posts">
                  <input type="number" min={10} max={5000} step={10} className="field" value={form.max_results} onChange={(e) => setForm({ ...form, max_results: Number(e.target.value) })} />
                </Field>
              </div>
            </div>

            {API_BASE_IS_LOCAL && (
              <p className="mt-4 break-all text-[11px] text-slate-700" suppressHydrationWarning>Backend: {API_BASE}</p>
            )}
            {error && (
              <div className="mt-4 rounded-lg border border-red-900/60 bg-red-950/40 px-3 py-2.5 text-sm text-red-300">{error}</div>
            )}
            <button
              onClick={runAnalysis}
              disabled={running}
              className="mt-5 flex h-10 w-full items-center justify-center gap-2 rounded-lg bg-accent text-sm font-semibold text-slate-950 transition hover:bg-teal-200 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Play size={15} />
              Run analysis
            </button>
          </div>
        </div>
      </main>
    );
  }

  // ── Running state: centered progress ─────────────────────────────────────
  if (running) {
    return (
      <main className="flex min-h-screen flex-col items-center bg-ink text-slate-100">
        <div className="mx-auto flex w-full max-w-md flex-col px-6 pt-24">
          <h1 className="mb-1 text-xl font-semibold text-white">Running analysis…</h1>
          <p className="mb-10 text-sm text-slate-500">{form.subreddit} · {form.k} topics · {form.max_results} posts</p>

          <div className="rounded-xl border border-white/[0.06] bg-slate-900/70 p-6">
            <div className="grid gap-3">
              {PROGRESS_STEPS.map((step, index) => {
                const done = index < progressIndex || progressStatus === "completed";
                const active = index === progressIndex && running;
                return (
                  <div key={step} className="flex items-center gap-3">
                    <span className={`h-2 w-2 shrink-0 rounded-full ${done ? "bg-accent" : active ? "bg-amber animate-pulse" : "bg-slate-700"}`} />
                    <span className={`text-sm ${done ? "text-slate-200" : active ? "text-white" : "text-slate-600"}`}>{step}</span>
                    {active && <Loader2 className="ml-auto animate-spin text-amber" size={14} />}
                  </div>
                );
              })}
            </div>
            {collectionProgress && (progressIndex === 0 || progressIndex === 1) && (
              <div className="mt-5 border-t border-white/[0.06] pt-4 text-xs text-slate-400">
                <div>{collectionProgress.posts_collected} / {collectionProgress.max_posts} posts collected</div>
                {collectionProgress.eta_seconds !== null && (
                  <div className="mt-1 text-slate-600">ETA {formatDuration(collectionProgress.eta_seconds)}</div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>
    );
  }

  // ── Results state: full-viewport graph ───────────────────────────────────
  const totalDocs = Object.values(result!.corpus_stats).find((v) => typeof v === "number" && (v as number) > 10) as number ?? topics.reduce((s, t) => s + t.doc_count, 0);

  return (
    <main className="flex h-screen flex-col overflow-hidden bg-ink text-slate-100">
      {/* Top bar */}
      <header className="flex shrink-0 items-center justify-between border-b border-white/[0.06] px-6 py-3">
        <div className="flex items-center gap-4">
          <h1 className="text-sm font-semibold text-white">Post-Mohs Recovery Analysis</h1>
          <span className="text-xs text-slate-600">{form.subreddit} · {topics.length} topics</span>
        </div>
        <div className="flex items-center gap-3">
          {primaryExport && (
            <a
              href={`${API_BASE}${primaryExport.url}`}
              className="flex items-center gap-1.5 rounded-lg border border-accent/30 bg-teal-950/30 px-3 py-1.5 text-xs text-slate-200 transition hover:border-accent/60"
            >
              <FileSpreadsheet size={12} className="text-accent" />
              Download workbook
            </a>
          )}
          {result!.export_links.length > 0 && (
            <div className="flex items-center gap-2">
              {result!.export_links.slice(0, 3).map((link) => (
                <a key={link.url} href={`${API_BASE}${link.url}`} className="text-xs text-slate-600 hover:text-slate-300">
                  <Download size={12} />
                </a>
              ))}
            </div>
          )}
          <button
            onClick={reset}
            className="flex items-center gap-1.5 text-xs text-slate-600 transition hover:text-slate-300"
          >
            <RotateCcw size={12} /> New analysis
          </button>
        </div>
      </header>

      {/* Corpus stats strip */}
      <div className="flex shrink-0 items-center gap-8 border-b border-white/[0.04] bg-slate-900/30 px-6 py-2.5">
        {Object.entries(result!.corpus_stats).map(([key, value]) => (
          <div key={key} className="flex items-baseline gap-1.5">
            <span className="text-sm font-semibold text-white">{String(value)}</span>
            <span className="text-[11px] text-slate-600">{key.replaceAll("_", " ")}</span>
          </div>
        ))}
      </div>

      {/* Graph — fills remaining space */}
      <div className="relative min-h-0 flex-1">
        <TopicGraph topics={topics} totalDocs={totalDocs} />
      </div>
    </main>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <label className="grid gap-1.5">
      <span className="text-xs text-slate-400">{label}</span>
      {children}
    </label>
  );
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!value)}
      className="flex items-center justify-between rounded-lg bg-slate-800/50 px-3 py-2.5 text-sm text-slate-300 transition hover:bg-slate-800"
    >
      <span>{label}</span>
      <span className={`relative h-5 w-9 rounded-full transition-colors ${value ? "bg-accent" : "bg-slate-700"}`}>
        <span className={`absolute top-1 h-3 w-3 rounded-full bg-slate-950 transition-all ${value ? "left-5" : "left-1"}`} />
      </span>
    </button>
  );
}

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) return "calculating…";
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

function formatJsonList(value: string): string {
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) return parsed.join(", ");
  } catch {
    if (value.startsWith("[") && value.endsWith("]")) {
      const items = value.slice(1, -1).split(/',\s*'/).map((s) => s.replace(/^['"]|['"]$/g, "").trim()).filter(Boolean);
      if (items.length > 1) return items.join(", ");
    }
    return value;
  }
  return value;
}
