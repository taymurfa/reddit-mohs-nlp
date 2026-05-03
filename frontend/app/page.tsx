"use client";

import dynamic from "next/dynamic";
import { Download, Loader2, Play, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "http://127.0.0.1:8010";
const DEFAULT_LDA_STOPWORDS = [
  "mohs",
  "mohs surgery",
  "surgery",
  "skin",
  "cancer",
  "dermatologist",
  "doctor",
  "basal cell carcinoma",
  "squamous cell carcinoma",
];
const CATEGORIES = [
  "Management/recovery",
  "Clinical presentation",
  "Procedure/reconstruction",
  "Emotion/anxiety",
  "Cost/access",
  "Information appraisal",
];
const PROGRESS_STEPS = [
  "Collecting Reddit data",
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
  example_documents: Array<{
    id: string;
    type: string;
    date: string;
    score: number;
    permalink: string;
    text: string;
  }>;
  label: string;
  category: string;
  llm_topic_title?: string;
  llm_summary?: string;
  llm_explanation?: string;
  official_practice_area?: string;
  comparison_guidance?: string;
  llm_summary_source?: string;
};

type FigureSpec = {
  id: string;
  title: string;
  spec: { data: unknown[]; layout: Record<string, unknown>; frames?: unknown[] };
};

type AnalysisResult = {
  corpus_stats: Record<string, string | number>;
  topics: Topic[];
  category_percentages: Array<Record<string, string | number>>;
  sentiment_summary: {
    overall_distribution: Record<string, number>;
    by_topic: Array<Record<string, string | number>>;
  };
  treatment_sentiment: Array<Record<string, string | number>>;
  shared_domains: Array<Record<string, string | number>>;
  figures: FigureSpec[];
  export_links: Array<{ name: string; url: string }>;
};

type AnalysisJob = {
  job_id: string;
  status: "queued" | "running" | "completed" | "failed";
  step_index: number;
  step: string;
  steps: string[];
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
    max_results: 100,
    include_comments: true,
    sample_mode: true,
  });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [running, setRunning] = useState(false);
  const [progressIndex, setProgressIndex] = useState(-1);
  const [progressStatus, setProgressStatus] = useState<AnalysisJob["status"] | "idle">("idle");
  const [error, setError] = useState("");

  const payload = useMemo(
    () => ({
      ...form,
      keywords: form.keywords
        .split(",")
        .map((item) => item.trim().replace(/^"|"$/g, ""))
        .filter(Boolean),
      k: Number(form.k),
      max_results: Number(form.max_results),
    }),
    [form],
  );

  async function runAnalysis() {
    setRunning(true);
    setProgressStatus("queued");
    setError("");
    setResult(null);
    setProgressIndex(-1);
    try {
      const response = await fetch(`${API_BASE}/analysis-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.detail ?? "Analysis failed");
      }
      let finished = false;
      while (!finished) {
        await new Promise((resolve) => window.setTimeout(resolve, 700));
        const statusResponse = await fetch(`${API_BASE}/analysis-jobs/${data.job_id}`);
        const job: AnalysisJob = await statusResponse.json();
        if (!statusResponse.ok) {
          throw new Error((job as unknown as { detail?: string }).detail ?? "Could not read analysis status");
        }
        setProgressStatus(job.status);
        setProgressIndex(job.step_index);
        if (job.status === "completed") {
          if (!job.result) {
            throw new Error("Analysis completed without results");
          }
          setResult(job.result);
          setTopics(job.result.topics);
          finished = true;
        }
        if (job.status === "failed") {
          throw new Error(job.error ?? "Analysis failed");
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setRunning(false);
    }
  }

  function updateTopic(index: number, key: "label" | "category", value: string) {
    setTopics((items) => items.map((topic, i) => (i === index ? { ...topic, [key]: value } : topic)));
  }

  return (
    <main className="min-h-screen bg-ink text-slate-100">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-4 py-6 lg:px-6">
        <header className="flex flex-col gap-2 border-b border-line pb-5 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-3xl font-semibold tracking-normal text-white">Mohs Reddit NLP Analysis</h1>
            <p className="mt-2 max-w-3xl text-sm leading-6 text-slate-400">
              One-step LDA topic modeling, sentiment analysis, treatment mentions, shared domains, figures, and exports from Reddit public JSON data.
            </p>
          </div>
          <div className="rounded border border-line bg-panel px-3 py-2 text-xs text-slate-300">FastAPI + gensim + VADER + Plotly</div>
        </header>

        <section className="grid gap-6 lg:grid-cols-[380px_1fr]">
          <aside className="h-fit border border-line bg-panel p-4 shadow-glow">
            <div className="mb-4 flex items-center gap-2 text-sm font-semibold uppercase tracking-wide text-accent">
              <SlidersHorizontal size={16} /> Analysis Inputs
            </div>
            <div className="grid gap-4">
              <Label text="Subreddit">
                <input className="field" value={form.subreddit} onChange={(e) => setForm({ ...form, subreddit: e.target.value })} />
              </Label>
              <Label text="LDA filter words">
                <textarea className="field min-h-24" value={form.keywords} onChange={(e) => setForm({ ...form, keywords: e.target.value })} />
              </Label>
              <div className="grid grid-cols-2 gap-3">
                <Label text="Start date">
                  <input type="date" className="field" value={form.start_date} onChange={(e) => setForm({ ...form, start_date: e.target.value })} />
                </Label>
                <Label text="End date">
                  <input type="date" className="field" value={form.end_date} onChange={(e) => setForm({ ...form, end_date: e.target.value })} />
                </Label>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <Label text="Topics k">
                  <input type="number" min={2} max={50} className="field" value={form.k} onChange={(e) => setForm({ ...form, k: Number(e.target.value) })} />
                </Label>
                <Label text="Max posts">
                  <input type="number" min={10} max={5000} className="field" value={form.max_results} onChange={(e) => setForm({ ...form, max_results: Number(e.target.value) })} />
                </Label>
              </div>
              <Toggle label="Include comments" value={form.include_comments} onChange={(value) => setForm({ ...form, include_comments: value })} />
              <Toggle label="Sample data mode" value={form.sample_mode} onChange={(value) => setForm({ ...form, sample_mode: value })} />
              <button
                onClick={runAnalysis}
                disabled={running}
                className="flex h-11 items-center justify-center gap-2 rounded bg-accent px-4 text-sm font-semibold text-slate-950 transition hover:bg-teal-200 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {running ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} />}
                Run Analysis
              </button>
            </div>

            {(running || result) && (
              <div className="mt-5 border-t border-line pt-4">
                <div className="mb-3 flex items-center justify-between gap-3 text-xs font-semibold uppercase tracking-wide text-slate-400">
                  <span>Progress</span>
                  <span>{progressStatus === "idle" ? "Ready" : progressStatus}</span>
                </div>
                <div className="grid gap-2">
                  {PROGRESS_STEPS.map((step, index) => (
                    <div key={step} className="flex items-center gap-2 text-sm">
                      <span className={`h-2 w-2 rounded-full ${index < progressIndex || progressStatus === "completed" ? "bg-accent" : index === progressIndex ? "bg-amber" : "bg-slate-600"}`} />
                      <span className={index <= progressIndex || progressStatus === "completed" ? "text-slate-100" : "text-slate-500"}>{step}</span>
                      {running && index === progressIndex && <Loader2 className="ml-auto animate-spin text-amber" size={14} />}
                    </div>
                  ))}
                </div>
              </div>
            )}
            {error && <div className="mt-4 border border-red-500/40 bg-red-950/40 p-3 text-sm text-red-200">{error}</div>}
          </aside>

          <section className="grid gap-6">
            {!result && (
              <div className="flex min-h-96 items-center justify-center border border-line bg-panel p-8 text-center text-slate-400">
                Configure the form and run the pipeline to populate corpus statistics, LDA topics, sentiment panels, figures, and exports.
              </div>
            )}

            {result && (
              <>
                <Panel title="Corpus Statistics">
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-5">
                    {Object.entries(result.corpus_stats).map(([key, value]) => (
                      <div key={key} className="border border-line bg-slate-950/30 p-3">
                        <div className="text-xs uppercase tracking-wide text-slate-500">{key.replaceAll("_", " ")}</div>
                        <div className="mt-1 text-lg font-semibold text-white">{String(value)}</div>
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel title="LDA Topics Table">
                  <div className="overflow-x-auto">
                    <table className="w-full min-w-[980px] border-collapse text-sm">
                      <thead>
                        <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-slate-500">
                          <th className="py-2 pr-3">Topic</th>
                          <th className="py-2 pr-3">Label</th>
                          <th className="py-2 pr-3">Category</th>
                          <th className="py-2 pr-3">Keywords</th>
                          <th className="py-2 pr-3">LLM summary</th>
                          <th className="py-2 pr-3">Docs</th>
                          <th className="py-2 pr-3">%</th>
                          <th className="py-2 pr-3">Distinct</th>
                          <th className="py-2">Representative examples</th>
                        </tr>
                      </thead>
                      <tbody>
                        {topics.map((topic, index) => (
                          <tr key={topic.topic} className="border-b border-line/70 align-top">
                            <td className="py-3 pr-3 font-semibold text-accent">T{topic.topic + 1}</td>
                            <td className="py-3 pr-3">
                              <input className="table-field w-36" value={topic.label} onChange={(e) => updateTopic(index, "label", e.target.value)} />
                            </td>
                            <td className="py-3 pr-3">
                              <select className="table-field w-52" value={topic.category} onChange={(e) => updateTopic(index, "category", e.target.value)}>
                                {CATEGORIES.map((category) => (
                                  <option key={category}>{category}</option>
                                ))}
                              </select>
                            </td>
                            <td className="max-w-xs py-3 pr-3 text-slate-300">{topic.keywords.join(", ")}</td>
                            <td className="max-w-sm py-3 pr-3 text-slate-300">
                              <div className="font-medium text-slate-100">{topic.llm_topic_title || topic.label}</div>
                              <div className="mt-1 text-slate-400">{topic.llm_summary}</div>
                              {topic.official_practice_area && <div className="mt-2 text-xs text-amber">{topic.official_practice_area}</div>}
                              {topic.llm_summary_source && <div className="mt-2 text-[11px] uppercase tracking-wide text-slate-600">{topic.llm_summary_source}</div>}
                            </td>
                            <td className="py-3 pr-3">{topic.doc_count}</td>
                            <td className="py-3 pr-3">{topic.percentage}</td>
                            <td className="py-3 pr-3">{topic.distinctiveness}</td>
                            <td className="max-w-md py-3 text-slate-400">
                              <div className="grid gap-2">
                                {(topic.example_documents?.length ? topic.example_documents : [{ id: `${topic.topic}-rep`, type: "", date: "", score: 0, permalink: "", text: topic.representative_document }]).map((example) => (
                                  <div key={example.id} className="border border-line/70 bg-slate-950/30 p-2">
                                    <div className="mb-1 flex flex-wrap gap-2 text-[11px] uppercase tracking-wide text-slate-500">
                                      {example.type && <span>{example.type}</span>}
                                      {example.date && <span>{example.date}</span>}
                                      {Number.isFinite(example.score) && <span>score {example.score}</span>}
                                      {example.permalink && (
                                        <a href={example.permalink} target="_blank" rel="noreferrer" className="text-accent hover:text-teal-200">
                                          source
                                        </a>
                                      )}
                                    </div>
                                    <div>{example.text}</div>
                                  </div>
                                ))}
                              </div>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Panel>

                <Panel title="Figures Panel">
                  <div className="grid gap-4 xl:grid-cols-2">
                    {result.figures.map((figure) => (
                      <div key={figure.id} className="border border-line bg-slate-950/30 p-3">
                        <Plot data={figure.spec.data} layout={{ ...figure.spec.layout, autosize: true, height: 360 }} config={{ responsive: true, displaylogo: false, toImageButtonOptions: { format: "png" } }} className="w-full" />
                      </div>
                    ))}
                  </div>
                </Panel>

                <div className="grid gap-6 xl:grid-cols-2">
                  <Panel title="Sentiment Panel">
                    <DataTable rows={Object.entries(result.sentiment_summary.overall_distribution).map(([sentiment, percentage]) => ({ sentiment, percentage }))} />
                  </Panel>
                  <Panel title="Treatment/Product Sentiment">
                    <DataTable rows={result.treatment_sentiment} />
                  </Panel>
                </div>

                <div className="grid gap-6 xl:grid-cols-2">
                  <Panel title="Shared Domains">
                    <DataTable rows={result.shared_domains} />
                  </Panel>
                  <Panel title="Export Panel">
                    <div className="grid gap-2 sm:grid-cols-2">
                      {result.export_links.map((link) => (
                        <a key={link.url} href={`${API_BASE}${link.url}`} className="flex items-center gap-2 border border-line bg-slate-950/40 px-3 py-2 text-sm text-slate-200 hover:border-accent hover:text-accent">
                          <Download size={16} /> {link.name}
                        </a>
                      ))}
                    </div>
                  </Panel>
                </div>
              </>
            )}
          </section>
        </section>
      </div>
    </main>
  );
}

function Label({ text, children }: { text: string; children: React.ReactNode }) {
  return (
    <label className="grid gap-1 text-sm">
      <span className="text-xs font-semibold uppercase tracking-wide text-slate-400">{text}</span>
      {children}
    </label>
  );
}

function Toggle({ label, value, onChange }: { label: string; value: boolean; onChange: (value: boolean) => void }) {
  return (
    <button type="button" onClick={() => onChange(!value)} className="flex items-center justify-between border border-line bg-slate-950/40 px-3 py-2 text-sm">
      <span>{label}</span>
      <span className={`relative h-5 w-9 rounded-full ${value ? "bg-accent" : "bg-slate-700"}`}>
        <span className={`absolute top-1 h-3 w-3 rounded-full bg-slate-950 transition ${value ? "left-5" : "left-1"}`} />
      </span>
    </button>
  );
}

function Panel({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="border border-line bg-panel p-4 shadow-glow">
      <h2 className="mb-4 text-sm font-semibold uppercase tracking-wide text-accent">{title}</h2>
      {children}
    </section>
  );
}

function DataTable({ rows }: { rows: Array<Record<string, string | number>> }) {
  if (!rows.length) {
    return <div className="text-sm text-slate-500">No rows.</div>;
  }
  const keys = Object.keys(rows[0]);
  return (
    <div className="max-h-96 overflow-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-line text-left text-xs uppercase tracking-wide text-slate-500">
            {keys.map((key) => (
              <th key={key} className="py-2 pr-3">{key.replaceAll("_", " ")}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-b border-line/70">
              {keys.map((key) => (
                <td key={key} className="py-2 pr-3 text-slate-300">{String(row[key])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
