"use client";

import dynamic from "next/dynamic";
import { BarChart3, ChevronDown, ChevronRight, Download, FileSpreadsheet, Loader2, Network, Play, SlidersHorizontal } from "lucide-react";
import { useMemo, useState } from "react";

const Plot = dynamic(() => import("react-plotly.js"), { ssr: false });

const API_BASE = (
  process.env.NEXT_PUBLIC_API_BASE_URL ??
  process.env.NEXT_PUBLIC_API_BASE ??
  "http://127.0.0.1:8010"
).replace(/\/$/, "");
const API_BASE_IS_LOCAL = API_BASE.includes("127.0.0.1") || API_BASE.includes("localhost");
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
  "Collecting Reddit posts",
  "Collecting Reddit comments",
  "Cleaning text",
  "Running LDA",
  "Running sentiment",
  "Generating figures",
  "Exporting results",
];
const TOPIC_SECTIONS = ["Summary", "Evidence", "Recommendations", "Examples", "Edit"] as const;
type TopicSection = (typeof TOPIC_SECTIONS)[number];

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
  evidence_source_ids?: string;
  notable_recommendations?: string;
  cautions_or_uncertainties?: string;
  official_practice_area?: string;
  comparison_guidance?: string;
  llm_summary_source?: string;
  llm_error?: string;
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
  collection_progress?: {
    phase: string;
    posts_collected: number;
    comments_collected: number;
    max_posts: number;
    comment_fetches: number;
    comments_target: number;
    eta_seconds: number | null;
    elapsed_seconds: number;
    collection_status: string;
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
    max_results: 100,
    include_comments: true,
    sample_mode: true,
  });
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [topics, setTopics] = useState<Topic[]>([]);
  const [running, setRunning] = useState(false);
  const [progressIndex, setProgressIndex] = useState(-1);
  const [progressStatus, setProgressStatus] = useState<AnalysisJob["status"] | "idle">("idle");
  const [collectionProgress, setCollectionProgress] = useState<AnalysisJob["collection_progress"] | null>(null);
  const [error, setError] = useState("");
  const [inputsOpen, setInputsOpen] = useState(true);
  const [selectedTopicIndex, setSelectedTopicIndex] = useState(0);
  const [activeTopicSection, setActiveTopicSection] = useState<TopicSection>("Summary");

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
    if (message === "Failed to fetch" || message.includes("fetch")) {
      return `Could not reach the backend at ${API_BASE}. Confirm Vercel has NEXT_PUBLIC_API_BASE_URL set to your Render URL, then redeploy Vercel. Also confirm the Render /health URL opens.`;
    }
    return message;
  }

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
    setCollectionProgress(null);
    try {
      const response = await fetch(`${API_BASE}/analysis-jobs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await parseJsonResponse<{ job_id: string; detail?: string }>(response);
      if (!response.ok) {
        throw new Error(data.detail ?? "Analysis failed");
      }
      let finished = false;
      while (!finished) {
        await new Promise((resolve) => window.setTimeout(resolve, 700));
        const statusResponse = await fetch(`${API_BASE}/analysis-jobs/${data.job_id}`);
        const job = await parseJsonResponse<AnalysisJob & { detail?: string }>(statusResponse);
        if (!statusResponse.ok) {
          throw new Error(job.detail ?? "Could not read analysis status");
        }
        setProgressStatus(job.status);
        setProgressIndex(job.step_index);
        setCollectionProgress(job.collection_progress ?? null);
        if (job.status === "completed") {
          if (!job.result) {
            throw new Error("Analysis completed without results");
          }
          setResult(job.result);
          setTopics(job.result.topics);
          setSelectedTopicIndex(0);
          setActiveTopicSection("Summary");
          setInputsOpen(false);
          finished = true;
        }
        if (job.status === "failed") {
          throw new Error(job.error ?? "Analysis failed");
        }
      }
    } catch (err) {
      setError(networkErrorMessage(err));
    } finally {
      setRunning(false);
    }
  }

  function updateTopic(index: number, key: "label" | "category", value: string) {
    setTopics((items) => items.map((topic, i) => (i === index ? { ...topic, [key]: value } : topic)));
  }

  const primaryExport = result?.export_links.find((link) => link.name === "final_topics_comparison.xlsx");
  const selectedTopic = topics[selectedTopicIndex] ?? topics[0];

  return (
    <main className="min-h-screen bg-ink text-slate-100">
      <div className="mx-auto flex w-full max-w-[1500px] flex-col gap-5 px-4 py-5 lg:px-6">
        <header className="flex flex-col gap-4 border-b border-line pb-5 md:flex-row md:items-end md:justify-between">
          <div>
            <div className="text-xs font-semibold uppercase tracking-[0.18em] text-accent">Reddit recovery advice mining</div>
            <h1 className="mt-2 text-2xl font-semibold tracking-normal text-white md:text-3xl">Post-Mohs Recovery Topic Analysis</h1>
            <p className="mt-2 max-w-4xl text-sm leading-6 text-slate-400">
              Collect Reddit discussions, isolate management and recovery recommendations, model common advice themes, summarize topics, and export a comparison workbook for official post-op guidance review.
            </p>
          </div>
          <div className="flex flex-wrap gap-2 text-xs text-slate-300">
            <span className="rounded border border-line bg-panel px-3 py-2">FastAPI</span>
            <span className="rounded border border-line bg-panel px-3 py-2">gensim LDA</span>
            <span className="rounded border border-line bg-panel px-3 py-2">LLM summaries</span>
          </div>
        </header>

        <section className={`grid gap-5 ${result ? "lg:grid-cols-[260px_1fr]" : "lg:grid-cols-[360px_1fr]"}`}>
          <aside className="h-fit border border-line bg-panel p-4 shadow-glow lg:sticky lg:top-5">
            <button
              type="button"
              onClick={() => setInputsOpen((value) => !value)}
              className="flex w-full items-center justify-between gap-3 text-left text-xs font-semibold uppercase tracking-[0.18em] text-accent"
            >
              <span className="flex items-center gap-2"><SlidersHorizontal size={16} /> Analysis Inputs</span>
              {result && (inputsOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />)}
            </button>
            {result && !inputsOpen && (
              <div className="mt-4 grid gap-3 text-xs text-slate-400">
                <div className="border border-line bg-slate-950/40 p-3">
                  <div className="text-slate-500">Subreddit</div>
                  <div className="mt-1 font-medium text-slate-100">{form.subreddit}</div>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <div className="border border-line bg-slate-950/40 p-3">
                    <div className="text-slate-500">Topics</div>
                    <div className="mt-1 font-medium text-slate-100">{form.k}</div>
                  </div>
                  <div className="border border-line bg-slate-950/40 p-3">
                    <div className="text-slate-500">Max posts</div>
                    <div className="mt-1 font-medium text-slate-100">{form.max_results}</div>
                  </div>
                </div>
                <button
                  type="button"
                  onClick={() => setInputsOpen(true)}
                  className="border border-line bg-slate-950/40 px-3 py-2 text-left text-slate-200 hover:border-accent"
                >
                  Edit analysis inputs
                </button>
              </div>
            )}
            {(!result || inputsOpen) && <div className="mt-4 grid gap-4">
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
                  <input type="number" min={10} max={5000} step={10} className="field" value={form.max_results} onChange={(e) => setForm({ ...form, max_results: Number(e.target.value) })} />
                </Label>
              </div>
              <div className="text-xs leading-5 text-slate-500">
                Larger runs are allowed. The backend slows Reddit requests automatically; runs above 100 posts can take several minutes, especially with comments enabled.
              </div>
              <Toggle label="Include comments" value={form.include_comments} onChange={(value) => setForm({ ...form, include_comments: value })} />
              <Toggle label="Sample data mode" value={form.sample_mode} onChange={(value) => setForm({ ...form, sample_mode: value })} />
              <div className={`break-all border p-2 text-[11px] leading-4 ${API_BASE_IS_LOCAL ? "border-amber/40 bg-amber/10 text-amber" : "border-line bg-slate-950/40 text-slate-500"}`}>
                Backend: {API_BASE}
              </div>
              <button
                onClick={runAnalysis}
                disabled={running}
                className="flex h-11 items-center justify-center gap-2 rounded bg-accent px-4 text-sm font-semibold text-slate-950 transition hover:bg-teal-200 disabled:cursor-not-allowed disabled:opacity-60"
              >
                {running ? <Loader2 className="animate-spin" size={18} /> : <Play size={18} />}
                Run Analysis
              </button>
            </div>}

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
                {collectionProgress && (progressIndex === 0 || progressIndex === 1) && (
                  <div className="mt-3 border border-line bg-slate-950/40 p-3 text-xs text-slate-300">
                    <div className="font-semibold text-slate-100">
                      {progressIndex === 0
                        ? `${collectionProgress.posts_collected}/${collectionProgress.max_posts} posts collected`
                        : `${collectionProgress.comment_fetches}/${collectionProgress.comments_target || collectionProgress.posts_collected} post comment trees fetched`}
                    </div>
                    <div className="mt-1">
                      {collectionProgress.posts_collected}/{collectionProgress.max_posts} posts collected; {collectionProgress.comments_collected} comments collected
                    </div>
                    <div className="mt-1 text-slate-400">
                      {collectionProgress.eta_seconds !== null
                        ? `Estimated time remaining: ${formatDuration(collectionProgress.eta_seconds)}`
                        : "Estimated time remaining: calculating..."}
                    </div>
                    <div className="mt-1 text-slate-500">{collectionProgress.collection_status}</div>
                  </div>
                )}
              </div>
            )}
            {error && <div className="mt-4 border border-red-500/40 bg-red-950/40 p-3 text-sm text-red-200">{error}</div>}
          </aside>

          <section className="grid gap-5">
            {!result && (
              <div className="flex min-h-[520px] items-center justify-center border border-line bg-panel p-8 text-center text-slate-400">
                <div>
                  <BarChart3 className="mx-auto mb-4 text-slate-600" size={42} />
                  <div className="text-base font-medium text-slate-200">No analysis loaded</div>
                  <div className="mt-2 max-w-xl text-sm leading-6">
                    Configure the run, then generate focused recovery-advice topics, representative examples, charts, and export files.
                  </div>
                </div>
              </div>
            )}

            {result && (
              <>
                <Panel title="Corpus Statistics">
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-4 2xl:grid-cols-6">
                    {Object.entries(result.corpus_stats).map(([key, value]) => (
                      <div key={key} className="border border-line bg-slate-950/30 p-3">
                        <div className="text-xs uppercase tracking-wide text-slate-500">{key.replaceAll("_", " ")}</div>
                        <div className="mt-1 text-lg font-semibold text-white">{String(value)}</div>
                      </div>
                    ))}
                  </div>
                </Panel>

                <Panel title="Recovery Advice Topic Map">
                  <div className="grid gap-4 xl:grid-cols-[minmax(360px,0.95fr)_minmax(420px,1.05fr)]">
                    <TopicGraph
                      topics={topics}
                      selectedIndex={selectedTopicIndex}
                      onSelect={(index) => {
                        setSelectedTopicIndex(index);
                        setActiveTopicSection("Summary");
                      }}
                    />
                    {selectedTopic && (
                      <TopicDetail
                        topic={selectedTopic}
                        index={selectedTopicIndex}
                        activeSection={activeTopicSection}
                        onSectionChange={setActiveTopicSection}
                        onUpdate={updateTopic}
                      />
                    )}
                  </div>
                </Panel>

                {primaryExport && (
                  <a href={`${API_BASE}${primaryExport.url}`} className="flex items-center justify-between gap-3 border border-accent/40 bg-teal-950/20 px-4 py-3 text-sm text-slate-100 hover:border-accent">
                    <span className="flex items-center gap-2">
                      <FileSpreadsheet size={18} className="text-accent" />
                      Download final topics comparison workbook
                    </span>
                    <span className="text-xs text-slate-400">{primaryExport.name}</span>
                  </a>
                )}

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

function formatDuration(seconds: number) {
  if (!Number.isFinite(seconds) || seconds < 0) {
    return "calculating...";
  }
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;
  if (minutes <= 0) {
    return `${remainingSeconds}s`;
  }
  return `${minutes}m ${remainingSeconds}s`;
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

function TopicGraph({ topics, selectedIndex, onSelect }: { topics: Topic[]; selectedIndex: number; onSelect: (index: number) => void }) {
  if (!topics.length) {
    return <div className="border border-line bg-slate-950/30 p-4 text-sm text-slate-500">No topics available.</div>;
  }

  return (
    <div className="relative min-h-[460px] overflow-hidden border border-line bg-slate-950/30 p-4">
      <div className="mb-3 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-slate-500">
        <Network size={15} /> Click a topic node
      </div>
      <div className="topic-graph">
        <div className="topic-core">
          <div className="text-xs uppercase tracking-wide text-slate-500">Corpus</div>
          <div className="mt-1 text-lg font-semibold text-white">{topics.reduce((sum, topic) => sum + topic.doc_count, 0)} docs</div>
        </div>
        {topics.map((topic, index) => {
          const angle = (index / Math.max(topics.length, 1)) * Math.PI * 2 - Math.PI / 2;
          const radius = topics.length <= 5 ? 36 : 39;
          const x = 50 + Math.cos(angle) * radius;
          const y = 50 + Math.sin(angle) * radius;
          const active = index === selectedIndex;
          return (
            <button
              type="button"
              key={topic.topic}
              onClick={() => onSelect(index)}
              className={`topic-node ${active ? "topic-node-active" : ""}`}
              style={{ left: `${x}%`, top: `${y}%` }}
            >
              <span className="text-[11px] uppercase tracking-wide">Topic {topic.topic + 1}</span>
              <span className="line-clamp-2 text-sm font-semibold">{topic.llm_topic_title || topic.label}</span>
              <span className="text-[11px] text-slate-400">{topic.percentage}% | {topic.doc_count} docs</span>
            </button>
          );
        })}
      </div>
      <div className="pointer-events-none absolute inset-x-8 bottom-6 h-px bg-gradient-to-r from-transparent via-line to-transparent" />
    </div>
  );
}

function TopicDetail({
  topic,
  index,
  activeSection,
  onSectionChange,
  onUpdate,
}: {
  topic: Topic;
  index: number;
  activeSection: TopicSection;
  onSectionChange: (section: TopicSection) => void;
  onUpdate: (index: number, key: "label" | "category", value: string) => void;
}) {
  const examples = topic.example_documents?.length
    ? topic.example_documents
    : [{ id: `${topic.topic}-rep`, type: "", date: "", score: 0, permalink: "", text: topic.representative_document }];

  return (
    <article className="border border-line bg-slate-950/25 p-4">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line pb-4">
        <div>
          <div className="text-xs font-semibold uppercase tracking-[0.16em] text-accent">Topic {topic.topic + 1}</div>
          <h3 className="mt-1 text-xl font-semibold text-white">{topic.llm_topic_title || topic.label}</h3>
        </div>
        <div className="flex flex-wrap gap-2 text-xs text-slate-400">
          <span className="border border-line bg-panel px-2 py-1">{topic.doc_count} docs</span>
          <span className="border border-line bg-panel px-2 py-1">{topic.percentage}%</span>
          <span className="border border-line bg-panel px-2 py-1">distinct {topic.distinctiveness}</span>
        </div>
      </div>

      <div className="mt-4 flex flex-wrap gap-2">
        {TOPIC_SECTIONS.map((section) => (
          <button
            type="button"
            key={section}
            onClick={() => onSectionChange(section)}
            className={`flex items-center gap-1 border px-3 py-2 text-xs font-medium uppercase tracking-wide transition ${
              activeSection === section ? "border-accent bg-teal-950/30 text-accent" : "border-line bg-panel text-slate-400 hover:border-slate-500 hover:text-slate-100"
            }`}
          >
            {activeSection === section ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            {section}
          </button>
        ))}
      </div>

      <div className="mt-4 min-h-[300px] border border-line bg-ink/40 p-4">
        {activeSection === "Summary" && (
          <div>
            <div className="mb-3 flex flex-wrap gap-2">
              {topic.keywords.map((keyword) => (
                <span key={keyword} className="rounded border border-line bg-panel px-2 py-1 text-xs text-slate-300">
                  {keyword.replaceAll("_", " ")}
                </span>
              ))}
            </div>
            <p className="text-sm leading-6 text-slate-300">{topic.llm_summary}</p>
            {topic.llm_explanation && <p className="mt-3 text-sm leading-6 text-slate-400">{topic.llm_explanation}</p>}
            {topic.official_practice_area && (
              <div className="mt-4 border border-amber/30 bg-amber/10 p-3 text-xs leading-5 text-amber">
                {topic.official_practice_area}
              </div>
            )}
            {topic.llm_summary_source && <div className="mt-3 text-[11px] uppercase tracking-wide text-slate-600">Summary source: {topic.llm_summary_source}</div>}
            {topic.llm_error && <div className="mt-2 text-xs leading-5 text-red-300">LLM fallback reason: {topic.llm_error}</div>}
          </div>
        )}

        {activeSection === "Evidence" && (
          <div className="grid gap-3 text-sm leading-6 text-slate-300">
            <div className="border border-line bg-panel p-3">
              <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Evidence source IDs</div>
              {topic.evidence_source_ids ? formatJsonList(topic.evidence_source_ids) : "No evidence IDs returned."}
            </div>
            <div className="border border-line bg-panel p-3">
              <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Cautions and uncertainty</div>
              {topic.cautions_or_uncertainties ? formatJsonList(topic.cautions_or_uncertainties) : "No cautions returned."}
            </div>
          </div>
        )}

        {activeSection === "Recommendations" && (
          <div className="grid gap-3 text-sm leading-6 text-slate-300">
            <div className="border border-line bg-panel p-3">
              <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Notable recommendations</div>
              {topic.notable_recommendations ? formatJsonList(topic.notable_recommendations) : "No recommendations returned."}
            </div>
            {topic.comparison_guidance && (
              <div className="border border-line bg-panel p-3">
                <div className="mb-1 text-xs uppercase tracking-wide text-slate-500">Comparison guidance</div>
                {topic.comparison_guidance}
              </div>
            )}
          </div>
        )}

        {activeSection === "Examples" && (
          <div className="grid max-h-[430px] gap-3 overflow-auto pr-1">
            {examples.map((example) => (
              <div key={example.id} className="border border-line/70 bg-slate-950/40 p-3">
                <div className="mb-2 flex flex-wrap gap-2 text-[11px] uppercase tracking-wide text-slate-500">
                  {example.type && <span>{example.type}</span>}
                  {example.date && <span>{example.date}</span>}
                  {Number.isFinite(example.score) && <span>score {example.score}</span>}
                  {example.permalink && (
                    <a href={example.permalink} target="_blank" rel="noreferrer" className="text-accent hover:text-teal-200">
                      source
                    </a>
                  )}
                </div>
                <div className="text-sm leading-6 text-slate-400">{example.text}</div>
              </div>
            ))}
          </div>
        )}

        {activeSection === "Edit" && (
          <div className="grid gap-3 md:grid-cols-2">
            <label className="grid gap-1 text-xs uppercase tracking-wide text-slate-500">
              Editable label
              <input className="table-field w-full" value={topic.label} onChange={(e) => onUpdate(index, "label", e.target.value)} />
            </label>
            <label className="grid gap-1 text-xs uppercase tracking-wide text-slate-500">
              Category
              <select className="table-field w-full" value={topic.category} onChange={(e) => onUpdate(index, "category", e.target.value)}>
                {CATEGORIES.map((category) => (
                  <option key={category}>{category}</option>
                ))}
              </select>
            </label>
          </div>
        )}
      </div>
    </article>
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

function formatJsonList(value: string) {
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) {
      return parsed.join(", ");
    }
  } catch {
    if (value.startsWith("[") && value.endsWith("]")) {
      const items = value
        .slice(1, -1)
        .split(/',\s*'/)
        .map((item) => item.replace(/^['"]|['"]$/g, "").trim())
        .filter(Boolean);
      if (items.length > 1) {
        return items.join(", ");
      }
    }
    return value;
  }
  return value;
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
