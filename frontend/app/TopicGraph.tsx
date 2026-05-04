"use client";

import * as d3 from "d3";
import React, { useEffect, useRef, useState } from "react";

type Topic = {
  topic: number;
  keywords: string[];
  doc_count: number;
  percentage: number;
  label: string;
  llm_topic_title?: string;
  llm_summary?: string;
  llm_explanation?: string;
  notable_recommendations?: string;
  cautions_or_uncertainties?: string;
  official_practice_area?: string;
  representative_document?: string;
  example_documents?: Array<{ id: string; type: string; date: string; score: number; permalink: string; text: string }>;
};

const TOPIC_COLORS = [
  "#5eead4", "#818cf8", "#fb923c", "#f472b6",
  "#a3e635", "#38bdf8", "#e879f9", "#34d399",
];

function linkifyCitations(
  text: string,
  docs: Array<{ id: string; type: string; date: string; score: number; permalink: string; text: string }>,
  topicNumber: number,
): React.ReactNode[] {
  const lookup: Record<string, string> = {};
  docs.forEach((doc, i) => {
    lookup[`T${topicNumber}-E${i + 1}`] = doc.permalink;
  });
  const parts = text.split(/(T\d+-E\d+)/g);
  return parts.map((part, i) => {
    const permalink = lookup[part];
    if (permalink) {
      return (
        <a
          key={i}
          href={permalink.startsWith("http") ? permalink : `https://reddit.com${permalink}`}
          target="_blank"
          rel="noopener noreferrer"
          className="font-medium underline decoration-dotted underline-offset-2 hover:decoration-solid"
          style={{ color: "inherit" }}
        >
          {part}
        </a>
      );
    }
    return part;
  });
}

function formatJsonList(value: string): string {
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) return parsed.filter(Boolean).join(" · ");
  } catch { /* noop */ }
  return value;
}

// Wrap SVG text into multiple tspan lines
function wrapText(
  textEl: d3.Selection<SVGTextElement, unknown, null, undefined>,
  words: string[],
  lineHeight: number,
  maxWidth: number,
) {
  textEl.text(null);
  let line: string[] = [];
  let lineNumber = 0;
  const x = textEl.attr("x") ?? "0";
  const dy = parseFloat(textEl.attr("dy") ?? "0");
  let tspan = textEl.append("tspan").attr("x", x).attr("dy", `${dy}px`);

  for (const word of words) {
    line.push(word);
    tspan.text(line.join(" "));
    const node = tspan.node();
    if (node && node.getComputedTextLength() > maxWidth && line.length > 1) {
      line.pop();
      tspan.text(line.join(" "));
      line = [word];
      lineNumber++;
      tspan = textEl.append("tspan").attr("x", x).attr("dy", `${lineHeight}px`).text(word);
    }
  }
  // Vertically centre the block of lines
  textEl.attr("dy", `${dy - (lineNumber * lineHeight) / 2}px`);
}

export default function TopicGraph({ topics, totalDocs }: { topics: Topic[]; totalDocs: number }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const [selected, setSelected] = useState<Topic | null>(null);
  const [dims, setDims] = useState<{ w: number; h: number } | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;
    const measure = () => {
      const { clientWidth: w, clientHeight: h } = containerRef.current!;
      if (w > 0 && h > 0) setDims({ w, h });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  useEffect(() => {
    if (!svgRef.current || !topics.length || !dims) return;

    const { w: W, h: H } = dims;
    const cx = W / 2;
    const cy = H / 2;
    const corpusR = 38;
    const topicR = topics.map((t) => Math.max(28, Math.min(54, 18 + (t.doc_count / Math.max(totalDocs, 1)) * 160)));

    const svg = d3.select(svgRef.current);
    svg.selectAll("*").remove();
    svg.attr("width", W).attr("height", H);

    // ── Defs ──────────────────────────────────────────────────────────────────
    const defs = svg.append("defs");

    const grad = defs.append("radialGradient").attr("id", "bg-grad").attr("cx", "50%").attr("cy", "50%").attr("r", "50%");
    grad.append("stop").attr("offset", "0%").attr("stop-color", "#0f1620").attr("stop-opacity", 1);
    grad.append("stop").attr("offset", "100%").attr("stop-color", "#090b10").attr("stop-opacity", 1);

    topics.forEach((_, i) => {
      const color = TOPIC_COLORS[i % TOPIC_COLORS.length];
      const f = defs.append("filter").attr("id", `glow-${i}`).attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
      f.append("feGaussianBlur").attr("in", "SourceGraphic").attr("stdDeviation", "6").attr("result", "blur");
      const merge = f.append("feMerge");
      merge.append("feMergeNode").attr("in", "blur");
      merge.append("feMergeNode").attr("in", "SourceGraphic");
      const fa = defs.append("filter").attr("id", `aura-${i}`).attr("x", "-100%").attr("y", "-100%").attr("width", "300%").attr("height", "300%");
      fa.append("feGaussianBlur").attr("in", "SourceGraphic").attr("stdDeviation", "14").attr("result", "blur");
      fa.append("feFlood").attr("flood-color", color).attr("flood-opacity", "0.15").attr("result", "color");
      fa.append("feComposite").attr("in", "color").attr("in2", "blur").attr("operator", "in");
    });

    const corpusF = defs.append("filter").attr("id", "glow-corpus").attr("x", "-50%").attr("y", "-50%").attr("width", "200%").attr("height", "200%");
    corpusF.append("feGaussianBlur").attr("in", "SourceGraphic").attr("stdDeviation", "8").attr("result", "blur");
    const cm = corpusF.append("feMerge");
    cm.append("feMergeNode").attr("in", "blur");
    cm.append("feMergeNode").attr("in", "SourceGraphic");

    // ── Background ────────────────────────────────────────────────────────────
    svg.append("rect").attr("width", W).attr("height", H).attr("fill", "url(#bg-grad)");

    const gridG = svg.append("g");
    [0.18, 0.32, 0.46, 0.58].forEach((r) => {
      gridG.append("circle")
        .attr("cx", cx).attr("cy", cy)
        .attr("r", Math.min(W, H) * r)
        .attr("fill", "none")
        .attr("stroke", "rgba(255,255,255,0.035)")
        .attr("stroke-dasharray", "3 7");
    });

    // ── Simulation nodes ──────────────────────────────────────────────────────
    type SimNode = d3.SimulationNodeDatum & {
      id: string;
      kind: "corpus" | "topic" | "keyword";
      ti?: number;
      r: number;
      color: string;
      label: string;
      pct?: string;
    };

    const orbitR = Math.min(W, H) * 0.29;
    const nodes: SimNode[] = [
      { id: "corpus", kind: "corpus", r: corpusR, color: "rgba(255,255,255,0.25)", label: String(totalDocs), x: cx, y: cy, fx: cx, fy: cy },
      ...topics.map((t, i) => {
        const angle = (i / topics.length) * Math.PI * 2 - Math.PI / 2;
        return {
          id: `topic-${i}`,
          kind: "topic" as const,
          ti: i,
          r: topicR[i],
          color: TOPIC_COLORS[i % TOPIC_COLORS.length],
          label: t.llm_topic_title || t.label,
          pct: `${t.percentage}%`,
          x: cx + Math.cos(angle) * orbitR,
          y: cy + Math.sin(angle) * orbitR,
        };
      }),
      ...topics.flatMap((t, ti) => {
        const baseAngle = (ti / topics.length) * Math.PI * 2 - Math.PI / 2;
        const kwOrbitR = Math.min(W, H) * 0.44;
        return t.keywords.slice(0, 3).map((kw, ki) => {
          const angle = baseAngle + (ki - 1) * 0.28;
          return {
            id: `kw-${ti}-${ki}`,
            kind: "keyword" as const,
            ti,
            r: 4,
            color: TOPIC_COLORS[ti % TOPIC_COLORS.length],
            label: kw.replaceAll("_", " "),
            x: cx + Math.cos(angle) * kwOrbitR,
            y: cy + Math.sin(angle) * kwOrbitR,
          };
        });
      }),
    ];

    type SimLink = d3.SimulationLinkDatum<SimNode> & { kind: "corpus-topic" | "topic-kw" };
    const links: SimLink[] = [
      ...topics.map((_, i) => ({ source: "corpus", target: `topic-${i}`, kind: "corpus-topic" as const })),
      ...topics.flatMap((t, ti) =>
        t.keywords.slice(0, 3).map((_, ki) => ({ source: `topic-${ti}`, target: `kw-${ti}-${ki}`, kind: "topic-kw" as const }))
      ),
    ];

    // ── Link elements (drawn before nodes) ───────────────────────────────────
    const linkG = svg.append("g");
    const linkEls = linkG.selectAll("line")
      .data(links)
      .join("line")
      .attr("stroke", (d) => {
        const target = nodes.find((n) => n.id === (typeof d.target === "string" ? d.target : (d.target as SimNode).id));
        return target?.color ?? "#fff";
      })
      .attr("stroke-opacity", (d) => d.kind === "corpus-topic" ? 0.18 : 0.1)
      .attr("stroke-width", (d) => d.kind === "corpus-topic" ? 1.5 : 0.8);

    // ── Keyword node elements ─────────────────────────────────────────────────
    const kwG = svg.append("g");
    const kwNodes = nodes.filter((n) => n.kind === "keyword");
    const kwEls = kwG.selectAll("g")
      .data(kwNodes)
      .join("g");
    kwEls.append("circle")
      .attr("r", 4)
      .attr("fill", (d) => d.color + "30")
      .attr("stroke", (d) => d.color)
      .attr("stroke-width", 0.8)
      .attr("stroke-opacity", 0.5);
    kwEls.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", -9)
      .attr("fill", (d) => d.color)
      .attr("font-size", "9px")
      .attr("fill-opacity", 0.55)
      .attr("pointer-events", "none")
      .text((d) => d.label.length > 18 ? d.label.slice(0, 16) + "…" : d.label);

    // ── Topic node elements ───────────────────────────────────────────────────
    const topicG = svg.append("g");
    const topicNodes = nodes.filter((n) => n.kind === "topic");
    const topicEls = topicG.selectAll<SVGGElement, SimNode>("g")
      .data(topicNodes)
      .join("g")
      .attr("cursor", "pointer")
      .on("click", (_, d) => {
        const t = topics[d.ti!];
        setSelected((s) => s?.topic === t.topic ? null : t);
      });

    topicEls.append("circle") // aura
      .attr("r", (d) => d.r + 20)
      .attr("fill", (d) => d.color)
      .attr("fill-opacity", 0.06)
      .attr("filter", (d) => `url(#aura-${d.ti})`);

    topicEls.append("circle") // outer ring
      .attr("r", (d) => d.r + 8)
      .attr("fill", "none")
      .attr("stroke", (d) => d.color)
      .attr("stroke-width", 0.5)
      .attr("stroke-opacity", (d) => selected?.topic === d.ti ? 0.6 : 0.18)
      .attr("stroke-dasharray", (d) => selected?.topic === d.ti ? "none" : "4 4");

    topicEls.append("circle") // main
      .attr("r", (d) => d.r)
      .attr("fill", (d) => d.color + "18")
      .attr("stroke", (d) => d.color)
      .attr("stroke-width", (d) => selected?.topic === d.ti ? 2 : 1.2)
      .attr("filter", (d) => `url(#glow-${d.ti})`);

    for (let k = 0; k < 3; k++) {
      const a = (k / 3) * Math.PI * 2;
      topicEls.append("line")
        .attr("x1", 0).attr("y1", 0)
        .attr("x2", (d) => Math.cos(a) * d.r * 0.7)
        .attr("y2", (d) => Math.sin(a) * d.r * 0.7)
        .attr("stroke", (d) => d.color)
        .attr("stroke-opacity", 0.12)
        .attr("stroke-width", 0.8);
    }

    // Percentage badge
    topicEls.append("text")
      .attr("text-anchor", "middle")
      .attr("dy", 4)
      .attr("fill", (d) => d.color)
      .attr("font-size", "13px")
      .attr("font-weight", "700")
      .attr("pointer-events", "none")
      .text((d) => d.pct ?? "");

    // Label below — wrapped
    topicEls.each(function (d) {
      const labelEl = d3.select(this).append("text")
        .attr("text-anchor", "middle")
        .attr("x", "0")
        .attr("dy", `${d.r + 16}px`)
        .attr("fill", d.color)
        .attr("font-size", "11px")
        .attr("font-weight", "500")
        .attr("pointer-events", "none");
      wrapText(labelEl, d.label.split(" "), 14, 110);
    });

    // ── Corpus center node ────────────────────────────────────────────────────
    const corpusG = svg.append("g").attr("transform", `translate(${cx},${cy})`);
    corpusG.append("circle").attr("r", corpusR + 14).attr("fill", "rgba(255,255,255,0.03)").attr("stroke", "rgba(255,255,255,0.06)").attr("stroke-width", 1);
    corpusG.append("circle").attr("r", corpusR)
      .attr("fill", "rgba(255,255,255,0.05)")
      .attr("stroke", "rgba(255,255,255,0.25)")
      .attr("stroke-width", 1.5)
      .attr("filter", "url(#glow-corpus)");
    corpusG.append("text").attr("text-anchor", "middle").attr("dy", -5)
      .attr("fill", "rgba(255,255,255,0.8)").attr("font-size", "14px").attr("font-weight", "700")
      .text(totalDocs);
    corpusG.append("text").attr("text-anchor", "middle").attr("dy", 12)
      .attr("fill", "rgba(255,255,255,0.35)").attr("font-size", "9px")
      .text("docs");

    // ── Force simulation ──────────────────────────────────────────────────────
    const nodeById = new Map(nodes.map((n) => [n.id, n]));
    const resolvedLinks = links.map((l) => ({
      ...l,
      source: nodeById.get(typeof l.source === "string" ? l.source : (l.source as SimNode).id)!,
      target: nodeById.get(typeof l.target === "string" ? l.target : (l.target as SimNode).id)!,
    }));

    const sim = d3.forceSimulation(nodes)
      .force("link", d3.forceLink(resolvedLinks).id((d) => (d as SimNode).id).distance((l) => {
        const s = l.source as SimNode;
        const t = l.target as SimNode;
        return s.kind === "corpus" || t.kind === "corpus" ? Math.min(W, H) * 0.28 : 80;
      }).strength(0.6))
      .force("charge", d3.forceManyBody().strength((d) => {
        const n = d as SimNode;
        return n.kind === "corpus" ? -600 : n.kind === "topic" ? -300 : -40;
      }))
      .force("collide", d3.forceCollide((d) => (d as SimNode).r + 30).strength(0.8))
      .force("center", d3.forceCenter(cx, cy).strength(0.05))
      .alphaDecay(0.025);

    const pad = 60;
    sim.on("tick", () => {
      // clamp nodes inside viewport
      nodes.forEach((n) => {
        if (n.fx !== undefined && n.fx !== null) return;
        n.x = Math.max(pad + n.r, Math.min(W - pad - n.r, n.x ?? cx));
        n.y = Math.max(pad + n.r, Math.min(H - pad - n.r, n.y ?? cy));
      });

      linkEls
        .attr("x1", (d) => (d.source as SimNode).x ?? 0)
        .attr("y1", (d) => (d.source as SimNode).y ?? 0)
        .attr("x2", (d) => (d.target as SimNode).x ?? 0)
        .attr("y2", (d) => (d.target as SimNode).y ?? 0);

      (kwEls as d3.Selection<SVGGElement, SimNode, SVGGElement, unknown>)
        .attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);

      (topicEls as d3.Selection<SVGGElement, SimNode, SVGGElement, unknown>)
        .attr("transform", (d) => `translate(${d.x ?? 0},${d.y ?? 0})`);
    });

    // ── Drag ─────────────────────────────────────────────────────────────────
    const drag = d3.drag<SVGGElement, SimNode>()
      .on("start", (event, d) => {
        if (!event.active) sim.alphaTarget(0.3).restart();
        d.fx = d.x;
        d.fy = d.y;
      })
      .on("drag", (event, d) => {
        d.fx = Math.max(pad + d.r, Math.min(W - pad - d.r, event.x));
        d.fy = Math.max(pad + d.r, Math.min(H - pad - d.r, event.y));
      })
      .on("end", (event, d) => {
        if (!event.active) sim.alphaTarget(0);
        d.fx = null;
        d.fy = null;
      });

    (topicEls as d3.Selection<SVGGElement, SimNode, SVGGElement, unknown>).call(drag);
    (kwEls as d3.Selection<SVGGElement, SimNode, SVGGElement, unknown>).call(drag);

    return () => { sim.stop(); };
  }, [topics, totalDocs, dims, selected]);

  const recommendations = selected?.notable_recommendations ? formatJsonList(selected.notable_recommendations) : "";
  const cautions = selected?.cautions_or_uncertainties ? formatJsonList(selected.cautions_or_uncertainties) : "";

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden">
      <svg ref={svgRef} className="absolute inset-0" />

      {/* Detail panel */}
      {selected && (
        <div className="absolute bottom-0 left-0 right-0 max-h-[55%] overflow-y-auto border-t border-white/[0.08] bg-slate-950/95 px-8 py-6 backdrop-blur-md">
          <div className="mx-auto max-w-3xl">
            <div className="flex items-start justify-between gap-4">
              <div>
                <div className="text-[11px] font-medium" style={{ color: TOPIC_COLORS[selected.topic % TOPIC_COLORS.length] }}>
                  Topic {selected.topic + 1}
                </div>
                <h3 className="mt-0.5 text-lg font-semibold text-white">
                  {selected.llm_topic_title || selected.label}
                </h3>
              </div>
              <button onClick={() => setSelected(null)} className="mt-1 shrink-0 text-slate-600 hover:text-slate-300 text-lg leading-none">✕</button>
            </div>

            <div className="mt-3 flex flex-wrap gap-1.5">
              {selected.keywords.map((kw) => (
                <span key={kw} className="rounded bg-slate-800 px-2 py-0.5 text-[11px] text-slate-400">
                  {kw.replaceAll("_", " ")}
                </span>
              ))}
            </div>

            {selected.llm_explanation && (
              <p className="mt-4 text-sm leading-6 text-slate-400">
                {linkifyCitations(selected.llm_explanation, selected.example_documents ?? [], selected.topic + 1)}
              </p>
            )}
            <div className="mt-4 grid gap-3 sm:grid-cols-2">
              {recommendations && (
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wide text-slate-600">Recommendations</div>
                  <div className="mt-1 text-sm text-slate-300">{recommendations}</div>
                </div>
              )}
              {cautions && (
                <div>
                  <div className="text-[11px] font-medium uppercase tracking-wide text-slate-600">Cautions</div>
                  <div className="mt-1 text-sm text-slate-300">{cautions}</div>
                </div>
              )}
            </div>
            {selected.official_practice_area && (
              <div className="mt-4 rounded-lg bg-amber/10 px-3 py-2.5 text-xs leading-5 text-amber">
                {selected.official_practice_area}
              </div>
            )}

            {selected.example_documents && selected.example_documents.length > 0 && (
              <div className="mt-5">
                <div className="text-[11px] font-medium uppercase tracking-wide text-slate-600">Sources</div>
                <div className="mt-2 grid gap-2">
                  {selected.example_documents.slice(0, 3).map((doc) => (
                    <a
                      key={doc.id}
                      href={doc.permalink.startsWith("http") ? doc.permalink : `https://reddit.com${doc.permalink}`}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="group rounded-lg border border-white/[0.06] bg-slate-900/60 px-3 py-2.5 transition hover:border-white/[0.14]"
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] uppercase tracking-wide text-slate-600">{doc.type}</span>
                        <span className="text-[10px] text-slate-700">·</span>
                        <span className="text-[10px] text-slate-600">{doc.date}</span>
                        <span className="ml-auto text-[10px] text-slate-700 group-hover:text-slate-500">↗</span>
                      </div>
                      <p className="mt-1 line-clamp-2 text-xs leading-5 text-slate-400 group-hover:text-slate-300">
                        {doc.text}
                      </p>
                    </a>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>
      )}

      {!selected && (
        <div className="pointer-events-none absolute bottom-6 left-0 right-0 text-center text-xs text-slate-700">
          Click a topic node to read its interpretation · Drag to rearrange
        </div>
      )}
    </div>
  );
}
