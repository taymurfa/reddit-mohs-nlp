"use client";

import React, { useMemo, useState } from "react";
import { Search, X, ExternalLink, ArrowUpDown } from "lucide-react";

type RawDocument = {
  id: string;
  type: string;
  date: string;
  author: string;
  score: number;
  permalink: string;
  text: string;
  text: string;
  topic: number | null;
  thread_id: string | null;
};

type Topic = {
  topic: number;
  label: string;
  llm_topic_title?: string;
};

export default function DocumentsView({
  documents,
  topics,
  onClose,
}: {
  documents: RawDocument[];
  topics: Topic[];
  onClose: () => void;
}) {
  const [search, setSearch] = useState("");
  const [sortBy, setSortBy] = useState<"date" | "score" | "topic">("date");
  const [sortDesc, setSortDesc] = useState(true);
  const [visibleCount, setVisibleCount] = useState(50);
  const [expandedPosts, setExpandedPosts] = useState<Set<string>>(new Set());

  const toggleExpand = (postId: string) => {
    setExpandedPosts(prev => {
      const next = new Set(prev);
      if (next.has(postId)) {
        next.delete(postId);
      } else {
        next.add(postId);
      }
      return next;
    });
  };

  const filteredAndSortedPosts = useMemo(() => {
    let result = documents.filter(d => d.type === "post");
    
    if (search.trim()) {
      const q = search.toLowerCase();
      result = result.filter(
        (d) => d.text.toLowerCase().includes(q) || d.author.toLowerCase().includes(q)
      );
    }

    result = [...result].sort((a, b) => {
      let cmp = 0;
      if (sortBy === "date") {
        cmp = a.date.localeCompare(b.date);
      } else if (sortBy === "score") {
        cmp = a.score - b.score;
      } else if (sortBy === "topic") {
        const tA = a.topic ?? 999;
        const tB = b.topic ?? 999;
        cmp = tA - tB;
      }
      return sortDesc ? -cmp : cmp;
    });

    return result;
    return result;
  }, [documents, search, sortBy, sortDesc]);

  const visibleDocs = filteredAndSortedPosts.slice(0, visibleCount);

  // Group comments by thread_id
  const commentsByPostId = useMemo(() => {
    const map = new Map<string, RawDocument[]>();
    for (const doc of documents) {
      if (doc.type === "comment" && doc.thread_id) {
        const arr = map.get(doc.thread_id) || [];
        arr.push(doc);
        map.set(doc.thread_id, arr);
      }
    }
    // Sort each array by score desc
    for (const [_, arr] of map.entries()) {
      arr.sort((a, b) => b.score - a.score);
    }
    return map;
  }, [documents]);

  return (
    <div className="absolute inset-0 flex flex-col bg-ink">
      {/* Header & Controls */}
      <div className="flex flex-col gap-4 border-b border-white/[0.06] bg-slate-900/50 p-6">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-white">Source Documents</h2>
            <p className="text-xs text-slate-400 mt-1">
              {filteredAndSortedPosts.length} {filteredAndSortedPosts.length === 1 ? "post" : "posts"} found
            </p>
          </div>
          <button
            onClick={onClose}
            className="flex h-8 w-8 items-center justify-center rounded-full bg-white/[0.04] text-slate-400 hover:bg-white/[0.08] hover:text-white transition"
          >
            <X size={16} />
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-4">
          {/* Search */}
          <div className="relative flex-1 min-w-[200px]">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-slate-500" />
            <input
              type="text"
              placeholder="Search text or author..."
              value={search}
              onChange={(e) => {
                setSearch(e.target.value);
                setVisibleCount(50);
              }}
              className="h-9 w-full rounded-lg border border-white/[0.1] bg-slate-950/50 pl-9 pr-4 text-sm text-slate-200 outline-none focus:border-accent/50 focus:bg-slate-900"
            />
          </div>

          {/* Sort Dropdown */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-500">Sort by:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="h-9 rounded-lg border border-white/[0.1] bg-slate-950/50 px-3 text-sm text-slate-200 outline-none focus:border-accent/50"
            >
              <option value="date">Date</option>
              <option value="score">Score</option>
              <option value="topic">Topic</option>
            </select>
            <button
              onClick={() => setSortDesc(!sortDesc)}
              className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/[0.1] bg-slate-950/50 text-slate-400 hover:text-slate-200"
              title={sortDesc ? "Descending" : "Ascending"}
            >
              <ArrowUpDown size={14} />
            </button>
          </div>
        </div>
      </div>

      {/* List */}
      <div className="flex-1 overflow-y-auto p-6 scroll-smooth">
        <div className="mx-auto max-w-4xl space-y-4">
          {visibleDocs.map((doc) => {
            const topicLabel =
              doc.topic !== null
                ? topics[doc.topic]?.llm_topic_title || topics[doc.topic]?.label || `Topic ${doc.topic + 1}`
                : "No Topic";
            
            const comments = commentsByPostId.get(doc.id) || [];
            const isExpanded = expandedPosts.has(doc.id);

            return (
              <div key={doc.id} className="space-y-3">
                <div className="rounded-xl border border-white/[0.04] bg-slate-900/30 p-5 transition hover:border-white/[0.1]">
                  <div className="mb-3 flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-white/[0.04] pb-3 text-xs">
                    <span className="font-medium text-blue-400">
                      {doc.type.toUpperCase()}
                    </span>
                    <span className="text-slate-500">{doc.date}</span>
                    <span className="text-slate-400">Score: {doc.score}</span>
                    <span className="text-slate-400">u/{doc.author}</span>
                    
                    <div className="ml-auto flex items-center gap-3">
                      <span className={`rounded px-2 py-0.5 font-medium ${doc.topic !== null ? "bg-teal-950/50 text-teal-400" : "bg-slate-800 text-slate-500"}`}>
                        {topicLabel}
                      </span>
                      <a
                        href={doc.permalink.startsWith("http") ? doc.permalink : `https://reddit.com${doc.permalink}`}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="flex items-center gap-1 text-slate-500 hover:text-accent transition"
                      >
                        Reddit <ExternalLink size={12} />
                      </a>
                    </div>
                  </div>
                  <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-300">
                    {doc.text}
                  </p>
                  
                  {comments.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-white/[0.04]">
                      <button
                        onClick={() => toggleExpand(doc.id)}
                        className="text-xs font-medium text-slate-400 hover:text-white transition"
                      >
                        {isExpanded ? "Hide Comments" : `View Comments (${comments.length})`}
                      </button>
                    </div>
                  )}
                </div>

                {isExpanded && comments.length > 0 && (
                  <div className="ml-8 space-y-3 border-l-2 border-white/[0.04] pl-4">
                    {comments.map(comment => {
                      const commentTopicLabel = comment.topic !== null
                        ? topics[comment.topic]?.llm_topic_title || topics[comment.topic]?.label || `Topic ${comment.topic + 1}`
                        : "No Topic";
                      
                      return (
                        <div key={comment.id} className="rounded-xl border border-white/[0.02] bg-slate-900/10 p-4 transition hover:bg-slate-900/20">
                          <div className="mb-2 flex flex-wrap items-center gap-x-4 gap-y-2 text-xs">
                            <span className="font-medium text-amber-500/70">
                              COMMENT
                            </span>
                            <span className="text-slate-500">{comment.date}</span>
                            <span className="text-slate-400">Score: {comment.score}</span>
                            <span className="text-slate-400">u/{comment.author}</span>
                            
                            <div className="ml-auto flex items-center gap-3">
                              <span className={`rounded px-2 py-0.5 text-[10px] font-medium ${comment.topic !== null ? "bg-teal-950/40 text-teal-400/80" : "bg-slate-800/50 text-slate-500"}`}>
                                {commentTopicLabel}
                              </span>
                              <a
                                href={comment.permalink.startsWith("http") ? comment.permalink : `https://reddit.com${comment.permalink}`}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="text-slate-500 hover:text-accent transition"
                              >
                                <ExternalLink size={12} />
                              </a>
                            </div>
                          </div>
                          <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-400">
                            {comment.text}
                          </p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            );
          })}

          {visibleCount < filteredAndSortedPosts.length && (
            <button
              onClick={() => setVisibleCount((c) => c + 50)}
              className="mt-6 w-full rounded-lg border border-white/[0.06] bg-slate-900/50 py-3 text-sm font-medium text-slate-400 transition hover:bg-slate-800 hover:text-slate-200"
            >
              Load more posts
            </button>
          )}

          {filteredAndSortedPosts.length === 0 && (
            <div className="py-20 text-center text-sm text-slate-500">
              No posts matched your search.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
