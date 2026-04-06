"""
topic_graph.py
==============
Generate a standalone, interactive Obsidian-style force-directed graph
visualization of LDA topics and save it as a self-contained HTML file.

Intended to be called from 03_model.py after training, but can also be
run standalone if a saved model already exists in outputs/topics/.

Standalone usage (from project root):
    python scripts/topic_graph.py
"""

import os
import json
import logging
import numpy as np

log = logging.getLogger(__name__)

# ── Default output path ────────────────────────────────────────────────────────

GRAPH_OUTPUT_PATH = os.path.join("outputs", "figures", "topic_graph.html")
SIMILARITY_THRESHOLD = 0.10   # minimum cosine similarity to draw an edge


# ── Data extraction ────────────────────────────────────────────────────────────

def _cosine_similarity_matrix(topic_word: np.ndarray) -> np.ndarray:
    """
    Return the (num_topics × num_topics) pairwise cosine similarity matrix
    for the topic-word probability distributions.
    """
    norms = np.linalg.norm(topic_word, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0, 1e-10, norms)
    normalised = topic_word / safe_norms
    return normalised @ normalised.T   # shape: (num_topics, num_topics)


def extract_graph_data(model, corpus, dictionary, best_k: int,
                       subreddit_name: str = "MohsSurgery") -> dict:
    """
    Pull everything needed for the visualisation out of the trained model.

    Returns a dict with keys:
        subreddit   – subreddit name string
        k           – number of topics
        topics      – list of topic dicts (id, words, prevalence)
        edges       – list of edge dicts (source, target, similarity)
    """
    num_topics = model.num_topics

    # ── Topic-word matrix (shape: num_topics × vocab_size) ────────────────────
    topic_word = model.get_topics()   # already a numpy array in Gensim 4.x

    # ── Topic prevalence: average probability each topic gets across the corpus ─
    # For each document we get a topic distribution; we average those distributions.
    topic_totals = np.zeros(num_topics)
    n_docs = 0
    for bow in corpus:
        if not bow:
            continue
        # minimum_probability=0 ensures all topics are returned (not just dominant ones)
        doc_dist = dict(model.get_document_topics(bow, minimum_probability=0))
        for tid in range(num_topics):
            topic_totals[tid] += doc_dist.get(tid, 0.0)
        n_docs += 1

    prevalence = topic_totals / max(n_docs, 1)
    # Normalise so prevalences sum to 1 (handles floating-point drift)
    total = prevalence.sum()
    if total > 0:
        prevalence /= total

    # ── Top 10 words per topic ────────────────────────────────────────────────
    topics = []
    for tid in range(num_topics):
        top_words = model.show_topic(tid, topn=10)
        topics.append({
            "id":         tid,
            "words":      [{"word": w, "weight": round(float(p), 5)} for w, p in top_words],
            "prevalence": round(float(prevalence[tid]), 5),
        })

    # ── Pairwise cosine similarity → edges ────────────────────────────────────
    sim_matrix = _cosine_similarity_matrix(topic_word)
    edges = []
    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            sim = float(sim_matrix[i, j])
            if sim > SIMILARITY_THRESHOLD:
                edges.append({
                    "source":     i,
                    "target":     j,
                    "similarity": round(sim, 4),
                })

    log.info(
        f"Graph data: {len(topics)} topics, {len(edges)} edges "
        f"(similarity > {SIMILARITY_THRESHOLD})"
    )

    return {
        "subreddit": subreddit_name,
        "k":         best_k,
        "topics":    topics,
        "edges":     edges,
    }


# ── HTML template ──────────────────────────────────────────────────────────────
# Placeholders replaced at render time:
#   __DATA_JSON__   → serialised graph data object
#   __SUBREDDIT__   → subreddit name
#   __K_VALUE__     → best k

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Topic Graph \u2014 r/__SUBREDDIT__</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#12131a;color:#c8d0e0;font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;overflow:hidden;height:100vh}

/* ── Toolbar ── */
#toolbar{
  display:flex;align-items:center;padding:0 20px;gap:10px;
  background:rgba(14,15,22,.98);
  border-bottom:1px solid rgba(255,255,255,.06);
  height:48px;position:relative;z-index:10;
  user-select:none;
}
#title{font-size:13px;font-weight:600;color:#c8d0e0;letter-spacing:.03em}
#subtitle{font-size:11px;color:rgba(200,208,224,.38);margin-left:2px}
#divider{width:1px;height:18px;background:rgba(255,255,255,.1);margin:0 6px}
#node-count{font-size:11px;color:rgba(200,208,224,.38)}
#toggle-btn{
  margin-left:auto;
  background:rgba(55,138,221,.1);border:1px solid rgba(55,138,221,.32);
  color:rgb(55,138,221);padding:4px 15px;border-radius:5px;
  cursor:pointer;font-size:11px;font-weight:500;letter-spacing:.04em;
  transition:background .15s,border-color .15s;white-space:nowrap;
}
#toggle-btn:hover{background:rgba(55,138,221,.2);border-color:rgba(55,138,221,.55)}
#toggle-btn:disabled{opacity:.3;cursor:default;pointer-events:none}

/* ── Canvas ── */
#canvas{display:block;cursor:grab}
#canvas:active{cursor:grabbing}

/* ── Detail panel ── */
#detail{
  position:fixed;top:60px;right:18px;width:272px;
  background:rgba(13,14,21,.96);
  border:1px solid rgba(127,119,221,.25);
  border-radius:10px;padding:0;
  backdrop-filter:blur(14px);z-index:20;
  max-height:calc(100vh - 78px);overflow:hidden;
  box-shadow:0 12px 40px rgba(0,0,0,.55);
  transition:opacity .15s,transform .15s;
}
#detail.hidden{opacity:0;pointer-events:none;transform:translateY(-6px)}
#detail-inner{overflow-y:auto;max-height:calc(100vh - 78px);padding:15px 16px 14px}

/* detail header */
.ph{display:flex;justify-content:space-between;align-items:baseline;margin-bottom:11px;padding-bottom:10px;border-bottom:1px solid rgba(255,255,255,.07)}
.pt{font-size:14px;font-weight:700;color:rgb(127,119,221);letter-spacing:.02em}
.pp{font-size:11px;color:rgba(200,208,224,.45)}
.pw-label{font-size:10px;font-weight:600;letter-spacing:.07em;text-transform:uppercase;color:rgba(200,208,224,.35);margin-bottom:6px}

/* word rows */
.pw{display:flex;flex-direction:column;gap:5px}
.wr{display:grid;grid-template-columns:86px 1fr 52px;align-items:center;gap:8px}
.wl{font-size:11.5px;color:#c8d0e0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bb{background:rgba(255,255,255,.07);border-radius:2px;height:5px;overflow:hidden}
.bf{height:100%;border-radius:2px;background:linear-gradient(90deg,rgb(55,138,221),rgb(127,119,221));transition:width .3s ease}
.ww{font-size:9.5px;color:rgba(200,208,224,.4);text-align:right;font-variant-numeric:tabular-nums}

/* ── Hint ── */
#hint{
  position:fixed;bottom:14px;left:18px;
  font-size:10px;color:rgba(200,208,224,.22);
  pointer-events:none;letter-spacing:.1em;
}
</style>
</head>
<body>

<div id="toolbar">
  <span id="title">r/__SUBREDDIT__</span>
  <span id="subtitle">\u2014 LDA Topic Graph</span>
  <div id="divider"></div>
  <span id="node-count"></span>
  <button id="toggle-btn">Show Top 5</button>
</div>

<canvas id="canvas"></canvas>

<div id="detail" class="hidden">
  <div id="detail-inner"></div>
</div>

<div id="hint">drag \u00b7 scroll to zoom \u00b7 click to explore</div>

<script>
/* =========================================================
   Embedded graph data (generated by topic_graph.py)
   ========================================================= */
const DATA = __DATA_JSON__;

/* =========================================================
   Canvas & resize
   ========================================================= */
const canvas  = document.getElementById('canvas');
const ctx     = canvas.getContext('2d');
const toolbar = document.getElementById('toolbar');

function resize() {
  canvas.width  = window.innerWidth;
  canvas.height = window.innerHeight - toolbar.offsetHeight;
}
resize();
window.addEventListener('resize', resize);

/* =========================================================
   Colour palette
   ========================================================= */
const PALETTE = ['rgb(55,138,221)', 'rgb(127,119,221)'];
const BG      = '#12131a';

/* =========================================================
   State
   ========================================================= */
let nodes      = [];
let edges      = [];
let selectedId = null;
let showTop5   = false;

// Pan / zoom transform (world origin at canvas centre)
let tx = 0, ty = 0, zoom = 1;

// Interaction tracking
let dragNode  = null;
let isPanning = false;
let didMove   = false;
let lx = 0,  ly = 0;   // last mouse position

/* =========================================================
   Build node + edge lists
   ========================================================= */
function initGraph() {
  // Preserve existing positions when toggling
  const prevPos = {};
  nodes.forEach(n => { prevPos[n.id] = { x: n.x, y: n.y }; });

  const pool = showTop5
    ? [...DATA.topics].sort((a, b) => b.prevalence - a.prevalence).slice(0, 5)
    : DATA.topics;

  const vis = new Set(pool.map(t => t.id));
  const n   = pool.length;
  const R   = Math.min(canvas.width, canvas.height) * 0.26;

  nodes = pool.map((t, i) => {
    const angle = (2 * Math.PI * i / n) - Math.PI / 2;
    const prev  = prevPos[t.id];
    return {
      id    : t.id,
      x     : prev ? prev.x : R * Math.cos(angle),
      y     : prev ? prev.y : R * Math.sin(angle),
      vx    : 0,
      vy    : 0,
      // radius scales with sqrt of prevalence so area ~ prevalence
      radius: 11 + Math.sqrt(t.prevalence) * 115,
      color : PALETTE[t.id % 2],
      topic : t,
    };
  });

  edges = DATA.edges.filter(e => vis.has(e.source) && vis.has(e.target));

  document.getElementById('node-count').textContent =
    nodes.length + (nodes.length === 1 ? ' topic' : ' topics');
}

/* =========================================================
   Physics simulation
   ========================================================= */
const REPULSION   = 10000;
const GRAVITY     = 0.042;
const SPRING_K    = 0.007;
const BASE_REST   = 210;
const DAMPING     = 0.80;
const MAX_SPEED   = 9;

function byId(id) { return nodes.find(n => n.id === id); }

function tick() {
  for (const n of nodes) {
    if (dragNode && dragNode.id === n.id) continue;

    let fx = 0, fy = 0;

    // Repulsion from every other node (weighted by combined radii so big nodes
    // push harder and are pushed harder — prevents overlap)
    for (const m of nodes) {
      if (m === n) continue;
      const dx = n.x - m.x, dy = n.y - m.y;
      const d2 = dx * dx + dy * dy + 1;
      const d  = Math.sqrt(d2);
      const f  = (REPULSION * (n.radius + m.radius) * 0.04) / d2;
      fx += f * dx / d;
      fy += f * dy / d;
    }

    // Gentle gravity toward world origin (keeps graph from drifting away)
    fx -= GRAVITY * n.x;
    fy -= GRAVITY * n.y;

    // Spring forces along edges (similar topics are pulled closer)
    for (const e of edges) {
      let other = null;
      if      (e.source === n.id) other = byId(e.target);
      else if (e.target === n.id) other = byId(e.source);
      if (!other) continue;

      const dx   = other.x - n.x, dy = other.y - n.y;
      const d    = Math.sqrt(dx * dx + dy * dy) || 1;
      // stronger similarity → shorter rest length → pulled closer
      const rest = BASE_REST * (1 - e.similarity * 0.45);
      const f    = SPRING_K * (d - rest);
      fx += f * dx / d;
      fy += f * dy / d;
    }

    // Apply forces with velocity damping (weighted feel, not bouncy)
    n.vx = (n.vx + fx) * DAMPING;
    n.vy = (n.vy + fy) * DAMPING;

    const spd = Math.sqrt(n.vx * n.vx + n.vy * n.vy);
    if (spd > MAX_SPEED) { n.vx = n.vx / spd * MAX_SPEED; n.vy = n.vy / spd * MAX_SPEED; }

    n.x += n.vx;
    n.y += n.vy;
  }
}

/* =========================================================
   Rendering
   ========================================================= */
function render() {
  const W = canvas.width, H = canvas.height;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = BG;
  ctx.fillRect(0, 0, W, H);

  ctx.save();
  // World origin at canvas centre; pan/zoom applied on top
  ctx.translate(tx + W / 2, ty + H / 2);
  ctx.scale(zoom, zoom);

  /* ── Edges ── */
  for (const e of edges) {
    const src = byId(e.source), tgt = byId(e.target);
    if (!src || !tgt) continue;

    const hi = selectedId !== null &&
               (e.source === selectedId || e.target === selectedId);

    // Opacity and thickness scale with similarity strength
    const alpha = hi
      ? Math.min(0.92, e.similarity * 2.2 + 0.18)
      : Math.min(0.42, e.similarity * 1.1 + 0.05);
    const lw = hi
      ? Math.max(1.2, e.similarity * 5.5)
      : Math.max(0.4, e.similarity * 2.8);

    ctx.beginPath();
    ctx.moveTo(src.x, src.y);
    ctx.lineTo(tgt.x, tgt.y);
    ctx.strokeStyle = hi
      ? `rgba(175,135,255,${alpha})`
      : `rgba(85,100,170,${alpha})`;
    ctx.lineWidth = lw;
    ctx.stroke();
  }

  /* ── Nodes ── */
  for (const n of nodes) {
    const sel = n.id === selectedId;

    ctx.save();

    // Glow halo when selected
    if (sel) {
      ctx.shadowBlur  = 34;
      ctx.shadowColor = n.color;
    }

    ctx.beginPath();
    ctx.arc(n.x, n.y, n.radius, 0, Math.PI * 2);
    ctx.fillStyle   = n.color;
    ctx.globalAlpha = sel ? 1.0 : 0.85;
    ctx.fill();

    if (sel) {
      ctx.shadowBlur  = 0;
      ctx.strokeStyle = 'rgba(255,255,255,0.65)';
      ctx.lineWidth   = 1.8;
      ctx.stroke();
    }

    ctx.restore();

    // Topic number label (inside node)
    ctx.save();
    const fs = Math.max(9, Math.min(15, n.radius * 0.62));
    ctx.font         = `600 ${fs}px system-ui,sans-serif`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillStyle    = 'rgba(255,255,255,0.88)';
    ctx.fillText('T' + (n.id + 1), n.x, n.y);
    ctx.restore();

    // Prevalence % label (just below node)
    ctx.save();
    ctx.font         = '9.5px system-ui,sans-serif';
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'top';
    ctx.fillStyle    = 'rgba(200,208,224,0.38)';
    ctx.fillText((n.topic.prevalence * 100).toFixed(1) + '%', n.x, n.y + n.radius + 5);
    ctx.restore();
  }

  ctx.restore();
}

/* =========================================================
   Animation loop
   ========================================================= */
function loop() {
  tick();
  render();
  requestAnimationFrame(loop);
}

/* =========================================================
   Coordinate helpers
   ========================================================= */
function toWorld(sx, sy) {
  return {
    x: (sx - canvas.width  / 2 - tx) / zoom,
    y: (sy - canvas.height / 2 - ty) / zoom,
  };
}

function hitTest(wx, wy) {
  // Iterate reversed so visually topmost node wins
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n  = nodes[i];
    const dx = wx - n.x, dy = wy - n.y;
    if (Math.sqrt(dx * dx + dy * dy) <= n.radius + 5) return n;
  }
  return null;
}

/* =========================================================
   Mouse events
   ========================================================= */
canvas.addEventListener('mousedown', e => {
  const toolbarH = toolbar.offsetHeight;
  const w = toWorld(e.clientX, e.clientY - toolbarH);
  const hit = hitTest(w.x, w.y);
  didMove = false;
  if (hit) { dragNode = hit; }
  else      { isPanning = true; }
  lx = e.clientX; ly = e.clientY;
  e.preventDefault();
});

canvas.addEventListener('mousemove', e => {
  const dx = e.clientX - lx, dy = e.clientY - ly;
  if (Math.abs(dx) + Math.abs(dy) > 2) didMove = true;

  if (dragNode) {
    const toolbarH = toolbar.offsetHeight;
    const w = toWorld(e.clientX, e.clientY - toolbarH);
    dragNode.x = w.x; dragNode.y = w.y;
    dragNode.vx = 0;  dragNode.vy = 0;
  } else if (isPanning) {
    tx += dx; ty += dy;
  }
  lx = e.clientX; ly = e.clientY;
});

canvas.addEventListener('mouseup', () => {
  if (dragNode && !didMove) {
    // Pure click — toggle selection
    if (dragNode.id === selectedId) {
      selectedId = null;
      hidePanel();
    } else {
      selectedId = dragNode.id;
      showPanel(dragNode.topic);
    }
  }
  dragNode  = null;
  isPanning = false;
});

canvas.addEventListener('mouseleave', () => {
  dragNode = null; isPanning = false;
});

// Scroll to zoom (centred on cursor position)
canvas.addEventListener('wheel', e => {
  e.preventDefault();
  const factor = e.deltaY < 0 ? 1.12 : 0.89;
  const newZoom = Math.max(0.12, Math.min(7, zoom * factor));

  // Zoom toward cursor
  const toolbarH = toolbar.offsetHeight;
  const cx = e.clientX - canvas.width  / 2;
  const cy = e.clientY - toolbarH - canvas.height / 2;
  tx = cx - (cx - tx) * (newZoom / zoom);
  ty = cy - (cy - ty) * (newZoom / zoom);
  zoom = newZoom;
}, { passive: false });

/* =========================================================
   Detail panel
   ========================================================= */
const panel      = document.getElementById('detail');
const panelInner = document.getElementById('detail-inner');

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function showPanel(topic) {
  const maxW = Math.max(...topic.words.map(w => w.weight));
  panelInner.innerHTML =
    '<div class="ph">' +
      '<span class="pt">Topic ' + (topic.id + 1) + '</span>' +
      '<span class="pp">' + (topic.prevalence * 100).toFixed(1) + '% of corpus</span>' +
    '</div>' +
    '<div class="pw-label">Top words</div>' +
    '<div class="pw">' +
    topic.words.map(w =>
      '<div class="wr">' +
        '<span class="wl">' + esc(w.word) + '</span>' +
        '<div class="bb"><div class="bf" style="width:' +
          (w.weight / maxW * 100).toFixed(1) + '%"></div></div>' +
        '<span class="ww">' + w.weight.toFixed(4) + '</span>' +
      '</div>'
    ).join('') +
    '</div>';
  panel.classList.remove('hidden');
}

function hidePanel() { panel.classList.add('hidden'); }

/* =========================================================
   Toggle: all topics vs top 5 by prevalence
   ========================================================= */
const toggleBtn = document.getElementById('toggle-btn');

// Disable button if there are 5 or fewer topics (no point toggling)
if (DATA.topics.length <= 5) {
  toggleBtn.disabled = true;
  toggleBtn.title = 'All topics are already shown (k \u2264 5)';
}

toggleBtn.addEventListener('click', () => {
  showTop5 = !showTop5;
  toggleBtn.textContent = showTop5 ? 'Show All' : 'Show Top 5';

  // If selected node is no longer visible, clear selection
  if (selectedId !== null) {
    const visible = showTop5
      ? [...DATA.topics].sort((a, b) => b.prevalence - a.prevalence).slice(0, 5)
      : DATA.topics;
    if (!visible.find(t => t.id === selectedId)) {
      selectedId = null;
      hidePanel();
    }
  }
  initGraph();
});

/* =========================================================
   Boot
   ========================================================= */
initGraph();
loop();
</script>
</body>
</html>"""


# ── Main export function ───────────────────────────────────────────────────────

def generate_topic_graph(
    model,
    corpus,
    dictionary,
    best_k: int,
    output_path: str = GRAPH_OUTPUT_PATH,
    subreddit_name: str = "MohsSurgery",
) -> None:
    """
    Extract topic data from the trained LDA model, build the graph data
    structure, and write a fully self-contained HTML visualisation to disk.

    Parameters
    ----------
    model         : trained Gensim LdaModel
    corpus        : BoW corpus (list of doc vectors)
    dictionary    : Gensim Dictionary used to build the corpus
    best_k        : number of topics in the model (used in the page title)
    output_path   : where to save the HTML file
    subreddit_name: displayed in the page title
    """
    log.info("Extracting graph data from LDA model ...")
    graph_data = extract_graph_data(model, corpus, dictionary, best_k, subreddit_name)

    # Serialise to JSON — json.dumps handles all escaping safely
    data_json = json.dumps(graph_data, ensure_ascii=False)

    # Inject into template
    html = (
        _HTML_TEMPLATE
        .replace("__DATA_JSON__",  data_json)
        .replace("__SUBREDDIT__",  subreddit_name)
        .replace("__K_VALUE__",    str(best_k))
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    log.info(f"Topic graph saved to {output_path}")


# ── Standalone runner ──────────────────────────────────────────────────────────

def _standalone():
    """
    Load the best saved model from outputs/topics/ and regenerate the graph.
    Useful if you want to tweak the visualisation without retraining.
    """
    import glob
    from gensim.models import LdaModel
    import gensim.corpora as corpora

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )

    model_dir  = os.path.join("outputs", "topics")
    dict_path  = os.path.join(model_dir, "dictionary.gensim")

    # Find the saved model file (gensim saves .index, .expElogbeta.npy, etc.)
    index_files = glob.glob(os.path.join(model_dir, "lda_k*.index"))
    if not index_files:
        raise FileNotFoundError(
            f"No saved LDA model found in {model_dir}. "
            "Run 03_model.py first."
        )

    # Pick the model with the highest k that was saved (fallback to any)
    model_base = sorted(index_files)[-1].replace(".index", "")
    best_k     = int(os.path.basename(model_base).replace("lda_k", ""))

    log.info(f"Loading model from {model_base} (k={best_k}) ...")
    model      = LdaModel.load(model_base)
    dictionary = corpora.Dictionary.load(dict_path)

    # Rebuild a minimal corpus from the dictionary (no text needed for prevalence
    # in standalone mode — we use equal weights as a placeholder)
    log.info("No corpus available in standalone mode; using uniform prevalence.")
    num_topics = model.num_topics
    graph_data = {
        "subreddit": "MohsSurgery",
        "k":         best_k,
        "topics": [],
        "edges":  [],
    }

    topic_word = model.get_topics()
    from numpy.linalg import norm as np_norm
    norms = np.linalg.norm(topic_word, axis=1, keepdims=True)
    safe  = np.where(norms == 0, 1e-10, norms)
    sim   = (topic_word / safe) @ (topic_word / safe).T

    uniform_prev = 1.0 / num_topics
    for tid in range(num_topics):
        top_words = model.show_topic(tid, topn=10)
        graph_data["topics"].append({
            "id":         tid,
            "words":      [{"word": w, "weight": round(float(p), 5)} for w, p in top_words],
            "prevalence": round(uniform_prev, 5),
        })

    for i in range(num_topics):
        for j in range(i + 1, num_topics):
            s = float(sim[i, j])
            if s > SIMILARITY_THRESHOLD:
                graph_data["edges"].append({"source": i, "target": j, "similarity": round(s, 4)})

    data_json = json.dumps(graph_data, ensure_ascii=False)
    html = (
        _HTML_TEMPLATE
        .replace("__DATA_JSON__", data_json)
        .replace("__SUBREDDIT__", "MohsSurgery")
        .replace("__K_VALUE__",   str(best_k))
    )

    out = GRAPH_OUTPUT_PATH
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    log.info(f"Saved to {out}")


if __name__ == "__main__":
    _standalone()
