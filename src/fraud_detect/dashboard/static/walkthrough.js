/* "Explain a node": one test node's score, walked through step by step, with a
 * glossary of every label it uses. Reads the same saved explanations as the
 * /explain page (its two API calls); computes nothing new about the model.
 * No libraries and no external requests: charts are SVG built here, and every
 * value reaches the page as a text node, never as markup. */
"use strict";

(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const state = { models: [], key: null, draw: "any", card: null, revealed: false };
  const N_SHOWN = 6; // contributions drawn one by one; the rest are summed

  // ---- DOM helpers ------------------------------------------------------
  function fill(node, attrs, kids) {
    for (const [name, value] of Object.entries(attrs || {})) {
      if (value === null || value === undefined || value === false) continue;
      if (name === "text") node.textContent = String(value);
      else if (name === "on") {
        for (const [event, fn] of Object.entries(value)) node.addEventListener(event, fn);
      } else node.setAttribute(name, value === true ? "" : String(value));
    }
    for (const kid of kids.flat()) {
      if (kid === null || kid === undefined || kid === false) continue;
      node.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
    }
    return node;
  }
  const h = (tag, attrs, ...kids) => fill(document.createElement(tag), attrs, kids);
  const s = (tag, attrs, ...kids) =>
    fill(document.createElementNS(SVG_NS, tag), attrs, kids);
  function put(id, ...kids) {
    $(id).replaceChildren(...kids.flat().filter(Boolean));
  }
  const errorBox = (message) => h("div", { class: "error", text: message });

  // ---- Number formats ---------------------------------------------------
  const MINUS = "−";
  const DASH = "—";
  const missing = (v) => v === null || v === undefined || Number.isNaN(v);
  const signed = (v, d = 2) =>
    missing(v) ? DASH : (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(d);
  // Scores near 0 or 1 and tiny score changes must not round to look exact.
  function prob(v) {
    if (missing(v)) return DASH;
    if (v >= 0.9995) return "> 0.999";
    if (v < 0.0005) return "< 0.001";
    return v.toFixed(3);
  }
  function delta(v) {
    if (missing(v)) return DASH;
    if (v === 0) return "0";
    if (Math.abs(v) < 0.0005) return (v < 0 ? MINUS : "+") + Math.abs(v).toExponential(1);
    return signed(v, 3);
  }
  const logit = (p) => Math.log(p / (1 - p));
  const sigmoid = (z) => 1 / (1 + Math.exp(-z));

  async function api(path, options) {
    const response = await fetch(path, options);
    let body;
    try {
      body = await response.json();
    } catch (e) {
      throw new Error("the server sent something unexpected (" + response.status + ")");
    }
    if (!response.ok) throw new Error(body.error || String(response.status));
    return body;
  }

  // ---- Sources and glossary ---------------------------------------------
  // Numbered in this order on the page; walkthrough.html cites [12] by number.
  const SOURCES = [
    "Dou, Liu, Sun, Deng, Peng and Yu. Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters. CIKM 2020. arXiv:2008.08692, DOI 10.1145/3340531.3411903. The source of both datasets' graphs and preprocessed features.",
    "Rayana and Akoglu. Collective Opinion Spam Detection: Bridging Review Networks and Metadata. KDD 2015. YelpChi's reviews, labels and 32 features.",
    "Zhang, Yin, Chen, Nguyen Hung, Huang and Cui. GCN-Based User Representation Learning for Unifying Robust Recommendation and Fraudster Detection. SIGIR 2020. arXiv:2005.10150. The groups behind Amazon's 25 user features.",
    "McAuley and Leskovec. From Amateurs to Connoisseurs: Modeling the Evolution of User Expertise through Online Reviews. WWW 2013. arXiv:1303.4402. The Amazon reviews and helpfulness votes.",
    "Lundberg and Lee. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017. arXiv:1705.07874.",
    "Lundberg et al. From local explanations to global understanding with explainable AI for trees. Nature Machine Intelligence 2, 56–67 (2020). DOI 10.1038/s42256-019-0138-9.",
    "Jullum, Redelmeier and Aas. groupShapley: Efficient prediction explanation with Shapley values for feature groups. arXiv:2106.12228.",
    "Veličković, Fedus, Hamilton, Liò, Bengio and Hjelm. Deep Graph Infomax. ICLR 2019. arXiv:1809.10341.",
    "This project: docs/RESEARCH_PLAN.md §11, the checks of Amazon's f19 and the other #36 columns (2026-10-19 and 2026-10-26).",
    "This project: docs/EXPLAINABILITY_PLAN.md, how the SHAP values, group Shapley and neighbour influence on this page are computed.",
    "This project: docs/PHASE2_GNN_PLAN.md and docs/GNN_MODELS_PLAN.md, the neighbour-mean, degree, DGI and GNN-score blocks; docs/RESEARCH_PLAN.md decision log (2026-10-08) for the threshold rule.",
    "Jesus et al. How can I choose an explainer? FAccT 2021, DOI 10.1145/3442188.3445941; Amarasinghe et al. AAAI 2024, DOI 10.1609/aaai.v38i19.30082; Bansal et al. CHI 2021, DOI 10.1145/3411764.3445717.",
  ];
  const SRC = { care: 1, yelp: 2, amzfeat: 3, amzvotes: 4, shap: 5, tree: 6, group: 7, dgi: 8, f19: 9, explain: 10, blocks: 11 };

  // [title, text, source numbers]
  const GLOSSARY = {
    score: ["score", "The model's probability that this node is fraud, from 0 to 1.", [SRC.explain]],
    threshold: ["threshold", "The score at or above which a node is flagged. Chosen on training data only (the best F1 on out-of-fold training predictions), never on test nodes.", [SRC.blocks]],
    base: ["start (base)", "The model's average log-odds over a background sample of training nodes. Every explanation starts here.", [SRC.tree, SRC.explain]],
    logodds: ["log-odds", "The model's raw output. score = 1 / (1 + e^−log-odds). SHAP values add up in log-odds, so the bars are drawn in log-odds.", [SRC.tree]],
    shap: ["SHAP value", "How far one input moved this node's log-odds away from the base: exact Shapley values of a stated game, with absent inputs filled in from the background sample (interventional TreeSHAP). They add up exactly to the score; that is arithmetic, not proof the explanation is faithful.", [SRC.shap, SRC.tree, SRC.explain]],
    group: ["group Shapley", "The same fair share, given to a whole block of inputs at once. Computed for a sample of test nodes; for other nodes the block's SHAP values are summed, and the page says which.", [SRC.group, SRC.explain]],
    influence: ["neighbour influence", "How much the score changes when one neighbour is removed and the neighbour means are recomputed. Necessity-style: redundant neighbours can each look unimportant.", [SRC.explain]],
    own: ["own features", "The node's own columns, without anything from the graph.", [SRC.care]],
    nbr: ["neighbour mean", "The average of one column over the node's neighbours in one relation. Built from features only, never from labels.", [SRC.blocks]],
    deg: ["degree (log)", "log(1 + the number of neighbours) in one relation.", [SRC.blocks]],
    nbrrw: ["rewired neighbour mean", "A neighbour mean on a graph rewired at random so every node keeps its degree: a control for real structure.", [SRC.blocks]],
    gnnssl: ["DGI embedding", "A summary of the node's neighbourhood from a Deep Graph Infomax encoder trained without labels, one per relation, reduced by PCA (fitted on training nodes) to a few numbers. SHAP credits these numbers, not edges or nodes.", [SRC.dgi, SRC.blocks]],
    gnnrand: ["untrained-GNN embedding", "The same encoder as the DGI embedding with random weights: a control for whether training the encoder matters.", [SRC.blocks]],
    gnnsup: ["supervised GNN score", "The out-of-fold score of a GraphSAGE model trained on training labels (Model 6), used as one input column.", [SRC.blocks]],
    "amazon:feat": ["fNN (own), Amazon", "One of the user's 25 handcrafted features. The dataset does not say which column is which.", [SRC.amzfeat, SRC.care]],
    "amazon:f19": ["f19 (own), Amazon", "Probably a helpfulness-vote count. Amazon's label is the share of helpful votes, so f19 likely leaks the label.", [SRC.f19, SRC.amzvotes]],
    "amazon:rating": ["f13, f15, f18 (own), Amazon", "Consistent with rating statistics in this project's checks; not documented in the dataset.", [SRC.f19]],
    "amazon:label": ["fraud label, Amazon", "Fraud: a user with fewer than 20% helpful votes. Benign: more than 80%.", [SRC.care, SRC.amzvotes]],
    "amazon:UPU": ["U-P-U", "Users who reviewed at least one of the same products.", [SRC.care]],
    "amazon:USU": ["U-S-U", "Users who gave at least one same star rating within one week.", [SRC.care]],
    "amazon:UVU": ["U-V-U", "Users whose review texts are among the 5% most similar (TF-IDF).", [SRC.care]],
    "yelpchi:feat": ["fNN (own), YelpChi", "One of the review's 32 handcrafted features, from the text and behaviour of the review, its user and its product. The dataset does not say which column is which.", [SRC.yelp, SRC.care]],
    "yelpchi:label": ["fraud label, YelpChi", "Fraud: Yelp's own filter flagged the review as spam. Not verified fraud.", [SRC.yelp, SRC.care]],
    "yelpchi:RUR": ["R-U-R", "Reviews posted by the same user.", [SRC.care]],
    "yelpchi:RSR": ["R-S-R", "Reviews of the same product with the same star rating.", [SRC.care]],
    "yelpchi:RTR": ["R-T-R", "Reviews of the same product posted in the same month.", [SRC.care]],
  };
  const KIND_WORDS = [
    ["rewired neighbour mean", "nbrrw"],
    ["neighbour mean", "nbr"],
    ["DGI embedding", "gnnssl"],
    ["untrained-GNN embedding", "gnnrand"],
    ["supervised GNN score", "gnnsup"],
    ["degree", "deg"],
  ];

  // The glossary keys a label from the server uses (explain.column_label and player_label).
  function termsOf(label, dataset) {
    const keys = [];
    const own = /^(f\d+) \(own\)$/.exec(label);
    const base = /^(f\d+)[ ,]/.exec(label);
    if (own) keys.push(ownKey(own[1], dataset));
    else if (base) keys.push(ownKey(base[1], dataset));
    if (label === "own features") keys.push("own");
    for (const [words, key] of KIND_WORDS) {
      if (label.toLowerCase().includes(words.toLowerCase())) {
        keys.push(key);
        break;
      }
    }
    if (/mean over/.test(label) && !keys.includes("nbr") && !keys.includes("nbrrw")) keys.push("nbr");
    for (const rel of label.match(/\b[A-Z]-[A-Z]-[A-Z]\b/g) || []) keys.push(dataset + ":" + rel.replaceAll("-", ""));
    return keys.filter((k) => GLOSSARY[k]);
  }
  function ownKey(feature, dataset) {
    if (dataset === "amazon" && feature === "f19") return "amazon:f19";
    if (dataset === "amazon" && ["f13", "f15", "f18"].includes(feature)) return "amazon:rating";
    return dataset + ":feat";
  }

  // A label shown as a dotted term; hovering or focusing it lights up its card.
  let used = new Set();
  function term(text, keys) {
    const list = (Array.isArray(keys) ? keys : [keys]).filter((k) => GLOSSARY[k]);
    list.forEach((k) => used.add(k));
    if (!list.length) return h("span", { text });
    const light = (on) => () => {
      for (const k of list) {
        const card = $("g-" + k.replace(":", "-"));
        if (card) card.classList.toggle("on", on);
      }
    };
    return h("span", {
      class: "term",
      tabindex: "0",
      text,
      on: { pointerenter: light(true), pointerleave: light(false), focus: light(true), blur: light(false) },
    });
  }
  function renderGlossary() {
    const order = Object.keys(GLOSSARY).filter((k) => used.has(k));
    put("glossary-body", order.map((k) => {
      const [title, text, sources] = GLOSSARY[k];
      return h("div", { class: "g-card", id: "g-" + k.replace(":", "-") },
        h("b", { text: title }),
        h("span", { text }),
        h("div", { class: "src" }, "Source: ", sources.flatMap((n, i) => [i ? ", " : "", h("a", { href: "#src-" + n, text: "[" + n + "]" })]))
      );
    }));
  }
  function renderSources() {
    put("sources-list", SOURCES.map((text, i) => h("li", { id: "src-" + (i + 1), text })));
  }

  // ---- Models -----------------------------------------------------------
  const current = () => state.models.find((m) => m.key === state.key);
  const noun = (m) => (m.dataset === "yelpchi" ? "review" : "user");
  const DRAWS = [
    ["any", "any"],
    ["fraud", "fraud"],
    ["benign", "benign"],
    ["true_positive", "caught fraud"],
    ["false_positive", "false alarm"],
    ["false_negative", "missed fraud"],
  ];
  function renderModelButtons() {
    put("model-buttons", state.models.map((m) =>
      h("button", {
        type: "button",
        "aria-pressed": String(m.key === state.key),
        text: (m.dataset_label || m.dataset) + (m.label ? " · " + m.label : "") + (m.state === "ready" ? "" : " (" + m.state + ")"),
        on: { click: () => { state.key = m.key; state.card = null; renderModelButtons(); renderAll(); } },
      })
    ));
    put("draw-buttons", DRAWS.map(([key, text]) =>
      h("button", {
        type: "button",
        "aria-pressed": String(key === state.draw),
        text,
        on: { click: () => { state.draw = key; renderModelButtons(); } },
      })
    ));
    const m = current();
    $("draw-button").disabled = !m || m.state !== "ready";
    const notes = [];
    if (m && m.state !== "ready") notes.push(h("div", { class: "notice", text: m.message || "This model's explanation is still loading." }));
    if (m && m.dataset === "amazon" && m.variant === "all") {
      notes.push(h("div", { class: "notice", text: "All 25 features, including f19, a probable label leak. Choose the \"without f19\" model to see the explanation without it." }));
    }
    put("model-note", notes);
  }

  async function drawNode() {
    const m = current();
    if (!m || m.state !== "ready") return;
    $("draw-button").disabled = true;
    try {
      const card = await api("/api/explain/node", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: m.key, draw: state.draw }),
      });
      if (card.model !== state.key) return;
      state.card = card;
      state.revealed = false;
      renderAll();
    } catch (e) {
      put("sample-body", errorBox("Could not draw a node: " + e.message));
    } finally {
      $("draw-button").disabled = false;
    }
  }

  // ---- The six steps ----------------------------------------------------
  function renderAll() {
    used = new Set();
    const m = current();
    const c = state.card;
    if (!m || !c) {
      put("sample-body", h("p", { class: "empty", text: "Choose a model and draw a test node." }));
      for (const id of ["words-body", "features-body", "blocks-body", "neighbours-body"]) put(id);
      for (const id of ["features-lede", "blocks-lede", "neighbours-lede"]) $(id).textContent = "";
    } else {
      renderSample(m, c);
      renderWords(m, c);
      renderFeatures(m, c);
      renderBlocks(m, c);
      renderNeighbours(m, c);
    }
    for (const k of ["score", "threshold", "base", "logodds", "shap", "group", "influence"]) used.add(k);
    renderGlossary();
  }

  function scoreBar(score, threshold) {
    const W = 400;
    const svg = s("svg", { viewBox: `0 0 ${W} 22`, preserveAspectRatio: "none", role: "img", "aria-label": "score " + prob(score) + ", threshold " + prob(threshold) });
    svg.append(s("rect", { class: "score-track", x: 0, y: 6, width: W, height: 10, rx: 5 }));
    svg.append(s("rect", { class: "score-fill", x: 0, y: 6, width: Math.max(2, score * W), height: 10, rx: 5 }));
    svg.append(s("line", { class: "score-thr", x1: threshold * W, x2: threshold * W, y1: 1, y2: 21 }));
    return svg;
  }

  function renderSample(m, c) {
    const label = c.label === 1 ? "fraud" : "benign";
    const right = (c.label === 1) === c.flagged;
    put("sample-body",
      h("div", { class: "sample-row" },
        h("div", {}, h("div", { class: "small muted", text: "Test " + noun(m) }), h("div", { class: "id", text: "node " + c.node_id })),
        h("div", { class: "score-wrap" },
          h("div", { class: "small muted" }, term("Score", "score"), " ", h("b", { text: prob(c.prob) }), " · ", term("threshold", "threshold"), " " + prob(c.threshold)),
          scoreBar(c.prob, c.threshold)
        ),
        h("span", { class: "pill " + (c.flagged ? "flagged" : "cleared"), text: c.flagged ? "Flagged" : "Not flagged" }),
        state.revealed
          ? h("span", { class: "small" }, "True ", term("label", m.dataset + ":label"), ": ", h("b", { text: label }), right ? " (the model is right)" : " (the model is wrong)")
          : h("button", { type: "button", class: "ghost", text: "Reveal true label", on: { click: () => { state.revealed = true; renderAll(); } } })
      ),
      h("p", { class: "small muted", text: "Drawn at random from " + c.pool.toLocaleString("en") + " test nodes (" + (DRAWS.find((d) => d[0] === c.draw) || [, c.draw])[1] + "), seed " + c.seed + ". The model was not fitted on it." })
    );
  }

  // Blocks: group Shapley where computed for this node, otherwise summed SHAP.
  function blockValues(c) {
    const exact = c.in_group_shapley_sample && c.blocks.every((b) => !missing(b.group_shapley));
    return { exact, rows: c.blocks.map((b) => ({ ...b, v: exact ? b.group_shapley : b.summed_shap })) };
  }

  function renderWords(m, c) {
    const quoted = (x) => "“" + x.label + "”";
    const sorted = c.contributions.slice().sort((a, b) => b.shap - a.shap);
    const up = sorted.filter((x) => x.shap > 0).slice(0, 2);
    const down = sorted.filter((x) => x.shap < 0).slice(-1);
    const cls = c.flagged ? "fraud" : "benign";
    const dir = c.flagged ? 1 : -1;
    const p = prob(c.prob);
    const scoreText = p.startsWith(">") ? "above " + p.slice(2) : p.startsWith("<") ? "below " + p.slice(2) : "of " + p;
    let t = "The model gives this " + noun(m) + " a score " + scoreText + ", " +
      (c.flagged ? "at or above" : "below") + " the " + prob(c.threshold) + " threshold, so it is " +
      (c.flagged ? "flagged" : "not flagged") + ". The average training " + noun(m) +
      " starts at a score of " + prob(c.base_prob) + ". ";
    const clauses = [];
    if (up.length) clauses.push("The biggest pushes toward fraud come from " + up.map(quoted).join(" and "));
    if (down.length) clauses.push(quoted(down[0]) + " pulls it towards benign");
    if (clauses.length) t += clauses.join("; ") + ". ";
    const { rows } = blockValues(c);
    const toward = rows.filter((b) => b.v * dir > 0);
    const total = toward.reduce((a, b) => a + Math.abs(b.v), 0);
    const graph = toward.filter((b) => b.name !== "own").reduce((a, b) => a + Math.abs(b.v), 0);
    if (total > 0) {
      t += "Its neighbours in the graph account for about " + Math.round((100 * graph) / total) +
        "% of the total push towards " + cls + ".";
    }
    put("words-body", h("p", { class: "words", text: t }));
  }

  function renderFeatures(m, c) {
    const shown = c.contributions.slice(0, N_SHOWN);
    const restN = c.n_rest + c.contributions.length - shown.length;
    const rest = c.rest + c.contributions.slice(N_SHOWN).reduce((a, x) => a + x.shap, 0);
    const steps = [];
    let at = c.base_value;
    for (const x of shown) {
      steps.push({ x, from: at, to: at + x.shap, shap: x.shap });
      at += x.shap;
    }
    if (restN > 0) {
      steps.push({ x: null, label: restN + " other inputs", from: at, to: at + rest, shap: rest });
      at += rest;
    }
    const thr = logit(c.threshold);
    const ends = steps.flatMap((p) => [p.from, p.to]).concat([c.base_value, c.margin, thr]);
    const pad = (Math.max(...ends) - Math.min(...ends)) * 0.05 || 0.5;
    const lo = Math.min(...ends) - pad;
    const hi = Math.max(...ends) + pad;
    const W = 400;
    const X = (v) => ((v - lo) / (hi - lo)) * W;
    const barSvg = (draw) => {
      const svg = s("svg", { viewBox: `0 0 ${W} 16`, preserveAspectRatio: "none", "aria-hidden": "true" });
      draw(svg);
      return svg;
    };
    const mark = (v) => (svg) => svg.append(s("rect", { class: "bar-mark", x: X(v) - 1, y: 0, width: 2, height: 16 }));
    $("features-lede").replaceChildren(
      "Each bar is one input's ", term("SHAP value", "shap"), ", in ", term("log-odds", "logodds"),
      ". They add up from the ", term("start", "base"), " to this node's score; the dashed line is the threshold."
    );
    const rows = [
      h("div", { class: "wf-row" }, term("start (base)", "base"), barSvg(mark(c.base_value)), h("span", { class: "val", text: prob(c.base_prob) })),
    ];
    for (const p of steps) {
      const left = X(Math.min(p.from, p.to));
      const width = Math.max(1.5, Math.abs(X(p.to) - X(p.from)));
      rows.push(h("div", { class: "wf-row" },
        p.x ? term(p.x.label, termsOf(p.x.label, m.dataset)) : h("span", { class: "muted", text: p.label }),
        barSvg((svg) => svg.append(s("rect", { class: p.shap >= 0 ? "bar-up" : "bar-down", x: left, y: 2, width, height: 12, rx: 2 }))),
        h("span", { class: "val", text: signed(p.shap) })
      ));
    }
    rows.push(h("div", { class: "wf-row total" }, term("score", "score"),
      barSvg((svg) => {
        mark(at)(svg);
        svg.append(s("line", { class: "bar-thr", x1: X(thr), x2: X(thr), y1: 0, y2: 16 }));
      }),
      h("span", { class: "val", text: prob(sigmoid(at)) })
    ));
    put("features-body", rows,
      h("div", { class: "key-row" },
        h("span", {}, h("span", { class: "swatch up" }), "toward fraud"),
        h("span", {}, h("span", { class: "swatch down" }), "toward benign"),
        h("span", { text: "start and score columns are scores; the others are log-odds" })
      )
    );
  }

  function renderBlocks(m, c) {
    const { exact, rows } = blockValues(c);
    $("blocks-lede").replaceChildren(
      exact ? term("Group Shapley", "group") : "Summed SHAP values per block (this node is not in the ",
      exact ? " per block of inputs, in log-odds: how much the node's own columns and each part of the graph pushed its score." : "",
      exact ? "" : term("group Shapley", "group"),
      exact ? "" : " sample), in log-odds."
    );
    const maxAbs = Math.max(...rows.map((b) => Math.abs(b.v)), 1e-9);
    const W = 400;
    const zero = W / 2;
    put("blocks-body", rows.map((b) =>
      h("div", { class: "blk-row" },
        term(b.label, termsOf(b.label, m.dataset)),
        (() => {
          const svg = s("svg", { viewBox: `0 0 ${W} 16`, preserveAspectRatio: "none", "aria-hidden": "true" });
          const w = (Math.abs(b.v) / maxAbs) * (W / 2 - 4);
          svg.append(s("rect", { class: b.v >= 0 ? "bar-up" : "bar-down", x: b.v >= 0 ? zero : zero - w, y: 2, width: Math.max(1.5, w), height: 12, rx: 2 }));
          svg.append(s("line", { class: "zero-line", x1: zero, x2: zero, y1: 0, y2: 16 }));
          return svg;
        })(),
        h("span", { class: "val", text: signed(b.v) })
      )
    ));
  }

  function renderNeighbours(m, c) {
    const n = c.neighbours;
    if (!n || !n.available) {
      $("neighbours-lede").textContent = "";
      put("neighbours-body", h("p", { class: "empty", text: "Neighbour influence was not computed for this model." + (m.influence_note ? " " + m.influence_note : "") }));
      return;
    }
    $("neighbours-lede").replaceChildren(
      "The largest ", term("neighbour influences", "influence"),
      " in each relation: the change in this node's score if that one neighbour were removed."
    );
    put("neighbours-body", n.relations.map((r) =>
      h("div", { class: "nb-rel" },
        h("h3", {}, term(r.label, termsOf(r.label, m.dataset)), " · " + r.degree.toLocaleString("en") + " neighbours"),
        r.top.length
          ? r.top.slice(0, 3).map((t) =>
              h("div", { class: "nb-row" },
                h("span", { text: "node " + t.neighbour }),
                h("span", { class: "val", text: delta(t.delta_prob) }),
                h("span", { class: "muted", text: missing(t.neighbour_score) ? "" : (t.neighbour_score >= c.threshold ? "flagged" : "not flagged") + " by the model (score " + prob(t.neighbour_score) + ")" + (t.out_of_range ? "; outside the training range" : "") })
              ))
          : h("p", { class: "small muted", text: "No neighbours in this relation." })
      )
    ));
  }

  // ---- Theme and start --------------------------------------------------
  function setupTheme() {
    const themes = ["auto", "light", "dark"];
    let theme = "auto";
    try {
      theme = localStorage.getItem("dashboard-theme") || "auto";
    } catch (e) {
      theme = "auto";
    }
    const apply = () => {
      if (theme === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", theme);
      $("theme-toggle").textContent = "Theme: " + theme;
    };
    $("theme-toggle").addEventListener("click", () => {
      theme = themes[(themes.indexOf(theme) + 1) % themes.length];
      try {
        localStorage.setItem("dashboard-theme", theme);
      } catch (e) {
        /* the theme still changes for this visit */
      }
      apply();
    });
    apply();
  }

  async function poll() {
    let answer;
    try {
      answer = await api("/api/explain/models");
    } catch (e) {
      put("sample-body", errorBox("Could not reach the server: " + e.message));
      return;
    }
    state.models = answer.models;
    if (!state.key) {
      // Amazon with all 25 features first (with its f19 note), as on the rest of the dashboard.
      const wanted = state.models.find((m) => m.dataset === "amazon" && m.variant === "all");
      state.key = (wanted || state.models.find((m) => m.state === "ready") || state.models[0]).key;
    }
    $("header-meta").textContent = "commit " + (answer.commit || DASH);
    put("page-footer", h("p", { class: "small muted", text: "Built from the saved outputs of scripts/dfa/shap_explain.py and scripts/explain_report.py, split 0. Row-level: group only." }));
    renderModelButtons();
    renderAll();
    if (state.models.some((m) => m.state === "loading")) setTimeout(poll, 1500);
  }

  setupTheme();
  renderSources();
  $("controls").addEventListener("submit", (e) => e.preventDefault());
  $("draw-button").addEventListener("click", drawNode);
  poll();
})();
