/* The explanations page. No libraries and no external requests: charts are SVG
 * built here, and every value reaches the page as a text node, never as markup. */
"use strict";

(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const state = { models: [], key: null, draw: "any", stratum: "all", history: [] };

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
  const num = (v, d = 3) => (missing(v) ? DASH : Number(v).toFixed(d).replace("-", MINUS));
  const signed = (v, d = 3) =>
    missing(v) ? DASH : (v < 0 ? MINUS : "+") + Math.abs(v).toFixed(d);
  const pct = (v, d = 1) => (missing(v) ? DASH : (100 * v).toFixed(d) + "%");
  const int = (v) => (missing(v) ? DASH : Number(v).toLocaleString("en"));
  function probText(v) {
    if (missing(v)) return DASH;
    if (v === 0 || v === 1) return String(v);
    if (v < 0.001) return v.toExponential(1);
    if (v > 0.999) return "1 " + MINUS + " " + (1 - v).toExponential(1);
    return v.toFixed(3);
  }
  const logit = (p) => Math.log(p / (1 - p));
  const yesNo = (v) => (v === true || v === "True" ? "yes" : v === false || v === "False" ? "no" : String(v));

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

  // ---- Tooltip ----------------------------------------------------------
  function showTip(event, title, rows) {
    const tip = $("tooltip");
    tip.replaceChildren(
      h("div", { class: "tip-title", text: title }),
      ...rows.map(([label, value]) =>
        h("div", { class: "tip-row" }, h("span", { text: label }), h("b", { text: value }))
      )
    );
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    let x = event.clientX + 14;
    let y = event.clientY + 14;
    if (event.type === "focus") {
      const r = event.target.getBoundingClientRect();
      x = r.left + r.width / 2;
      y = r.top;
    }
    tip.style.left = Math.max(8, Math.min(x, window.innerWidth - box.width - 8)) + "px";
    tip.style.top = Math.max(8, Math.min(y, window.innerHeight - box.height - 8)) + "px";
  }
  const hideTip = () => ($("tooltip").hidden = true);
  const tipHandlers = (title, rows) => ({
    pointermove: (e) => showTip(e, title, rows),
    pointerleave: hideTip,
    focus: (e) => showTip(e, title, rows),
    blur: hideTip,
  });

  // ---- Tables -----------------------------------------------------------
  // columns: [label, key or function, format, numeric?]
  function table(columns, rows, emptyText) {
    if (!rows || rows.length === 0) return h("p", { class: "empty small", text: emptyText || "Not computed for this model." });
    return h(
      "table",
      { class: "compact" },
      h("thead", {}, h("tr", {}, columns.map(([label, , , numeric]) => h("th", { class: numeric ? "num-cell" : null, text: label })))),
      h(
        "tbody",
        {},
        rows.map((row) =>
          h(
            "tr",
            {},
            columns.map(([, key, format, numeric]) => {
              const value = typeof key === "function" ? key(row) : row[key];
              const shown = format ? format(value, row) : value;
              return h("td", { class: numeric ? "num-cell" : null }, shown instanceof Node ? shown : String(shown ?? DASH));
            })
          )
        )
      )
    );
  }

  // ---- Charts -----------------------------------------------------------
  function niceTicks(lo, hi, n) {
    const span = hi - lo || 1;
    const raw = span / n;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((x) => span / x <= n) || 10 * mag;
    const out = [];
    for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(+t.toFixed(10));
    return out;
  }

  // A log-odds waterfall: base value, the top contributions, the rest, the score.
  function waterfall(card) {
    const steps = [];
    let at = card.base_value;
    for (const c of card.contributions) {
      steps.push({ label: c.label, sub: c.block, from: at, to: at + c.shap, shap: c.shap, c });
      at += c.shap;
    }
    if (card.n_rest > 0) {
      steps.push({ label: "other " + card.n_rest + " columns", sub: "summed", from: at, to: at + card.rest, shap: card.rest });
      at += card.rest;
    }
    const thr = logit(card.threshold);
    const ends = steps.flatMap((x) => [x.from, x.to]).concat([card.base_value, card.margin, thr]);
    let lo = Math.min(...ends);
    let hi = Math.max(...ends);
    const pad = (hi - lo) * 0.06 || 1;
    lo -= pad;
    hi += pad;
    const W = 760, L = 250, R = 80, rowH = 24, top = 26;
    const plotW = W - L - R;
    const x = (v) => L + ((v - lo) / (hi - lo)) * plotW;
    const H = top + rowH * (steps.length + 2) + 30;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "waterfall", role: "img", "aria-label": "SHAP waterfall in log-odds" });
    for (const t of niceTicks(lo, hi, 6)) {
      svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top - 6, y2: H - 24 }));
      svg.append(s("text", { class: "axis", x: x(t), y: H - 8, "text-anchor": "middle", text: num(t, 1) }));
    }
    svg.append(s("line", { class: "thr", x1: x(thr), x2: x(thr), y1: top - 10, y2: H - 24 }));
    svg.append(s("text", { class: "ref-label", x: x(thr), y: top - 12, "text-anchor": "middle", text: "threshold " + probText(card.threshold) }));
    const totalRow = (i, label, value, sub) => {
      const y = top + i * rowH;
      const g = s("g", { class: "row", tabindex: 0 });
      g.append(s("text", { class: "row-label", x: L - 8, y: y + 15, "text-anchor": "end", text: label }));
      g.append(s("circle", { class: "bar total", cx: x(value), cy: y + rowH / 2, r: 5 }));
      g.append(s("text", { class: "row-value", x: W - R + 8, y: y + 15, text: num(value, 2) }));
      for (const [ev, fn] of Object.entries(tipHandlers(label, [["log-odds", num(value, 3)], ["probability", probText(1 / (1 + Math.exp(-value)))], ["", sub]]))) g.addEventListener(ev, fn);
      svg.append(g);
    };
    totalRow(0, "start: base value", card.base_value, "average score over the SHAP background (training nodes)");
    steps.forEach((st, i) => {
      const y = top + (i + 1) * rowH;
      const up = st.shap >= 0;
      const g = s("g", { class: "row", tabindex: 0 });
      g.append(s("rect", { class: "hit", x: 0, y, width: W, height: rowH }));
      g.append(s("text", { class: "row-label", x: L - 8, y: y + 15, "text-anchor": "end", text: st.label.length > 38 ? st.label.slice(0, 37) + "…" : st.label }));
      const a = x(Math.min(st.from, st.to));
      const w = Math.max(1.5, Math.abs(x(st.to) - x(st.from)));
      g.append(s("rect", { class: "bar " + (up ? "up" : "down"), x: a, y: y + 5, width: w, height: rowH - 10, rx: 2 }));
      g.append(s("line", { class: "link", x1: x(st.to), x2: x(st.to), y1: y + rowH - 5, y2: y + rowH + 5 }));
      g.append(s("text", { class: "row-value", x: W - R + 8, y: y + 15, text: signed(st.shap, 2) }));
      const rows = [["SHAP (log-odds)", signed(st.shap, 4)], ["block", st.sub]];
      if (st.c) {
        rows.push(["input value", missing(st.c.value) ? "not shown" : num(st.c.value, 4)]);
        rows.push(["rank over all test nodes", missing(st.c.overall_rank) ? DASH : "#" + st.c.overall_rank]);
      }
      for (const [ev, fn] of Object.entries(tipHandlers(st.label, rows))) g.addEventListener(ev, fn);
      svg.append(g);
    });
    totalRow(steps.length + 1, "end: model score", card.margin, "this node's log-odds; probability " + probText(card.prob));
    return svg;
  }

  // Mean |SHAP| per column with its interval; colour by the signed mean.
  function importanceChart(rows) {
    if (!rows.length) return h("p", { class: "empty small", text: "Not computed for this model." });
    const W = 760, L = 290, R = 70, rowH = 22, top = 8;
    const hi = Math.max(...rows.map((r) => r.high ?? r.mean_abs_shap)) * 1.05 || 1;
    const plotW = W - L - R;
    const x = (v) => L + (v / hi) * plotW;
    const H = top + rowH * rows.length + 28;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "imp", role: "img", "aria-label": "Mean absolute SHAP per column" });
    for (const t of niceTicks(0, hi, 5)) {
      svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: H - 22 }));
      svg.append(s("text", { class: "axis", x: x(t), y: H - 6, "text-anchor": "middle", text: num(t, 2) }));
    }
    rows.forEach((r, i) => {
      const y = top + i * rowH;
      const g = s("g", { class: "row", tabindex: 0 });
      g.append(s("rect", { class: "hit", x: 0, y, width: W, height: rowH }));
      g.append(s("text", { class: "row-label", x: L - 8, y: y + 15, "text-anchor": "end", text: r.label.length > 42 ? r.label.slice(0, 41) + "…" : r.label }));
      g.append(s("rect", { class: "bar " + (r.mean_signed_shap >= 0 ? "up" : "down"), x: L, y: y + 5, width: Math.max(1, x(r.mean_abs_shap) - L), height: rowH - 10, rx: 2 }));
      if (!missing(r.low) && !missing(r.high)) g.append(s("line", { class: "whisker", x1: x(r.low), x2: x(r.high), y1: y + rowH / 2, y2: y + rowH / 2 }));
      g.append(s("text", { class: "row-value", x: W - R + 8, y: y + 15, text: num(r.mean_abs_shap, 3) }));
      const tip = [
        ["mean |SHAP|", num(r.mean_abs_shap, 4)],
        ["95% interval", num(r.low, 4) + " to " + num(r.high, 4)],
        ["mean signed SHAP", signed(r.mean_signed_shap, 4)],
        ["rank (interval)", r.rank + " (" + r.rank_low + "–" + r.rank_high + ")"],
        ["block", r.block],
      ];
      if (r.unstable === true || r.unstable === "True") tip.push(["", "rank unstable (interval spans > 5 places)"]);
      for (const [ev, fn] of Object.entries(tipHandlers(r.label, tip))) g.addEventListener(ev, fn);
      svg.append(g);
    });
    return svg;
  }

  // ---- Model view -------------------------------------------------------
  const current = () => state.models.find((m) => m.key === state.key);

  function renderModelButtons() {
    put(
      "model-buttons",
      state.models.map((m) =>
        h("button", {
          type: "button",
          "aria-pressed": String(m.key === state.key),
          text: m.label ? m.dataset_label.split(" ")[0] + ": " + m.label : m.dataset_label + " (" + m.state + ")",
          on: { click: () => selectModel(m.key) },
        })
      )
    );
  }

  function selectModel(key) {
    state.key = key;
    state.history = [];
    renderModelButtons();
    renderModel();
    put("card");
    put("draw-history");
  }

  function renderModel() {
    const m = current();
    if (!m) return;
    if (m.state !== "ready") {
      put("model-state", m.state === "loading" ? h("p", { class: "notice neutral", text: "Loading this model's saved explanation…" }) : errorBox("Not available: " + m.message));
      for (const id of ["model-stats", "model-checks", "model-notes", "importance-chart", "blocks-table", "interactions-table", "correlated-box", "roar-table", "randomisation-table", "seeds-table", "sensitivity-table", "neighbours-body"]) put(id);
      $("model-lede").textContent = "";
      $("draw-button").disabled = true;
      return;
    }
    $("draw-button").disabled = false;
    put("model-state");
    $("model-lede").textContent =
      m.label + ", " + m.dataset_label + ", split 0, " + m.budget_label + ". " +
      "SHAP is on the model's log-odds, " + m.shap_mode + " TreeSHAP against " + int(m.n_background) + " background training nodes.";
    const stat = (k, v) => h("div", { class: "stat" }, h("div", { class: "k", text: k }), h("div", { class: "v", text: v }));
    put(
      "model-stats",
      stat("model", m.model + " (" + m.selection + ")"),
      stat("test nodes", int(m.n_test) + " (" + int(m.n_fraud) + " fraud)"),
      stat("threshold", probText(m.threshold)),
      stat("base value", num(m.base_value, 2) + " log-odds (" + probText(m.base_prob) + ")"),
      stat("input columns", int(m.n_columns))
    );
    put("model-checks", m.checks.map((c) => h("li", { class: c.ok ? "check-pass" : "check-fail" }, (c.ok ? "✓ " : "✗ ") + c.name + ": " + c.detail)));
    $("model-checks-box").open = !m.checks.every((c) => c.ok);
    put("model-notes", m.notes.map((n) => h("li", { text: n })));
    put("draw-buttons", m.draws.map((d) => h("button", { type: "button", "aria-pressed": String(d === state.draw), text: DRAW_LABELS[d] || d, on: { click: () => { state.draw = d; renderModel(); } } })));
    renderOverall(m);
    renderTests(m);
    renderNeighbours(m);
  }

  const DRAW_LABELS = {
    any: "any",
    fraud: "fraud",
    benign: "benign",
    true_positive: "caught fraud",
    false_positive: "false alarm",
    false_negative: "missed fraud",
  };

  function renderOverall(m) {
    const strata = ["all"].concat(Object.keys(m.strata).filter((k) => m.strata[k].length));
    if (!strata.includes(state.stratum)) state.stratum = "all";
    put("stratum-buttons", strata.map((k) => h("button", { type: "button", "aria-pressed": String(k === state.stratum), text: m.stratum_labels[k] || k, on: { click: () => { state.stratum = k; renderOverall(m); } } })));
    put("importance-legend",
      h("span", { class: "key" }, h("span", { class: "key-swatch up" }), "pushes towards fraud on average"),
      h("span", { class: "key" }, h("span", { class: "key-swatch down" }), "pushes away from fraud on average"),
      h("span", { class: "key small muted", text: "whisker: 95% interval" })
    );
    const rows = state.stratum === "all" ? m.importance : m.strata[state.stratum];
    put("importance-chart", importanceChart(rows || []));
    const groups = m.groups.slice().sort((a, b) => (a.label > b.label ? 1 : -1) || (a.measure > b.measure ? 1 : -1));
    put("blocks-table", table([
      ["block", "label"],
      ["measure", "measure"],
      ["columns", "n_columns", int, true],
      ["mean |SHAP|", "mean_abs_shap", (v) => num(v, 3), true],
      ["95% interval", (r) => num(r.low, 3) + " to " + num(r.high, 3), null, true],
      ["mean signed", "mean_signed_shap", (v) => signed(v, 3), true],
    ], groups));
    put("interactions-table", table([
      ["column a", "a"],
      ["column b", "b"],
      ["mean |interaction|", "mean_abs_interaction", (v) => num(v, 4), true],
      ["own × graph", "own_x_graph", yesNo],
    ], m.interactions));
    put("correlated-box", m.correlated.length ? [
      h("h3", { class: "sub spaced", text: "Issue #36's correlated columns as one group" }),
      h("p", { class: "small muted", text: "f13, f15, f18 and f19 share credit unpredictably, so their combined importance is shown too." }),
      h("div", { class: "table-scroll" }, table([["group", "name"], ["mean |SHAP|", "mean_abs_shap", (v) => num(v, 3), true], ["95% interval", (r) => num(r.low, 3) + " to " + num(r.high, 3), null, true], ["rank", "rank", null, true]], m.correlated)),
    ] : null);
  }

  function renderTests(m) {
    put("roar-table", table([
      ["k removed", "k", null, true],
      ["drop, top-SHAP k", "top_k_drop", (v) => num(v, 3), true],
      ["drop, random k", "random_drop", (v) => num(v, 3), true],
      ["gap", "gap", (v) => signed(v, 3), true],
      ["top-SHAP hurts more", "top_k_hurts_more", yesNo],
    ], m.roar));
    put("randomisation-table", table([
      ["fit", "run"],
      ["CV PR-AUC", "pr_auc", (v) => num(v, 3), true],
      ["positive rate", "prevalence", (v) => num(v, 3), true],
      ["total mean |SHAP|", "total_mean_abs", (v) => num(v, 3), true],
      ["rank agreement with real", "spearman_with_true", (v) => num(v, 2), true],
    ], m.randomisation));
    put("seeds-table", table([
      ["refit", "refit"],
      ["Kendall tau", "kendall_tau", (v) => num(v, 2), true],
      ["top-10 overlap", "top10_jaccard", (v) => num(v, 2), true],
      ["largest top-10 rank move", "top10_max_rank_move", null, true],
    ], m.seed_stability));
    put("sensitivity-table", table([
      ["compared with", "comparison"],
      ["mode", "feature_perturbation"],
      ["background", "n_background", int, true],
      ["Kendall tau", "kendall_tau", (v) => num(v, 2), true],
      ["top-10 overlap", "top10_jaccard", (v) => num(v, 2), true],
    ], m.sensitivity));
  }

  function renderNeighbours(m) {
    if (!m.has_influence) {
      put("neighbours-body", h("p", { class: "notice neutral", text:
        "Not computed for this model. " + (m.influence_note ||
        "Its graph information reaches the model through GNN embeddings, so removing a neighbour means recomputing the encoder; that was done only for the pilot-budget model (see docs/EXPLAINABILITY_PLAN.md).") }));
      return;
    }
    put("neighbours-body",
      h("div", { class: "panel panel-pad" },
        h("h3", { class: "sub", text: "By relation" }),
        h("p", { class: "small muted", text: "Mean |change in score| when one neighbour is removed, split into the part from the neighbour means and the part from the degree column. 'Outside training range' counts removals that leave a value the model never saw in training." }),
        h("div", { class: "table-scroll" }, table([
          ["relation", "relation_label"],
          ["test nodes", "n_targets", int, true],
          ["median degree", "median_target_degree", (v) => num(v, 0), true],
          ["mean |Δ score|", "mean_abs_delta_prob", (v) => num(v, 4), true],
          ["from means", "mean_abs_mean_part", (v) => num(v, 4), true],
          ["from degree", "mean_abs_degree_part", (v) => num(v, 4), true],
          ["outside training range", "share_out_of_range", (v) => pct(v), true],
        ], m.neighbour_relations))),
      h("div", { class: "grid-2" },
        h("div", { class: "panel panel-pad" },
          h("h3", { class: "sub", text: "Nodes whose removal moves most scores" }),
          h("p", { class: "small muted", text: "Summed |Δ score| over the test nodes each one neighbours. Hubs win on count, so the mean per test node is shown beside it." }),
          h("div", { class: "table-scroll" }, table([
            ["node", "neighbour", null, true],
            ["relation", "relation_label"],
            ["test nodes", "n_targets", int, true],
            ["sum |Δ|", "sum_abs", (v) => num(v, 3), true],
            ["mean |Δ|", "mean_abs", (v) => num(v, 4), true],
            ["towards fraud", "towards_fraud", int, true],
            ["away", "away_from_fraud", int, true],
          ], m.neighbour_top_nodes))),
        h("div", { class: "panel panel-pad" },
          h("h3", { class: "sub", text: "Does simple structure explain it?" }),
          h("p", { class: "small muted", text: "Rank correlation of a node's summed influence with its degree, and the overlap of the top 100 with the 100 highest-degree nodes. High values mean centrality alone reproduces the ranking." }),
          h("div", { class: "table-scroll" }, table([
            ["relation", (r) => r.relation.toUpperCase().split("").join("-")],
            ["vs degree", "spearman_sum_vs_degree", (v) => num(v, 2), true],
            ["vs flagged neighbours", "spearman_sum_vs_flagged_neighbours", (v) => num(v, 2), true],
            ["top-100 overlap", "top100_overlap_with_highest_degree", (v) => pct(v, 0), true],
          ], m.neighbour_structure)),
          h("h3", { class: "sub spaced", text: "Leave-one-out vs Shapley over neighbours" }),
          h("p", { class: "small muted", text: "On sampled test nodes with few neighbours: low agreement means neighbours are redundant, so leave-one-out undersells them." }),
          h("div", { class: "table-scroll" }, table([
            ["relation", (r) => r.relation.toUpperCase().split("").join("-")],
            ["test nodes", "n_targets", int, true],
            ["median rank agreement", "median_spearman", (v) => num(v, 2), true],
            ["same top neighbour", "share_same_top_neighbour", (v) => pct(v, 0), true],
          ], m.neighbour_redundancy))
        )
      )
    );
  }

  // ---- One node ---------------------------------------------------------
  const OUTCOMES = {
    true_positive: ["caught fraud", "correct"],
    false_positive: ["false alarm", "wrong"],
    false_negative: ["missed fraud", "wrong"],
    true_negative: ["correctly cleared", "correct"],
  };

  async function drawNode(seedOverride) {
    const m = current();
    if (!m || m.state !== "ready") return;
    const typed = $("seed-input").value.trim();
    let seed = seedOverride;
    if (seed === undefined && typed !== "") {
      if (!/^\d+$/.test(typed)) {
        put("card", errorBox("The seed must be a whole number."));
        return;
      }
      seed = Number(typed);
    }
    $("draw-button").disabled = true;
    try {
      const body = { model: m.key, draw: state.draw };
      if (seed !== undefined) body.seed = seed;
      const card = await api("/api/explain/node", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (card.model !== state.key) return;
      renderCard(m, card);
      if (seedOverride === undefined) {
        state.history.unshift(card);
        state.history = state.history.slice(0, 10);
        renderHistory(m);
      }
    } catch (e) {
      put("card", errorBox("Could not explain a node: " + e.message));
    } finally {
      $("draw-button").disabled = false;
    }
  }

  function renderHistory(m) {
    put("draw-history", state.history.length ? [h("span", { class: "small muted", text: "Earlier draws: " })].concat(
      state.history.map((c) => h("button", {
        type: "button",
        class: OUTCOMES[c.outcome][1],
        title: "draw " + c.draw + ", seed " + c.seed,
        text: String(c.node_id),
        on: { click: () => renderCard(m, c) },
      }))
    ) : []);
  }

  function renderCard(m, c) {
    const [outcomeText, outcomeClass] = OUTCOMES[c.outcome];
    const head = h("div", { class: "card-head" },
      h("span", { class: "id", text: "node " + c.node_id }),
      h("span", { class: "outcome " + outcomeClass, text: outcomeText }),
      h("span", { class: "small", text: "label: " + (c.label === 1 ? "fraud" : "benign") }),
      h("span", { class: "small", text: "score " + probText(c.prob) + (c.flagged ? " ≥ " : " < ") + "threshold " + probText(c.threshold) }),
      h("span", { class: "small muted", text: "drawn from " + (c.draw === "any" ? "all " + int(c.pool) + " test nodes" : int(c.pool) + " test nodes that are " + (DRAW_LABELS[c.draw] || c.draw)) + ", seed " + c.seed })
    );
    const contributions = table([
      ["column", "label"],
      ["block", "block"],
      ["input value", "value", (v) => (missing(v) ? "not shown" : num(v, 4)), true],
      ["SHAP", "shap", (v) => h("span", { class: v >= 0 ? "push-up" : "push-down", text: signed(v, 3) }), true],
      ["rank overall", "overall_rank", (v) => (missing(v) ? DASH : "#" + v), true],
    ], c.contributions);
    const blocks = table([
      ["block", "label"],
      ["summed SHAP", "summed_shap", (v) => h("span", { class: v >= 0 ? "push-up" : "push-down", text: signed(v, 3) }), true],
      ["group Shapley", "group_shapley", (v) => (missing(v) ? DASH : signed(v, 3)), true],
    ], c.blocks);
    const nearest = table([
      ["training node", "node_id", null, true],
      ["label", "label", (v) => (v === 1 ? "fraud" : "benign")],
      ["model score", "prob", probText, true],
      ["SHAP distance", "distance", (v) => num(v, 2), true],
    ], c.nearest_training);
    put("card",
      h("div", { class: "panel panel-pad result" },
        head,
        h("p", { class: "small muted", text:
          "Read the chart from the top: the score starts at the base value, each bar pushes it towards fraud (orange) or away (blue), and it ends at this node's score. " +
          "The bars add up exactly (gap " + c.additivity_gap.toExponential(1) + " log-odds). Values are log-odds; the dashed line is the threshold." }),
        h("div", { class: "chart-scroll" }, waterfall(c)),
        h("div", { class: "card-grid" },
          h("div", {},
            h("h3", { class: "sub", text: "The " + c.contributions.length + " largest pushes" }),
            h("p", { class: "small muted", text: "Input values are df-analyze's prepared inputs (after its scaling), not raw features." }),
            h("div", { class: "table-scroll" }, contributions)),
          h("div", {},
            h("h3", { class: "sub", text: "By block" }),
            h("p", { class: "small muted", text: c.in_group_shapley_sample
              ? "This node is in the group-Shapley sample, so each block's exact value as one player is shown too."
              : "Group Shapley was computed on 1,000 sampled test nodes; this one is not among them." }),
            h("div", { class: "table-scroll" }, blocks),
            h("h3", { class: "sub spaced", text: "Most similar training nodes" }),
            h("p", { class: "small muted", text: "Closest in SHAP space: training nodes whose scores were made up the same way. Their labels are training labels." }),
            h("div", { class: "table-scroll" }, nearest))
        ),
        neighbourSection(m, c)
      )
    );
  }

  function neighbourSection(m, c) {
    const box = h("div", { class: "spaced-top" }, h("h3", { class: "sub", text: "Neighbours whose removal moves this score most" }));
    if (!c.neighbours.available) {
      box.append(h("p", { class: "small muted", text: "Not computed for this model (see 'Neighbour influence overall' below)." }));
      return box;
    }
    box.append(h("p", { class: "small muted", text:
      "Δ score = this node's score minus its score with that neighbour removed and its neighbour means recomputed. Positive: the neighbour pushes it towards fraud. " +
      "The neighbour's own score is the model's, never its label." }));
    for (const r of c.neighbours.relations) {
      const block = h("div", { class: "rel-block" }, h("h4", { text: r.label + " · degree " + int(r.degree) + (r.n_computed && r.n_computed < r.degree ? " (" + int(r.n_computed) + " computed)" : "") + " · total |Δ| " + num(r.total_abs, 4) }));
      if (!r.top.length) block.append(h("p", { class: "small muted", text: "No neighbours in this relation." }));
      else block.append(h("div", { class: "table-scroll" }, table([
        ["neighbour", "neighbour", null, true],
        ["Δ score", "delta_prob", (v) => h("span", { class: v >= 0 ? "push-up" : "push-down", text: signed(v, 4) }), true],
        ["from means", "mean_part", (v) => signed(v, 4), true],
        ["from degree", "degree_part", (v) => signed(v, 4), true],
        ["neighbour's score", "neighbour_score", probText, true],
        ["", "out_of_range", (v) => (v ? h("span", { class: "badge warn", text: "outside training range" }) : "")],
      ], r.top)));
      box.append(block);
    }
    return box;
  }

  // ---- Start ------------------------------------------------------------
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
      put("model-state", errorBox("Could not reach the server: " + e.message));
      return;
    }
    const before = current();
    state.models = answer.models;
    if (!state.key) {
      const ready = state.models.find((m) => m.state === "ready");
      state.key = (ready || state.models[0]).key;
    }
    $("header-meta").textContent = "commit " + (answer.commit || DASH);
    put("page-footer", h("p", { class: "small muted", text: "Built from the saved outputs of scripts/dfa/shap_explain.py and scripts/explain_report.py, split 0. Row-level: group only." }));
    renderModelButtons();
    const now = current();
    if (!before || before.state !== now.state) renderModel();
    if (state.models.some((m) => m.state === "loading")) setTimeout(poll, 1500);
  }

  setupTheme();
  $("draw-button").addEventListener("click", () => drawNode());
  $("seed-input").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      e.preventDefault();
      drawNode();
    }
  });
  poll();
})();
