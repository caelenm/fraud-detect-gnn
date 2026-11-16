/* The GAT attention page. No libraries and no external requests: the heatmap
 * and the maps are SVG built here, and every value reaches the page as a text
 * node or an attribute, never as markup. */
"use strict";

(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const state = { models: [], key: null, draw: "any", layer: 1, head: "mean", card: null, history: [] };

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
  const pct = (v, d = 0) => (missing(v) ? DASH : (100 * v).toFixed(d) + "%");
  const int = (v) => (missing(v) ? DASH : Number(v).toLocaleString("en"));
  const times = (v, d = 1) => (missing(v) ? DASH : Number(v).toFixed(d) + "×");
  function probText(v) {
    if (missing(v)) return DASH;
    if (v === 0 || v === 1) return String(v);
    if (v < 0.001) return v.toExponential(1);
    if (v > 0.999) return "1 " + MINUS + " " + (1 - v).toExponential(1);
    return v.toFixed(3);
  }
  function weightText(v) {
    if (missing(v)) return DASH;
    if (v >= 0.995) return "1.00";
    if (v >= 0.01) return v.toFixed(2);
    return v.toExponential(0);
  }

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
  function withTip(node, title, rows) {
    node.setAttribute("tabindex", "0");
    node.addEventListener("pointermove", (e) => showTip(e, title, rows));
    node.addEventListener("pointerleave", hideTip);
    node.addEventListener("focus", (e) => showTip(e, title, rows));
    node.addEventListener("blur", hideTip);
    return node;
  }

  // ---- Heatmap ----------------------------------------------------------
  // Shading: how far a value is from what equal weights give. Ratios are
  // shaded on a log2 scale (2× and ½× look equally strong); focus by 1 − focus.
  const shade = (strength) => Math.max(0.06, Math.min(0.85, strength));
  const ratioShade = (r) => (missing(r) || r <= 0 ? 0 : shade(Math.abs(Math.log2(r)) / 2));
  const HEAT_COLUMNS = [
    {
      label: ["focus", "(1 = equal weights)"],
      value: (r) => r.focus,
      text: (r) => num(r.focus, 2),
      cls: () => "heat-down",
      strength: (r) => (missing(r.focus) ? 0 : shade((1 - r.focus) * 4)),
      tip: (r, m) => [["focus, head-averaged", num(r.focus, 3)]].concat(
        r.heads.map((f, i) => ["head " + (i + 1), num(f, 3)])
      ),
    },
    {
      label: ["top weight", "vs 1 / degree"],
      value: (r) => r.top_vs_equal,
      text: (r) => times(r.top_vs_equal),
      cls: (r) => (r.top_vs_equal >= 1 ? "heat-up" : "heat-down"),
      strength: (r) => ratioShade(r.top_vs_equal),
      tip: (r) => [
        ["largest weight (mean)", num(r.max_alpha, 4)],
        ["1 / degree (mean)", num(r.uniform_max, 4)],
        ["mean degree", num(r.mean_degree, 1)],
      ],
    },
    {
      label: ["relation term", "vs own-features term"],
      value: (r) => r.relation_to_root,
      text: (r) => times(r.relation_to_root),
      cls: (r) => (r.relation_to_root >= 1 ? "heat-up" : "heat-down"),
      strength: (r) => ratioShade(r.relation_to_root),
      tip: (r) => [["‖relation term‖ / ‖own term‖", num(r.relation_to_root, 3)], ["", "above 1: this layer leans on the relation more than on the node itself"]],
    },
    {
      label: ["fraud neighbours' share", "fraud targets"],
      value: (r) => r.fraud_ratio_fraud,
      text: (r) => times(r.fraud_ratio_fraud, 2),
      cls: (r) => (r.fraud_ratio_fraud >= 1 ? "heat-up" : "heat-down"),
      strength: (r) => ratioShade(r.fraud_ratio_fraud),
      tip: (r) => [
        ["attention on fraud neighbours", num(r.fraud_attention_fraud, 3)],
        ["their share of labelled neighbours", num(r.fraud_uniform_fraud, 3)],
        ["training fraud nodes with both kinds", int(r.fraud_n_fraud)],
      ],
    },
    {
      label: ["fraud neighbours' share", "benign targets"],
      value: (r) => r.fraud_ratio_benign,
      text: (r) => times(r.fraud_ratio_benign, 2),
      cls: (r) => (r.fraud_ratio_benign >= 1 ? "heat-up" : "heat-down"),
      strength: (r) => ratioShade(r.fraud_ratio_benign),
      tip: (r) => [
        ["attention on fraud neighbours", num(r.fraud_attention_benign, 3)],
        ["their share of labelled neighbours", num(r.fraud_uniform_benign, 3)],
        ["training benign nodes with both kinds", int(r.fraud_n_benign)],
      ],
    },
  ];

  function heatmap(m) {
    const rows = m.heat;
    if (!rows.length) return h("p", { class: "empty small", text: "No summary rows." });
    const L = 120, cellW = 132, cellH = 34, top = 44;
    const W = L + cellW * HEAT_COLUMNS.length + 4;
    const H = top + cellH * rows.length + 6;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "heat", role: "img", "aria-label": "Attention by layer and relation" });
    HEAT_COLUMNS.forEach((col, j) => {
      const x = L + j * cellW + cellW / 2;
      svg.append(s("text", { class: "col-label", x, y: 16, "text-anchor": "middle", text: col.label[0] }));
      svg.append(s("text", { class: "col-sub", x, y: 32, "text-anchor": "middle", text: col.label[1] }));
    });
    rows.forEach((r, i) => {
      const y = top + i * cellH;
      if (i > 0 && r.layer !== rows[i - 1].layer) svg.append(s("line", { class: "layer-rule", x1: 0, x2: W, y1: y, y2: y }));
      svg.append(s("text", { class: "row-label", x: L - 10, y: y + cellH / 2 + 5, "text-anchor": "end", text: "layer " + r.layer + " · " + r.relation_label }));
      HEAT_COLUMNS.forEach((col, j) => {
        const value = col.value(r);
        const g = s("g", { class: "cell" });
        g.append(s("rect", { class: "heat-bg", x: L + j * cellW + 2, y: y + 2, width: cellW - 4, height: cellH - 4, rx: 4 }));
        if (!missing(value)) {
          g.append(s("rect", { class: col.cls(r), x: L + j * cellW + 2, y: y + 2, width: cellW - 4, height: cellH - 4, rx: 4, "fill-opacity": col.strength(r).toFixed(3) }));
        }
        g.append(s("text", { class: "cell-value", x: L + j * cellW + cellW / 2, y: y + cellH / 2 + 5, "text-anchor": "middle", text: missing(value) ? DASH : col.text(r) }));
        withTip(g, "Layer " + r.layer + ", " + r.relation_label + ": " + col.label.join(", "), col.tip(r, m));
        svg.append(g);
      });
    });
    return svg;
  }

  // ---- Node maps --------------------------------------------------------
  const KIND_LABELS = {
    train_fraud: "training node, fraud",
    train_benign: "training node, benign",
    test: "test node (label not shown)",
    unlabelled: "unlabelled node",
  };

  const weightOf = (n) => (state.head === "mean" ? n.alpha : n.heads[state.head]);

  function egoMap(m, rel) {
    const W = 340, H = 340, cx = W / 2, cy = H / 2, R = 118;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: "ego", role: "img", "aria-label": rel.relation_label + " attention map" });
    const ns = rel.neighbours;
    if (!ns.length) {
      svg.append(s("circle", { class: "target", cx, cy, r: 14 }));
      svg.append(s("text", { class: "empty-note", x: cx, y: cy + 40, "text-anchor": "middle", text: "no neighbours in this relation" }));
      return svg;
    }
    const placed = ns.map((n, i) => {
      const angle = -Math.PI / 2 + (2 * Math.PI * i) / Math.max(ns.length, 3);
      return { n, x: cx + R * Math.cos(angle), y: cy + R * Math.sin(angle), angle };
    });
    // Equal weight, for reference: a dashed ring whose label gives 1 / degree.
    svg.append(s("circle", { class: "ring", cx, cy, r: R }));
    for (const p of placed) {
      const w = weightOf(p.n);
      svg.append(s("line", { class: "edge " + p.n.kind, x1: p.x, y1: p.y, x2: cx, y2: cy, "stroke-width": (0.8 + 18 * Math.sqrt(Math.max(w, 0))).toFixed(2) }));
    }
    svg.append(s("circle", { class: "target", cx, cy, r: 14 }));
    for (const p of placed) {
      const w = weightOf(p.n);
      const g = s("g", { class: "nbr" });
      g.append(s("circle", { class: "node " + p.n.kind, cx: p.x, cy: p.y, r: 9 }));
      const lx = cx + (R + 26) * Math.cos(p.angle);
      const ly = cy + (R + 26) * Math.sin(p.angle) + 4;
      g.append(s("text", { class: "weight", x: lx, y: ly, "text-anchor": "middle", text: weightText(w) }));
      const rows = [
        ["weight " + (state.head === "mean" ? "(mean of heads)" : "(head " + (state.head + 1) + ")"), num(w, 4)],
        ["equal weight (1 / degree)", num(rel.equal_weight, 4)],
        ["rank by mean weight", "#" + p.n.rank + " of " + int(rel.degree)],
        ["kind", KIND_LABELS[p.n.kind] || p.n.kind],
      ].concat(p.n.heads.map((x, i) => ["head " + (i + 1), num(x, 4)]));
      withTip(g, "neighbour " + p.n.node_id, rows);
      svg.append(g);
    }
    return svg;
  }

  function relationCaption(rel) {
    const parts = [int(rel.degree) + (rel.degree === 1 ? " neighbour" : " neighbours")];
    if (rel.degree > rel.n_shown) parts.push("the top " + rel.n_shown + " carry " + pct(rel.shown_weight) + " of the weight");
    if (!missing(rel.equal_weight)) parts.push("equal weight " + weightText(rel.equal_weight));
    if (!missing(rel.focus)) parts.push("focus " + num(rel.focus, 2));
    if (!missing(rel.relation_to_root)) parts.push("relation term " + times(rel.relation_to_root) + " the own term");
    return parts.join(" · ");
  }

  // ---- Model view -------------------------------------------------------
  const current = () => state.models.find((m) => m.key === state.key);

  function renderModelButtons() {
    put("model-buttons", state.models.map((m) =>
      h("button", {
        type: "button",
        "aria-pressed": String(m.key === state.key),
        text: m.dataset_label + (m.state === "ready" ? "" : " (" + m.state + ")"),
        on: { click: () => selectModel(m.key) },
      })
    ));
  }

  function selectModel(key) {
    state.key = key;
    state.history = [];
    state.card = null;
    renderModelButtons();
    renderModel();
    put("card");
    put("draw-history");
  }

  const DRAW_LABELS = {
    any: "any",
    fraud: "fraud",
    benign: "benign",
    true_positive: "caught fraud",
    false_positive: "false alarm",
    false_negative: "missed fraud",
  };

  function renderModel() {
    const m = current();
    if (!m) return;
    if (m.state !== "ready") {
      put("model-state", m.state === "loading" ? h("p", { class: "notice neutral", text: "Loading the saved attention read-out…" }) : errorBox("Not available: " + m.message));
      for (const id of ["model-stats", "model-checks", "heatmap", "heat-legend", "draw-buttons"]) put(id);
      $("model-lede").textContent = "";
      $("draw-button").disabled = true;
      return;
    }
    $("draw-button").disabled = false;
    put("model-state");
    $("model-lede").textContent =
      m.label + ", " + m.dataset_label + ", split 0. " + m.layers + " layers, " + m.heads + " heads per relation, hidden " +
      m.setting.hidden + ", dropout " + m.setting.dropout + ", learning rate " + m.setting.lr + ".";
    const stat = (k, v) => h("div", { class: "stat" }, h("div", { class: "k", text: k }), h("div", { class: "v", text: v }));
    put("model-stats",
      stat("refit", "fold " + m.fold_model + " model, best epoch " + m.best_epoch),
      stat("held-out fold PR-AUC", num(m.refit_fold_pr_auc, 4) + " (run " + num(m.saved_fold_pr_auc, 4) + ")"),
      stat("test nodes", int(m.n_test) + " (" + int(m.n_fraud) + " fraud)"),
      stat("threshold", probText(m.threshold)),
      stat("neighbours kept", "top " + m.top_k + " per relation")
    );
    put("model-checks", m.checks.map((c) => h("li", { class: c.ok ? "check-pass" : "check-fail" }, (c.ok ? "✓ " : "✗ ") + c.name + ": " + c.detail)));
    $("model-checks-box").open = !m.checks.every((c) => c.ok);
    put("heat-legend",
      h("span", { class: "key" }, h("span", { class: "key-swatch up" }), "above what equal weights give"),
      h("span", { class: "key" }, h("span", { class: "key-swatch down" }), "below it, or more focused"),
      h("span", { class: "key small muted", text: "darker: further from equal weights" })
    );
    put("heatmap", heatmap(m));
    put("draw-buttons", m.draws.map((d) => h("button", { type: "button", "aria-pressed": String(d === state.draw), text: DRAW_LABELS[d] || d, on: { click: () => { state.draw = d; renderModel(); } } })));
  }

  // ---- One node ---------------------------------------------------------
  const OUTCOMES = {
    true_positive: ["caught fraud", "correct"],
    false_positive: ["false alarm", "wrong"],
    false_negative: ["missed fraud", "wrong"],
    true_negative: ["correctly cleared", "correct"],
  };

  async function drawNode() {
    const m = current();
    if (!m || m.state !== "ready") return;
    const typed = $("seed-input").value.trim();
    let seed;
    if (typed !== "") {
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
      const card = await api("/api/attention/node", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (card.model !== state.key) return;
      state.card = card;
      renderCard(m, card);
      state.history.unshift(card);
      state.history = state.history.slice(0, 10);
      renderHistory(m);
    } catch (e) {
      put("card", errorBox("Could not draw a node: " + e.message));
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
        on: { click: () => { state.card = c; renderCard(m, c); } },
      }))
    ) : []);
  }

  function toggles(m, c) {
    const layerButtons = c.layers.map((l) => h("button", {
      type: "button", "aria-pressed": String(l.layer === state.layer), text: "layer " + l.layer,
      on: { click: () => { state.layer = l.layer; renderCard(m, c); } },
    }));
    const heads = ["mean"].concat([...Array(m.heads).keys()]);
    const headButtons = heads.map((hd) => h("button", {
      type: "button", "aria-pressed": String(hd === state.head), text: hd === "mean" ? "mean of heads" : "head " + (hd + 1),
      on: { click: () => { state.head = hd; renderCard(m, c); } },
    }));
    return h("div", { class: "map-toggles" },
      h("div", { class: "segmented", role: "group", "aria-label": "Layer" }, layerButtons),
      h("div", { class: "segmented wrap-buttons", role: "group", "aria-label": "Head" }, headButtons));
  }

  function mapLegend() {
    const key = (cls, text) => h("span", { class: "key" }, h("span", { class: "key-dot " + cls }), text);
    return h("div", { class: "legend" },
      key("train_fraud", "training node, fraud"),
      key("train_benign", "training node, benign"),
      key("test", "test node (label not shown)"),
      key("unlabelled", "unlabelled"),
      h("span", { class: "key small muted", text: "line width: attention weight; number: the weight" }));
  }

  function renderCard(m, c) {
    if (!c.layers.some((l) => l.layer === state.layer)) state.layer = 1;
    if (state.head !== "mean" && state.head >= m.heads) state.head = "mean";
    const [outcomeText, outcomeClass] = OUTCOMES[c.outcome];
    const head = h("div", { class: "card-head" },
      h("span", { class: "id", text: "node " + c.node_id }),
      h("span", { class: "outcome " + outcomeClass, text: outcomeText }),
      h("span", { class: "small", text: "label: " + (c.label === 1 ? "fraud" : "benign") }),
      h("span", { class: "small", text: "run score " + probText(c.prob) + (c.flagged ? " ≥ " : " < ") + "threshold " + probText(c.threshold) }),
      h("span", { class: "small", text: "refit fold-0 model " + probText(c.refit_prob) }),
      h("span", { class: "small muted", text: "drawn from " + (c.draw === "any" ? "all " + int(c.pool) + " test nodes" : int(c.pool) + " test nodes that are " + (DRAW_LABELS[c.draw] || c.draw)) + ", seed " + c.seed })
    );
    const layer = c.layers.find((l) => l.layer === state.layer);
    const maps = layer.relations.map((rel) =>
      h("figure", { class: "ego-box" },
        h("figcaption", {}, h("b", { text: rel.relation_label }), h("span", { class: "small muted", text: " " + relationCaption(rel) })),
        egoMap(m, rel)));
    put("card",
      h("div", { class: "panel panel-pad result" },
        head,
        h("p", { class: "small muted", text:
          "The drawn node is in the centre of each map, with its strongest " + m.top_k + " neighbours in that relation around it, strongest at the top and then clockwise. " +
          "Ranks use the mean over heads; picking one head re-weights the same neighbours. The dashed ring has no meaning beyond placement." }),
        toggles(m, c),
        mapLegend(),
        h("div", { class: "ego-grid" }, maps)
      )
    );
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
      answer = await api("/api/attention/models");
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
    put("page-footer", h("p", { class: "small muted", text: "Built from the saved outputs of scripts/gat_attention.py, split 0. Row-level: group only." }));
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
