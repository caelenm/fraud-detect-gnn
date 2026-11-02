/* The dashboard's page. No libraries and no external requests: charts are SVG
 * built here, and every value reaches the page as a text node, never as markup.
 * The one exception is the neighbourhood pictures, which the server builds with
 * compare.ego_svg (values escaped there), parsed as SVG, never as HTML. */
"use strict";

(function () {
  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);
  const state = { ds: "amazon", variant: "all", budget: "pilot", zoom: false };
  let meta = null;
  let lastMethods = null;
  let loadToken = 0;

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
  function errorBox(error) {
    return h("div", { class: "error", text: "Could not load this section: " + error.message });
  }

  // ---- Number formats ---------------------------------------------------
  const MINUS = "−";
  const num = (v, d = 3) => (v === null || v === undefined ? "—" : Number(v).toFixed(d));
  function signed(v, d = 3) {
    if (v === null || v === undefined) return "—";
    const text = Math.abs(v).toFixed(d);
    if (Number(text) === 0) return (v < 0 ? MINUS : "+") + text;
    return (v < 0 ? MINUS : "+") + text;
  }
  const meanSd = (mean, sd, d = 3) =>
    num(mean, d) + (sd === null || sd === undefined ? "" : " ± " + num(sd, d));
  function pValue(p) {
    if (p === null || p === undefined) return "—";
    if (p < 0.001) return "< 0.001";
    return p.toPrecision(2);
  }
  function probText(v) {
    if (v === 0 || v === 1) return String(v);
    if (v < 0.001) return v.toExponential(1);
    if (v > 0.999) return "1 " + MINUS + " " + (1 - v).toExponential(1);
    return v.toFixed(3);
  }
  const VERDICT_ICONS = {
    supported: "✓",
    not_distinguishable: "≈",
    worse_significant: "▼",
    worse_mean: "▽",
    descriptive: "·",
  };
  const verdictChip = (kind, label) =>
    h("span", { class: "verdict " + kind }, VERDICT_ICONS[kind] || "", " ", label);

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
  const tooltip = () => $("tooltip");
  function showTip(event, title, rows) {
    const tip = tooltip();
    tip.replaceChildren(
      h("div", { class: "tip-title", text: title }),
      ...rows.map(([label, value]) =>
        h("div", { class: "tip-row" }, h("span", { text: label }), h("b", { text: value }))
      )
    );
    tip.hidden = false;
    const box = tip.getBoundingClientRect();
    let x, y;
    if (event.type === "focus") {
      const r = event.target.getBoundingClientRect();
      x = r.left + r.width / 2;
      y = r.top;
    } else {
      x = event.clientX + 14;
      y = event.clientY + 14;
    }
    tip.style.left = Math.max(8, Math.min(x, window.innerWidth - box.width - 8)) + "px";
    tip.style.top = Math.max(8, Math.min(y, window.innerHeight - box.height - 8)) + "px";
  }
  function hideTip() {
    tooltip().hidden = true;
  }
  function tipHandlers(title, rows) {
    return {
      pointermove: (e) => showTip(e, title, rows),
      pointerleave: hideTip,
      focus: (e) => showTip(e, title, rows),
      blur: hideTip,
    };
  }

  // ---- Controls and URL state -------------------------------------------
  function readUrl() {
    const q = new URLSearchParams(window.location.search);
    if (q.get("ds")) state.ds = q.get("ds");
    if (q.get("variant")) state.variant = q.get("variant");
    if (q.get("budget")) state.budget = q.get("budget");
    state.zoom = q.get("zoom") === "1";
  }
  function writeUrl() {
    const q = new URLSearchParams({ ds: state.ds, variant: state.variant, budget: state.budget });
    if (state.zoom) q.set("zoom", "1");
    window.history.replaceState(null, "", "?" + q.toString());
  }
  function datasetInfo(name) {
    return meta.datasets.find((d) => d.name === name);
  }
  function normaliseState() {
    if (!datasetInfo(state.ds)) state.ds = meta.datasets[0].name;
    const variants = datasetInfo(state.ds).variants.map((v) => v.name);
    if (!variants.includes(state.variant)) state.variant = meta.default_variant;
    if (!meta.budgets.some((b) => b.name === state.budget)) state.budget = meta.default_budget;
  }
  function renderControls() {
    put(
      "dataset-buttons",
      meta.datasets.map((d) =>
        h("button", {
          type: "button",
          "aria-pressed": d.name === state.ds ? "true" : "false",
          text: d.label,
          on: {
            click: () => {
              state.ds = d.name;
              normaliseState();
              renderControls();
              loadDatasetViews();
            },
          },
        })
      )
    );
    const info = datasetInfo(state.ds);
    const select = $("variant-select");
    select.replaceChildren(
      ...info.variants.map((v) =>
        h("option", { value: v.name, selected: v.name === state.variant, text: v.label })
      )
    );
    $("variant-control").hidden = info.variants.length < 2;
    const budget = $("budget-select");
    budget.replaceChildren(
      ...meta.budgets.map((b) =>
        h("option", { value: b.name, selected: b.name === state.budget, text: b.label })
      )
    );
    $("zoom-toggle").checked = state.zoom;
    writeUrl();
  }
  function wireControls() {
    $("controls").addEventListener("submit", (e) => e.preventDefault());
    $("variant-select").addEventListener("change", (e) => {
      state.variant = e.target.value;
      writeUrl();
      syncAmazonPanel();
      loadDatasetViews();
    });
    $("budget-select").addEventListener("change", (e) => {
      state.budget = e.target.value;
      writeUrl();
      loadDatasetViews();
    });
    $("zoom-toggle").addEventListener("change", (e) => {
      state.zoom = e.target.checked;
      writeUrl();
      if (lastMethods) renderLeaderboard(lastMethods);
    });
    const themes = ["auto", "light", "dark"];
    let theme = "auto";
    try {
      theme = localStorage.getItem("dashboard-theme") || "auto";
    } catch (e) {
      theme = "auto";
    }
    function applyTheme() {
      if (theme === "auto") document.documentElement.removeAttribute("data-theme");
      else document.documentElement.setAttribute("data-theme", theme);
      $("theme-toggle").textContent = "Theme: " + theme;
    }
    $("theme-toggle").addEventListener("click", () => {
      theme = themes[(themes.indexOf(theme) + 1) % themes.length];
      try {
        localStorage.setItem("dashboard-theme", theme);
      } catch (e) {
        /* the setting just lasts for this page */
      }
      applyTheme();
    });
    applyTheme();
  }

  // ---- Header and "read this first" ---------------------------------------
  function renderHeader() {
    const o = meta.overview;
    const rows = [
      ["Runs", String(o.n_runs)],
      ["Period", (o.first_started || "").slice(0, 10) + " to " + (o.last_started || "").slice(0, 10)],
      ["Commit", (meta.commit || "unknown").slice(0, 9)],
    ];
    put("header-meta", rows.flatMap(([k, v]) => [h("span", { text: k }), h("b", { text: v })]));
    put("general-caveats", meta.general_caveats.map((t) => h("li", { text: t })));
    $("verdict-rule").textContent = meta.verdict_rule;
    put("not-run", meta.not_run.map((t) => h("li", { text: t })));
    put("demo-notes", meta.demo_notes.map((t) => h("li", { text: t })));
    const banner = $("check-banner");
    banner.hidden = meta.failed_checks.length === 0;
    if (meta.failed_checks.length) {
      banner.textContent =
        meta.failed_checks.length + " of " + meta.n_checks +
        " sanity checks failed, so do not trust this page until they are fixed: " +
        meta.failed_checks.join("; ") + ".";
    }
    put(
      "page-footer",
      h("span", {
        text: "Built from results/runs/ at commit " + (meta.commit || "unknown") +
          ". Row-level data on this page stays on this machine: the server binds 127.0.0.1 and makes no external requests.",
      })
    );
  }

  // ---- Key questions -------------------------------------------------------
  function earlierNote(row) {
    if (!row.replaces || !row.replaces.length) return null;
    const text = row.replaces
      .map((r) => r.n_splits + " splits: " + r.verdict_label + " (" + signed(r.mean) + ", p " + pValue(r.p) + ")")
      .join("; ");
    return h("span", { class: "earlier", text: "Replaces " + text });
  }
  function renderQuestions(contrasts, methods) {
    $("questions-lede").textContent =
      "On " + datasetInfo(state.ds).label + ", " + methods.variant_label +
      ". Pilot budget. A question with no row was not run for this view.";
    if (!contrasts.questions.length) {
      put("questions-body", h("div", { class: "panel empty", text: "No pre-registered contrast answers these questions for this view." }));
      return;
    }
    const table = h(
      "table",
      {},
      h("thead", {}, h("tr", {},
        h("th", { class: "wrap-cell", text: "Question" }),
        h("th", { class: "l", text: "Contrast" }),
        h("th", { text: "Test Δ PR-AUC" }),
        h("th", { text: "Splits" }),
        h("th", { text: "CV p" }),
        h("th", { class: "l", text: "Verdict" })
      )),
      h("tbody", {}, contrasts.questions.map((q) =>
        h("tr", {},
          h("td", { class: "wrap-cell" }, q.question, earlierNote(q)),
          h("td", { class: "l", text: q.b_id + " " + MINUS + " " + q.a_id }),
          h("td", { text: signed(q.mean) + (q.sd === null ? "" : " ± " + num(q.sd)) }),
          h("td", { text: q.n_splits }),
          h("td", { text: pValue(q.p) }),
          h("td", { class: "l" }, verdictChip(q.verdict_kind, q.verdict_label))
        )
      ))
    );
    put("questions-body", h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, table)));
  }

  // ---- Section 1: the leaderboard ------------------------------------------
  function glyph(kind, cls) {
    const svg = s("svg", { width: 22, height: 14, viewBox: "0 0 22 14", "aria-hidden": "true" });
    if (kind === "mean") {
      svg.append(s("line", { x1: 2, x2: 20, y1: 7, y2: 7, class: "interval " + cls }));
      svg.append(s("circle", { cx: 11, cy: 7, r: 5, class: "mean " + cls }));
    } else if (kind === "split") {
      svg.append(s("circle", { cx: 11, cy: 7, r: 3, class: "split-dot " + cls }));
    } else if (kind === "cv") {
      svg.append(s("path", { d: "M11 2 L16 7 L11 12 L6 7 Z", class: "cv-mark" }));
    } else if (kind === "ref") {
      svg.append(s("line", { x1: 11, x2: 11, y1: 0, y2: 14, class: "ref" }));
    } else if (kind === "square") {
      svg.append(s("rect", { x: 6, y: 2, width: 10, height: 10, rx: 1, class: "mean " + cls }));
    } else if (kind === "triangle") {
      svg.append(s("path", { d: "M11 1 L17 12 L5 12 Z", class: "mean " + cls }));
    }
    return svg;
  }
  const legendKey = (kind, cls, text) => h("span", { class: "key" }, glyph(kind, cls), text);

  function ticks(lo, hi, count) {
    const raw = (hi - lo) / count;
    const steps = [0.005, 0.01, 0.02, 0.025, 0.05, 0.1, 0.2, 0.25, 0.5];
    const step = steps.find((v) => v >= raw) || raw;
    const out = [];
    const first = Math.ceil((lo - 1e-9) / step);
    for (let i = first; i * step <= hi + 1e-9; i++) out.push(Number((i * step).toFixed(4)));
    return out;
  }
  function decimals(values) {
    return values.every((v) => Math.abs(v * 10 - Math.round(v * 10)) < 1e-6) ? 1
      : values.every((v) => Math.abs(v * 100 - Math.round(v * 100)) < 1e-6) ? 2 : 3;
  }

  function renderLeaderboard(view) {
    const rows = view.rows;
    if (!rows.length) {
      put("leaderboard-legend");
      put("leaderboard", h("p", { class: "empty", text: "No run at this budget for this view. Try the pilot budget." }));
      return;
    }
    const noSkill = rows[0].no_skill;
    const all = rows.flatMap((r) => [...r.test_per_split, ...r.cv_per_split]);
    let lo = 0, hi = 1;
    if (state.zoom) {
      lo = Math.max(0, Math.floor((Math.min(...all) - 0.005) / 0.01) * 0.01);
      hi = Math.min(1, Math.ceil((Math.max(...all) + 0.005) / 0.01) * 0.01);
    }
    const W = 1000, L = 390, R = 70, top = 34, rowH = 30, groupH = 26, axisH = 34;
    const families = view.families.filter((f) => rows.some((r) => r.family === f));
    const height = top + rows.length * rowH + families.length * groupH + axisH;
    const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    const svg = s("svg", {
      viewBox: "0 0 " + W + " " + height,
      role: "img",
      "aria-label": "Test and CV PR-AUC per setup. The table below has every value.",
    });
    const tickValues = ticks(lo, hi, 8);
    const d = decimals(tickValues);
    const bottom = height - axisH;
    for (const t of tickValues) {
      svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top - 6, y2: bottom }));
      svg.append(s("text", { x: x(t), y: bottom + 18, "text-anchor": "middle", text: t.toFixed(d) }));
    }
    svg.append(s("text", { x: (L + W - R) / 2, y: bottom + 32, "text-anchor": "middle", text: state.zoom ? "PR-AUC (axis zoomed: it does not start at 0)" : "PR-AUC" }));
    svg.append(s("text", { x: W - R + 8, y: top - 14, text: "test" , class: "row-label" }));
    if (noSkill >= lo && noSkill <= hi) {
      svg.append(s("line", { class: "ref", x1: x(noSkill), x2: x(noSkill), y1: top - 6, y2: bottom }));
      svg.append(s("text", { class: "ref-label", x: x(noSkill) + 5, y: top - 14, text: "no skill " + num(noSkill) }));
    } else {
      svg.append(s("text", { class: "ref-label", x: L, y: top - 14, text: "no skill " + num(noSkill) + " is left of this axis" }));
    }
    let y = top;
    const best = view.best && view.best.budget === view.budget ? view.best.feature_set : null;
    for (const family of families) {
      svg.append(s("text", { class: "group-title", x: 0, y: y + 17, text: family }));
      y += groupH;
      for (const r of rows.filter((row) => row.family === family)) {
        const cy = y + rowH / 2;
        const tip = [
          ["Test PR-AUC, mean ± SD", meanSd(r.test_pr_auc, r.test_pr_auc_sd)],
          ["CV PR-AUC, mean", num(r.cv_pr_auc)],
          ["Per split", r.test_per_split.map((v) => num(v)).join(", ")],
          ["Splits", String(r.n_splits)],
          ["No skill", num(r.no_skill)],
        ];
        const row = s("g", {
          class: "row",
          tabindex: 0,
          "aria-label": r.display_id + ", " + r.label + ": test " + meanSd(r.test_pr_auc, r.test_pr_auc_sd) + ", CV " + num(r.cv_pr_auc),
          on: tipHandlers(r.display_id + " " + r.label, tip),
        });
        if (r.feature_set === best) {
          row.append(s("rect", { class: "band", x: 0, y: y + 1, width: W, height: rowH - 2, rx: 4 }));
        }
        row.append(s("rect", { class: "hit", x: 0, y: y, width: W, height: rowH }));
        row.append(s("text", { class: "row-id", x: 6, y: cy + 4, text: r.display_id }));
        row.append(s("text", { class: "row-label", x: 78, y: cy + 4, text: r.label }));
        if (r.test_pr_auc_sd !== null) {
          row.append(s("line", { class: "interval s1", x1: x(r.test_pr_auc - r.test_pr_auc_sd), x2: x(r.test_pr_auc + r.test_pr_auc_sd), y1: cy, y2: cy }));
        }
        for (const v of r.test_per_split) {
          row.append(s("circle", { class: "split-dot s1", cx: x(v), cy: cy, r: 3 }));
        }
        const cvX = x(r.cv_pr_auc);
        row.append(s("path", { class: "cv-mark", d: "M" + cvX + " " + (cy - 5) + " L" + (cvX + 5) + " " + cy + " L" + cvX + " " + (cy + 5) + " L" + (cvX - 5) + " " + cy + " Z" }));
        row.append(s("circle", { class: "mean s1", cx: x(r.test_pr_auc), cy: cy, r: 5.5 }));
        row.append(s("text", { class: "row-value", x: W - R + 8, y: cy + 4, text: num(r.test_pr_auc) }));
        svg.append(row);
        y += rowH;
      }
    }
    svg.append(s("line", { class: "axis", x1: L, x2: W - R, y1: bottom, y2: bottom }));
    put("leaderboard", svg);
    put(
      "leaderboard-legend",
      legendKey("mean", "s1", "Test PR-AUC, mean ± SD over splits"),
      legendKey("split", "s1", "One split"),
      legendKey("cv", "", "CV PR-AUC, mean"),
      legendKey("ref", "", "No skill (the test base rate)")
    );
  }

  function renderBest(view) {
    const best = view.best;
    if (!best) {
      put("best-callout");
      return;
    }
    const lines = [
      h("p", {},
        h("b", { text: "Highest mean CV PR-AUC: " + best.display_id + " " }),
        best.label + ". CV " + meanSd(best.cv_pr_auc, best.cv_pr_auc_sd) + ", test " +
        meanSd(best.test_pr_auc, best.test_pr_auc_sd) + ", " + best.n_splits + " splits, " + best.budget + " budget."
      ),
    ];
    if (best.close.length) {
      const text = best.close
        .map((c) => c.display_id + " (CV " + num(c.cv_pr_auc) + (c.verdict ? "; " + c.verdict : "; not contrasted") + ")")
        .join(", ");
      lines.push(h("p", { text: "Not a clear winner. Within one SD of it: " + text + "." }));
    } else {
      lines.push(h("p", { text: "No other setup is within one SD of it in CV. A CV ranking is not a test; see the contrasts below." }));
    }
    lines.push(h("p", { text: best.rule }));
    put("best-callout", h("div", { class: "callout" }, lines));
  }

  function renderMethodsTable(view) {
    const rows = view.rows;
    if (!rows.length) {
      put("methods-table", h("p", { class: "empty", text: "No run at this budget for this view." }));
      return;
    }
    const best = view.best && view.best.budget === view.budget ? view.best.feature_set : null;
    const most = view.coverage.most_splits;
    const body = [];
    for (const family of view.families) {
      const members = rows.filter((r) => r.family === family);
      if (!members.length) continue;
      body.push(h("tr", { class: "family-row" }, h("td", { colspan: 8, text: family })));
      for (const r of members) {
        body.push(
          h("tr", { class: r.feature_set === best ? "best" : null },
            h("td", {},
              h("b", { text: r.display_id }),
              r.feature_set === best ? h("span", { class: "sub-label", text: "best by CV" }) : null
            ),
            h("td", { class: "wrap-cell" },
              r.label,
              r.label_derived_column ? h("span", { class: "sub-label", text: "Uses a column built from training labels" }) : null,
              r.n_backfilled ? h("span", { class: "sub-label", text: r.n_backfilled + " of " + r.n_splits + " runs recorded later (backfill)" }) : null
            ),
            h("td", {},
              r.n_splits + (view.budget === "pilot" && r.n_splits < most ? " of " + most : "")
            ),
            h("td", { text: meanSd(r.test_pr_auc, r.test_pr_auc_sd) }),
            h("td", { text: r.test_per_split.map((v) => num(v)).join(" / ") }),
            h("td", { text: num(r.cv_pr_auc) }),
            h("td", { text: num(r.test_auroc) }),
            h("td", { class: "l" },
              h("details", { class: "entries" },
                h("summary", { text: [...new Set(r.model_a)].join(", ") }),
                h("ul", {}, r.run_ids.map((id, i) => h("li", { text: "split " + r.splits[i] + ": " + r.model_a[i] + " · " + id })))
              )
            )
          )
        );
      }
    }
    put(
      "methods-table",
      h("table", {},
        h("thead", {}, h("tr", {},
          h("th", { text: "Setup" }),
          h("th", { class: "wrap-cell", text: "What it adds" }),
          h("th", { text: "Splits" }),
          h("th", { text: "Test PR-AUC" }),
          h("th", { text: "Per split" }),
          h("th", { text: "CV PR-AUC" }),
          h("th", { text: "AUROC" }),
          h("th", { class: "l", text: "Model A, database entries" })
        )),
        h("tbody", {}, body)
      )
    );
  }

  function renderCoverage(view) {
    const items = [];
    const c = view.coverage;
    if (c.fewer_splits.length) {
      items.push(h("li", { text: "Fewer splits than this dataset's " + c.most_splits + ": " + c.fewer_splits.map((f) => f.display_id + " (" + f.n_splits + ")").join(", ") + ". Means over different splits are not strictly comparable." }));
    }
    const pilot = c.not_run.filter((n) => n.budget === "pilot").map((n) => n.display_id);
    const full = c.not_run.filter((n) => n.budget === "full").map((n) => n.display_id);
    if (pilot.length) items.push(h("li", { text: "Not run at the pilot budget for this view: " + pilot.join(", ") + "." }));
    if (full.length) items.push(h("li", { text: "In the config but not run at the full budget for this view: " + full.join(", ") + "." }));
    if (view.has_label_column) items.push(h("li", { text: meta.label_column_note }));
    put("coverage", items.length ? h("div", { class: "panel panel-pad" }, h("h3", { class: "sub", text: "What this table leaves out" }), h("ul", { class: "plain" }, items)) : null);
  }

  function renderMethods(view) {
    $("methods-lede").textContent =
      datasetInfo(view.dataset).label + ", " + view.variant_label + ", " +
      view.budget + " budget. Pilot and full runs are never mixed in one row.";
    put(
      "dataset-caveats",
      view.caveats.map((t) => h("div", { class: "notice", text: t }))
    );
    renderBest(view);
    renderLeaderboard(view);
    renderMethodsTable(view);
    renderCoverage(view);
  }

  // ---- Contrast tables -----------------------------------------------------
  function contrastTable(rows, options) {
    const maxSplits = Math.max(...rows.map((r) => r.deltas.length));
    return h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, h("table", {},
      h("thead", {}, h("tr", {},
        h("th", { class: "l", text: "Contrast" }),
        h("th", { text: "Test Δ per split" }),
        h("th", { text: "Mean ± SD" }),
        h("th", { text: "Headroom" }),
        h("th", { text: "CV p" }),
        options.holm ? h("th", { text: "Holm p" }) : null,
        h("th", { class: "l", text: "Verdict" }),
        options.source ? h("th", { class: "wrap-cell", text: "From" }) : null
      )),
      h("tbody", {}, rows.map((r) =>
        h("tr", {},
          h("td", { class: "l" },
            h("b", { text: contrastName(r) }),
            h("span", { class: "sub-label", text: r.b_label + " vs " + r.a_label }),
            earlierNote(r)
          ),
          h("td", { text: r.deltas.map((v) => signed(v)).join(" / ") + (r.deltas.length < maxSplits ? " (" + r.deltas.length + ")" : "") }),
          h("td", { text: signed(r.mean) + (r.sd === null ? "" : " ± " + num(r.sd)) }),
          h("td", { text: Math.round(r.headroom * 100) + "%" }),
          h("td", { text: pValue(r.p) }),
          options.holm ? h("td", { text: pValue(r.holm_p) }) : null,
          h("td", { class: "l" }, verdictChip(r.verdict_kind, r.verdict_label)),
          options.source ? h("td", { class: "wrap-cell" }, h("span", { class: "sub-label", text: r.description }), h("span", { class: "sub-label", text: r.run_id })) : null
        )
      ))
    )));
  }
  function contrastName(r) {
    if (r.a_variant === r.b_variant && r.a_budget === r.b_budget) {
      return r.b_id + " " + MINUS + " " + r.a_id;
    }
    const tag = (id, variant, budget) => id + " [" + variant + ", " + budget + "]";
    return tag(r.b_id, r.b_variant, r.b_budget) + " " + MINUS + " " + tag(r.a_id, r.a_variant, r.a_budget);
  }
  function renderVerdicts(view) {
    const { primary, other, descriptive } = view.families;
    const parts = [];
    if (primary.length) {
      parts.push(h("h3", { class: "sub spaced", text: "Primary family, " + primary[0].n_splits + " splits (Holm within it)" }));
      parts.push(contrastTable(primary, { holm: true }));
    }
    if (other.length) {
      const inner = [contrastTable(other, { source: true })];
      parts.push(
        h("details", { class: "more", open: other.length <= 14 },
          h("summary", { text: "Other pre-registered contrasts (" + other.length + ")" }),
          inner
        )
      );
    }
    if (descriptive.length) {
      parts.push(
        h("details", { class: "more" },
          h("summary", { text: "Descriptive only: one split, or across budgets (" + descriptive.length + ")" }),
          h("p", { class: "small muted", text: "One split has no spread, so these carry no verdict." }),
          contrastTable(descriptive, { source: true })
        )
      );
    }
    if (!parts.length) parts.push(h("div", { class: "panel empty", text: "No contrast was pre-registered for this view." }));
    put("verdicts-body", parts);

    const sens = [];
    const check = view.default_check;
    if (check) {
      sens.push(h("h3", { class: "sub spaced", text: "Sensitivity check: default settings (not the verdicts)" }));
      sens.push(h("p", { class: "small muted", text: "Pre-registered on 2026-11-02, proposed after the tuned verdicts were known. Model A stays the best tuned row; this asks how much the pilot's tuning moves the result. Entries: " + check.run_ids.join(", ") }));
      sens.push(h("div", { class: "grid-2" },
        h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, h("table", {},
          h("thead", {}, h("tr", {},
            h("th", { text: "Setup" }), h("th", { text: "Test, tuned" }), h("th", { text: "Test, default" }), h("th", { text: "Default − tuned" }))),
          h("tbody", {}, check.scores.map((r) => h("tr", {},
            h("td", { text: r.setup }), h("td", { text: num(r.test_tuned) }), h("td", { text: num(r.test_default) }), h("td", { text: signed(r.test_default - r.test_tuned) }))))
        ))),
        h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, h("table", {},
          h("thead", {}, h("tr", {},
            h("th", { class: "l", text: "At default settings" }), h("th", { text: "Mean ± SD" }), h("th", { text: "CV p" }), h("th", { class: "l", text: "Verdict" }))),
          h("tbody", {}, check.contrasts.map((r) => h("tr", {},
            h("td", { class: "l", text: r.contrast }), h("td", { text: signed(r.mean) + " ± " + num(r.sd) }), h("td", { text: pValue(r.p) }), h("td", { class: "l" }, verdictChip(r.verdict_kind, r.verdict_label)))))
        )))
      ));
    }
    if (view.tuned_vs_default.length) {
      sens.push(h("h3", { class: "sub spaced", text: "Best CV PR-AUC, tuned and default settings" }));
      sens.push(h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, h("table", {},
        h("thead", {}, h("tr", {},
          h("th", { text: "Setup" }), h("th", { text: "Tuned" }), h("th", { text: "Default" }), h("th", { text: "Tuned − default" }))),
        h("tbody", {}, view.tuned_vs_default.map((r) => h("tr", {},
          h("td", { text: r.setup }), h("td", { text: num(r.tuned) }), h("td", { text: num(r.default) }), h("td", { text: signed(r.tuned - r.default) }))))
      ))));
    }
    put("sensitivity", sens);
  }

  // ---- Section 2: across datasets ------------------------------------------
  const SERIES = [
    { key: "amazon", cls: "s1", glyph: "mean", label: "Amazon, all 25 features" },
    { key: "yelpchi", cls: "s2", glyph: "square", label: "YelpChi, all 32 features" },
    { key: "amazon_no19", cls: "s3", glyph: "triangle", label: "Amazon, without f19" },
  ];
  function marker(series, cx, cy, value) {
    if (series.glyph === "square") {
      return s("rect", { class: "mean " + series.cls, x: cx - 5, y: cy - 5, width: 10, height: 10, rx: 1 });
    }
    if (series.glyph === "triangle") {
      return s("path", { class: "mean " + series.cls, d: "M" + cx + " " + (cy - 6) + " L" + (cx + 6) + " " + (cy + 5) + " L" + (cx - 6) + " " + (cy + 5) + " Z" });
    }
    return s("circle", { class: "mean " + series.cls, cx: cx, cy: cy, r: 5.5, "data-value": value });
  }
  const seriesLegend = () =>
    h("div", { class: "legend" }, SERIES.map((x) => legendKey(x.glyph, x.cls, x.label)));

  function skillChart(setups) {
    const W = 1000, L = 390, R = 30, top = 12, rowH = 32, axisH = 44;
    const height = top + setups.length * rowH + axisH;
    const x = (v) => L + v * (W - L - R);
    const bottom = height - axisH;
    const svg = s("svg", { viewBox: "0 0 " + W + " " + height, role: "img", "aria-label": "Normalised skill per setup and dataset. The table below has every value." });
    for (const t of [0, 0.2, 0.4, 0.6, 0.8, 1]) {
      svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: bottom }));
      svg.append(s("text", { x: x(t), y: bottom + 18, "text-anchor": "middle", text: t.toFixed(1) }));
    }
    svg.append(s("text", { x: (L + W - R) / 2, y: bottom + 36, "text-anchor": "middle", text: "Normalised skill: 0 is the no-skill line, 1 is perfect" }));
    setups.forEach((row, i) => {
      const y = top + i * rowH, cy = y + rowH / 2;
      const tip = SERIES.filter((v) => row[v.key]).map((v) => [v.label, "skill " + num(row[v.key].normalised_skill) + ", PR-AUC " + num(row[v.key].test_pr_auc) + ", " + row[v.key].n_splits + " splits"]);
      const g = s("g", { class: "row", tabindex: 0, "aria-label": row.setup + ": " + tip.map((t) => t.join(" ")).join("; "), on: tipHandlers(row.setup + " " + row.label, tip) });
      g.append(s("rect", { class: "hit", x: 0, y: y, width: W, height: rowH }));
      g.append(s("text", { class: "row-id", x: 6, y: cy + 4, text: row.setup }));
      g.append(s("text", { class: "row-label", x: 78, y: cy + 4, text: row.label }));
      g.append(s("line", { class: "connector", x1: x(row.amazon.normalised_skill), x2: x(row.yelpchi.normalised_skill), y1: cy, y2: cy }));
      for (const v of SERIES) {
        if (row[v.key]) g.append(marker(v, x(row[v.key].normalised_skill), cy));
      }
      svg.append(g);
    });
    return svg;
  }

  function gainChart(gains) {
    const values = gains.flatMap((g) => SERIES.flatMap((v) => (g[v.key] ? g[v.key].deltas : [])));
    const lo = Math.floor(Math.min(0, ...values) * 20) / 20;
    const hi = Math.ceil(Math.max(0, ...values) * 20) / 20;
    const W = 1000, L = 390, R = 30, top = 12, rowH = 40, axisH = 44;
    const height = top + gains.length * rowH + axisH;
    const x = (v) => L + ((v - lo) / (hi - lo)) * (W - L - R);
    const bottom = height - axisH;
    const svg = s("svg", { viewBox: "0 0 " + W + " " + height, role: "img", "aria-label": "Test PR-AUC gain over B per setup and dataset. The table below has every value." });
    const tickValues = ticks(lo, hi, 8);
    const d = decimals(tickValues);
    for (const t of tickValues) {
      svg.append(s("line", { class: "grid", x1: x(t), x2: x(t), y1: top, y2: bottom }));
      svg.append(s("text", { x: x(t), y: bottom + 18, "text-anchor": "middle", text: signed(t, d) }));
    }
    svg.append(s("line", { class: "ref", x1: x(0), x2: x(0), y1: top, y2: bottom }));
    svg.append(s("text", { x: (L + W - R) / 2, y: bottom + 36, "text-anchor": "middle", text: "Test PR-AUC minus B's, on the splits both ran (raw PR-AUC, so compare within a dataset)" }));
    gains.forEach((row, i) => {
      const y = top + i * rowH, cy = y + rowH / 2;
      const tip = SERIES.filter((v) => row[v.key]).map((v) => [v.label, signed(row[v.key].mean) + (row[v.key].sd === null ? "" : " ± " + num(row[v.key].sd)) + ", " + row[v.key].n_splits + " splits"]);
      const g = s("g", { class: "row", tabindex: 0, "aria-label": row.setup + " minus B: " + tip.map((t) => t.join(" ")).join("; "), on: tipHandlers(row.setup + " " + MINUS + " B", tip) });
      g.append(s("rect", { class: "hit", x: 0, y: y, width: W, height: rowH }));
      g.append(s("text", { class: "row-id", x: 6, y: cy + 4, text: row.setup }));
      g.append(s("text", { class: "row-label", x: 78, y: cy + 4, text: row.label }));
      SERIES.forEach((v, j) => {
        const gain = row[v.key];
        if (!gain) return;
        const off = (j - 1) * 11;
        for (const delta of gain.deltas) {
          g.append(s("circle", { class: "split-dot " + v.cls, cx: x(delta), cy: cy + off, r: 2.5 }));
        }
        g.append(marker(v, x(gain.mean), cy + off));
      });
      svg.append(g);
    });
    return svg;
  }

  function renderCross(cross) {
    if (!cross.setups.length) {
      put("cross-body", h("div", { class: "panel empty", text: "No setup has run on both datasets at the pilot budget." }));
      return;
    }
    const am = (r) => r.amazon;
    const parts = [
      h("div", { class: "notice", text: "Amazon has 5 splits and YelpChi 3. Amazon's all-feature scores include f19, which likely leaks the label, and YelpChi has no counterpart to it; the without-f19 markers are beside them where those runs exist. All runs are at the pilot budget." }),
      h("h3", { class: "sub", text: "Skill on a shared scale" }),
      h("p", { class: "small muted", text: "The base rates differ (Amazon " + num(am(cross.setups[0]).no_skill) + ", YelpChi " + num(cross.setups[0].yelpchi.no_skill) + "), so raw PR-AUC is not comparable between datasets. Skill = (PR-AUC − no skill) / (1 − no skill)." }),
      h("div", { class: "panel chart-panel" }, seriesLegend(), h("div", { class: "chart-scroll" }, skillChart(cross.setups))),
      h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, h("table", {},
        h("thead", {}, h("tr", {},
          h("th", { text: "Setup" }), h("th", { class: "wrap-cell", text: "What it adds" }),
          h("th", { text: "Amazon skill" }), h("th", { text: "Amazon PR-AUC" }),
          h("th", { text: "YelpChi skill" }), h("th", { text: "YelpChi PR-AUC" }),
          h("th", { text: "Amazon without f19, skill" }), h("th", { text: "Without f19, PR-AUC" }))),
        h("tbody", {}, cross.setups.map((r) => h("tr", {},
          h("td", {}, h("b", { text: r.setup })), h("td", { class: "wrap-cell", text: r.label }),
          h("td", { text: num(r.amazon.normalised_skill) }), h("td", { text: meanSd(r.amazon.test_pr_auc, r.amazon.test_pr_auc_sd) }),
          h("td", { text: num(r.yelpchi.normalised_skill) }), h("td", { text: meanSd(r.yelpchi.test_pr_auc, r.yelpchi.test_pr_auc_sd) }),
          h("td", { text: r.amazon_no19 ? num(r.amazon_no19.normalised_skill) : "not run" }),
          h("td", { text: r.amazon_no19 ? meanSd(r.amazon_no19.test_pr_auc, r.amazon_no19.test_pr_auc_sd) : "not run" }))))
      ))),
      h("h3", { class: "sub spaced", text: "Gain over B, per dataset" }),
      h("p", { class: "small muted", text: "Small dots are single splits. E6 uses no graph, so its row is not a graph gain; it is here because it is the GNN's control." }),
      h("div", { class: "panel chart-panel" }, seriesLegend(), h("div", { class: "chart-scroll" }, gainChart(cross.gains))),
      h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, h("table", {},
        h("thead", {}, h("tr", {},
          h("th", { text: "Setup − B" }), h("th", { class: "wrap-cell", text: "What it adds" }),
          h("th", { text: "Amazon" }), h("th", { text: "YelpChi" }), h("th", { text: "Amazon without f19" }))),
        h("tbody", {}, cross.gains.map((r) => {
          const cell = (g) => (g ? signed(g.mean) + (g.sd === null ? "" : " ± " + num(g.sd)) + " (" + g.n_splits + ")" : "not run");
          return h("tr", {},
            h("td", {}, h("b", { text: r.setup })), h("td", { class: "wrap-cell", text: r.label }),
            h("td", { text: cell(r.amazon) }), h("td", { text: cell(r.yelpchi) }), h("td", { text: cell(r.amazon_no19) }));
        }))
      ))),
      h("h3", { class: "sub spaced", text: "Do the verdicts agree?" }),
      h("p", { class: "small muted", text: "All features, pilot budget, the current verdict on each dataset. Two verdicts agree only if they are the same kind: a significant drop and a tiny lower mean do not." }),
      h("div", { class: "panel table-panel" }, h("div", { class: "table-scroll" }, h("table", {},
        h("thead", {}, h("tr", {},
          h("th", { class: "l", text: "Contrast" }),
          h("th", { text: "Amazon Δ" }), h("th", { class: "l", text: "Amazon verdict" }),
          h("th", { text: "YelpChi Δ" }), h("th", { class: "l", text: "YelpChi verdict" }),
          h("th", { class: "l", text: "" }))),
        h("tbody", {}, cross.agreement.map((r) => h("tr", {},
          h("td", { class: "l" }, h("b", { text: r.contrast }), h("span", { class: "sub-label", text: r.b_label + " vs " + r.a_label })),
          h("td", { text: signed(r.amazon.mean) + " (" + r.amazon.n_splits + ")" }),
          h("td", { class: "l" }, verdictChip(r.amazon.verdict_kind, r.amazon.verdict_label)),
          h("td", { text: signed(r.yelpchi.mean) + " (" + r.yelpchi.n_splits + ")" }),
          h("td", { class: "l" }, verdictChip(r.yelpchi.verdict_kind, r.yelpchi.verdict_label)),
          h("td", { class: "l" }, h("span", { class: "badge " + (r.agrees ? "good" : "warn"), text: r.agrees ? "agrees" : "differs" })))))
      ))),
    ];
    put("cross-body", parts);
  }

  // ---- Section 3: classify -------------------------------------------------
  const panels = {}; // per dataset: { variant, results, history, tally, node }
  function panelFor(dataset) {
    if (!panels[dataset]) {
      panels[dataset] = {
        variant: "all",
        results: {},
      };
    }
    return panels[dataset];
  }
  function syncAmazonPanel() {
    if (state.variant === "all" || state.variant === "no19") {
      panelFor("amazon").variant = state.variant;
      if (lastStatus) renderPanels(lastStatus);
    }
  }
  function resultsFor(panel, key) {
    if (!panel.results[key]) {
      panel.results[key] = { current: null, history: [], tally: { n: 0, correct: 0, fp: 0, fn: 0 }, error: null, busy: false };
    }
    return panel.results[key];
  }
  function statTile(k, v) {
    return h("div", { class: "stat" }, h("div", { class: "k", text: k }), h("div", { class: "v", text: v }));
  }

  function resultCard(result) {
    const position = result.prob * 100;
    const tickSide = result.threshold < 0.12 ? "left" : result.threshold > 0.88 ? "right" : "";
    const fill = h("div", { class: "prob-fill" });
    fill.style.width = position + "%";
    const tick = h("div", { class: "prob-tick" });
    tick.style.left = result.threshold * 100 + "%";
    const tickLabel = h("div", { class: "prob-tick-label " + tickSide, text: "threshold " + num(result.threshold) });
    tickLabel.style.left = result.threshold * 100 + "%";
    const egos = result.neighbourhoods.map((n) => {
      // compare.ego_svg is written for HTML, so as XML it needs its namespace.
      const markup = n.svg.replace("<svg ", '<svg xmlns="' + SVG_NS + '" ');
      const parsed = new DOMParser().parseFromString(markup, "image/svg+xml");
      const svg = parsed.querySelector("parsererror") ? null : document.importNode(parsed.documentElement, true);
      const share = n.train_fraud_share === null ? "no training neighbour" : Math.round(n.train_fraud_share * 100) + "% of " + n.n_train_neighbours + " training neighbours are fraud";
      return h("div", {}, svg, h("div", { class: "ego-cap", text: n.relation + ": " + n.degree + " neighbours" + (n.degree > n.n_shown ? ", " + n.n_shown + " shown" : "") }), h("div", { class: "ego-cap", text: share }));
    });
    const base = result.baseline;
    return h("div", { class: "result" },
      h("div", { class: "result-head" },
        h("span", {}, "Node ", h("span", { class: "id", text: result.node_id })),
        h("span", { class: "outcome " + (result.correct ? "correct" : "wrong") }, result.correct ? "✓ " : "✕ ", result.outcome)
      ),
      h("div", { class: "prob-bar" }, fill, tick, tickLabel),
      h("div", { class: "facts-row" },
        h("span", {}, "P(fraud) ", h("b", { text: probText(result.prob) })),
        h("span", {}, "Flagged ", h("b", { text: result.predicted ? "yes" : "no" })),
        h("span", {}, "True label ", h("b", { text: result.actual ? "fraud" : "not fraud" })),
        h("span", {}, "Confidence ", h("b", { text: num(result.confidence) }))
      ),
      h("div", { class: "facts-row small" },
        result.mode === "live"
          ? h("span", { text: "Live prediction; equals df-analyze's saved value (difference " + result.difference.toExponential(1) + ")." })
          : h("span", { text: "Saved test prediction, not computed live." })
      ),
      base
        ? h("div", { class: "baseline-line" },
            h("b", { text: "What did B say? " }),
            "P(fraud) " + probText(base.prob) + " at its threshold " + num(base.threshold) + ": " + base.outcome + ". ",
            base.changed ? h("b", { text: "The graph changed the decision." }) : "Same decision."
          )
        : null,
      h("h4", { text: "Most unusual features, against the training nodes" }),
      h("div", { class: "table-scroll" }, h("table", { class: "compact" },
        h("thead", {}, h("tr", {}, h("th", { text: "Feature" }), h("th", { text: "Value" }), h("th", { text: "Training percentile" }))),
        h("tbody", {}, result.top_features.map((f) => h("tr", {},
          h("td", { class: "mono", text: f.name }), h("td", { text: Number(f.value).toPrecision(4) }), h("td", { text: Math.round(f.percentile * 100) + "%" }))))
      )),
      egos.length ? h("h4", { text: "Its neighbourhood, per relation" }) : null,
      egos.length ? h("div", { class: "egos" }, egos) : null,
      egos.length
        ? h("div", { class: "ego-legend" },
            h("span", {}, h("span", { class: "swatch fraud" }), "training, fraud"),
            h("span", {}, h("span", { class: "swatch benign" }), "training, not fraud"),
            h("span", {}, h("span", { class: "swatch other" }), "test or unlabelled"))
        : null
    );
  }

  async function classify(panel, key, draw, seedField) {
    const results = resultsFor(panel, key);
    const text = seedField.value.trim();
    let seed = null;
    if (text !== "") {
      seed = Number(text);
      if (!Number.isInteger(seed) || seed < 0 || seed > 4294967295) {
        results.error = "The seed must be a whole number from 0 to 4294967295.";
        renderPanels(lastStatus);
        return;
      }
    }
    results.busy = true;
    results.error = null;
    renderPanels(lastStatus);
    try {
      const result = await api("/api/classify", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ target: key, draw: draw, seed: seed }),
      });
      results.current = result;
      results.history = [result, ...results.history].slice(0, 10);
      if (draw === "any") {
        results.tally.n += 1;
        if (result.correct) results.tally.correct += 1;
        else if (result.predicted) results.tally.fp += 1;
        else results.tally.fn += 1;
      }
    } catch (e) {
      results.error = e.message;
    }
    results.busy = false;
    panel.seed = text;
    renderPanels(lastStatus);
  }

  function targetPanel(dataset, entry, panel) {
    const label = datasetInfo(dataset) ? datasetInfo(dataset).label : dataset;
    const head = [h("h3", { text: label })];
    const options = lastStatus.panels.filter((p) => p.dataset === dataset);
    if (options.length > 1) {
      head.push(
        h("div", { class: "segmented" }, options.map((p) =>
          h("button", {
            type: "button",
            "aria-pressed": p.key === entry.key ? "true" : "false",
            text: p.variant_label,
            on: { click: () => { panel.variant = p.variant; renderPanels(lastStatus); } },
          })
        ))
      );
    }
    if (entry.state === "unavailable") {
      return h("div", { class: "panel classify-panel" }, head, h("p", { class: "small" }, h("b", { text: "Not available here. " }), entry.message));
    }
    if (!entry.target) {
      return h("div", { class: "panel classify-panel" }, head, h("p", { class: "small muted", text: "Loading this run's outputs…" }));
    }
    const t = entry.target;
    const m = t.test_metrics;
    const results = resultsFor(panel, entry.key);
    const seedInput = h("input", { type: "text", inputmode: "numeric", placeholder: "none", "aria-label": "Seed for a reproducible draw", value: panel.seed || null });
    const ready = entry.state === "ready" && !results.busy;
    const button = (cls, draw, text) =>
      h("button", { type: "button", class: cls, disabled: !ready, text: text, on: { click: () => classify(panel, entry.key, draw, seedInput) } });
    const checksOk = t.checks_ok;
    return h("div", { class: "panel classify-panel" },
      head,
      h("p", { class: "panel-sub", text: t.best.display_id + " " + t.best.label + ". Model A on split 0: " + t.model + " (" + t.selection + "), " + t.variant_label + "." }),
      h("div", { class: "status-line" },
        t.mode === "live"
          ? h("span", { class: "badge good", text: "✓ Live, verified on all " + t.n_test.toLocaleString() + " test rows" })
          : h("span", { class: "badge warn", text: entry.state === "loading" ? "Starting the model…" : "Saved predictions" }),
        h("span", { class: "badge " + (checksOk ? "good" : "bad"), text: (checksOk ? "✓ " : "✕ ") + t.checks.length + " checks" })
      ),
      t.note ? h("p", { class: "small muted", text: t.note }) : null,
      t.best.close.length
        ? h("p", { class: "small muted", text: "Not a clear winner: " + t.best.close.map((c) => c.display_id).join(", ") + " are within one SD in CV." })
        : null,
      h("div", { class: "stats" },
        statTile("Test PR-AUC", num(m.pr_auc)),
        statTile("No skill", num(m.no_skill)),
        statTile("F1", num(m.f1)),
        statTile("Precision", num(m.precision)),
        statTile("Recall", num(m.recall))
      ),
      h("p", { class: "small muted", text: "Whole test set at threshold " + num(t.threshold) + ": " + m.tp + " frauds caught, " + m.fn + " missed, " + m.fp + " false alarms, " + m.tn + " correct passes." }),
      h("div", { class: "actions" },
        button("primary", "any", "Random test node"),
        button("ghost", "fraud", "Random fraud"),
        button("ghost", "benign", "Random not fraud"),
        h("label", { class: "field" }, "Seed (optional)", seedInput)
      ),
      results.error ? h("div", { class: "error", text: results.error }) : null,
      results.current ? resultCard(results.current) : null,
      results.history.length
        ? h("div", {},
            h("div", { class: "history" }, results.history.map((r) =>
              h("button", {
                type: "button",
                class: r.correct ? "correct" : "wrong",
                title: r.outcome + ", P(fraud) " + probText(r.prob),
                text: r.node_id,
                on: { click: () => { results.current = r; renderPanels(lastStatus); } },
              })
            )),
            h("div", { class: "tally", text: results.tally.n
              ? "Plain random draws this session: " + results.tally.n + " (" + results.tally.correct + " correct, " + results.tally.fp + " false positive, " + results.tally.fn + " false negative). Not a performance estimate; fraud and not-fraud buttons are not counted."
              : "Fraud and not-fraud buttons are not counted in any tally: they pick the class on purpose." })
          )
        : null,
      h("details", { class: "more" },
        h("summary", { text: "Checks on this panel" }),
        h("ul", { class: "checks-list" }, t.checks.map((c) =>
          h("li", {}, h("span", { class: c.ok ? "check-pass" : "check-fail", text: c.ok ? "✓ " : "✕ " }), c.name + ": " + c.detail)
        ))
      )
    );
  }

  let lastStatus = null;
  function renderPanels(status) {
    lastStatus = status;
    const datasets = [...new Set(status.panels.map((p) => p.dataset))];
    // A rebuild must not lose what is being typed into a seed box.
    const typed = document.activeElement && document.activeElement.tagName === "INPUT" && document.activeElement.type === "text";
    if (typed) return;
    put("classify-panels", datasets.map((dataset) => {
      const panel = panelFor(dataset);
      const options = status.panels.filter((p) => p.dataset === dataset);
      const entry = options.find((p) => p.variant === panel.variant) || options[0];
      return targetPanel(dataset, entry, panel);
    }));
  }
  async function pollStatus() {
    try {
      const status = await api("/api/status");
      renderPanels(status);
      if (status.panels.some((p) => p.state === "loading")) {
        window.setTimeout(pollStatus, 1500);
      }
    } catch (e) {
      put("classify-panels", errorBox(e));
    }
  }

  // ---- Checks --------------------------------------------------------------
  function renderChecks(checks) {
    const o = meta.overview;
    const passed = checks.filter((c) => c.ok).length;
    put("database-facts",
      h("div", { class: "facts-row" },
        h("span", {}, "Sanity checks passed ", h("b", { text: passed + " of " + checks.length })),
        h("span", {}, "Database entries ", h("b", { text: o.n_entries })),
        h("span", {}, "Runs ", h("b", { text: o.n_runs + " (" + Object.entries(o.datasets).map(([k, v]) => k + " " + v).join(", ") + ")" })),
        h("span", {}, "Pilot / full ", h("b", { text: o.budgets.pilot + " / " + o.budgets.full })),
        h("span", {}, "Backfilled ", h("b", { text: o.n_backfilled })),
        h("span", {}, "Machines ", h("b", { text: o.devices.join(", ") }))
      )
    );
    put("checks-table", h("table", {},
      h("thead", {}, h("tr", {}, h("th", { class: "l", text: "Check" }), h("th", { class: "l", text: "Result" }), h("th", { class: "wrap-cell", text: "Detail" }))),
      h("tbody", {}, checks.map((c) => h("tr", {},
        h("td", { class: "l", text: c.name }),
        h("td", { class: "l" }, h("span", { class: c.ok ? "check-pass" : "check-fail", text: c.ok ? "✓ pass" : "✕ FAIL" })),
        h("td", { class: "wrap-cell" }, c.detail, c.problems.map((p) => h("span", { class: "problem", text: p })))
      )))
    ));
    const failed = meta.failed_entries;
    put("failed-entries", failed && failed.length
      ? h("div", { class: "panel panel-pad" },
          h("h3", { class: "sub", text: "Started but not finished" }),
          h("ul", { class: "plain" }, failed.map((f) => h("li", {}, f.description + ". ", f.notes || "", h("span", { class: "sub-label", text: f.run_id })))))
      : null);
  }

  // ---- Loading -------------------------------------------------------------
  async function loadDatasetViews() {
    const token = ++loadToken;
    const q = "ds=" + encodeURIComponent(state.ds) + "&variant=" + encodeURIComponent(state.variant);
    const scoped = ["questions-body", "best-callout", "leaderboard", "methods-table", "verdicts-body", "sensitivity"];
    scoped.forEach((id) => $(id).classList.add("stale"));
    try {
      const [methods, contrasts] = await Promise.all([
        api("/api/methods?" + q + "&budget=" + encodeURIComponent(state.budget)),
        api("/api/contrasts?" + q),
      ]);
      if (token !== loadToken) return; // a newer choice has been made
      lastMethods = methods;
      renderMethods(methods);
      renderQuestions(contrasts, methods);
      renderVerdicts(contrasts);
    } catch (e) {
      if (token === loadToken) put("methods-table", errorBox(e));
    }
    scoped.forEach((id) => $(id).classList.remove("stale"));
  }
  async function start() {
    readUrl();
    wireControls();
    try {
      meta = await api("/api/datasets");
    } catch (e) {
      document.querySelector("main").replaceChildren(errorBox(e));
      return;
    }
    normaliseState();
    renderControls();
    renderHeader();
    syncAmazonPanelFirst();
    await loadDatasetViews();
    try {
      renderCross(await api("/api/cross"));
    } catch (e) {
      put("cross-body", errorBox(e));
    }
    try {
      renderChecks((await api("/api/checks")).checks);
    } catch (e) {
      put("checks-table", errorBox(e));
    }
    pollStatus();
  }
  function syncAmazonPanelFirst() {
    if (state.variant === "no19") panelFor("amazon").variant = "no19";
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
