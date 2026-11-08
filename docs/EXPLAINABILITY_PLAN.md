# Explaining the best model per dataset

This is the plan and method for the explanation work (branch `exp/explain`). Results are in [`README.md`](../README.md) and the results database (kind `explain`). The code is `scripts/dfa/shap_explain.py` (df-analyze's environment, with `shap` added for the run only), `scripts/explain_report.py`, `src/fraud_detect/models/explain.py`, `explain_eval.py` and `src/fraud_detect/report/explain.py`. It is not a pipeline stage: it adds no stage outputs, so finished runs and bundles do not rerun.

## What is explained

Each dataset's best model, chosen on training CV, never on test (invariant 8). Its exported Model A is explained only if it is a LightGBM or CatBoost model; otherwise the script stops.

- **Amazon, substantive explanation:** E1 without f19 (`m2_own_nbr_no19`, full budget). f19 is a probable label leak (issue #36), so this is the run to read for what the model does.
- **Amazon, leakage audit:** E1 with every feature. An explanation saying "f19 matters" mostly confirms the leak and says little about fraud behaviour. The full-budget export of this model is on the group's desktop, so the audit here uses `pilot_m2_own_nbr` and says so.
- **YelpChi:** E2 (`pilot_m5_own_gnnssl`), the best shared CV at the pilot budget, the largest budget with every setup. At the full budget E1 and E2 are not distinguishable (E2 − E1 +0.003 ± 0.030), and the full-budget exports are also on the desktop.

## What each result means, and does not

| Output | What it is | What it is not |
|---|---|---|
| SHAP values (`shap`) | Exact Shapley values of the **interventional** game defined by a stated background of training rows, in log-odds | Faithful by construction. Different games give different values, especially for correlated columns (Sundararajan & Najmi 2020; Kumar et al. 2020; Janzing et al. 2020), and complete, linear attributions can provably fail to beat random guessing at inferring model behaviour (Bilodeau et al. 2024) |
| Additivity check | An **implementation check**: base + sum of SHAP = margin, and the base is the mean margin over the background | Evidence of faithfulness. It holds by construction for correct code, so it only catches errors such as a wrong output space or column order |
| Summed SHAP per block | Complete and additive | The Shapley value of the block as one player. Cross-block interactions stay split as the per-column game splits them |
| Group Shapley (`group_shapley`) | Exact Shapley values with blocks as players (all 2^p coalitions) | A per-column explanation. For DGI's PCA columns, this block level is the only reading reported |
| Path-dependent TreeSHAP (`sensitivity`, `interactions`) | A different game, using tree cover statistics. It is what LightGBM's `pred_contrib` gives. Always labelled, never mixed with the main run | "The same Tree SHAP". Disagreement between modes is information. shap provides interaction values only in this mode |
| Permutation importance | A diagnostic next to SHAP: it tracks loss where mean \|SHAP\| tracks output movement | A validation of SHAP. Explanation methods often disagree (Krishna et al. 2024) |
| Neighbour influence (`neighbours`) | **Necessity-style influence under mean aggregation**: how a target's score moves when one neighbour leaves one relation, split into a mean part and a degree part | "The important nodes". It misses neighbours that are redundant with each other, its inputs can leave the training data, and totals favour hubs (about 1/degree per target), so it is reported with a normalised view, a redundancy check, a range guard and a structural baseline |

## Quality tests

All of these use training data only.

- **ROAR-lite** (`roar`; Hooker et al. 2019): in each saved fold, rank columns by SHAP from a model fit without that fold, on that model's own rows; retrain with fixed hyperparameters after removing each block, the correlated group, the top k columns and k random columns; compare held-out PR-AUC. It retrains so the model never sees inputs it was not trained on. Issue #36 (0.919 to 0.518) is one point of this test. `roar_extended` adds k = 16, 32, 64 and all graph columns together; these were added after k ≤ 8 showed no separation from random on Amazon, so they are follow-up.
- **Model-randomisation check** (`randomisation`; Adebayo et al. 2018): refit on permuted labels. The ranking should lose its structure.
- **Stability:** group-bootstrap rank intervals (a rank interval wider than 5 places is flagged unstable), other background seeds and sizes, 3 refit seeds, other good models within one SD of Model A's CV PR-AUC (`rashomon`; refits only, no tuning), and across split repeats.

## How this differs from attention

SHAP and leave-one-out work after training, outside the model, from its inputs and outputs. Attention weights are parameters inside an architecture such as GAT or HGT, per layer and head, and describe how a layer mixes neighbours' messages. They are not, by default, a faithful measure of input importance (Jain & Wallace 2019); whether they can serve as one depends on the test (Wiegreffe & Pinter 2019). **No model in this ladder uses attention:** GraphSAGE and the R-GCN-style rung use mean or symmetric normalisation, and HGT was not run. So the comparison is conceptual. Adding a GAT needs the group's approval.

## Presentation

The audience is the course group and its reviewers, so the page aims at understanding the model and its failures. It claims no usefulness for analysts: controlled studies with fraud analysts found no reliable decision benefit once the design matched deployment (Jesus et al. 2021; Amarasinghe et al. 2024), and explanations can raise over-reliance (Bansal et al. 2021).

## Leakage and privacy

- Background rows, the ROAR and randomisation checks, and the threshold use training data and training CV only. SHAP on test nodes is reporting only, and no feature or model choice comes from it. Any later choice must use training-CV SHAP.
- Graph influence uses features, never labels. "Flagged neighbours" uses the model's scores.
- Row-level outputs (`explain/*.parquet`, `case_cards.json`, `explain.html`) stay in `outputs/` behind a group-only banner. The results database gets `tables/db_summary.csv` only.

## df-analyze's prepared columns

df-analyze saves no preprocessing object. Its prepared columns turned out to be the input columns unchanged, apart from a binary column becoming one indicator (residuals below 1e-11 on Amazon's and YelpChi's train and test rows). `shap_explain.py` rebuilds every node's input from the own block and the graph and **stops** if the rebuilt test rows differ from df-analyze's by more than 1e-6.

## A problem in shap that the implementation check found

shap 0.54's interventional TreeSHAP can send a LightGBM row the wrong way at a split when its value and the threshold agree in single precision. The evidence: the misses are per row, of one leaf's worth, the same for any background, and shap's copy of the tree still predicts correctly. Moving a value by one double-precision step did not clear a miss; moving it apart in single precision cleared all of them. That second fix can change the model's output when several thresholds share one single-precision value, so it is not used. Instead:

- A test row that misses is set to NaN, left out of every summary and counted on the page. YelpChi E2 lost 24 of 18,383 rows; Amazon's split 0 pilot model lost 1 of 3,456.
- A background row that could miss would spoil every row a little, so background rows equal to a threshold in single precision are not used.
- If more than 1% of rows miss, the run stops. YelpChi's E1 did (6.7%), so it uses path-dependent TreeSHAP with `--path-dependent`, labelled on its page. Its neighbour influence does not use SHAP.

The cause was inferred from those tests; shap's wheel does not include the C source to read.

## Run it

```bash
uv run --directory "$DF_ANALYZE_DIR" --python '>=3.13.11,<3.14' --with shap \
  python "$PWD/scripts/dfa/shap_explain.py" --dataset amazon --feature-set m2_own_nbr_no19
```

```bash
uv run scripts/explain_report.py --dataset amazon --feature-set m2_own_nbr_no19 \
  --audit pilot_m2_own_nbr --split-set pilot_m2_own_nbr_no19
```

## Not done

- **Node-level influence for YelpChi E2.** Its graph columns are DGI embeddings, so influence passes through a 2-layer encoder and a PCA. The encoder weights are not saved. The seeded CPU run should be reproducible, so the planned route is to retrain, assert it reproduces the saved embeddings, and score candidates chosen by a stated proxy, labelled approximate. Node-level influence on YelpChi here uses E1 (`pilot_m2_own_nbr`) instead, whose neighbour columns can be recomputed exactly.
- **The GNNExplainer cross-check** on the GraphSAGE rung (optional). The trained GNN weights are not saved, so it would need a retrain.
- **Full-budget explanations** of Amazon's all-features E1 and YelpChi's E2, whose exports are on the group's desktop.

## References

Titles, venues and DOIs were checked during the literature review, except where noted.

1. Lundberg & Lee. A Unified Approach to Interpreting Model Predictions. NeurIPS 2017. arXiv:1705.07874.
2. Lundberg et al. From local explanations to global understanding with explainable AI for trees. *Nature Machine Intelligence* 2, 56–67 (2020). DOI 10.1038/s42256-019-0138-9.
3. Kumar et al. Problems with Shapley-value-based explanations as feature importance measures. ICML 2020, PMLR 119:5491–5500.
4. Janzing, Minorics & Blöbaum. Feature relevance quantification in explainable AI: A causal problem. AISTATS 2020, PMLR 108:2907–2916.
5. Sundararajan & Najmi. The Many Shapley Values for Model Explanation. ICML 2020. arXiv:1908.08474.
6. Chen et al. True to the Model or True to the Data? arXiv:2006.16234 (workshop paper, not a full peer-reviewed paper).
7. Bilodeau et al. Impossibility theorems for feature attribution. *PNAS* 121(2):e2304406120 (2024). DOI 10.1073/pnas.2304406120.
8. Krishna et al. The Disagreement Problem in Explainable Machine Learning: A Practitioner's Perspective. *TMLR* (2024).
9. Hooker et al. A Benchmark for Interpretability Methods in Deep Neural Networks. NeurIPS 2019, pp. 9734–9745.
10. Adebayo et al. Sanity Checks for Saliency Maps. NeurIPS 2018. arXiv:1810.03292.
11. Ying et al. GNNExplainer. NeurIPS 2019. arXiv:1903.03894.
12. Duval & Malliaros. GraphSVX: Shapley Value Explanations for Graph Neural Networks. ECML PKDD 2021, pp. 302–318. DOI 10.1007/978-3-030-86520-7_19.
13. Faber, Moghaddam & Wattenhofer. When Comparing to Ground Truth is Wrong: On Evaluating GNN Explanation Methods. KDD 2021, pp. 332–341. DOI 10.1145/3447548.3467283.
14. Amara et al. GraphFramEx: Towards Systematic Evaluation of Explainability Methods for GNNs. LoG 2022, PMLR 198.
15. Amara, El-Assady & Ying. GInX-Eval: Towards In-Distribution Evaluation of GNN Explanations. arXiv:2309.16223 (preprint or workshop paper).
16. Rao et al. xFraud: Explainable Fraud Transaction Detection. *PVLDB* 15(3):427–436 (2022).
17. Jain & Wallace. Attention is not Explanation. NAACL-HLT 2019. DOI 10.18653/v1/N19-1357.
18. Wiegreffe & Pinter. Attention is not not Explanation. EMNLP-IJCNLP 2019, pp. 11–20. DOI 10.18653/v1/D19-1002.
19. Tang et al. GADBench: Revisiting and Benchmarking Supervised Graph Anomaly Detection. NeurIPS 2023 Datasets & Benchmarks, vol. 36, pp. 29628–29653.
20. Jesus et al. How can I choose an explainer? An Application-grounded Evaluation of Post-hoc Explanations. FAccT 2021, pp. 805–815. DOI 10.1145/3442188.3445941.
21. Amarasinghe et al. On the Importance of Application-Grounded Experimental Design for Evaluating Explainable ML Methods. AAAI 2024, 38(19):20921–20929. DOI 10.1609/aaai.v38i19.30082.
22. Bansal et al. Does the Whole Exceed its Parts? The Effect of AI Explanations on Complementary Team Performance. CHI 2021. DOI 10.1145/3411764.3445717.
