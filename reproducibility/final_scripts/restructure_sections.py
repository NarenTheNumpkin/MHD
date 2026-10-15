"""Rebuild main.tex into Abstract, Introduction, Literature Review, Methodology, Results and Discussion, Conclusion."""
from pathlib import Path

old = Path("ieee/main.tex").read_text()
L = old.split("\n")


def line(sub):
    hits = [i for i, l in enumerate(L) if sub in l]
    assert len(hits) == 1, (sub, len(hits))
    return hits[0]


def para(sub):
    return L[line(sub)]


def block(start, end):
    i = line(start)
    j = next(k for k in range(i, len(L)) if end in L[k])
    return "\n".join(L[i:j + 1])


def figure(label, path, caption, star=True):
    env = "figure*" if star else "figure"
    width = r"\textwidth" if star else r"\columnwidth"
    return (f"\\begin{{{env}}}[t]\n\\centering\n\\includegraphics[width={width}]{{figures/{path}}}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{{env}}}")


preamble = block(r"\documentclass[conference]{IEEEtran}", r"\end{IEEEkeywords}")
preamble = preamble.replace("\\usepackage{tikz}\n\\usetikzlibrary{arrows.meta,positioning}\n", "")

intro = "\n\n".join([para("Synthetic data are increasingly proposed as a way to share clinical tables"),
                      para("This work began as the development of a constraint-guided diffusion model"),
                      para("Our contributions are threefold."),
                      r"The rest of the paper is organized as follows. Section~\ref{sec:literature} reviews tabular generators, dependency-aware and constraint-aware generation, and the evaluation of synthetic data. Section~\ref{sec:method} describes the datasets, relation reparameterization, the model, the comparators, the seven training and evaluation pitfalls, the metrics and the experimental design. Section~\ref{sec:results} reports and discusses the results, and Section~\ref{sec:conclusion} concludes."])

literature = "\n\n".join([
    r"\subsection*{Tabular Data Generators}", para("Tabular data generators range from statistical copulas"),
    r"\subsection*{Dependency-Aware and Constraint-Aware Generation}", para("showed that common generators do not preserve functional dependencies"),
    r"\subsection*{Evaluation of Synthetic Data}", para("Evaluation commonly combines classifier two-sample tests")])

pitfalls = "\n\n".join(next(l for l in L if l.startswith(f"The {k} pitfall")) for k in ("first", "second", "third", "fourth", "fifth", "sixth", "seventh"))
pitfalls = pitfalls.replace("Section VI shows that the update budget", r"Section~\ref{sec:results} shows that the update budget")

overview = (r"MedHyperDiff is a mixed-type diffusion model with Gaussian diffusion for numerical values, absorbing-mask diffusion for categories and explicit missingness indicators (Fig.~\ref{fig:pipeline}). Its denoiser is a feature transformer with an attention bias from a bootstrap-stable association graph and nonlinear hyperedge message passing over supplied clinical relations. Relation reparameterization, described below, replaces one participant of each supplied laboratory relation by a residual before diffusion and rebuilds it after decoding. The study has five parts: a development case study that exposed the pitfalls, a controlled study that quantifies them, an experiment that decomposes the training correction, a first relation test on blood counts, and a robustness study across survey cycles, relation types and sample sizes."
            + "\n\n" + figure("fig:pipeline", "fig_pipeline.pdf",
                              "Overview of MedHyperDiff with relation reparameterization and its evaluation. Blue: components of the proposed method; green: evaluation and the knowledge-equivalent post-processing controls against which the method is compared."))

datasets = "\n\n".join([
    r"The development case study used four public UCI cohorts \cite{uci}: the Indian Liver Patient Dataset, heart failure, chronic kidney disease and early-stage diabetes.",
    para("The case study cannot separate the pitfalls from one model's idiosyncrasies").replace(
        "The case study cannot separate the pitfalls from one model's idiosyncrasies or from the noise of small test folds. The controlled study was specified in a dated protocol before any generator was fitted to its data. It uses",
        "The controlled study, which was specified in a dated protocol before any generator was fitted to its data, uses").replace(
        "and the UCI thyroid registry (9,172 records,", r"and the UCI thyroid registry \cite{thyroid} (9,172 records,"),
    r"The relation studies use complete blood count and chemistry panels of NHANES \cite{nhanes}. The first relation test uses the blood count panel of 2017 to 2018 (4,865 adults), whose analyzer measures red cell count, hemoglobin and mean corpuscular volume and computes hematocrit, mean corpuscular hemoglobin and mean corpuscular hemoglobin concentration, while the five white cell differential percentages sum to 100, all up to rounding of unrounded internal values. The panel had not been used before, but 4,753 of its participants also belong to the biochemistry population of the controlled study.",
    para("A second study, again specified in a dated protocol before fitting").replace(
        "A second study, again specified in a dated protocol before fitting, asked whether the relation result replicates in other survey cycles, whether it extends to relations of another kind, how it changes with sample size, and whether reparameterization also helps other generators. Four confirmation datasets",
        "The robustness study uses four confirmation datasets, which"),
    r"All datasets are publicly available and de-identified; no new data were collected and no participants were contacted. The source files are retrieved by scripts that check SHA-256 digests, and the source code with archived snapshots for each study, the dated protocols and amendments, run manifests and the scripts that regenerate every table and figure will be released with the published article."])
assert "The robustness study uses four confirmation datasets, which" in datasets
datasets = datasets.replace("The robustness study uses four confirmation datasets, which were built from NHANES public files:",
                            "The robustness study uses four confirmation datasets built from NHANES public files:")

relation_method = block(r"A supplied relation $k$ links a derived variable", r"over assignments with distinct targets that admit an acyclic reconstruction order")
model_training = block(r"In MedHyperDiff each residual and the participants it depends on", r"\end{table}")
assert "tab:config" in model_training
metric_block = block(r"We score relation $k$ by", r"the Wasserstein distance between absolute residuals of synthetic and held-out reference records")

comparators = (r"Five published generators serve as comparators: TabDiff \cite{shi2025tabdiff}, TabDDPM \cite{kotelnikov2023tabddpm} with its upstream training recipe, CTGAN and TVAE \cite{xu2019ctgan}, and a Gaussian copula, together with bootstrap resampling of training rows as a copy control. Their configurations were fixed in advance, and TabDiff and TabDDPM are adaptations of pinned upstream code. In the relation studies each published generator is evaluated as generated and with post-processings of the same output that use the same knowledge of the relations: deterministic re-derivation in the reconstruction direction of our rule, re-derivation of every derived variable in its textbook direction, which corresponds to rule-based reconstruction, and derivation plus real residuals, resampled either independently for each relation or jointly as the complete residual vector of one randomly drawn fit row with every residual observed, leaving a residual missing where the synthetic row lacks its participants. Each published generator is also fitted with reparameterization added, which tests the method as a plug-in. The textbook and joint controls were added after the confirmation outputs had been examined and were evaluated from saved outputs with the full metric set.")

metrics = (r"Realism is measured by C2ST, the AUROC of a random forest that separates synthetic from held-out real records, where 0.5 means indistinguishable. Utility is the AUROC of a random forest for the class label trained on synthetic and tested on real records (TSTR). Membership risk is the larger of the AUROCs of a nearest-distance attack and a density-ratio attack in the style of DOMIAS \cite{vanbreugel2023domias}, and every arm, including the proposed model, counts as eligible for privacy-constrained comparisons only if its mean membership AUROC is at most 0.60. The controlled study also reports numerical Wasserstein and association distances, rule violation rates and their gap to real held-out rates, rankings compared by the fraction of generator pairs whose order reverses and by Kendall's $\tau$ with bootstrap intervals over training samples, and the agreement of single two-sample tests on 50 to 1,000 records per source with the full reference."
           + "\n\n" + metric_block + " For chemistry and blood counts we also compare the joint distribution of relation-dependent clinical categories by total-variation distance: anemia by sex-specific hemoglobin thresholds, mean corpuscular volume class, hypochromia and low hemoglobin concentration for blood counts, and body mass index class, albuminuria stage, albumin to globulin ratio below one and osmolality range for chemistry.")
assert "We score relation" in metrics

design = "\n\n".join([
    r"The development case study audited ten development iterations of MedHyperDiff, applied the corrections one at a time on three development partitions of the liver dataset with three seeds each, and then evaluated a comparison specified in advance with equal tuning budgets once on reserved test folds of the four UCI cohorts.",
    para("Each population is split once into a reference half and a pool, grouped by identical predictor profile"),
    para("Every continuous-output generator emits raw values that are decoded after sampling in three ways"),
    r"The training decomposition, specified before it was run, refits MedHyperDiff on the 20 samples of the controlled study with eight training variants: the legacy recipe, the corrected recipe, the corrected recipe returning the checkpoint with the lowest validation loss instead of the final weights, the same with patience-based stopping or with the validation loss averaged over 16 corruption draws, the corrected recipe with the legacy number of updates, the corrected recipe with a constant learning rate and raw weights, and the legacy recipe rebuilt inside the fixed-budget loop.",
    r"In the first relation test the variant was chosen on a separate development pool with a rule fixed in advance and evaluated on a never-fitted reference half of 2,433 records with five samples of nominal size 250 and 1,000. During development, reconstructing the basophil fraction or the mean corpuscular hemoglobin concentration, which are narrow quantities, raised C2ST to between 0.84 and 0.88; the rule of Eq.~(\ref{eq:choice}) instead reconstructs red cell count, hemoglobin, mean corpuscular hemoglobin and the neutrophil percentage.",
    para("Each dataset was split once into a never-fitted reference half and a pool. Samples of nominal size 250"),
])
design = design.replace(" Published generators were evaluated as generated, with deterministic re-derivation in the reconstruction direction of our rule, with derivation plus independently resampled residuals, and fitted with reparameterization added. After the run and two external reviews we added two controls that use only saved outputs and the same knowledge of the relations: re-derivation of every derived variable in its textbook direction, which corresponds to rule-based reconstruction, and joint residual resampling, which gives each synthetic row the complete residual vector of one randomly drawn fit row with every residual observed and leaves a residual missing where the synthetic row lacks its participants. Both were evaluated with the full metric set. Every arm, including our model, counts as eligible for privacy-constrained comparisons only if its mean membership AUROC is at most 0.60.", "")
assert "After the run and two external reviews" not in design

methodology = "\n\n".join([
    r"\subsection*{Overview}", overview,
    r"\subsection*{Datasets}", datasets,
    r"\subsection*{Relation Reparameterization}", relation_method,
    r"\subsection*{Model and Training}", model_training,
    r"\subsection*{Comparators and Knowledge-Equivalent Controls}", comparators,
    r"\subsection*{Training and Evaluation Pitfalls}",
    r"We examine seven choices that can change which generator appears best; each is described with its correction, which together form the corrected pipeline.",
    pitfalls,
    r"\subsection*{Evaluation Metrics}", metrics,
    r"\subsection*{Experimental Design}", design])
methodology = methodology.replace(r"\subsection*{Choosing the Reconstructed Participant}", r"\subsection*{Choosing the Reconstructed Participant}")

# Results ---------------------------------------------------------------------------------------------------------------
case = "\n\n".join([
    para("The audit found that 13 of 36 fits in one iteration stopped during the initial plateau"),
    para("apply the corrections one at a time on three development partitions of the liver dataset"),
    block(r"\caption{Correcting one pitfall at a time", r"Bootstrap copy control & 0.493").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    figure("fig:ablation", "fig_ablation.pdf",
           r"Effect of each correction on the liver dataset development partitions (mean and standard deviation across three partitions). Blue shades accumulate the corrections; red bars are the original recipe and the corrected model with the original training loop; gray bars are references. Numeric Wasserstein is scaled by the training standard deviation and the rule gap is the difference between synthetic and real violation rates."),
    para("After correction, a comparison specified in advance with equal tuning budgets")])

uncorrected = "\n\n".join([
    para("We ranked eight generator identities by realism under an uncorrected pipeline"),
    block(r"\caption{Agreement between uncorrected and corrected pipelines", r"Thyroid, 1,000 & 26\%").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    figure("fig:pipeline_rev", "fig_reversals.pdf",
           r"Generator pairs whose order reverses between uncorrected and corrected pipelines among eight generator identities, for the realism ranking, the privacy-aware ranking and rule scoring (mean and 95\% bootstrap interval over five training samples).", star=False)])

training = "\n\n".join([
    para("With the legacy recipe, 30\\% of MedHyperDiff fits kept a checkpoint"),
    block(r"\caption{Decomposition of the training correction", r"Legacy recipe & 0.951").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    para("The earlier TabDDPM adapter loop on the upstream architecture diverged")])
training = training.replace("A follow-up experiment, specified before it was run, decomposed this difference on the same samples", "The decomposition experiment separates this difference on the same samples")

precision = "\n\n".join([
    para("Decoding changed some generators by up to 0.31 and others not at all"),
    block(r"\caption{C2ST by decoding", r"TabDDPM & 0.744 & 0.738").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    figure("fig:precision", "fig_precision.pdf",
           r"C2ST of raw outputs and of outputs decoded to the recorded resolution (mean of five training samples; 0.5 is indistinguishable).")])

rules = "\n\n".join([
    para("No learned generator reproduced the supplied relations.").replace(
        "Scoring by violation counts rather than by the gap to real rates reversed 11\\% to 22\\% of generator pairs but not the winner.",
        "Scoring by violation counts rather than by the gap to real rates reversed 11\\% to 22\\% of pairs among all variant and post-processing combinations and 24\\% to 39\\% of pairs among the eight generator identities of Fig.~\\ref{fig:pipeline_rev}, but in every sample the winner stayed the same."),
    para("Across generators, lower C2ST went with higher membership AUROC"),
    figure("fig:memo", "fig_memorization.pdf",
           r"Realism against membership risk for each generator (mean of five training samples). The shaded band marks a membership AUROC above 0.60 on at least one of two attacks.")])
assert "24\\% to 39\\%" in rules

evalsize = "\n\n".join([
    para("report the probability that a single two-sample test orders a generator pair"),
    block(r"\caption{Probability that one evaluation orders a pair correctly", r"Median single-draw SD").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    figure("fig:evalsize", "fig_evaluation_size.pdf",
           r"Agreement between a single two-sample test and the full reference ordering as a function of evaluation size. The dashed line marks 90\%.", star=False)])

first = "\n\n".join([
    block(r"\caption{Blood count 2017 to 2018 (mean of five samples).", r"& Membership & 0.509 & 0.560").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    figure("fig:relation", "fig_first_test.pdf",
           r"Relation distance on the 2017 to 2018 blood counts (logarithmic scale). For each post-processing the bar shows the best of the five published generators and the whisker reaches their median. Rule and textbook re-der.: deterministic re-derivation in the reconstruction direction of our rule or in the textbook direction; indep. and joint residuals: real residuals resampled independently per relation or jointly per record; ours: MedHyperDiff with relation reparameterization."),
    para("summarize the result. With $n = 250$ the model reproduced the four relations")])

robust = "\n\n".join([
    block(r"\begin{table*}[t]", r"\end{table*}"),
    r"\subsection*{Relation Fidelity Across Survey Cycles}",
    para("For the approximate blood count relations the model had the lowest mean relation distance"),
    para("For the exact chemistry relations the order reversed."),
    para("shows what approximate relations require."),
    block(r"\caption{Relation diagnostics: relation distance", r"TabDiff, as generated & 10.56").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    figure("fig:robust", "fig_robustness.pdf",
           r"Relation distance (top, logarithmic scale) and C2ST (bottom) on the four confirmation datasets by nominal sample size (mean over samples). Re-derivation is shown in the reconstruction direction of our rule and in the textbook direction; joint residuals are real residual vectors resampled after generation. The dashed line is MedHyperDiff with a fixed budget of 5,000 updates."),
    r"\subsection*{Realism, Utility and Sample Size}",
    para("Applying the membership threshold to every arm, the model was the most realistic eligible generator"),
    para("The proportional training budget explains the change from the first test."),
    para("Utility was measured as the AUROC of a random forest for diagnosed diabetes"),
    r"\subsection*{Clinical Categories and Membership}",
    para("We also compared the joint distribution of relation-dependent clinical categories").replace(
        "We also compared the joint distribution of relation-dependent clinical categories: anemia by sex-specific hemoglobin thresholds, mean corpuscular volume class, hypochromia and low hemoglobin concentration for blood counts, and body mass index class, albuminuria stage, albumin to globulin ratio below one and osmolality range for chemistry. Against",
        "For the relation-dependent clinical categories, against"),
    para("The mean membership AUROC of the model, the larger of a nearest-distance"),
    r"\subsection*{Reparameterization as a Plug-In and Computation}",
    para("Added to TabDiff, TabDDPM and the copula, reparameterization lowered the relation distance"),
    block(r"\caption{Computation of the diffusion models", r"TabDDPM & 0.66 M").join([r"\begin{table}[t]" + "\n", "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n" + r"\end{table}"]),
    para("shows that the model's results did not come from a larger model or more computation")])
assert "For the relation-dependent clinical categories, against" in robust

discussion = "\n\n".join([
    para("The seven pitfalls change which generator appears best."),
    para("The relation studies separate three situations."),
    para("Beyond relations, the proposed model was the most realistic generator below the membership threshold"),
    para("The results suggest a corrected protocol with seven elements.")])

results = "\n\n".join([
    r"\subsection*{Correcting a Development Project}", case,
    r"\subsection*{Uncorrected Versus Corrected Pipelines}", uncorrected,
    r"\subsection*{Training Choices}", training,
    r"\subsection*{Recording Resolution}", precision,
    r"\subsection*{Rules and Memorization}", rules,
    r"\subsection*{Evaluation Size}", evalsize,
    r"\subsection*{Relation Fidelity on the 2017 to 2018 Blood Counts}", first,
    r"\subsection*{Robustness Across Survey Cycles, Relation Types and Sizes}", robust,
    r"\subsection*{Discussion}", discussion,
    r"\subsection*{Limitations}", para("All relation datasets come from NHANES, analyzed without survey weights")])
results = results.replace("This model used 5,000 updates at every sample size, which the next section shows to be the cause of the latter.",
                          "This model used 5,000 updates at every sample size, which the robustness study below shows to be the cause of the latter.")

conclusion = para("Routine training and evaluation choices can determine which synthetic data generator appears best") + " " + \
    r"Future work should isolate the contribution of the hypergraph with a matched ablation, compare tuned baselines on untouched data with participant-consistent splits, apply stronger membership attacks, and evaluate hospital data from other laboratories and an intended clinical use."

bib = block(r"\begin{thebibliography}{00}", r"\end{thebibliography}")
bib = bib.replace(r"\end{thebibliography}",
                  r"\bibitem{uci} M. Kelly, R. Longjohn, and K. Nottingham, ``The UCI Machine Learning Repository,'' [Online]. Available: https://archive.ics.uci.edu" + "\n" +
                  r"\bibitem{thyroid} R. Quinlan, ``Thyroid Disease,'' UCI Machine Learning Repository, 1986." + "\n" + r"\end{thebibliography}")

doc = "\n\n".join([
    preamble,
    r"\section{Introduction}" + "\n\\label{sec:intro}", intro,
    r"\section{Literature Review}" + "\n\\label{sec:literature}", literature,
    r"\section{Methodology}" + "\n\\label{sec:method}", methodology,
    r"\section{Results and Discussion}" + "\n\\label{sec:results}", results,
    r"\section{Conclusion}" + "\n\\label{sec:conclusion}", conclusion,
    bib, r"\end{document}", ""])
Path("ieee/main_restructured.tex").write_text(doc)
print("sections:", [l for l in doc.split("\n") if l.startswith("\\section")])
