# Before submitting: items only the authors can complete

The manuscript (`main.pdf`) has no known factual errors against the saved results. The items below cannot be filled in from the project and must be done by the authors.

## Required items

- **Code repository.** Upload `release/medhyperdiff-paper-release.zip` (or its contents) to a public repository or an archive such as GitHub or Zenodo. In Methodology, under "Datasets", the last sentence says the code "will be released with the published article". Replace that with the link or DOI.
- **License.** Choose a license for the code before the release is public. The release bundles neither baseline; it tells users to clone TabDDPM (MIT license) and TabDiff from their upstream repositories at pinned commits.
- **Funding statement.** For example: "This work received no specific funding." Only use this if it is true.
- **Conflict-of-interest statement.**
- **Author contributions.** Many journals require this; the format depends on the venue.
- **Ethics.** The manuscript states that only public, de-identified data were used. This statement is in Methodology, under "Datasets". Confirm with VIT whether any institutional statement is required.
- **Approval.** All co-authors and the guide should approve the final claims.

## Venue

- **Section structure.** As requested, the manuscript contains only these numbered sections: Introduction, Literature Review, Methodology, Results and Discussion, and Conclusion. The abstract, index terms and references are also kept. Many venues require separate end statements, such as Acknowledgment, Data Availability, Funding and Conflict of Interest. Add these after the Conclusion only if the chosen venue asks for them.

- **Length.** The paper is 13 pages in IEEE conference format. Most IEEE conferences allow 6 to 8 pages. Journals such as the Journal of Biomedical Informatics or PLOS Digital Health accept this length but use their own templates.
- **Reformatting.** Reformat to the chosen venue's template, and move the case-study figures and Table VI to a supplement if space is limited.

## Optional, but likely requested by journal reviewers

- A matched ablation of the final model with and without the graph and the hypergraph.
- Tuned baselines on untouched NHANES cycles (2009–10, 2007–08) with participant-consistent splits.
- Stronger membership attacks.
- A downstream task tied to an intended clinical use.
- An external cohort.
