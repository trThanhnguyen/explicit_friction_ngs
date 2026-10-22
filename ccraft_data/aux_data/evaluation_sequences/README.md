# evaluation_sequences — the 40-cell evaluation grid

The garment/motion grid the paper evaluates on: four garment templates from HOOD crossed with ten
AMASS CMU motions, giving 40 matched garment–motion sequences.

```
datasplits/
  comparison_seqs_to_snug.csv    the snug split (tshirt, tank, longsleeve)
  comparison_seqs_to_ssch.csv    the ssch split (dress)
pose_sequences/*.pkl             one SMPL pose sequence per motion, at h = 1/30 s
rest_geometries/{snug,ssch}.pkl  rest garment geometry per split
betas.pkl                        body shape parameters
verify_report.json               the conversion check for this tree
```

Every sequence carries its own body shape, so rollouts must be run with `--betas_source native`;
`tools/validate_cells.py` and `evaluate.py` already default to that. `rest_geometries` is indexed
by `betas_id` and is used only to build the rest state.

Numbers measured on this grid are not comparable with numbers measured on a full data split — the
grid is a matched subset, chosen so that every method is evaluated on exactly the same cells.

The pose sequences are derived from the CMU split of AMASS. See INSTALL.md for the AMASS download
and its licence terms.
