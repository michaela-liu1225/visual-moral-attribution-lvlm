# Chapter 4 recomputation

The script chapter4_recompute.py reproduces the descriptive results in
Chapter 4 from:

- the three canonical all-results CSV files in the project root;
- the restricted raw Qualtrics response ZIP supplied with --survey-zip; and
- variant_console_transcript.txt, a public copy of the retained visual-test
  console record with terminal user/host strings and provider response IDs
  removed. The model labels, confidence values, and brief reasons used by the
  parser are unchanged.

Run it from the project root:

    python analysis/chapter4_recompute.py --survey-zip /path/to/qualtrics-export.zip

The script writes JSON to standard output. The Qualtrics export is not copied
into this directory because it contains platform-generated identifiers,
network-derived metadata, response IDs, and timestamps.
