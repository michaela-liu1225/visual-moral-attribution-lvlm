# Public dissertation copy

`dissertation_public_access.pdf` is derived from the final 93-page Overleaf
project downloaded on 14 September 2026. It preserves the dissertation text,
tables, references, captions, and pagination structure while making four
public-release changes:

1. the student number is removed;
2. UCL header and footer artwork is omitted;
3. every study image is replaced by a labelled neutral placeholder with the
   same aspect ratio; and
4. the unused legacy `arydshln` package is omitted to avoid its known conflict
   with current `array`/`tabularx` releases.

The complete examination PDF remains outside this public repository because it
contains third-party images and a student identifier.

The source is in `source/`. From that directory, a compatible TeX installation
can build the public copy with:

```bash
tectonic Report.tex
```

The included `Figures/` files are generated placeholders, not the original
study images.
