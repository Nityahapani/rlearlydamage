"""QA of the presentation split and its links; never trains or changes inference."""
from pathlib import Path
import hashlib
import json
import re
import subprocess
from urllib.parse import unquote, urlparse

import pymupdf
from build_primary_artifact_index import BASELINE, ROOT, build


def frozen(name):
    return subprocess.check_output(["git", "show", BASELINE + ":" + name], cwd=ROOT).decode()


def main():
    old = frozen("paper/main.tex")
    supplement = (ROOT / "paper/supplement.tex").read_text()
    labels = lambda s: set(re.findall(r"\\label\{([^}]+)\}", s))
    assert labels(old) <= labels(supplement), "A historical labelled result was removed"
    graphics = lambda s: re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", s)
    assert sorted(graphics(old)) == sorted(graphics(supplement)), "Historical figures changed"
    assert old.count(r"\begin{table}") == supplement.count(r"\begin{table}")
    assert "../results/summary/p14_single_result.tex" in supplement
    # The main numerical artifact, including every secondary/failed control, is untouched.
    name = "results/summary/p14_single_result-analysis.json"
    assert (ROOT / name).read_text() == frozen(name)
    csv_text, index = build(ROOT)
    assert csv_text == (ROOT / index["csv"]).read_text()
    assert index == json.loads((ROOT / "docs/p14_primary_artifact_index.json").read_text())
    report = {"role": "presentation and reproducibility QA, not new experimental evidence",
              "baseline": BASELINE, "all_old_label_count": len(labels(old)),
              "old_literal_table_count_retained": old.count(r"\begin{table}"),
              "old_graphic_uses_retained": len(graphics(old)),
              "primary_numerical_artifact_unchanged": True, "documents": {}}
    for stem in ["main", "supplement"]:
        path = ROOT / f"paper/{stem}.pdf"
        doc = pymupdf.open(path)
        text = "\n".join(page.get_text() for page in doc)
        assert "??" not in text, f"Unresolved label in {stem}"
        log = (ROOT / f"paper/{stem}.log").read_text()
        for error in ["Overfull", "undefined references", "Undefined control sequence",
                      "Float too large", "Too many unprocessed floats", "Label(s) may have changed"]:
            assert error not in log, (stem, error)
        source = (ROOT / f"paper/{stem}.tex").read_text()
        cited = {key.strip() for m in re.finditer(r"\\cite\w*(?:\[[^\]]*\])*\{([^}]+)\}", source)
                 for key in m.group(1).split(",")}
        bbl = (ROOT / f"paper/{stem}.bbl").read_text()
        assert all("{" + key + "}" in bbl for key in cited)
        links = 0
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                prefix = "https://github.com/Nityahapani/rlearlydamage/blob/"
                if not uri.startswith(prefix):
                    continue
                parsed = urlparse(uri)
                version, filename = unquote(parsed.path).split("/blob/", 1)[1].split("/", 1)
                assert (ROOT / filename).is_file(), uri
                if version != "main":
                    frozen_bytes = subprocess.check_output(["git", "show", version + ":" + filename], cwd=ROOT)
                    if parsed.fragment.startswith("L"):
                        numbers = [int(n) for n in re.findall(r"\d+", parsed.fragment)]
                        assert all(1 <= n <= len(frozen_bytes.splitlines()) for n in numbers), uri
                links += 1
        report["documents"][stem] = {"pages": len(doc), "bytes": path.stat().st_size,
                                              "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                              "citation_keys_checked": len(cited),
                                              "direct_artifact_links_checked": links}
    assert report["documents"]["main"]["pages"] <= 10
    report["primary_export"] = {"paired_rows": 32, "raw_groups": 8,
                                  "requested_seed_cells": 352, "full_donor_trees_archived": False,
                                  "fingerprints_are_four_seed_batches": True}
    (ROOT / "docs/presentation_validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
