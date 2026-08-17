"""Every label in a documentation diagram is checked against the box holding it.

SVG `<text>` does not wrap, shrink, or clip. Whether a label fits inside its
`<rect>` is decided entirely by the rendered width of the glyphs, and that width
depends on which font the *viewer* happens to have. These diagrams ask for
`Segoe UI, Arial, sans-serif`, so a reader on Windows sees the narrowest option
the author designed against, and a reader on Linux gets a substitute that runs
5-7% wider. A label authored to "just fit" therefore renders correctly for the
author and crosses its border for everyone else.

That is not hypothetical: `An environment and a policy model` cleared its box by
17px in Segoe UI and by 3.4px in DejaVu Sans, and two more labels overflowed
outright. Eyeballing the picture on the machine that drew it cannot catch this.

So this test measures real glyph extents in a headless browser -- the browser
does the text layout, so tspans, whitespace collapsing, and `text-anchor` need no
reimplementation here -- and requires each label to keep enough clearance that a
wider substitute font still fits. Lengthening a label past what its box can hold
fails here, which is the point.
"""
import html
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# The widest realistic substitute (DejaVu Sans, the usual Linux default) renders
# these strings 5-7% wider than Segoe UI. Demand 8% of the label's own width as
# headroom so the substitution still fits. Centred text grows away from both
# borders at once, so it only needs half that on either side.
FALLBACK_GROWTH = 0.08
MIN_CLEARANCE_PX = 2.0

# Rects smaller than this are decorative (colour-key swatches, icon bars) and are
# never the container of the label whose centre happens to land on them.
MIN_CONTAINER_AREA = 2000

_MEASURE_JS = r"""
const FILES = __FILES__;
const MIN_AREA = __MIN_AREA__;
const out = [];
const stage = document.getElementById('stage');

(async () => {
  for (const rel of FILES) {
    let text;
    try {
      text = await (await fetch(new URL(rel, document.baseURI).href)).text();
    } catch (e) {
      out.push({file: rel, error: 'fetch failed: ' + e});
      continue;
    }
    const parsed = new DOMParser().parseFromString(text, 'image/svg+xml');
    if (parsed.querySelector('parsererror')) {
      out.push({file: rel, error: 'not well-formed XML'});
      continue;
    }
    stage.replaceChildren(document.importNode(parsed.documentElement, true));
    const svg = stage.firstElementChild;

    const boxes = [...svg.querySelectorAll('rect')].map(r => {
      const b = r.getBBox();
      return {b, inset: (parseFloat(r.getAttribute('stroke-width')) || 0) / 2,
              area: b.width * b.height};
    }).filter(r => r.area >= MIN_AREA);

    const labels = [];
    for (const t of svg.querySelectorAll('text')) {
      if (!t.textContent.trim()) continue;
      let tb;
      try { tb = t.getBBox(); } catch (e) { continue; }
      if (!(tb.width > 0)) continue;
      const cx = tb.x + tb.width / 2, cy = tb.y + tb.height / 2;
      const holding = boxes.filter(r =>
        cx >= r.b.x && cx <= r.b.x + r.b.width &&
        cy >= r.b.y && cy <= r.b.y + r.b.height);
      if (!holding.length) continue;  // free-standing caption, no box to overflow
      const c = holding.reduce((a, b) => (a.area < b.area ? a : b));
      const left = tb.x - (c.b.x + c.inset);
      const right = (c.b.x + c.b.width - c.inset) - (tb.x + tb.width);
      labels.push({
        text: t.textContent.replace(/\s+/g, ' ').trim(),
        width: +tb.width.toFixed(2),
        clearance: +Math.min(left, right).toFixed(2),
        anchor: t.getAttribute('text-anchor') || 'start',
      });
    }
    out.push({file: rel, labels});
  }
  document.getElementById('out').textContent = JSON.stringify(out);
})();
"""

_PAGE = """<body><pre id="out"></pre>
<div id="stage" style="position:absolute;left:-99999px;top:0"></div>
<script>%s</script></body>
"""

_BROWSERS = (
    "chrome",
    "chromium",
    "chromium-browser",
    "google-chrome",
    "msedge",
    r"C:/Program Files/Google/Chrome/Application/chrome.exe",
    r"C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
)


def _browser() -> str | None:
    for name in _BROWSERS:
        found = shutil.which(name) or (name if os.path.isfile(name) else None)
        if found:
            return found
    return None


def _diagrams() -> list[str]:
    return sorted(
        p.relative_to(ROOT).as_posix()
        for p in ROOT.glob("**/docs/assets/*.svg")
        if ".local" not in p.parts and "build" not in p.parts
    )


def _measure(files: list[str]) -> list[dict]:
    """Render each diagram in a headless browser and return its label extents."""
    browser = _browser()
    if browser is None:
        pytest.skip("no Chrome/Chromium/Edge found to measure text extents")

    js = (
        _MEASURE_JS
        .replace("__FILES__", json.dumps(files))
        .replace("__MIN_AREA__", str(MIN_CONTAINER_AREA))
    )
    with tempfile.TemporaryDirectory() as tmp:
        # The harness fetches the SVGs as siblings, so it must live at the root.
        page = ROOT / f".svg-label-check-{os.getpid()}.html"
        page.write_text(_PAGE % js, encoding="utf-8")
        try:
            proc = subprocess.run(
                [
                    browser, "--headless", "--disable-gpu", "--no-sandbox",
                    "--allow-file-access-from-files", "--virtual-time-budget=20000",
                    f"--user-data-dir={tmp}", "--dump-dom", page.as_uri(),
                ],
                capture_output=True, text=True, timeout=180,
            )
        finally:
            page.unlink(missing_ok=True)

    match = re.search(r'<pre id="out">(.*?)</pre>', proc.stdout, re.DOTALL)
    if not match or not match.group(1).strip():
        pytest.skip(f"browser produced no measurements (exit {proc.returncode})")
    return json.loads(html.unescape(match.group(1)))


def _required(label: dict) -> float:
    """Clearance this label needs to survive a 8%-wider substitute font.

    Centred text spreads growth across both borders, so each side absorbs half.
    """
    share = 0.5 if label["anchor"] == "middle" else 1.0
    return max(MIN_CLEARANCE_PX, label["width"] * FALLBACK_GROWTH * share)


@pytest.fixture(scope="module")
def measured() -> dict[str, dict]:
    files = _diagrams()
    assert files, "no diagrams found under **/docs/assets/"
    return {r["file"]: r for r in _measure(files)}


def test_every_diagram_was_measured(measured):
    """Guard the harness: a silent fetch or parse failure must not pass as green."""
    broken = {f: r["error"] for f, r in measured.items() if "error" in r}
    assert not broken, f"diagrams could not be measured: {broken}"

    empty = [f for f, r in measured.items() if not r.get("labels")]
    assert not empty, f"no boxed labels measured, so nothing was checked: {empty}"


@pytest.mark.parametrize("diagram", _diagrams())
def test_labels_fit_their_boxes_in_a_wider_font(measured, diagram):
    tight = [
        f"{lab['clearance']:.1f}px clearance, needs {_required(lab):.1f}px"
        f" -- {lab['text'][:60]!r}"
        for lab in measured[diagram]["labels"]
        if lab["clearance"] < _required(lab)
    ]
    assert not tight, (
        f"{diagram}: label(s) too close to their box to survive a wider "
        "substitute font. Widen the box (or the canvas) rather than shrinking "
        "the text:\n  " + "\n  ".join(tight)
    )
