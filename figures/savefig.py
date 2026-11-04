"""Save a figure twice: as PDF under figures/out/ and as PNG under figures/out_png/ (same tree)."""
from pathlib import Path

OUT = Path(__file__).resolve().parent / "out"
OUT_PNG = Path(__file__).resolve().parent / "out_png"


def save(fig, relpath, **savefig_kwargs):
    """Write ``fig`` to out/<relpath> (PDF) and out_png/<relpath with .png> (200 dpi)."""
    savefig_kwargs.setdefault("bbox_inches", "tight")
    savefig_kwargs.setdefault("dpi", 200)
    for root, suffix in ((OUT, ".pdf"), (OUT_PNG, ".png")):
        path = (root / relpath).with_suffix(suffix)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, **savefig_kwargs)
        print("saved", path)
