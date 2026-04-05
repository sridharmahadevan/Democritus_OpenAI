#!/usr/bin/env python3
import argparse, csv, json, subprocess
from pathlib import Path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Folder with LCM JSON files (e.g., figs/sweep_wapo)")
    ap.add_argument("--pngdir", default="", help="Folder with PNGs (default: same as indir)")
    ap.add_argument("--pattern", default="*.json", help="Which JSONs to label (default: *.json)")
    ap.add_argument("--labels", default="labels.csv", help="CSV to write (inside indir)")
    ap.add_argument("--resume", action="store_true", help="Skip files already labeled")
    args = ap.parse_args()

    indir = Path(args.indir)
    pngdir = Path(args.pngdir) if args.pngdir else indir
    labels_path = indir / args.labels

    done = set()
    if args.resume and labels_path.exists():
        with labels_path.open("r", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                done.add(row["file"])

    files = sorted(indir.glob(args.pattern))
    files = [p for p in files if p.parent == indir]  # avoid subdirs

    write_header = not labels_path.exists()
    with labels_path.open("a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "focus", "decision"])
        if write_header:
            w.writeheader()

        for p in files:
            if args.resume and p.name in done:
                continue

            lcm = json.loads(p.read_text(encoding="utf-8"))
            focus = lcm.get("focus", "")

            png = pngdir / (p.stem + ".png")
            print("\n" + "-" * 80)
            print(f"FILE:  {p.name}")
            print(f"FOCUS: {focus}")
            print(f"PNG:   {png if png.exists() else '(missing)'}")

            if png.exists():
                # macOS: open image in Preview (non-blocking)
                subprocess.run(["open", str(png)], check=False)

            while True:
                ans = input("Keep this model? [y]es / [n]o / [s]kip / [q]uit > ").strip().lower()
                if ans in ("y", "n", "s", "q"):
                    break

            if ans == "q":
                print("Quitting.")
                return

            decision = {"y": "keep", "n": "reject", "s": "skip"}[ans]
            w.writerow({"file": p.name, "focus": focus, "decision": decision})
            f.flush()
            print(f"→ recorded: {decision}")

    print(f"\nDone. Labels written to {labels_path}")

if __name__ == "__main__":
    main()
