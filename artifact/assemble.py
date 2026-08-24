"""Assemble the artifact: concatenate parts, inline figures as data URIs, validate."""
import base64, re, sys
from pathlib import Path
from html.parser import HTMLParser

A = Path(__file__).resolve().parent
FIGDIR = Path("/home/user/rydberg-trackb/results/track_b/artifact")
PARTS = ["p1","p2","p3","p4","p5","p6","p7","p8a","p8b","p8c","p8d","p9","p10","p11","p12","p13"]
VOID = {"area","base","br","col","embed","hr","img","input","link","meta","param",
        "source","track","wbr"}

def inline(html):
    missing = []
    def sub(m):
        name = m.group(1)
        p = FIGDIR / f"{name}.png"
        if not p.exists():
            missing.append(name); return ""
        b64 = base64.b64encode(p.read_bytes()).decode()
        return f"data:image/png;base64,{b64}"
    out = re.sub(r"@@FIG:([A-Za-z0-9_]+)@@", sub, html)
    return out, missing

class Check(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True); self.stack=[]; self.err=[]
    def handle_starttag(self, tag, attrs):
        if tag not in VOID: self.stack.append(tag)
    def handle_endtag(self, tag):
        if tag in VOID: return
        if not self.stack: self.err.append(f"stray </{tag}>"); return
        if self.stack[-1]==tag: self.stack.pop()
        elif tag in self.stack:
            while self.stack and self.stack[-1]!=tag:
                self.err.append(f"unclosed <{self.stack.pop()}> before </{tag}>")
            if self.stack: self.stack.pop()
        else: self.err.append(f"stray </{tag}>")

def main():
    missing_parts = [p for p in PARTS if not (A/f"{p}.html").exists()]
    if missing_parts: print("MISSING PARTS:", missing_parts)
    body = "\n".join((A/f"{p}.html").read_text() for p in PARTS if (A/f"{p}.html").exists())
    body, missing = inline(body)
    if missing: print("!! MISSING FIGURES:", sorted(set(missing)))

    out = A/"artifact.html"; out.write_text(body)
    size = out.stat().st_size
    print(f"assembled: {size/1e6:.2f} MB  ({len(PARTS)-len(missing_parts)} parts)")

    c = Check(); c.feed(body)
    if c.stack: print("!! UNCLOSED AT EOF:", c.stack[:12])
    if c.err:   print("!! TAG ERRORS:", c.err[:12])
    if not c.stack and not c.err: print("tag balance: OK")

    # forbidden constructs for the artifact wrapper
    for bad in ("<!doctype", "<html", "<head>", "<body"):
        if bad in body.lower(): print(f"!! contains forbidden {bad}")
    # external hosts
    ext = re.findall(r'(?:src|href)="(https?://[^"]+)"', body)
    ext = [u for u in ext if "fonts.googleapis.com" not in u and "fonts.gstatic.com" not in u]
    if ext: print("!! EXTERNAL REFS:", ext[:5])
    else: print("external refs: none (fonts only)")
    # theme tokens
    for need in (':root{', 'prefers-color-scheme:dark', '[data-theme="dark"]'):
        print(f"theme {need!r}: {'OK' if need in body.replace(' ','') or need in body else 'MISSING'}")
    if size > 16e6: print("!! OVER 16MB")
    # placeholders left
    left = re.findall(r"@@[A-Z]+:[^@]*@@", body)
    if left: print("!! PLACEHOLDERS LEFT:", left[:5])

if __name__ == "__main__": main()
