import re
import sys

src = open('/home/kyle/schwa_spgc/paper_draft.tex').read()

# Parse \bibitem[Display(Year)]{key} → {key: "Display, Year"}
cites = {}
for m in re.finditer(r'\\bibitem\[([^\]]+)\]\{([^}]+)\}', src):
    display, key = m.group(1), m.group(2)
    m2 = re.match(r'(.+?)\((\d{4}[a-z]?)\)', display)
    if m2:
        author = m2.group(1).strip().rstrip(',').strip()
        year = m2.group(2)
        author = author.replace(r'\&', '&').replace(r'\ ', ' ')
        cites[key] = (author, year)
    else:
        cites[key] = (display, '')

def fmt(keys, paren=True):
    parts = []
    for k in keys:
        k = k.strip()
        if k in cites:
            a, y = cites[k]
            parts.append(f"{a}, {y}" if y else a)
        else:
            parts.append(k)
    joined = '; '.join(parts)
    return f"({joined})" if paren else joined

def repl_citep(m):
    keys = m.group(1).split(',')
    return fmt(keys, paren=True)

def repl_citet(m):
    keys = m.group(1).split(',')
    parts = []
    for k in keys:
        k = k.strip()
        if k in cites:
            a, y = cites[k]
            parts.append(f"{a} ({y})" if y else a)
        else:
            parts.append(k)
    return '; '.join(parts)

out = re.sub(r'\\citep\{([^}]+)\}', repl_citep, src)
out = re.sub(r'\\citet\{([^}]+)\}', repl_citet, out)

open('/home/kyle/schwa_spgc/paper_inlined.tex', 'w').write(out)
print(f"Resolved {len(cites)} bibitems")
print("Sample keys:", list(cites.keys())[:3])
