"""Flatten LaTeX inline math to plain Unicode for DOCX viewers that
don't render OMML well (Google Docs, many mobile previewers).

Reads paper_inlined.tex, writes paper_docx.tex with \(...\) content
converted to Unicode inline text.
"""
import re
import sys

src = open('/home/kyle/schwa_spgc/paper_inlined.tex').read()

UNICODE_SYMBOLS = [
    (r'\\eta', 'η'),
    (r'\\ell', 'ℓ'),
    (r'\\ge(?![a-z])', '≥'),
    (r'\\le(?![a-z])', '≤'),
    (r'\\pm', '±'),
    (r'\\approx', '≈'),
    (r'\\ldots', '…'),
    (r'\\times', '×'),
    (r'\\mid', '|'),
    (r'\\text\{([^}]*)\}', r'\1'),
    (r'\\texttt\{([^}]*)\}', r'\1'),
    (r'\\emph\{([^}]*)\}', r'\1'),
    (r'\\textbf\{([^}]*)\}', r'\1'),
    (r'\\overline\{([^}]+)\}', r'\1̄'),
    (r'\\lvert', '|'),
    (r'\\rvert', '|'),
    (r'\\,', ' '),
    (r'\\;', ' '),
    (r'\\!', ''),
    (r'\\ ', ' '),
    (r'\{=\}', '='),
    (r'\{,\}', ','),
]

SUPERSCRIPTS = {'0':'⁰','1':'¹','2':'²','3':'³','4':'⁴','5':'⁵','6':'⁶','7':'⁷','8':'⁸','9':'⁹','+':'⁺','-':'⁻','n':'ⁿ'}
SUBSCRIPTS   = {'0':'₀','1':'₁','2':'₂','3':'₃','4':'₄','5':'₅','6':'₆','7':'₇','8':'₈','9':'₉','s':'ₛ','n':'ₙ','w':'w'}

def apply_scripts(s):
    def sup(m):
        body = m.group(1)
        return ''.join(SUPERSCRIPTS.get(c, f'^{c}') for c in body)
    def sub(m):
        body = m.group(1)
        return ''.join(SUBSCRIPTS.get(c, f'_{c}') for c in body)
    s = re.sub(r'\^\{([^}]+)\}', sup, s)
    s = re.sub(r'\^(\w)', lambda m: SUPERSCRIPTS.get(m.group(1), '^'+m.group(1)), s)
    s = re.sub(r'_\{([^}]+)\}', sub, s)
    s = re.sub(r'_(\w)', lambda m: SUBSCRIPTS.get(m.group(1), '_'+m.group(1)), s)
    return s

def flatten(m):
    content = m.group(1)
    for pat, rep in UNICODE_SYMBOLS:
        content = re.sub(pat, rep, content)
    content = apply_scripts(content)
    # Strip remaining braces (after scripts have consumed theirs)
    content = re.sub(r'\{([^{}]*)\}', r'\1', content)
    # Clean up residual spacing macros
    content = content.strip()
    return content

out = re.sub(r'\\\(([^)]+?)\\\)', flatten, src)
# Also handle display math \[...\] the same way (rare here)
out = re.sub(r'\\\[([\s\S]+?)\\\]', flatten, out)

open('/home/kyle/schwa_spgc/paper_docx.tex', 'w').write(out)
print(f'wrote paper_docx.tex ({len(out)} chars)')

# Spot-check the NLTK line
for ln in out.split('\n'):
    if 'NLTK multi-source (' in ln:
        print('  sample:', repr(ln.strip()[:120]))
        break
