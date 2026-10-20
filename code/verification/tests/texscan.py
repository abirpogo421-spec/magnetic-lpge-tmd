"""Number scanner for the paper source (protocol sec. 5, allow-list policy).

scan(text) -> list of dict(value, start, end, context, inmath, display, exempt, rule)
 1. strips comments, \\claim{id}{...} (recorded separately), and the arguments of \\label \\ref \\eqref \\cref \\cite*
    \\includegraphics \\input \\url \\href \\hspace \\vspace \\setlength, optional [..] arguments and tabular column specs;
 2. scans for numerals and the number words three..twenty;
 3. applies the exemption rules (a)-(f).
Positions refer to the ORIGINAL text, so callers can map hits back to lines.
"""
import re

STRIP_ARG = ["label", "ref", "eqref", "cref", "Cref", "cite", "citep", "citet", "citealp", "includegraphics", "input",
             "url", "href", "hspace", "vspace", "setlength", "bibliography", "bibliographystyle", "graphicspath",
             "usepackage", "documentclass", "setcounter", "begin", "end", "newcommand", "renewcommand", "color",
             "affiliation", "email", "keywords", "pacs", "TODO", "textcolor"]
WORDS = ["three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "thirteen", "fourteen",
         "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty"]
DISPLAY_ENVS = ["equation", "equation*", "align", "align*", "eqnarray", "gather", "multline"]
NUM_RE = re.compile(r"(?<![A-Za-z0-9.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?![A-Za-z]*\d)")
WORD_RE = re.compile(r"\b(" + "|".join(WORDS) + r")\b", re.I)
REF_WORDS = re.compile(r"(Fig\.|Figs\.|Figure|Sec\.|Secs\.|Section|Eq\.|Eqs\.|Table|Tables|panel|panels|Appendix|Ref\.|Refs\.)[~\s]*\(?$")


def _mask(text, spans, ch=" "):
    t = list(text)
    for a, b in spans:
        for i in range(a, b):
            if t[i] != "\n":
                t[i] = ch
    return "".join(t)


def _brace_end(text, i):
    """text[i] == '{' -> index after the matching '}'."""
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{" and (j == 0 or text[j - 1] != "\\"):
            depth += 1
        elif text[j] == "}" and text[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    return len(text)


def claims_in(text):
    """All \\claim{id}{printed} occurrences: list of (id, printed, start, end)."""
    out = []
    for m in re.finditer(r"\\claim\{([^{}]+)\}\{", text):
        e = _brace_end(text, m.end() - 1)
        out.append((m.group(1), text[m.end():e - 1], m.start(), e))
    return out


def math_regions(text):
    """(start, end, display) for $..$, $$..$$, \\(..\\), \\[..\\] and display environments."""
    regs = []
    for env in DISPLAY_ENVS:
        for m in re.finditer(r"\\begin\{" + re.escape(env) + r"\}(.*?)\\end\{" + re.escape(env) + r"\}", text, re.S):
            regs.append((m.start(), m.end(), True))
    for m in re.finditer(r"\\\[(.*?)\\\]", text, re.S):
        regs.append((m.start(), m.end(), True))
    covered = _mask(text, [(a, b) for a, b, _ in regs], "#")
    i = 0
    while i < len(covered):
        if covered[i] == "$" and (i == 0 or covered[i - 1] != "\\"):
            dbl = covered[i:i + 2] == "$$"
            j = covered.find("$$" if dbl else "$", i + (2 if dbl else 1))
            while j > 0 and covered[j - 1] == "\\":
                j = covered.find("$", j + 1)
            if j < 0:
                break
            regs.append((i, j + (2 if dbl else 1), dbl))
            i = j + (2 if dbl else 1)
        else:
            i += 1
    for m in re.finditer(r"\\\((.*?)\\\)", text, re.S):
        regs.append((m.start(), m.end(), False))
    return sorted(regs)


def strip(text):
    """Mask comments, claim macros and non-content arguments (same length as text)."""
    spans = []
    for m in re.finditer(r"(?<!\\)%.*", text):
        spans.append((m.start(), m.end()))
    for _, _, a, b in claims_in(text):
        spans.append((a, b))
    for cmd in STRIP_ARG:
        for m in re.finditer(r"\\" + cmd + r"\*?(\s*\[[^\]]*\])*\s*\{", text):
            e = _brace_end(text, m.end() - 1)
            if cmd == "begin":                     # keep environment content, drop the name and a tabular spec
                name_end = e
                spans.append((m.start(), name_end))
                rest = text[name_end:name_end + 1]
                env = text[m.end():name_end - 1]
                if env.startswith("tabular") and rest == "{":
                    spans.append((name_end, _brace_end(text, name_end)))
                continue
            spans.append((m.start(), e))
    for m in re.finditer(r"\\setcounter\{[^{}]*\}\{[^{}]*\}", text):
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"\\\\\[[^\]]*\]", text):        # \\[6pt]
        spans.append((m.start(), m.end()))
    for m in re.finditer(r"\\begin\{(figure|table)\*?\}\[[^\]]*\]", text):
        spans.append((m.start(), m.end()))
    return _mask(text, spans)


def _in(regs, i):
    for a, b, d in regs:
        if a <= i < b:
            return True, d
    return False, False


def scan(text):
    s = strip(text)
    nocom = _mask(text, [(m.start(), m.end()) for m in re.finditer(r"(?<!\\)%.*", text)])
    nocom = _mask(nocom, [(a, b) for _, _, a, b in claims_in(nocom)], "x")
    regs = math_regions(nocom)
    bd = text.find("\\begin{document}")
    if bd > 0:                                   # main.tex: only the body counts
        s = " " * bd + s[bd:]
    hits = []
    # digits glued to a control word (\\pm10, \\approx0.05, 900\\times900) must still be found: blank control words
    s_num = re.sub(r"\\[td]?frac\d\d", lambda m: " " * len(m.group(0)), s)          # \\tfrac12 = 1/2: exempt (a)
    s_num = re.sub(r"\\[A-Za-z]+", lambda m: " " * len(m.group(0)), s_num)
    for m in NUM_RE.finditer(s_num):
        val = m.group(1)
        a, b = m.start(1), m.end(1)
        inmath, disp = _in(regs, a)
        before, after = s[max(0, a - 12):a], s[b:b + 8]
        ctx = text[max(0, a - 40):b + 40].replace("\n", " ")
        rule = None
        isint = re.fullmatch(r"\d+", val) is not None
        if inmath:
            if re.search(r"[\^_]\{?[-+]?$", before) or re.search(r"[\^_]\{[^{}]*$", before):
                rule = "a: exponent/subscript"
            elif isint and int(val) <= 12 and (re.match(r"\s*(\\[A-Za-z]|[A-Za-z(\[|/])", after) or re.search(r"/\s*$", before)):
                rule = "a: small integer attached to a symbol"
            elif isint and disp:
                rule = "b: integer in display math"
            elif re.search(r"\\times\s*$", before) and val == "10":
                rule = "d: shared \\times10^{n}"
            elif isint and int(val) <= 12 and re.search(r"\\(frac|tfrac|dfrac)\{?\s*$", before):
                rule = "a: small integer in a fraction"
            elif isint and int(val) <= 12 and (re.search(r"\\[td]?frac\{[^{}]*\}\{\s*$", s[max(0, a - 40):a]) or re.search(r"\\sqrt\{\s*$", before)):
                rule = "a: small integer in a fraction/root"
            elif isint and int(val) <= 12 and re.search(r"\{\s*$", before) and re.match(r"\s*\}\s*\{", after):
                rule = "a: small integer in a fraction"
        if rule is None and isint and 1900 <= int(val) <= 2099:
            near = text[max(0, a - 60):b + 60]
            if re.search(r"\\cite|et al\.|[A-Z][a-z]+ \\textit", near):
                rule = "e: year near citation"
        if rule is None and REF_WORDS.search(s[max(0, a - 14):a]):
            rule = "f: reference number"
        if rule is None and re.search(r"(Mo|W|M|X)\s*$", before) and re.match(r"\s*\}", after):
            rule = "c: chemical subscript"
        hits.append(dict(value=val, start=a, end=b, context=ctx, inmath=inmath, display=disp, exempt=rule is not None, rule=rule))
    for m in WORD_RE.finditer(s):
        a, b = m.start(1), m.end(1)
        hits.append(dict(value=m.group(1).lower(), start=a, end=b, context=text[max(0, a - 40):b + 40].replace("\n", " "),
                         inmath=False, display=False, exempt=False, rule=None))
    return sorted(hits, key=lambda h: h["start"])


def line_of(text, pos):
    return text.count("\n", 0, pos) + 1
