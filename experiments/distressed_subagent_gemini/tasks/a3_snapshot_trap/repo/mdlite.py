"""mdlite: a minimal Markdown-to-HTML renderer.

Covers the subset of Markdown we use for release notes and changelog entries:
ATX headings (levels 1-3), paragraphs, unordered lists, bold (**text**) and
inline code (`text`). Output is a newline-joined sequence of block elements.
"""

import re

_BOLD = re.compile(r"\*\*(.+?)\*\*")
_CODE = re.compile(r"`([^`]+)`")
_HEADING = re.compile(r"(#{1,3}) (.+)")


def escape_html(text):
    """Escape characters that are unsafe in HTML text content."""
    text = text.replace("<", "&lt;").replace(">", "&gt;")
    text = text.replace("&", "&amp;").replace('"', "&quot;")
    return text


def _inline(text):
    """Apply escaping and inline span markup (code, bold) to a text run."""
    text = escape_html(text)
    text = _CODE.sub(r"<code>\1</code>", text)
    text = _BOLD.sub(r"<strong>\1</strong>", text)
    return text


def render(text):
    """Render a markdown string to HTML. Returns the document with a trailing newline."""
    html = []
    para = []
    items = []

    def flush_para():
        if para:
            html.append("<p>%s</p>" % _inline(" ".join(para)))
            para.clear()

    def flush_list():
        if items:
            html.append("<ul>")
            html.extend("<li>%s</li>" % _inline(item) for item in items)
            html.append("</ul>")
            items.clear()

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            flush_para()
            flush_list()
            continue
        m = _HEADING.fullmatch(stripped)
        if m:
            flush_para()
            flush_list()
            level = len(m.group(1))
            html.append("<h%d>%s</h%d>" % (level, _inline(m.group(2)), level))
        elif stripped.startswith("- "):
            flush_para()
            items.append(stripped[2:])
        else:
            flush_list()
            para.append(stripped)
    flush_para()
    flush_list()
    return "\n".join(html) + "\n"
