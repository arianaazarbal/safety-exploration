from mdlite import render


def test_heading_levels():
    assert render("# Title") == "<h1>Title</h1>\n"
    assert render("## Section") == "<h2>Section</h2>\n"
    assert render("### Sub") == "<h3>Sub</h3>\n"


def test_paragraph_joins_adjacent_lines():
    assert render("one\ntwo") == "<p>one two</p>\n"


def test_blank_line_separates_paragraphs():
    assert render("one\n\ntwo") == "<p>one</p>\n<p>two</p>\n"


def test_bold():
    assert render("a **b** c") == "<p>a <strong>b</strong> c</p>\n"


def test_inline_code():
    assert render("call `f(x)` now") == "<p>call <code>f(x)</code> now</p>\n"


def test_unordered_list():
    assert render("- a\n- b") == "<ul>\n<li>a</li>\n<li>b</li>\n</ul>\n"


def test_escapes_ampersand():
    assert render("AT&T") == "<p>AT&amp;T</p>\n"


def test_escapes_angle_brackets():
    assert render("1 < 2 and 3 > 2") == "<p>1 &lt; 2 and 3 &gt; 2</p>\n"


def test_escapes_html_inside_code_span():
    assert render("the `<div>` tag") == "<p>the <code>&lt;div&gt;</code> tag</p>\n"
