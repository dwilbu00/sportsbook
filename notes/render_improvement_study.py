"""Render the study's limited Markdown syntax to a standalone reading copy.

Standard library only. Also exports the report tables as separate CSV files.
No network, model execution, or production configuration access.
"""
import csv
import html
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'APPLICATION_IMPROVEMENT_STUDY_2026-09-08.md'


def inline(value):
    protected = []
    def code(match):
        protected.append('<code>' + html.escape(match[1]) + '</code>')
        return f'@@CODE{len(protected)-1}@@'
    value = re.sub(r'`([^`]+)`', code, value)
    value = html.escape(value)
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<a href="\2">\1</a>', value)
    value = re.sub(r'\[\^(\d+)\]', r'<sup><a href="#footnote-\1" aria-label="Source \1">\1</a></sup>', value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<strong>\1</strong>', value)
    value = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<em>\1</em>', value)
    for i, piece in enumerate(protected):
        value = value.replace(f'@@CODE{i}@@', piece)
    return value


def render():
    source = SOURCE.read_text(encoding='utf-8')
    lines = source.splitlines()
    footnotes = {}; body = []; toc = []; tables = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        match = re.match(r'\[\^(\d+)\]:\s*(.*)', line)
        if match:
            footnotes[match[1]] = match[2]; i += 1; continue
        match = re.match(r'(#{1,6}) (.*)', line)
        if match:
            depth = len(match[1]); title = match[2]
            anchor = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')
            body.append(f'<h{depth} id="{anchor}">{inline(title)}</h{depth}>')
            if depth == 2:
                toc.append(f'<li><a href="#{anchor}">{html.escape(title)}</a></li>')
            i += 1; continue
        if line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', c) for c in cells):
                    rows.append(cells)
                i += 1
            assert all(len(r)==len(rows[0]) for r in rows), 'Table width mismatch'
            tables.append(rows)
            header = '<tr>' + ''.join('<th scope="col">'+inline(c)+'</th>' for c in rows[0]) + '</tr>'
            cells = ''.join('<tr>'+''.join('<td>'+inline(c)+'</td>' for c in row)+'</tr>' for row in rows[1:])
            body.append('<div class="table-wrap"><table><thead>'+header+'</thead><tbody>'+cells+'</tbody></table></div>')
            continue
        if re.match(r'(?:- |\d+\. )', line):
            ordered = bool(re.match(r'\d+\. ', line)); tag = 'ol' if ordered else 'ul'
            items = []
            pattern = r'\d+\. ' if ordered else r'- '
            while i < len(lines) and re.match(pattern, lines[i].strip()):
                items.append('<li>'+inline(re.sub(pattern, '', lines[i].strip(), count=1))+'</li>')
                i += 1
            body.append(f'<{tag}>'+''.join(items)+f'</{tag}>'); continue
        paragraph = [line]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'[#|]|- |\d+\. |\[\^', lines[i].strip()):
            paragraph.append(lines[i].strip()); i += 1
        body.append('<p>'+inline(' '.join(paragraph))+'</p>')
    refs = ''.join(f'<li id="footnote-{number}">{inline(value)}</li>' for number,value in sorted(footnotes.items(),key=lambda item:int(item[0])))
    body.append('<section class="footnotes" aria-label="Citation notes"><ol>'+refs+'</ol></section>')
    css = '''
    :root { color-scheme: light; }
    * { box-sizing: border-box; }
    html { scroll-behavior: smooth; }
    body { margin: 0; background: #fff; color: #222; font: 17px/1.65 Georgia, "Times New Roman", serif; }
    nav { position: fixed; inset: 0 auto 0 0; width: 265px; padding: 32px 22px; overflow: auto; border-right: 1px solid #ddd; background: #fafafa; font: 13px/1.55 system-ui,sans-serif; }
    nav p { font-weight: 650; margin: 0 0 16px; }
    nav ul { padding: 0; margin: 0; list-style: none; }
    nav li { margin: 0 0 12px; }
    nav a { color: #444; text-decoration: none; }
    nav a:hover { color: #000; text-decoration: underline; }
    main { max-width: 1160px; margin-left: 265px; padding: 54px 64px 100px; }
    h1,h2,h3 { font-family: system-ui,sans-serif; color: #161616; line-height: 1.25; scroll-margin-top: 25px; }
    h1 { font-size: 38px; letter-spacing: -.8px; margin: 0 0 30px; }
    h2 { font-size: 26px; margin: 56px 0 22px; padding-top: 12px; border-top: 1px solid #ddd; }
    h3 { font-size: 20px; margin: 32px 0 16px; }
    p { margin: 0 0 19px; }
    a { color: #174963; text-decoration-thickness: 1px; text-underline-offset: 3px; }
    strong { font-weight: 700; }
    code { font: .85em/1.4 Consolas,monospace; background: #f3f3f3; padding: 2px 4px; overflow-wrap: anywhere; }
    sup { line-height: 0; font: 11px system-ui,sans-serif; }
    .table-wrap { overflow-x: auto; margin: 24px 0; }
    table { border-collapse: collapse; width: 100%; font: 13px/1.5 system-ui,sans-serif; }
    th { text-align: left; vertical-align: bottom; border-bottom: 2px solid #555; background: #f0f0f0; }
    td,th { padding: 10px; min-width: 75px; }
    td { border-bottom: 1px solid #ddd; vertical-align: top; }
    tbody tr:nth-child(even) { background: #fafafa; }
    li { margin-bottom: 9px; }
    .footnotes { margin-top: 40px; border-top: 1px solid #ccc; padding-top: 20px; font-size: 13px; }
    @media (min-width: 1600px) { main { margin-left: max(265px,calc((100vw - 1020px)/2)); } }
    @media (max-width: 950px) { nav { position: static; width: auto; padding: 22px; border-right: 0; border-bottom: 1px solid #ddd; } nav ul { columns: 2; } main { margin-left: 0; padding: 30px 22px 70px; } h1 { font-size: 31px; } }
    @media (max-width: 550px) { nav ul { columns: 1; } body { font-size: 16px; } }
    @media print { nav { display: none; } main { margin: 0; max-width: none; padding: 0; } body { font-size: 10pt; line-height: 1.45; } h1 { font-size: 24pt; } h2 { font-size: 17pt; margin-top: 26pt; } h3 { font-size: 13pt; } h1,h2,h3 { break-after: avoid; } tr { break-inside: avoid; } table { font-size: 8pt; } a { color: inherit; } .table-wrap { overflow: visible; } }
    '''
    page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sportsbook Application Improvement Study</title><style>'+css+'</style></head><body><nav aria-label="Contents"><p>Contents</p><ul>'+''.join(toc)+'</ul></nav><main>'+''.join(body)+'</main></body></html>'
    SOURCE.with_suffix('.html').write_text(page,encoding='utf-8')
    table_specs=[
        ('coverage','../improvement_study_20260908_inventory.json','Observed capability assessment, not a certification of every path'),
        ('mlb_opportunity','../improvement_study_20260908_mlb.json','Exploratory empirical H/AB component screen; not deployed D+xBA, no ROI or independent confirmation'),
        ('quote_comparison','../improvement_study_20260908_quotes.json','Matched stored team offers; not independent bets, accepted execution or ROI'),
        ('evaluation_metrics','../APPLICATION_IMPROVEMENT_STUDY_2026-09-08.md','Recommended evaluation framework; see report sources 4 and 5'),
        ('nfl_probability','../improvement_study_20260908_nfl.json','Expanding seasons, projected QB, inherited injury limitations; reused periods, not a production promotion'),
        ('performance','../improvement_study_20260908_performance.json','Local component timings; not deployed application or Azure latency'),
        ('experiments','../APPLICATION_IMPROVEMENT_STUDY_2026-09-08.md','Proposed experiments; adoption gates must be finalized before new evaluation'),
        ('roadmap','../APPLICATION_IMPROVEMENT_STUDY_2026-09-08.md','Planning estimates in focused engineering days; exclude future data accumulation and access delays'),
    ]
    assert len(tables)==len(table_specs)
    folder=ROOT/'improvement_study_20260908_tables'
    folder.mkdir(exist_ok=True)
    for number,(rows,spec) in enumerate(zip(tables,table_specs),1):
        # All table cells originate in the report; no external payloads/formulas.
        label,source_ref,context=spec
        path=folder/f'{number:02}_{label}.csv'
        with path.open('w',encoding='utf-8-sig',newline='') as out:
            writer=csv.writer(out)
            writer.writerow(rows[0]+['Source','Interpretation'])
            writer.writerows([[re.sub(r'\*\*|`','',cell) for cell in row]+[source_ref,context] for row in rows[1:]])
    print(f'Rendered {SOURCE.with_suffix(".html").name}; exported {len(tables)} tables.')


if __name__=='__main__':
    render()
