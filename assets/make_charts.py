"""results/ と diagnosis/ の実測値から、記事・README用のグラフ（SVG）を作る。

使い方（リポジトリ直下で）: python assets/make_charts.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def load(path):
    return json.loads((ROOT / path).read_text(encoding='utf-8'))


INK, SUB, GRID, TRACK, ACCENT, CARD = '#111827', '#6B7280', '#D1D5DB', '#EEF0F2', '#4D7C0F', '#FFFFFF'
FONT = "font-family=\"'Segoe UI','Hiragino Sans','Yu Gothic UI','Meiryo','Noto Sans JP',sans-serif\""
W = 880
TOTAL, SPEAKERS = 16, 8


def frame(h, title, subtitle, body):
    return '\n'.join([
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {h}" width="{W}" height="{h}" role="img" aria-label="{title}" {FONT}>',
        f'<rect x="0.5" y="0.5" width="{W - 1}" height="{h - 1}" rx="12" fill="{CARD}" stroke="{GRID}"/>',
        f'<text x="32" y="44" font-size="20" font-weight="700" fill="{INK}">{title}</text>',
        f'<text x="32" y="70" font-size="13.5" fill="{SUB}">{subtitle}</text>',
        *body, '</svg>'])


def write(name, svg):
    (ROOT / 'assets' / name).write_text(svg, encoding='utf-8', newline='\n')
    print('wrote', name)


diar = {r['name']: r for r in load('results/diarization-summary.json')}
native = load('results/native-clean-summary.json')
diag = {r['name']: r for r in load('diagnosis/summary.json') if r['threshold'] == 0.5}
tit = load('results/titanet-summary.json')
named = load('results/named-summary.json')
cast = load('experiment/manifest.json')['cast']
queries = load('results/titanet-queries.json')

# ---- 1. 条件別：正しく区別できた発話と検出人数 ----
rows = [
    ('順番発話・BGMなし', 'C++版 Q8', diar['clean']['turns_correct'], diar['clean']['detected_speaker_channels']),
    ('2人の重なりあり', 'C++版 Q8', diar['overlap']['turns_correct'], diar['overlap']['detected_speaker_channels']),
    ('順番発話・BGMあり', 'C++版 Q8', diar['bgm_minus20db']['turns_correct'], diar['bgm_minus20db']['detected_speaker_channels']),
    ('順番発話・BGMなし', 'Transformers版 FP32', native['correct_turns'], native['detected_speakers']),
    ('発話を最後まで残す', 'Transformers版 FP32', diag['conversation_full']['correct_turns'], diag['conversation_full']['detected_speakers']),
    ('自己紹介の先頭4秒', 'Transformers版 FP32', diag['introduction_4s']['correct_turns'], diag['introduction_4s']['detected_speakers']),
    ('自己紹介の先頭12秒', 'Transformers版 FP32', diag['introduction_12s']['correct_turns'], diag['introduction_12s']['detected_speakers']),
]
x0, bw, y0, rh = 250, 360, 124, 46
dx = 690
best = max(r[2] for r in rows)
body = [
    f'<text x="32" y="104" font-size="12.5" fill="{SUB}">条件</text>',
    f'<text x="{x0}" y="104" font-size="12.5" fill="{SUB}">正しく区別できた発話（満点{TOTAL}）</text>',
    f'<text x="{dx - 6}" y="104" font-size="12.5" fill="{SUB}">検出した人数（正解{SPEAKERS}人）</text>',
]
for i, (label, runtime, correct, detected) in enumerate(rows):
    y = y0 + i * rh
    w = correct / TOTAL * bw
    body += [
        f'<text x="32" y="{y + 13}" font-size="14" fill="{INK}">{label}</text>',
        f'<text x="32" y="{y + 30}" font-size="11.5" fill="{SUB}">{runtime}</text>',
        f'<rect x="{x0}" y="{y}" width="{bw}" height="22" rx="4" fill="{TRACK}"/>',
        f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="22" rx="4" fill="{ACCENT}"><title>{label}（{runtime}）: {correct}/{TOTAL}発話</title></rect>',
        f'<text x="{x0 + w + 10:.1f}" y="{y + 16}" font-size="14" font-weight="700" fill="{INK}">{correct}<tspan font-weight="400" fill="{SUB}"> / {TOTAL}</tspan></text>',
    ]
    for k in range(SPEAKERS):
        on = k < detected
        body.append(f'<circle cx="{dx + k * 15}" cy="{y + 11}" r="5" fill="{INK if on else "none"}" stroke="{INK if on else GRID}" stroke-width="1.5"/>')
    body.append(f'<text x="{dx + SPEAKERS * 15 + 4}" y="{y + 16}" font-size="13" fill="{INK}">{detected}人</text>')
yb = y0 + len(rows) * rh
body += [
    f'<line x1="{x0 + bw}" x2="{x0 + bw}" y1="{y0 - 8}" y2="{yb - 14}" stroke="{SUB}" stroke-dasharray="4 3"/>',
    f'<text x="{x0 + bw}" y="{yb + 4}" font-size="12" text-anchor="middle" fill="{SUB}">満点 {TOTAL}</text>',
]
write('chart-diarization.svg', frame(
    yb + 28, f'どの条件でも、正しく区別できたのは{TOTAL}発話中{best}発話まで',
    'Nemotron 3 Diarization。8人が2発話ずつ話す音声を、条件を変えて処理した結果', body))

# ---- 2. どこで失敗しているか：3つの試験の正答数 ----
steps = [
    ('声の照合だけ', '正しい区間を人が与え、TitaNetに8人から選ばせる', tit['closed_set_correct']),
    ('話者分離だけ', 'Nemotron 3が区間を分ける（順番発話・BGMなし）', diar['clean']['turns_correct']),
    ('2つをつなぐ', 'Nemotron 3の区間に、TitaNetで名前を付ける', named['correct']),
]
x0, bw, y0, rh = 400, 380, 104, 62
body = []
for i, (label, note, v) in enumerate(steps):
    y = y0 + i * rh
    w = v / TOTAL * bw
    body += [
        f'<text x="32" y="{y + 16}" font-size="15" font-weight="700" fill="{INK}">{label}</text>',
        f'<text x="32" y="{y + 35}" font-size="12" fill="{SUB}">{note}</text>',
        f'<rect x="{x0}" y="{y}" width="{bw}" height="28" rx="4" fill="{TRACK}"/>',
    ]
    if v:
        body.append(f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="28" rx="4" fill="{ACCENT}"><title>{label}: {v}/{TOTAL}発話</title></rect>')
    body.append(f'<text x="{x0 + w + 10:.1f}" y="{y + 20}" font-size="16" font-weight="700" fill="{INK}">{v}<tspan font-weight="400" fill="{SUB}"> / {TOTAL}</tspan></text>')
yb = y0 + len(steps) * rh
write('chart-pipeline.svg', frame(
    yb + 12, f'照合だけなら{steps[0][2]}発話正解。話者分離をはさむと{steps[2][2]}発話になる',
    f'{TOTAL}発話のうち、正しい話者（または正しい名前）を当てられた数', body))

# ---- 3. TitaNetの類似度と採用基準 ----
TH = tit['thresholds']['cosine']
lo, hi = 0.50, 0.80
x0, pw, y0, rh = 190, 560, 130, 36


def sx(v):
    return x0 + (v - lo) / (hi - lo) * pw


ids = list(cast)
yb = y0 + len(ids) * rh
body = []
for v in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
    body += [
        f'<line x1="{sx(v):.1f}" x2="{sx(v):.1f}" y1="{y0 - 10}" y2="{yb - 8}" stroke="{TRACK}"/>',
        f'<text x="{sx(v):.1f}" y="{yb + 10}" font-size="12" text-anchor="middle" fill="{SUB}">{v:.2f}</text>',
    ]
body += [
    f'<line x1="{sx(TH):.1f}" x2="{sx(TH):.1f}" y1="{y0 - 22}" y2="{yb - 8}" stroke="{INK}" stroke-dasharray="5 4" stroke-width="1.5"/>',
    f'<text x="{sx(TH) + 8:.1f}" y="{y0 - 16}" font-size="12" fill="{INK}">名前を出す基準 {TH}</text>',
    f'<text x="{x0 + pw / 2}" y="{yb + 32}" font-size="12.5" text-anchor="middle" fill="{SUB}">登録した声との類似度（cosine、1位の候補）</text>',
]
for i, sid in enumerate(ids):
    y = y0 + i * rh + 12
    body += [
        f'<text x="32" y="{y + 5}" font-size="14" fill="{INK}">{cast[sid]["name"]}</text>',
        f'<line x1="{x0}" x2="{x0 + pw}" y1="{y}" y2="{y}" stroke="{TRACK}" stroke-width="2"/>',
    ]
    for q in [q for q in queries if q['speaker'] == sid]:
        cx = sx(q['top1_cosine'])
        tip = f'{cast[sid]["name"]} → {q["top1_name"]} / {q["top1_cosine"]:.3f}'
        if q['correct_closed_set']:
            body.append(f'<circle cx="{cx:.1f}" cy="{y}" r="7" fill="{ACCENT}" stroke="{CARD}" stroke-width="2"><title>{tip}</title></circle>')
        else:
            body += [
                f'<circle cx="{cx:.1f}" cy="{y}" r="7" fill="{CARD}" stroke="{INK}" stroke-width="2"><title>{tip}</title></circle>',
                f'<path d="M{cx - 3:.1f} {y - 3}l6 6M{cx + 3:.1f} {y - 3}l-6 6" stroke="{INK}" stroke-width="2"/>',
                f'<text x="{cx + 12:.1f}" y="{y - 9}" font-size="11.5" fill="{INK}">{q["top1_name"]}と誤認</text>',
            ]
lx, ly = 32, yb + 58
body += [
    f'<circle cx="{lx + 7}" cy="{ly}" r="7" fill="{ACCENT}"/>',
    f'<text x="{lx + 22}" y="{ly + 5}" font-size="12.5" fill="{INK}">1位が本人（正解）</text>',
    f'<circle cx="{lx + 177}" cy="{ly}" r="7" fill="{CARD}" stroke="{INK}" stroke-width="2"/>',
    f'<path d="M{lx + 174} {ly - 3}l6 6M{lx + 180} {ly - 3}l-6 6" stroke="{INK}" stroke-width="2"/>',
    f'<text x="{lx + 192}" y="{ly + 5}" font-size="12.5" fill="{INK}">1位が別人（誤り）</text>',
]
cos = [q['top1_cosine'] for q in queries if q['correct_closed_set']]
write('chart-titanet.svg', frame(
    yb + 84, f'正解しても類似度は{min(cos):.2f}〜{max(cos):.2f}。基準{TH}の前後に集まる',
    f'TitaNet-Large。8人×2発話の{len(queries)}点。基準を超え、かつ2位と0.05以上の差があるときだけ名前を出す', body))
