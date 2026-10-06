#!/usr/bin/env python3
"""Render the bilingual visual abstract from the shipped aggregate evidence."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import tempfile


ASSETS = Path(__file__).resolve().parent
ROOT = ASSETS.parents[1]
arrival_path = ROOT / "evidence/arrival-analysis.json"
summary_path = ROOT / "evidence/numerical-theory-summary.json"
arrival = json.loads(arrival_path.read_text())
summary = json.loads(summary_path.read_text())


def interval(values):
    values = list(values)
    return [min(values), max(values)]


records = []
for run in arrival["runs"]:
    full = run["windows"]["full"]
    magnitudes = full["magnitudes"]
    records.append({
        "run": run["path"],
        "algorithm": run["algorithm"],
        "replica": run["replica"],
        "D_std_nats": magnitudes["difficulty"]["std"],
        "A_mean_nats": magnitudes["parameter_work"]["mean"],
        "Q2_mean_nats": magnitudes["quadratic"]["mean"],
        "epsilon_rms_nats": full["bands"]["raw"]["A_plus_Q2"]["absolute_rmse"],
    })

ranges = {}
for algorithm in ("sgd", "adam"):
    selected = [record for record in records if record["algorithm"] == algorithm]
    ranges[algorithm] = {
        key: interval(record[key] for record in selected)
        for key in ("D_std_nats", "A_mean_nats", "Q2_mean_nats", "epsilon_rms_nats")
    }
    ranges[algorithm]["response_relative_rms"] = interval(
        item["relative_rms"] for item in summary["slow_response"]
        if item["run"].endswith("-" + algorithm)
    )

data = {
    "schema": "rgi-visual-abstract-v1",
    "equation": "L = D + A + Q2 + epsilon",
    "scope": "Pythia-70M / FineWeb sample-10BT; retrospective local continuation",
    "table_window": [0, 512],
    "range_scope": "Three permutations of one shared data pool; not confidence intervals",
    "statistics": {"D": "standard deviation", "A": "mean", "Q2": "mean", "epsilon": "RMS"},
    "epsilon_definition": "L - D - A - Q2 = R + Q - Q2; not R alone",
    "records": records,
    "ranges": ranges,
    "common_reference_variance_explained": interval(
        item["common_reference_explained"] for item in summary["frozen_difficulty"]
    ),
    "paired_optimizer_difference_relative_rms": interval(
        item["relative_rms"] for item in summary["paired_incremental_failure"]
    ),
    "sources": {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (arrival_path, summary_path)
    },
}
ASSETS.mkdir(parents=True, exist_ok=True)
(ASSETS / "visual-abstract-data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")


def numeric_range(algorithm, metric, digits=6, signed=False):
    lower, upper = ranges[algorithm][metric]
    formatter = ("{:+." if signed else "{:.") + str(digits) + "f}"
    return formatter.format(lower).replace("-", r"$-$") + r"\ --\ " + formatter.format(upper).replace("-", r"$-$")


def percentage_range(values):
    return f"{values[0] * 100:.2f}--{values[1] * 100:.2f}\\%"


copy = {
    "en": {
        "title": "RGI explained through local response",
        "subtitle": "Shared batch difficulty dominates fluctuations; measured learning response changes with the optimizer.",
        "reference": r"Same true prefixes and targets; frozen continuation start $\theta_0$; $\Delta z=z_t-z_0$.",
        "table_scope": "Full window: 512 updates. Ranges span three permutations of one pool, not confidence intervals. Unit: nats.",
        "term_header": "Term / statistic",
        "measurement_header": "How it is measured",
        "difficulty": r"Frozen\\difficulty",
        "difficulty_detail": "Frozen model cross-entropy on the same batch.",
        "linear": "Linear work",
        "linear_detail": "Frozen batch gradient times actual parameter displacement.",
        "curvature": r"Softmax\\curvature",
        "curvature_detail": "Full 50,304-token vocabulary; average across target positions.",
        "residual": r"Measured\\remainder",
        "residual_detail": "Includes representation nonlinearity and curvature approximation.",
        "std": "batch standard deviation",
        "mean": "mean over batches",
        "rms": "RMS over batches",
        "result1": "Raw fluctuations",
        "result1_detail": "Common frozen checkpoint explains " + percentage_range(data["common_reference_variance_explained"]) + " of arriving-loss variance (near/far).",
        "result2": "Local response",
        "result2_detail": r"Negative $A$ lowers loss; positive $Q_2$ offsets part of the improvement.",
        "error": r"$A+Q_2$ relative RMS error: SGD " + percentage_range(ranges["sgd"]["response_relative_rms"]) + "; Adam " + percentage_range(ranges["adam"]["response_relative_rms"]) + " (near/far/full).",
        "limit": "Validation boundary",
        "limit_detail": "Paired optimizer-gap error " + percentage_range(data["paired_optimizer_difference_relative_rms"]) + r" exceeds the 10\% gate. Universal strong RGI remains unproved.",
        "footer": "Pythia-70M local continuation; observed displacements explain completed updates. The coefficient of D is an identity.",
    },
    "zh": {
        "title": "用局部响应解释 RGI",
        "subtitle": "共同 batch 难度主导损失波动；实测学习响应随优化器改变。",
        "reference": r"同一批真实前缀和目标；冻结 continuation 起点 $\theta_0$；$\Delta z=z_t-z_0$。",
        "table_scope": "全窗 512 次更新。范围跨同一数据池的三种排列，不是置信区间。单位为 nats。",
        "term_header": "具体项 / 统计量",
        "measurement_header": "如何测量",
        "difficulty": "冻结难度",
        "difficulty_detail": "冻结模型在同一批数据上的 cross-entropy。",
        "linear": "一阶作用",
        "linear_detail": "起点 batch 梯度与真实参数位移的点乘。",
        "curvature": "Softmax 曲率",
        "curvature_detail": "完整 50,304 词表；对目标位置取均值。",
        "residual": "实测余项",
        "residual_detail": "包含表征非线性与曲率近似误差。",
        "std": "batch 标准差",
        "mean": "batch 均值",
        "rms": "batch RMS",
        "result1": "原始波动",
        "result1_detail": "共同冻结 checkpoint 解释了 arriving-loss 方差的 " + percentage_range(data["common_reference_variance_explained"]) + "（近窗 / 远窗）。",
        "result2": "局部响应",
        "result2_detail": r"负 $A$ 使 loss 下降；正 $Q_2$ 抵消部分改善。",
        "error": r"$A+Q_2$ 相对 RMS 误差：SGD " + percentage_range(ranges["sgd"]["response_relative_rms"]) + "；Adam " + percentage_range(ranges["adam"]["response_relative_rms"]) + "（近窗 / 远窗 / 全窗）。",
        "limit": "验证边界",
        "limit_detail": "配对优化器差的误差为 " + percentage_range(data["paired_optimizer_difference_relative_rms"]) + r"，超过 10\% gate。普遍 strong RGI 尚未得到证明。",
        "footer": "Pythia-70M 局部 continuation；实际位移解释已完成的更新。D 的系数为 1 是分解恒等式。",
    },
}


def figure_source(language):
    labels = copy[language]
    parts = [r"""\documentclass[tikz,border=0pt]{standalone}
\usepackage{fontspec,xeCJK,amsmath}
\IfFontExistsTF{Arial}{\setmainfont{Arial}}{\setmainfont{Latin Modern Sans}}
\IfFontExistsTF{Noto Sans CJK SC}{\setCJKmainfont{Noto Sans CJK SC}}{\setCJKmainfont{FandolHei-Regular}}
\definecolor{ink}{HTML}{20252B}
\definecolor{muted}{HTML}{55616D}
\definecolor{accent}{HTML}{185D91}
\definecolor{rule}{HTML}{CED7DF}
\definecolor{wash}{HTML}{F4F7FA}
\begin{document}
\begin{tikzpicture}[x=1mm,y=-1mm]
\path[use as bounding box] (0,0) rectangle (300,257);
\fill[white] (0,0) rectangle (300,257);
"""]

    def text(x, y, content, width=276, size=11, color="ink", weight=""):
        parts.append(
            rf"\node[anchor=north west,align=left,text width={width}mm,inner sep=0pt,text={color},font=\fontsize{{{size}}}{{{size * 1.3}}}\selectfont {weight}] at ({x},{y}) {{{content}}};"
        )

    text(12, 10, labels["title"], size=25, weight=r"\bfseries")
    text(12, 25, labels["subtitle"], size=11.5, color="muted")
    parts.append(r"\node[anchor=center,text=ink,font=\fontsize{30}{36}\selectfont] at (150,46) {$L_t(B)=D(B)+A_t(B)+Q_{2,t}(B)+\varepsilon_t(B)$};")
    text(12, 59, labels["reference"], size=10.5, color="muted")
    text(12, 70, labels["table_scope"], size=10, color="muted")
    parts.append(r"\fill[wash] (12,80) rectangle (288,90);")
    text(15, 82.5, labels["term_header"], width=36, size=10, weight=r"\bfseries")
    text(58, 82.5, labels["measurement_header"], width=108, size=10, weight=r"\bfseries")
    text(183, 82.5, "SGD", width=45, size=11, weight=r"\bfseries")
    text(238, 82.5, "Adam", width=46, size=11, weight=r"\bfseries")
    parts.append(r"\draw[accent,line width=0.8pt] (12,90) -- (288,90);")
    for y in (115, 140, 165, 190):
        parts.append(rf"\draw[rule,line width=0.4pt] (12,{y}) -- (288,{y});")
    for x in (52, 177, 233):
        parts.append(rf"\draw[rule,line width=0.4pt] ({x},80) -- ({x},190);")

    rows = [
        ("D", "difficulty", "std", r"$\ell_{\theta_0}(B)$", "D_std_nats", 4, False),
        ("A", "linear", "mean", r"$g_B(\theta_0)^\top(\theta_t-\theta_0)$", "A_mean_nats", 6, True),
        ("Q_2", "curvature", "mean", r"$\frac12\,\operatorname{mean}_{i\in B}\operatorname{Var}_{p_{0,i}}(\Delta z_i)$", "Q2_mean_nats", 6, True),
        (r"\varepsilon", "residual", "rms", r"$L-D-A-Q_2=R+(Q-Q_2)$", "epsilon_rms_nats", 7, False),
    ]
    for index, (symbol, label, statistic, formula, metric, digits, signed) in enumerate(rows):
        y = 94 + 25 * index
        text(15, y, "$" + symbol + "$", width=10, size=17, color="accent")
        text(27, y + 1, labels[label], width=24, size=9.0, weight=r"\bfseries")
        text(15, y + 13, labels[statistic], width=36, size=8.5, color="muted")
        text(58, y + 0.5, formula, width=115, size=13)
        text(58, y + 12, labels[label + "_detail"], width=115, size=9.3, color="muted")
        for algorithm, x in (("sgd", 183), ("adam", 238)):
            text(x, y + 5, numeric_range(algorithm, metric, digits, signed), width=46, size=10.5)

    text(12, 198, labels["result1"], width=42, size=11, color="accent", weight=r"\bfseries")
    text(58, 198, labels["result1_detail"], width=230, size=11)
    text(12, 211, labels["result2"], width=42, size=11, color="accent", weight=r"\bfseries")
    text(58, 211, labels["result2_detail"], width=230, size=11)
    text(58, 218.5, labels["error"], width=230, size=9.3, color="muted")
    parts.append(r"\draw[rule,line width=0.4pt] (12,231) -- (288,231);")
    text(12, 235, labels["limit"], width=42, size=10, weight=r"\bfseries")
    text(58, 235, labels["limit_detail"], width=230, size=10)
    text(12, 248, labels["footer"], width=276, size=8.5, color="muted")
    parts.extend([r"\end{tikzpicture}", r"\end{document}"])
    return "\n".join(parts) + "\n"


for language in ("en", "zh"):
    stem = "visual-abstract-" + language
    with tempfile.TemporaryDirectory(prefix="rgi-visual-") as temporary:
        directory = Path(temporary)
        source = directory / (stem + ".tex")
        source.write_text(figure_source(language))
        result = subprocess.run(
            ["xelatex", "-no-shell-escape", "-interaction=nonstopmode", "-halt-on-error", "-output-directory", temporary, str(source)],
            capture_output=True, text=True,
        )
        if result.returncode:
            raise RuntimeError(result.stdout[-5000:])
        shutil.copyfile(directory / (stem + ".pdf"), ASSETS / (stem + ".pdf"))
    subprocess.run(["pdftocairo", "-svg", str(ASSETS / (stem + ".pdf")), str(ASSETS / (stem + ".svg"))], check=True)
    svg_path = ASSETS / (stem + ".svg")
    svg = svg_path.read_text().replace("<svg xmlns=", '<svg role="img" aria-labelledby="figure-title figure-description" xmlns=', 1)
    description = (
        "Four-term measured decomposition of local loss; shared difficulty dominates raw fluctuations, linear work and curvature explain local response, and the stricter paired optimizer-gap test fails. Values derive from included aggregate evidence."
        if language == "en" else
        "四项分解：共同难度主导原始波动，一阶作用和曲率解释局部响应，更严格的配对优化器差检验未通过。数值来自发布的汇总证据。"
    )
    marker = ">\n<defs>"
    assert marker in svg
    metadata = f'>\n<title id="figure-title">{copy[language]["title"]}</title>\n<desc id="figure-description">{description}</desc>\n<defs>'
    svg_path.write_text(svg.replace(marker, metadata, 1))
    subprocess.run(["pdftocairo", "-png", "-singlefile", "-r", "144", str(ASSETS / (stem + ".pdf")), str(ASSETS / stem)], check=True)
    print("Rendered", stem)
