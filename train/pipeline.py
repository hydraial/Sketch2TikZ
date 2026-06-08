"""
Pipeline with dual feedback loops:
  N2↔N3: Compile self-heal (max 2 retries, .log errors fed back)
  N4↔N5: Visual self-heal (max 1 retry, vision critic diagnosis fed back)
"""
import os, re, subprocess, time, json, base64, shutil

from dotenv import load_dotenv
load_dotenv(override=True)

from train.llm_caller import image_to_text, text_to_text, _create
from train.llm_caller import CODE_MODELS, VISION_MODELS, VISION_PLATFORMS, CODE_PLATFORMS
from train.contract import SampleResult
from train.prompts import load_prompts

XELATEX = os.getenv("XELATEX_PATH", "xelatex")

# ── Prompts (loaded per-difficulty from train/prompts/) ──
_VISION_PROMPT_CACHE = {}
_CODE_SYSTEM_CACHE = {}


# ── Temporary universal prompt override for A/B benchmark ──
# To restore per-difficulty loading, revert the two functions below.

_UNIVERSAL_VISION_PROMPT = (
    "Describe this diagram with maximum precision for TikZ code generation.\n\n"
    "OUTPUT FORMAT: Use a TikZ-like specification that maps directly to draw commands. "
    "Prefer notation the code generator can copy verbatim.\n\n"
    "FORMULAS: Every mathematical expression MUST be written in exact LaTeX notation "
    "(e.g. $\\sum_{i=1}^{n} x_i$, $\\frac{a}{b}$, $\\alpha$, $\\rightarrow$). "
    "Never describe formulas in plain English — output the exact LaTeX.\n\n"
    "SHAPES: Count and name every shape precisely. For each shape, state:\n"
    "- Type: rectangle, circle, ellipse, straight line, curved arrow, dashed line, etc.\n"
    "- Position: exact relative location (center, top-left, bottom-right, between X and Y)\n"
    "- Size: relative scale (large, small, same width as X, half the height of Y)\n"
    "- Style: solid, dashed, dotted, thick, thin, color, filled/hollow\n"
    "Use polar coords for circular diagrams: (angle:radius). Use cartesian for grid/bar/flowchart: (x,y).\n\n"
    "LINES & ARROWS: For every connector, state: start point, end point, "
    "direction (→, ←, ↔), style (straight, curved, right-angle), "
    "and any labels on or near it.\n\n"
    "TOPOLOGY & DEPTH: For every shape, explicitly state:\n"
    "- OPEN vs CLOSED: Is the shape fully enclosed, or does it have gaps / "
    "extending line segments that do NOT connect back to the start?\n"
    "- 3D STRUCTURE: If a polyhedron, count visible faces, edges, internal edges. "
    "Do NOT reduce it to a flat 2D triangle.\n"
    "- EXTENDING SEGMENTS: Are there lines that continue beyond the main body? "
    "State their direction and length.\n\n"
    "SPECIAL SHAPES:\n"
    "- For graphs: specify vertex symbols ($*$ vs filled dot vs circle). "
    "For self-loops: specify angular position (top/bottom/left/right) and relative size.\n"
    "- For symmetric arc-cutout shapes: describe arc centers, radii, and the resulting central shape.\n"
    "- For 3D isometric views: state projection type and viewing angles (e.g. 'tdplot_main_coords theta=60 phi=120').\n"
    "- For fractal/recursive patterns: state depth/order, branch colors, and symmetry.\n"
    "- For divided circles/wedges: state dividing line angles, whether double-stroked, and wedge sizes.\n"
    "- For curved/feedback arrows: specify exact start and end connection points, "
    "and whether clockwise or counterclockwise.\n\n"
    "OUTPUT CHECKLIST — verify ALL of the following are included:\n"
    "- Every visible line, segment, border, and outline (including outer bounding boxes)\n"
    "- Every arrowhead and its direction (from→to)\n"
    "- Every label with exact subscript/superscript notation\n"
    "- Every tick mark on axes, circles, or curves\n"
    "- Every dashed, dotted, or hatched line/region\n"
    "- Every filled region or shaded area (including pattern direction)\n"
    "- Every curved line or arc (with start/end points and routing)\n"
    "- Every node/vertex shape and its color\n"
    "- The outer bounding box or frame of the entire figure, if present\n\n"
    "Multi-panel figures: describe each panel (a), (b), (c) separately with its own layout.\n\n"
    "LAYOUT: Describe the overall spatial arrangement. Are elements in a row, "
    "column, grid, tree, or free-form? What is the relative spacing?"
)

_UNIVERSAL_CODE_SYSTEM = (
    "You are a TikZ LaTeX expert. Generate correct, compilable TikZ code.\n"
    "RULES:\n"
    "1) First line: \\documentclass[tikz, border=2pt]{standalone}\n"
    "2) Output ONLY raw LaTeX. No markdown, no explanation.\n"
    "3) No \\usepackage{inputenc}, \\usepackage{fontenc}, or [pdftex] driver.\n"
    "4) No \\ensuremath in node styles.\n"
    "5) Every formula in the description MUST appear verbatim in LaTeX math mode.\n"
    "6) \\draw[->] for arrows, \\node[draw,circle] for circled nodes, "
    "\\node[draw,rectangle] for boxes.\n"
    "7) Every shape in the description MUST be rendered. Count them: if the "
    "description says N circles, your code must have N circles.\n"
    "8) Lines: straight is --, curved is .. controls .., right-angle is -| or |-.\n"
    "9) Match the description's layout exactly: row, column, grid, or tree.\n"
    "10) OPEN SHAPES: If the description says a shape has gaps or extending segments, "
    "use \\draw to draw each edge individually. Do NOT use -- cycle to force closure.\n"
    "11) EXTENDING SEGMENTS: If the description mentions lines that extend beyond the main body, "
    "make those segments at least as long as the main shape itself. Do NOT draw tiny stub lines.\n"
    "12) FILLED DOTS: only place \\fill (X) circle (2pt) at exactly the vertices the description specifies. "
    "Do NOT add dots at every vertex automatically.\n"
    "13) ARC AND LINE LABELS: place labels via 'node[midway, above] {label}' directly on the \\draw command. "
    "NEVER place labels at separate unconnected coordinates.\n"
    "14) POLAR COORDINATES: for circular/radial diagrams, use (angle:radius). "
    "Define \\def\\R{2cm} for radius, use \\coordinate.\n"
    "15) COLORS: use \\definecolor{name}{HTML}{hex} for precise colors. "
    "Match the description's colors exactly — don't substitute generic 'red'.\n"
    "16) SELF-LOOPS: use edge [in=<angle>,out=<angle>,loop] with explicit angles "
    "(top=70/110, right=0/30, bottom=270/290, left=150/180).\n"
    "17) SYMMETRIC ARCS: for quarter-circle cutouts, chain arc commands with -- connectors. "
    "Ensure arcs share endpoints at edge midpoints.\n"
    "18) 3D PROJECTIONS: use \\usepackage{tikz-3dplot} + \\tdplotsetmaincoords{60}{120} + [tdplot_main_coords]. "
    "Include dashed projection wireframe.\n"
    "19) FRACTAL/RECURSIVE: use \\usetikzlibrary{lindenmayersystems} + \\pgfdeclarelindenmayersystem with production rules.\n"
    "20) DIVIDED CIRCLES: use double, double distance=2mm for parallel-line cut edges. "
    "Draw sectors with \\clip on the circle + radial lines at specified angles.\n"
    "21) MATRIX/TABLE: use \\usetikzlibrary{matrix,fit}. Use matrix of nodes with nodes in empty cells. "
    "Place operators between matrices via right=of <matrix> at mid-height.\n"
    "22) HATCHED/SHADED regions: use \\fill[pattern=north east lines, pattern color=...]. Do not omit.\n"
    "23) TICK MARKS on axes: use \\draw (x,ymin) -- (x,ymin-0.1) with explicit positions. Do not omit.\n"
    "24) CURVED FEEDBACK ARROWS: use .. controls +(left:Xcm) and +(left:Xcm) .. for smooth bends.\n"
    "25) Copy labels verbatim — subscripts, superscripts, primes, Greek letters. Do NOT rename variables.\n"
    "26) No unused packages, no commented-out blocks.\n"
    "27) NEVER use plot[samples>100] or \foreach with >200 iterations — this causes TeX 'Dimension too large' crash.\n"
    "28) In \pgfmathsetmacro, avoid large-number multiply-then-divide (e.g. 360*\k/\N). Reorder as \k/\N*360 or 360/\N*\k.\n"
    "EXAMPLE — simple node + arrow:\n"
    "\\documentclass[tikz, border=2pt]{standalone}\n"
    "\\begin{document}\n"
    "\\begin{tikzpicture}\n"
    "  \\node[draw, circle] (A) at (0,0) {$x_1$};\n"
    "  \\node[draw, rectangle] (B) at (2,1) {$\\sum_{i=1}^{n}$};\n"
    "  \\draw[->, thick] (A) -- (B);\n"
    "\\end{tikzpicture}\n"
    "\\end{document}\n"
    "EXAMPLE — open quadrilateral with extending diagonal legs:\n"
    "\\documentclass[tikz, border=2pt]{standalone}\n"
    "\\begin{document}\n"
    "\\begin{tikzpicture}\n"
    "  \\coordinate (TL) at (0,2);\n"
    "  \\coordinate (TR) at (2,2);\n"
    "  \\coordinate (BL) at (0,0);\n"
    "  \\coordinate (BR) at (2,0);\n"
    "  \\draw[thick] (TL) -- (BL);          % left vertical\n"
    "  \\draw[dashed] (TL) -- (TR);         % top dashed\n"
    "  \\draw[thick] (TR) -- (BR);          % right vertical\n"
    "  \\draw[thick] (TL) ++(-1.5,1.5) -- (TL);  % upper-left extending leg (LONG)\n"
    "  \\draw[thick] (BR) -- ++(1.5,-1.5);       % lower-right extending leg (LONG)\n"
    "  \\fill (TL) circle (2pt);\n"
    "  \\fill (TR) circle (2pt);\n"
    "\\end{tikzpicture}\n"
    "\\end{document}"
)


def get_vision_prompt(difficulty: str = "easy") -> str:
    return _UNIVERSAL_VISION_PROMPT


def get_code_system(difficulty: str = "easy") -> str:
    return _UNIVERSAL_CODE_SYSTEM


CRITIC_PROMPT = (
    "You are evaluating how well a generated figure matches a reference image. "
    "Image 1 is the REFERENCE (ground truth). Image 2 is the GENERATED output. "
    "Compare them on: shapes, line styles, colors, topology, connections, "
    "relative positions, spatial layout, and aspect ratio. "
    "CRITICAL: If shapes are severely stretched, squashed, or distorted vs "
    "the reference, assign score = 0.0. "
    "If key elements from the reference are entirely missing, score <= 1.0. "
    "If all elements present but positions/colors differ, score 1.0-2.0. "
    "If minor differences only, score 2.0-5.0. "
    "If there are any differences that are easily discernible to the naked eye—"
    "such as the absence of visually prominent lines—the score should not exceed 3.0. "
    "Output ONLY a JSON object, no markdown, no explanation:\n"
    '{"score": <float 1.0-5.0>, "is_pass": <true/false>, '
    '"diagnosis": "<one sentence describing the main difference>"}'
)


# ── Helpers ──────────────────────────────────────────
def _fix(code: str) -> str:
    code = re.sub(r'\\usepackage\[pdftex\]', r'\\usepackage', code)
    code = re.sub(r'\\usepackage\[pdftex,\s*', r'\\usepackage[', code)
    code = re.sub(r',\s*pdftex\]', r']', code)
    for pkg in ["MnSymbol", "mathrsfs"]:
        code = code.replace(r"\usepackage{" + pkg + "}", r"% removed")
        code = code.replace("," + pkg, "").replace(pkg + ",", "")
    # Cap samples to prevent TeX "Dimension too large" overflow
    code = re.sub(r'samples\s*=\s*(\d{3,})', r'samples=100', code)
    # Fix common PGF math overflow: reorder large-number multiplication
    # e.g. \pgfmathsetmacro{\t}{360*\k/\N} -> \pgfmathsetmacro{\t}{\k/\N*360}
    code = re.sub(
        r'(\\pgfmathsetmacro\{[^}]+\}\{)(\d+)(\*[^{}/\n]+/[^}]+)\}',
        lambda m: f'{m.group(1)}{m.group(3)[1:]}*{m.group(2)}' + "}",
        code,
    )
    return code


def _clean(raw: str) -> str:
    c = raw.strip()
    for p in ["```latex", "```tex", "```tikz", "```"]:
        if c.startswith(p): c = c[len(p):].strip()
    if c.endswith("```"): c = c[:-3].strip()
    return c


def _compile(tex_path: str, pdf_path: str) -> tuple:
    """Returns (ok: bool, errors: str)"""
    tex_abs = os.path.abspath(tex_path)
    log_abs = tex_abs.replace(".tex", ".log")
    try:
        subprocess.run([XELATEX, "-interaction=nonstopmode", tex_abs],
                       capture_output=True, text=True, timeout=60,
                       cwd=os.path.dirname(tex_abs) or ".")
        if os.path.exists(pdf_path) and os.path.getsize(pdf_path) > 0:
            return True, ""
        if os.path.exists(log_abs):
            with open(log_abs, "r", encoding="utf-8", errors="replace") as f:
                lines = [l.strip() for l in f if l.startswith("! ")]
                err_text = "\n".join(lines[-10:])
                return False, err_text
        return False, "(no log)"
    except subprocess.TimeoutExpired:
        return False, "Compile timeout"
    except FileNotFoundError:
        return False, f"XeLaTeX not found: {XELATEX}"


def _reduce_samples_for_overflow(code: str) -> str:
    """Emergency fix for Dimension too large: halve all samples values."""
    def halver(m):
        n = int(m.group(1))
        return f'samples={max(20, n // 2)}'
    return re.sub(r'samples\s*=\s*(\d+)', halver, code)


def _reduce_foreach_loops(code: str) -> str:
    """Emergency fix: reduce \foreach iteration counts > 100."""
    def cap(m):
        n = int(m.group(1))
        if n > 100:
            return f'{m.group(2)}100'
        return m.group(0)
    # Match \Nlines{120} or \def\Nlines{120}
    return re.sub(r'(\\[a-zA-Z]+\{)(\d{3,})(\})', cap, code)


def _gs() -> str:
    for name in ["gs", "gswin64c", "gswin64"]:
        f = shutil.which(name)
        if f: return f
    root = os.path.dirname(os.path.dirname(os.getenv("CONDA_PREFIX", "")))
    for sub in ["Library/bin/gs.exe", "Library/bin/gswin64c.exe"]:
        p = os.path.join(root, sub)
        if os.path.exists(p): return p
    raise FileNotFoundError("Ghostscript not found")


def _pdf_to_png(pdf_path: str, png_path: str) -> bool:
    try:
        subprocess.run([_gs(), "-dNOPAUSE", "-dBATCH", "-dSAFER",
                        "-sDEVICE=png16m", "-r150", "-dFirstPage=1", "-dLastPage=1",
                        f"-sOutputFile={png_path}", pdf_path],
                       capture_output=True, text=True, timeout=30)
        return os.path.exists(png_path) and os.path.getsize(png_path) > 0
    except Exception:
        return False


def _encode_img(path: str) -> str:
    with open(path, "rb") as f: data = base64.b64encode(f.read()).decode("utf-8")
    ext = os.path.splitext(path)[1].lower()
    return f"data:{'image/png' if ext=='.png' else 'image/jpeg'};base64,{data}"


def _internal_critic(original_path: str, pdf_path: str, output_dir: str) -> dict:
    """Internal visual critic for feedback loop (not the sealed judge)."""
    png_path = os.path.join(output_dir, "critic_internal.png")
    if not _pdf_to_png(pdf_path, png_path):
        return {"score": 0.0, "is_pass": False, "diagnosis": "PDF render failed"}
    b64_orig = _encode_img(original_path)
    b64_gen = _encode_img(png_path)
    critic_msgs = [{"role": "user", "content": [
        {"type": "text", "text": "Image 1 (REFERENCE):"},
        {"type": "image_url", "image_url": {"url": b64_orig}},
        {"type": "text", "text": "Image 2 (GENERATED):"},
        {"type": "image_url", "image_url": {"url": b64_gen}},
        {"type": "text", "text": CRITIC_PROMPT},
    ]}]

    # Try vision platforms in order via fallback
    raw = None
    for p in VISION_PLATFORMS:
        try:
            raw = _create(p, VISION_MODELS[p], critic_msgs, temperature=0.0, max_tokens=300)
            break
        except Exception:
            continue
    if raw is None:
        return {"score": 0.0, "is_pass": False, "diagnosis": "All critic platforms failed"}
    raw = raw.strip()
    if raw.startswith("```"): raw = raw.split("\n", 1)[-1].replace("```", "").strip()
    try:
        j = json.loads(raw)
        return {"score": float(j.get("score", 0)), "is_pass": bool(j.get("is_pass", False)),
                "diagnosis": str(j.get("diagnosis", ""))}
    except (json.JSONDecodeError, ValueError):
        return {"score": 0.0, "is_pass": False, "diagnosis": f"Critic parse failed: {raw[:100]}"}


# ── Main pipeline ────────────────────────────────────
def generate(image_path: str, index: int, output_dir: str = "output", difficulty: str = "easy") -> SampleResult:
    t_start = time.time()
    os.makedirs(output_dir, exist_ok=True)

    # Load per-difficulty prompts
    vision_prompt = get_vision_prompt(difficulty)
    code_system = get_code_system(difficulty)

    # N1: Vision description
    desc = image_to_text(image_path, vision_prompt,
                         platforms=VISION_PLATFORMS, temperature=0.0, max_tokens=1024)
    vision_time = round(time.time() - t_start, 1)

    tex_path = os.path.join(output_dir, f"gen_{index:04d}.tex")
    pdf_path = os.path.join(output_dir, f"gen_{index:04d}.pdf")

    msgs = [
        {"role": "system", "content": code_system},
        {"role": "user", "content": f"Generate TikZ code for:\n{desc}"},
    ]

    t_code = time.time()
    compile_ok = False
    compile_attempts = 0
    critic_first_score = 0.0
    critic_final_score = 0.0
    diagnosis = ""
    tikz = ""

    # ── N2↔N3 Compile self-heal loop (max 3 total attempts) ──
    for attempt in range(3):
        compile_attempts = attempt + 1
        if attempt == 0:
            # First attempt: let the code model see the original image too
            code_prompt = code_system + "\n\nGenerate TikZ code based on this description AND the original image:\n" + desc
            raw = image_to_text(image_path, code_prompt,
                                platforms=[p for p in CODE_PLATFORMS if p in VISION_MODELS],
                                temperature=0.0, max_tokens=4096)
        else:
            raw = text_to_text(msgs, platforms=CODE_PLATFORMS,
                               temperature=0.0, max_tokens=4096)
        tikz = _clean(raw)
        tikz = _fix(tikz)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tikz)
        ok, errors = _compile(tex_path, pdf_path)
        if ok:
            compile_ok = True
            break
        # Emergency auto-fix for Dimension too large (no LLM round needed)
        if "Dimension too large" in errors:
            tikz = _reduce_samples_for_overflow(tikz)
            tikz = _reduce_foreach_loops(tikz)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tikz)
            ok2, _ = _compile(tex_path, pdf_path)
            if ok2:
                compile_ok = True
                break
        msgs.append({"role": "user",
                     "content": f"Compile errors:\n{errors}\nFix and output complete code."})
    else:
        compile_ok = False

    codegen_time = round(time.time() - t_code, 1)

    # ── N4↔N5 Visual self-heal (1 pass) ──
    if compile_ok:
        c1 = _internal_critic(image_path, pdf_path, output_dir)
        critic_first_score = c1["score"]
        diagnosis = c1["diagnosis"]

        if not c1["is_pass"]:
            msgs.append({"role": "user",
                         "content": f"Visual review found these differences from the reference:\n"
                                    f"{diagnosis}\n\nMake ONLY minimal targeted fixes to address these "
                                    f"specific issues. Do NOT change anything that is already correct."})
            raw2 = text_to_text(msgs, platforms=CODE_PLATFORMS,
                                temperature=0.0, max_tokens=4096)
            tikz2 = _clean(raw2)
            tikz2 = _fix(tikz2)
            with open(tex_path, "w", encoding="utf-8") as f:
                f.write(tikz2)
            ok2, _ = _compile(tex_path, pdf_path)
            if ok2:
                tikz = tikz2
                c2 = _internal_critic(image_path, pdf_path, output_dir)
                critic_final_score = c2["score"]
                diagnosis = c2["diagnosis"]
            else:
                critic_final_score = 0.0
        else:
            critic_final_score = c1["score"]

    return SampleResult(
        index=index,
        compile_ok=compile_ok,
        compile_attempts=compile_attempts,
        gen_pdf_path=pdf_path if compile_ok else "",
        vision_time=vision_time,
        codegen_time=codegen_time,
        critic_score=critic_first_score,   # will be overwritten by test runner
        critic_pass=critic_first_score >= 3.0,
        diagnosis=diagnosis,
    )
