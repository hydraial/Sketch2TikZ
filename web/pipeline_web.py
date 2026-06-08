"""
Wrapped pipeline with progress callbacks for the web UI.
Does NOT modify train/pipeline.py — reuses its helpers and logic.
"""
import os, time, json
from typing import Callable, Optional

from train.pipeline import (
    get_vision_prompt, get_code_system, CRITIC_PROMPT,
    _fix, _clean, _compile, _pdf_to_png, _encode_img,
    _reduce_samples_for_overflow, _reduce_foreach_loops,
    _internal_critic as _orig_internal_critic,
)
from train.llm_caller import image_to_text, text_to_text, _create, VISION_MODELS, VISION_PLATFORMS, CODE_PLATFORMS


# ── Progress-aware generation ──────────────────────────────────

def generate_with_callbacks(
    image_path: str,
    output_dir: str,
    callbacks: Optional[Callable] = None,
    custom_prompt: Optional[str] = None,
    task_id: str = "",
    difficulty: str = "easy",
) -> dict:
    """
    Run the full Sketch2TikZ pipeline with optional progress callbacks.

    callbacks(stage: str, status: str, message: str, data: dict = None)
        stage   : "vision" | "codegen" | "compile" | "critic" | "done" | "error"
        status  : "running" | "retry" | "success" | "fail"
        message : human-readable description
        data    : optional extra info (attempt number, score, etc.)

    custom_prompt: if provided, skips the vision model and uses this directly.
    """

    def _cb(stage, status, message, data=None):
        if callbacks:
            try:
                callbacks(stage, status, message, data or {})
            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # ── Stage 1: Vision description ──
    if custom_prompt is not None:
        desc = custom_prompt
        _cb("vision", "success", "Using custom prompt (skipped auto description)", {"custom": True})
    else:
        _cb("vision", "running", "Analyzing sketch with vision model...")
        try:
            desc = image_to_text(image_path, get_vision_prompt(difficulty),
                                 platforms=VISION_PLATFORMS, temperature=0.0, max_tokens=1024)
            _cb("vision", "success", "Vision analysis complete", {"description": desc[:200]})
        except Exception as e:
            _cb("vision", "fail", f"Vision model failed: {e}")
            return {"error": f"Vision failed: {e}"}

    vision_time = round(time.time() - t_start, 1)

    tex_path = os.path.join(output_dir, "output.tex")
    pdf_path = os.path.join(output_dir, "output.pdf")
    png_path = os.path.join(output_dir, "output.png")

    code_system = get_code_system(difficulty)
    msgs = [
        {"role": "system", "content": code_system},
        {"role": "user", "content": f"Generate TikZ code for:\n{desc}"},
    ]

    # ── Stage 2: Code generation + compile self-heal ──
    t_code = time.time()
    compile_ok = False
    compile_attempts = 0
    critic_first_score = 0.0
    critic_final_score = 0.0
    diagnosis = ""
    tikz = ""

    for attempt in range(3):
        compile_attempts = attempt + 1
        _cb("codegen", "running", f"Generating TikZ code (attempt {compile_attempts}/3)...",
            {"attempt": compile_attempts})

        if attempt == 0 and custom_prompt is None:
            # First attempt: let code model see original image too
            code_prompt = code_system + "\n\nGenerate TikZ code based on this description AND the original image:\n" + desc
            try:
                raw = image_to_text(image_path, code_prompt,
                                    platforms=[p for p in CODE_PLATFORMS if p in VISION_MODELS],
                                    temperature=0.0, max_tokens=4096)
            except Exception as e:
                _cb("codegen", "fail", f"Code generation failed: {e}", {"attempt": compile_attempts})
                return {"error": f"Code generation failed: {e}"}
        else:
            try:
                raw = text_to_text(msgs, platforms=CODE_PLATFORMS,
                                   temperature=0.0, max_tokens=4096)
            except Exception as e:
                _cb("codegen", "fail", f"Code generation failed: {e}", {"attempt": compile_attempts})
                return {"error": f"Code generation failed: {e}"}

        tikz = _clean(raw)
        tikz = _fix(tikz)
        with open(tex_path, "w", encoding="utf-8") as f:
            f.write(tikz)

        _cb("compile", "running", f"Compiling LaTeX (attempt {compile_attempts}/3)...",
            {"attempt": compile_attempts})
        ok, errors = _compile(tex_path, pdf_path)
        if ok:
            compile_ok = True
            _cb("compile", "success", "LaTeX compiled successfully",
                {"attempt": compile_attempts})
            break
        else:
            _cb("compile", "retry", f"Compile error on attempt {compile_attempts}: {errors[:150]}",
                {"attempt": compile_attempts, "errors": errors})
            # Emergency auto-fix for Dimension too large (no LLM round needed)
            if "Dimension too large" in errors:
                _cb("compile", "running", "Auto-reducing samples/loops to fix overflow...")
                tikz = _reduce_samples_for_overflow(tikz)
                tikz = _reduce_foreach_loops(tikz)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tikz)
                ok2, _ = _compile(tex_path, pdf_path)
                if ok2:
                    compile_ok = True
                    _cb("compile", "success", "LaTeX compiled after auto-reducing samples",
                        {"attempt": compile_attempts})
                    break
            msgs.append({"role": "user",
                         "content": f"Compile errors:\n{errors}\nFix and output complete code."})
    else:
        compile_ok = False
        _cb("compile", "fail", "All 3 compile attempts failed", {"attempts": 3})

    codegen_time = round(time.time() - t_code, 1)

    # ── Stage 3: Visual critic + self-heal ──
    if compile_ok:
        _cb("critic", "running", "Running visual quality review...")
        c1 = _orig_internal_critic(image_path, pdf_path, output_dir)
        critic_first_score = c1["score"]
        diagnosis = c1["diagnosis"]
        _cb("critic", "success" if c1["is_pass"] else "retry",
            f"Visual review score: {critic_first_score:.1f}/5.0 — {diagnosis[:200]}",
            {"score": critic_first_score, "is_pass": c1["is_pass"], "diagnosis": diagnosis})

        if not c1["is_pass"]:
            _cb("codegen", "running", "Applying visual self-heal fixes...")
            msgs.append({"role": "user",
                         "content": f"Visual review found these differences from the reference:\n"
                                    f"{diagnosis}\n\nMake ONLY minimal targeted fixes to address these "
                                    f"specific issues. Do NOT change anything that is already correct."})
            try:
                raw2 = text_to_text(msgs, platforms=CODE_PLATFORMS,
                                    temperature=0.0, max_tokens=4096)
                tikz2 = _clean(raw2)
                tikz2 = _fix(tikz2)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tikz2)
                ok2, _ = _compile(tex_path, pdf_path)
                if ok2:
                    tikz = tikz2
                    c2 = _orig_internal_critic(image_path, pdf_path, output_dir)
                    critic_final_score = c2["score"]
                    diagnosis = c2["diagnosis"]
                    _cb("critic", "success" if c2["is_pass"] else "fail",
                        f"Self-heal score: {critic_final_score:.1f}/5.0 — {diagnosis[:200]}",
                        {"score": critic_final_score, "is_pass": c2["is_pass"], "diagnosis": diagnosis})
                else:
                    critic_final_score = 0.0
                    _cb("compile", "fail", "Self-heal compile failed")
            except Exception as e:
                _cb("codegen", "fail", f"Self-heal failed: {e}")
        else:
            critic_final_score = c1["score"]
    else:
        _cb("critic", "fail", "Skipped visual review (compile failed)")

    # Render PNG preview
    if compile_ok and os.path.exists(pdf_path):
        _pdf_to_png(pdf_path, png_path)

    total_time = round(time.time() - t_start, 1)

    result = {
        "task_id": task_id,
        "compile_ok": compile_ok,
        "compile_attempts": compile_attempts,
        "vision_time": vision_time,
        "codegen_time": codegen_time,
        "total_time": total_time,
        "critic_first_score": critic_first_score,
        "critic_final_score": critic_final_score,
        "critic_pass": critic_final_score >= 3.0,
        "diagnosis": diagnosis,
        "description": desc,
        "tikz_code": tikz if compile_ok else "",
        "paths": {
            "tex": tex_path if compile_ok else "",
            "pdf": pdf_path if compile_ok else "",
            "png": png_path if os.path.exists(png_path) else "",
        },
    }

    _cb("done", "success" if compile_ok else "fail",
        f"Generation complete in {total_time}s" if compile_ok else "Generation failed",
        result)
    return result


# ── Visual refine (user feedback) ──────────────────────────────

def refine_with_callbacks(
    image_path: str,
    prev_tex_path: str,
    prev_pdf_path: str,
    custom_prompt: str,
    output_dir: str,
    callbacks: Optional[Callable] = None,
    task_id: str = "",
    difficulty: str = "easy",
) -> dict:
    """
    Refine existing TikZ code based on user feedback + visual comparison.
    Reuses the same dual-feedback loops as generate_with_callbacks:
      - Compile self-heal (max 3 attempts)
      - Visual self-heal (1 pass)
    """

    def _cb(stage, status, message, data=None):
        if callbacks:
            try:
                callbacks(stage, status, message, data or {})
            except Exception:
                pass

    os.makedirs(output_dir, exist_ok=True)
    t_start = time.time()

    # Read previous code
    with open(prev_tex_path, "r", encoding="utf-8") as f:
        prev_code = f.read()

    # Convert previous PDF to PNG for visual comparison
    prev_png_path = os.path.join(output_dir, "prev_render.png")
    if not _pdf_to_png(prev_pdf_path, prev_png_path):
        _cb("vision", "fail", "Failed to render previous output for comparison")
        return {"error": "Failed to render previous PDF to PNG"}

    _cb("vision", "running", "Running AI visual review on previous output...")

    # Step 1: Run internal critic to get AI diagnosis
    c_prev = _orig_internal_critic(image_path, prev_pdf_path, output_dir)
    ai_diagnosis = c_prev["diagnosis"]
    ai_score = c_prev["score"]
    _cb("critic", "success" if c_prev["is_pass"] else "retry",
        f"AI review of previous output: {ai_score:.1f}/5.0 — {ai_diagnosis[:200]}",
        {"score": ai_score, "is_pass": c_prev["is_pass"], "diagnosis": ai_diagnosis})

    _cb("vision", "running", "Preparing visual refinement with AI diagnosis + user feedback...")

    # Build multimodal refine message (both AI diagnosis + user feedback)
    b64_orig = _encode_img(image_path)
    b64_prev = _encode_img(prev_png_path)

    code_system = get_code_system(difficulty)
    refine_system = (
        code_system + "\n\n"
        "You are REFINING existing TikZ code based on AI visual diagnosis AND user feedback. "
        "Make ONLY minimal targeted changes. Address BOTH the AI-identified issues and the user's requests. "
        "Preserve everything that is already correct. Do NOT rewrite from scratch."
    )

    feedback_text = (
        f"Current TikZ code:\n```latex\n{prev_code}\n```\n\n"
        f"[AI Visual Diagnosis]\n{ai_diagnosis}\n\n"
        f"[User Requested Changes]\n{custom_prompt}\n\n"
        "Please modify the code to address BOTH the AI-diagnosed issues and the user's requests. "
        "Output ONLY the complete, compilable LaTeX code."
    )

    refine_msgs = [
        {"role": "system", "content": refine_system},
        {"role": "user", "content": [
            {"type": "text", "text": "REFERENCE image (the sketch we want to match):"},
            {"type": "image_url", "image_url": {"url": b64_orig}},
            {"type": "text", "text": "CURRENT generated output:"},
            {"type": "image_url", "image_url": {"url": b64_prev}},
            {"type": "text", "text": feedback_text},
        ]}
    ]

    # Call vision-capable model (needs to see both images)
    _cb("codegen", "running", "Generating refined TikZ code based on visual feedback...")
    try:
        raw = None
        platforms_tried = [p for p in CODE_PLATFORMS if p in VISION_MODELS]
        for p in platforms_tried:
            try:
                raw = _create(p, VISION_MODELS[p], refine_msgs, temperature=0.0, max_tokens=4096)
                break
            except Exception as e:
                _cb("codegen", "retry", f"Platform {p} failed: {str(e)[:100]}")
                continue
        if raw is None:
            raise RuntimeError("All vision platforms failed for refinement")
        _cb("codegen", "success", "Refined code generated")
    except Exception as e:
        _cb("codegen", "fail", f"Refinement failed: {e}")
        return {"error": f"Refinement failed: {e}"}

    tex_path = os.path.join(output_dir, "output.tex")
    pdf_path = os.path.join(output_dir, "output.pdf")
    png_path = os.path.join(output_dir, "output.png")

    tikz = _clean(raw)
    tikz = _fix(tikz)
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tikz)

    # ── Compile self-heal loop ──
    t_code = time.time()
    compile_ok = False
    compile_attempts = 0
    critic_first_score = 0.0
    critic_final_score = 0.0
    diagnosis = ""

    for attempt in range(3):
        compile_attempts = attempt + 1
        _cb("compile", "running", f"Compiling LaTeX (attempt {compile_attempts}/3)...",
            {"attempt": compile_attempts})
        ok, errors = _compile(tex_path, pdf_path)
        if ok:
            compile_ok = True
            _cb("compile", "success", "LaTeX compiled successfully",
                {"attempt": compile_attempts})
            break
        else:
            _cb("compile", "retry", f"Compile error on attempt {compile_attempts}: {errors[:150]}",
                {"attempt": compile_attempts, "errors": errors})
            # Emergency auto-fix for Dimension too large (no LLM round needed)
            if "Dimension too large" in errors:
                _cb("compile", "running", "Auto-reducing samples/loops to fix overflow...")
                tikz = _reduce_samples_for_overflow(tikz)
                tikz = _reduce_foreach_loops(tikz)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tikz)
                ok2, _ = _compile(tex_path, pdf_path)
                if ok2:
                    compile_ok = True
                    _cb("compile", "success", "LaTeX compiled after auto-reducing samples",
                        {"attempt": compile_attempts})
                    break
            fix_msgs = [
                {"role": "system", "content": code_system + "\n\nFix compilation errors. Make minimal changes. Output complete code."},
                {"role": "user", "content": f"```latex\n{tikz}\n```\n\nCompile errors:\n{errors}\n\nFix and output complete code."}
            ]
            try:
                raw_fix = text_to_text(fix_msgs, platforms=CODE_PLATFORMS,
                                       temperature=0.0, max_tokens=4096)
                tikz = _clean(raw_fix)
                tikz = _fix(tikz)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tikz)
            except Exception as e:
                _cb("codegen", "fail", f"Fix attempt failed: {e}", {"attempt": compile_attempts})
    else:
        compile_ok = False
        _cb("compile", "fail", "All 3 compile attempts failed", {"attempts": 3})

    codegen_time = round(time.time() - t_code, 1)

    # ── Visual critic + self-heal ──
    if compile_ok:
        _cb("critic", "running", "Running visual quality review...")
        c1 = _orig_internal_critic(image_path, pdf_path, output_dir)
        critic_first_score = c1["score"]
        diagnosis = c1["diagnosis"]
        _cb("critic", "success" if c1["is_pass"] else "retry",
            f"Visual review score: {critic_first_score:.1f}/5.0 — {diagnosis[:200]}",
            {"score": critic_first_score, "is_pass": c1["is_pass"], "diagnosis": diagnosis})

        if not c1["is_pass"]:
            _cb("codegen", "running", "Applying visual self-heal fixes...")
            heal_msgs = [
                {"role": "system", "content": code_system},
                {"role": "user", "content": f"Current TikZ code:\n```latex\n{tikz}\n```\n\n"
                                          f"Visual review found these differences from the reference:\n"
                                          f"{diagnosis}\n\nMake ONLY minimal targeted fixes to address these "
                                          f"specific issues. Do NOT change anything that is already correct."}
            ]
            try:
                raw2 = text_to_text(heal_msgs, platforms=CODE_PLATFORMS,
                                    temperature=0.0, max_tokens=4096)
                tikz2 = _clean(raw2)
                tikz2 = _fix(tikz2)
                with open(tex_path, "w", encoding="utf-8") as f:
                    f.write(tikz2)
                ok2, _ = _compile(tex_path, pdf_path)
                if ok2:
                    tikz = tikz2
                    c2 = _orig_internal_critic(image_path, pdf_path, output_dir)
                    critic_final_score = c2["score"]
                    diagnosis = c2["diagnosis"]
                    _cb("critic", "success" if c2["is_pass"] else "fail",
                        f"Self-heal score: {critic_final_score:.1f}/5.0 — {diagnosis[:200]}",
                        {"score": critic_final_score, "is_pass": c2["is_pass"], "diagnosis": diagnosis})
                else:
                    critic_final_score = 0.0
                    _cb("compile", "fail", "Self-heal compile failed")
            except Exception as e:
                _cb("codegen", "fail", f"Self-heal failed: {e}")
        else:
            critic_final_score = c1["score"]
    else:
        _cb("critic", "fail", "Skipped visual review (compile failed)")

    # Render PNG preview
    if compile_ok and os.path.exists(pdf_path):
        _pdf_to_png(pdf_path, png_path)

    total_time = round(time.time() - t_start, 1)

    result = {
        "task_id": task_id,
        "compile_ok": compile_ok,
        "compile_attempts": compile_attempts,
        "codegen_time": codegen_time,
        "total_time": total_time,
        "critic_first_score": critic_first_score,
        "critic_final_score": critic_final_score,
        "critic_pass": critic_final_score >= 3.0,
        "diagnosis": diagnosis,
        "description": custom_prompt,
        "tikz_code": tikz if compile_ok else "",
        "paths": {
            "tex": tex_path if compile_ok else "",
            "pdf": pdf_path if compile_ok else "",
            "png": png_path if os.path.exists(png_path) else "",
        },
    }

    _cb("done", "success" if compile_ok else "fail",
        f"Refinement complete in {total_time}s" if compile_ok else "Refinement failed",
        result)
    return result


# ── Direct recompile (no AI call) ──────────────────────────────

def compile_tex_only(tikz_code: str, output_dir: str) -> dict:
    """Save TikZ code and compile it directly. Returns compilation result."""
    os.makedirs(output_dir, exist_ok=True)
    tex_path = os.path.join(output_dir, "output.tex")
    pdf_path = os.path.join(output_dir, "output.pdf")
    png_path = os.path.join(output_dir, "output.png")

    tikz = _fix(_clean(tikz_code))
    with open(tex_path, "w", encoding="utf-8") as f:
        f.write(tikz)

    ok, errors = _compile(tex_path, pdf_path)
    if ok:
        _pdf_to_png(pdf_path, png_path)

    return {
        "compile_ok": ok,
        "errors": errors,
        "tex_path": tex_path,
        "pdf_path": pdf_path if ok else "",
        "png_path": png_path if ok else "",
    }
