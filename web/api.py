"""FastAPI routes for the Sketch2TikZ web UI."""
import os, shutil, json
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, JSONResponse

from web.tasks import manager, UPLOADS_DIR
from web.pipeline_web import generate_with_callbacks

router = APIRouter()


@router.post("/api/generate")
async def api_generate(
    file: UploadFile = File(...),
    custom_prompt: Optional[str] = Form(None),
    difficulty: str = Form("easy"),
):
    """Upload a sketch and start generation. Returns task_id immediately."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, "File must be an image")

    task = manager.create_task(image_path="", custom_prompt=custom_prompt or None)
    image_path = os.path.join(task.output_dir, "input.png")

    # Save uploaded file
    with open(image_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    task.image_path = image_path

    # Build callback that feeds into task events
    def _callback(stage, status, message, data=None):
        task.emit(stage, status, message, data)

    # Submit to background thread
    manager.submit(
        task,
        generate_with_callbacks,
        image_path=image_path,
        output_dir=task.output_dir,
        callbacks=_callback,
        custom_prompt=task.custom_prompt,
        task_id=task.task_id,
        difficulty=difficulty,
    )

    return {"task_id": task.task_id, "status": "queued"}


@router.get("/api/tasks/{task_id}")
async def api_task_status(task_id: str):
    task = manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")
    return {
        "task_id": task.task_id,
        "status": task.status,
        "stage": task.stage,
        "stage_status": task.stage_status,
        "message": task.message,
        "progress": task.progress,
        "result": task.result,
        "created_at": task.created_at,
        "finished_at": task.finished_at,
    }


@router.get("/api/tasks/{task_id}/stream")
async def api_task_stream(task_id: str):
    """SSE endpoint for real-time progress."""
    import asyncio

    task = manager.get_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    async def _event_generator():
        last_index = 0
        while True:
            with task._lock:
                events = task.events[last_index:]
                last_index = len(task.events)
                is_done = task.status in ("done", "error")

            for ev in events:
                yield f"data: {json.dumps(ev)}\n\n"

            if is_done:
                yield f"data: {json.dumps({'stage': 'close', 'status': 'done', 'message': 'Stream closed'})}\n\n"
                break

            await asyncio.sleep(0.5)

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )


@router.post("/api/refine")
async def api_refine(
    prev_task_id: str = Form(...),
    custom_prompt: str = Form(...),
    difficulty: str = Form("easy"),
):
    """Refine a previous generation with user feedback + visual comparison.
    Reuses the original image from prev_task to ensure consistency."""
    prev_dir = os.path.join(UPLOADS_DIR, prev_task_id)
    prev_tex = os.path.join(prev_dir, "output.tex")
    prev_pdf = os.path.join(prev_dir, "output.pdf")
    prev_img = os.path.join(prev_dir, "input.png")
    if not os.path.exists(prev_tex) or not os.path.exists(prev_pdf):
        raise HTTPException(404, "Previous task result not found")
    if not os.path.exists(prev_img):
        raise HTTPException(404, "Previous task input image not found")

    task = manager.create_task(image_path="", custom_prompt=custom_prompt)
    image_path = os.path.join(task.output_dir, "input.png")
    shutil.copy2(prev_img, image_path)
    task.image_path = image_path

    def _callback(stage, status, message, data=None):
        task.emit(stage, status, message, data)

    from web.pipeline_web import refine_with_callbacks
    manager.submit(
        task,
        refine_with_callbacks,
        image_path=image_path,
        prev_tex_path=prev_tex,
        prev_pdf_path=prev_pdf,
        custom_prompt=custom_prompt,
        output_dir=task.output_dir,
        callbacks=_callback,
        task_id=task.task_id,
        difficulty=difficulty,
    )

    return {"task_id": task.task_id, "status": "queued"}


@router.get("/api/history")
async def api_history():
    return manager.list_history()


@router.get("/api/download/{task_id}/{fmt}")
async def api_download(task_id: str, fmt: str):
    """Download generated file: tex | pdf | png | input"""
    task = manager.get_task(task_id)
    if not task:
        # Try to construct path from history even if task object gone
        out_dir = os.path.join(UPLOADS_DIR, task_id)
    else:
        out_dir = task.output_dir

    file_map = {
        "tex": ("output.tex", "text/x-tex"),
        "pdf": ("output.pdf", "application/pdf"),
        "png": ("output.png", "image/png"),
        "input": ("input.png", "image/png"),
    }
    if fmt not in file_map:
        raise HTTPException(400, "Format must be tex, pdf, png, or input")

    fname, mime = file_map[fmt]
    path = os.path.join(out_dir, fname)
    if not os.path.exists(path):
        raise HTTPException(404, "File not found")

    return FileResponse(path, media_type=mime, filename=f"{task_id}_{fname}")


@router.get("/api/tasks/{task_id}/preview")
async def api_preview(task_id: str):
    """Return the rendered PNG preview (or input if no output yet)."""
    task = manager.get_task(task_id)
    out_dir = task.output_dir if task else os.path.join(UPLOADS_DIR, task_id)

    png_path = os.path.join(out_dir, "output.png")
    if os.path.exists(png_path):
        return FileResponse(png_path, media_type="image/png")

    # Fallback to input
    input_path = os.path.join(out_dir, "input.png")
    if os.path.exists(input_path):
        return FileResponse(input_path, media_type="image/png")

    raise HTTPException(404, "Preview not found")


@router.get("/api/tasks/{task_id}/code")
async def api_code(task_id: str):
    """Return the generated TikZ code as plain text."""
    task = manager.get_task(task_id)
    out_dir = task.output_dir if task else os.path.join(UPLOADS_DIR, task_id)

    tex_path = os.path.join(out_dir, "output.tex")
    if not os.path.exists(tex_path):
        raise HTTPException(404, "Code not found")

    with open(tex_path, "r", encoding="utf-8") as f:
        return JSONResponse({"code": f.read()})


@router.post("/api/compile")
async def api_compile(
    task_id: str = Form(...),
    code: str = Form(...),
):
    """Recompile user-edited TikZ code directly (no AI call)."""
    from web.pipeline_web import compile_tex_only

    task = manager.get_task(task_id)
    out_dir = task.output_dir if task else os.path.join(UPLOADS_DIR, task_id)

    result = compile_tex_only(code, out_dir)
    if result["compile_ok"]:
        import time as _time
        return {
            "compile_ok": True,
            "preview_url": f"/api/tasks/{task_id}/preview?t={int(_time.time())}"
        }
    else:
        raise HTTPException(422, detail={
            "compile_ok": False,
            "errors": result["errors"]
        })
