"""Video-script generator (Google ADK + Gemini + Imagen 4 Fast + Cloud TTS).

A small multi-agent pipeline that turns a TITLE (plus an optional short
description) into a professional, portrait (9:16) short-form video package — the
whole script in one go (story + consistent per-section image descriptions), and
for EACH scene a background photo plus a narration audio clip (generated in
parallel).

Everything for a project lives under a workdir, so work survives restarts and can
be resumed:

    workdir/<slug>/
        script.json     # title, topic, story, style, sections (text + desc)
        media.json      # manifest: per section -> imagePath + audioPath
        img/section-01.png   ...   # portrait 9:16 background photos (Imagen 4 Fast)
        audio/section-01.mp3 ...   # narration clips (Google Cloud TTS, Standard voice)
        audio/section-01.words.json ...   # per-word TTS timings (SSML <mark>) -> karaoke captions

You tell the agent what the video should be about — optionally with structure, e.g.
"top 3 … — 4 scenes: intro + one per item" — and it writes the WHOLE script in one
go: a punchy story plus a per-section image prompt sharing one visual style. It
shows the scenes and the style for you to confirm or change, then generates the
media, then renders the MP4.

Resuming a previous project is OFF by default. Set LOAD_LAST_PROJECT = True to have
the agent list saved projects on startup and offer to CONTINUE one — picking up
media that already exists instead of starting over.

  root_agent (video_script_agent)  — coordinator / scene planner
  media_generator_agent            — generates image + audio per scene, in parallel
  video_render_agent               — renders the final MP4 with Remotion

Run it from the project root, in single-agent mode so only this agent loads:
    adk web create_video_agent --port 8080
    adk run create_video_agent

Tip: set VIDEO_AGENT_FAKE_MEDIA=1 to dry-run the media step (writes tiny
placeholder files instead of calling — and paying for — Imagen / TTS).
"""

import asyncio
import base64
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape as _xml_escape

from google import genai
from google.genai import types
# v1beta1: same Standard-voice synthesis, but also returns per-word <mark>
# timepoints (for karaoke-style caption highlighting). <mark> tags aren't billed.
from google.cloud import texttospeech_v1beta1 as tts
from google.api_core.client_options import ClientOptions
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.adk.tools.agent_tool import AgentTool

# --- Authenticate with Vertex AI / Google Cloud via Application Default Credentials. ---
# Run `gcloud auth application-default login` once — no service account key needed.
os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "us-central1")
# GOOGLE_CLOUD_PROJECT is read from the environment — set it via Docker `-e`
# (or derived from your mounted gcloud config by the entrypoint), or in
# create_video_agent/.env for local dev. It is intentionally NOT hardcoded, so no
# project id ships in this repo.

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WORKDIR = PROJECT_ROOT / "workdir"   # all projects (json + media) live here
REMOTION_DIR = PROJECT_ROOT / "remotion"   # the Remotion project that renders the MP4

# Imagen 4 Fast on Vertex AI; portrait for phone / TikTok / YouTube Shorts.
IMAGE_MODEL = "imagen-4.0-fast-generate-001"
IMAGE_ASPECT_RATIO = "9:16"

# Google Cloud Text-to-Speech. Standard voices are the CHEAPEST tier
# (largest free allowance, ~$4 per 1M chars). MP3 keeps files small. Synthesis
# goes through the v1beta1 API so each clip also returns per-word <mark>
# timepoints used for karaoke captions (the <mark> tags themselves aren't billed).
TTS_LANGUAGE = "en-US"
TTS_VOICE = "en-US-Standard-C"

# How many sections (scenes) a video has. The user picks within
# [SECTIONS_MIN, SECTIONS_MAX]; SECTIONS_DEFAULT is used when they don't care.
SECTIONS_DEFAULT = 5
SECTIONS_MIN = 3
SECTIONS_MAX = 12

# Resume support. When False (the default) the agent never lists or reopens previous
# projects — every run starts a fresh video, and the list_projects / load_project
# tools aren't even registered. Set True to list saved projects on startup and offer
# to continue the most recent one.
LOAD_LAST_PROJECT = False

# Tiny placeholders used only for VIDEO_AGENT_FAKE_MEDIA dry-runs.
_FAKE_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+M9QDwADhgGA"
    "WjR9awAAAABJRU5ErkJggg=="
)
_FAKE_MP3 = b"ID3\x04\x00\x00\x00\x00\x00\x00"


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _fake() -> bool:
    """Dry-run media generation (no Imagen/TTS calls)?"""
    return (
        os.environ.get("VIDEO_AGENT_FAKE_MEDIA") == "1"
        or os.environ.get("VIDEO_AGENT_FAKE_IMAGES") == "1"
    )


def _upload_to_gcs(bucket: str, path: Path, object_name: str) -> str:
    """Upload `path` to gs://<bucket>/<object_name> and return its public https
    download URL.

    Content-Disposition=attachment makes a browser DOWNLOAD the file when the link
    is clicked (rather than trying to play it). google-cloud-storage is imported
    lazily so the dependency is only needed when VIDEO_BUCKET is set (i.e. on the
    Docker / Cloud Run deploy), not for plain local `adk web`."""
    from google.cloud import storage
    client = storage.Client(project=os.environ.get("GOOGLE_CLOUD_PROJECT") or None)
    blob = client.bucket(bucket).blob(object_name)
    blob.content_disposition = f'attachment; filename="{object_name}"'
    blob.cache_control = "no-store"  # re-renders reuse the name; don't serve a stale copy
    blob.upload_from_filename(str(path), content_type="video/mp4")
    return f"https://storage.googleapis.com/{bucket}/{object_name}"


def _slugify(title: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
    return slug or "video-script"


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def _paths(slug: str) -> dict:
    """All paths for a project slug."""
    base = WORKDIR / slug
    return {
        "base": base,
        "script": base / "script.json",
        "manifest": base / "media.json",
        "img": base / "img",
        "audio": base / "audio",
    }


def _write_json_atomic(path: Path, obj) -> None:
    """Write JSON to `path` atomically (temp file + os.replace), so an interrupted
    run can never leave a half-written script.json / media.json behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _sections_of(data: dict) -> list:
    """The section list from a script/manifest dict (tolerates the legacy key)."""
    return data.get("sections") or data.get("images") or []


def _load_manifest(slug: str):
    """The project's media.json (the AS-BUILT manifest), or None if not generated."""
    mp = _paths(slug)["manifest"]
    if not mp.is_file():
        return None
    try:
        return json.loads(mp.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _as_built(manifest) -> dict:
    """{section_no -> (text, imageDescription)} the existing media was generated from.

    media.json records, per section, the text and image description each file was
    actually produced from — so comparing it to the live script tells us what is
    stale.
    """
    out = {}
    for idx, s in enumerate((manifest or {}).get("sections") or [], start=1):
        out[idx] = (str(s.get("text", "")).strip(),
                    str(s.get("imageDescription", "")).strip())
    return out


def _media_states(slug: str, data: dict = None, manifest=None) -> dict:
    """Per-section media status: file existence + staleness vs the current script.

    A media file is STALE when the script's current text / imageDescription differs
    from what that file was generated from (recorded in media.json): audio is tied
    to `text`, the image to `imageDescription`. Editing a scene therefore marks only
    that scene's affected media stale, leaving every other scene fresh.

    Returns {i: {"image_exists","audio_exists","image_stale","audio_stale"}}.
    """
    p = _paths(slug)
    if data is None:
        data = json.loads(p["script"].read_text(encoding="utf-8"))
    if manifest is None:
        manifest = _load_manifest(slug)
    built = _as_built(manifest)
    states = {}
    for i, sec in enumerate(_sections_of(data), start=1):
        img = p["img"] / f"section-{i:02d}.png"
        aud = p["audio"] / f"section-{i:02d}.mp3"
        ie, ae = img.is_file(), aud.is_file()
        b_text, b_desc = built.get(i, (None, None))
        cur_text = str(sec.get("text", "")).strip()
        cur_desc = str(sec.get("imageDescription", "")).strip()
        # Stale only applies to a file that EXISTS and has an as-built record to
        # compare against (older projects without a manifest are treated as fresh).
        states[i] = {
            "image_exists": ie,
            "audio_exists": ae,
            "image_stale": bool(ie and b_desc is not None and b_desc != cur_desc),
            "audio_stale": bool(ae and b_text is not None and b_text != cur_text),
        }
    return states


def _list_slugs():
    """Project slugs that have a script.json, most recently modified first."""
    if not WORKDIR.is_dir():
        return []
    dirs = [d for d in WORKDIR.iterdir() if d.is_dir() and (d / "script.json").is_file()]
    dirs.sort(key=lambda d: (d / "script.json").stat().st_mtime, reverse=True)
    return [d.name for d in dirs]


def _resolve_slug(slug: str):
    slug = (slug or "").strip()
    if not slug:
        slugs = _list_slugs()
        return slugs[0] if slugs else None
    if (WORKDIR / slug / "script.json").is_file():
        return slug
    alt = _slugify(slug)
    if (WORKDIR / alt / "script.json").is_file():
        return alt
    return None


def _media_done(slug: str, total: int):
    """How many section images / audio clips already exist on disk."""
    p = _paths(slug)
    imgs = sum(1 for i in range(1, total + 1) if (p["img"] / f"section-{i:02d}.png").is_file())
    auds = sum(1 for i in range(1, total + 1) if (p["audio"] / f"section-{i:02d}.mp3").is_file())
    return imgs, auds


def _status(slug: str) -> dict:
    p = _paths(slug)
    data = json.loads(p["script"].read_text(encoding="utf-8"))
    secs = data.get("sections") or data.get("images") or []
    total = len(secs)
    imgs, auds = _media_done(slug, total)
    states = _media_states(slug, data)
    stale_img = sorted(i for i in states if states[i]["image_stale"])
    stale_aud = sorted(i for i in states if states[i]["audio_stale"])
    has_video = (p["base"] / "video.mp4").is_file()
    if total and imgs >= total and auds >= total and not (stale_img or stale_aud):
        state = "rendered" if has_video else "complete"
    elif imgs == 0 and auds == 0:
        state = "needs_media"
    else:
        # All present but something was edited and not yet regenerated -> partial.
        state = "partial"
    return {
        "slug": slug,
        "title": data.get("title", ""),
        "topic": data.get("topic", ""),
        "sections": total,
        "images_done": imgs,
        "audio_done": auds,
        "stale_images": stale_img,
        "stale_audio": stale_aud,
        "has_video": has_video,
        "status": state,
        "updated_at": datetime.fromtimestamp(
            p["script"].stat().st_mtime, timezone.utc
        ).isoformat(),
    }


def _parse_images(images_json: str):
    """Parse the section array from the model, tolerating code fences / stray prose."""
    cleaned = images_json.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```[a-zA-Z0-9]*\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("["), cleaned.rfind("]")
        if start != -1 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError as exc:
                raise ValueError(f"images_json is not valid JSON ({exc}).") from exc
        raise ValueError("images_json is not valid JSON.")


def _parse_targets(sections: str, total: int) -> set:
    """Parse "3,6" / "section 6" into a set of valid 1-based section numbers."""
    return {
        int(tok)
        for tok in re.split(r"[^0-9]+", sections or "")
        if tok and 1 <= int(tok) <= total
    }


def _build_image_prompt(description: str, style: str) -> str:
    parts = [description.strip()]
    if style and style.strip():
        parts.append(f"Consistent visual style across the whole video: {style.strip()}.")
    parts.append(
        "Vertical 9:16 portrait composition, full-bleed background suitable for a "
        "mobile short-form video (TikTok / YouTube Shorts / Reels), highly "
        "detailed, professional cinematic quality, no text, no watermark, no logos."
    )
    return " ".join(parts)


def _build_tts_ssml(text: str):
    """Build SSML with a <mark> after every word, for word-level timepoints.

    Returns (ssml, tokens). The mark after token i fires at the END of that
    token (~ the start of the next), so word i spans [mark i-1, mark i].
    """
    tokens = text.split()
    body = " ".join(
        f'{_xml_escape(tok)}<mark name="m{i}"/>' for i, tok in enumerate(tokens)
    )
    return f"<speak>{body}</speak>", tokens


def _words_from_timepoints(tokens, timepoints):
    """Turn TTS <mark> timepoints into [{word, start, end}] seconds (monotonic)."""
    times = {tp.mark_name: float(tp.time_seconds) for tp in timepoints}
    words, prev = [], 0.0
    for i, tok in enumerate(tokens):
        t = times.get(f"m{i}")
        end = prev if t is None else max(prev, t)
        words.append({"word": tok, "start": round(prev, 3), "end": round(end, 3)})
        prev = end
    return words


# --------------------------------------------------------------------------- #
# Per-kind media generation (run concurrently by generate_scene_media)
# --------------------------------------------------------------------------- #
def _generate_kind(kind: str, secs: list, paths: dict, style: str,
                   regen: set, fake: bool) -> dict:
    """Generate section images OR section audio for the sections in `regen`.

    `regen` is the set of 1-based section numbers to (re)generate for this kind;
    every other section is left exactly as it is on disk. The caller decides that
    set (missing + stale + explicitly targeted), so this function never has to know
    about resume/force semantics.
    Returns {"results": {i: {"path", "status"}}, "generated": n, "skipped": n}.
    """
    is_image = kind == "image"
    ext = "png" if is_image else "mp3"
    out_dir = paths["img"] if is_image else paths["audio"]
    out_dir.mkdir(parents=True, exist_ok=True)

    client, client_error = None, None
    if not fake:
        try:
            if is_image:
                client = genai.Client(
                    vertexai=True,
                    project=os.environ["GOOGLE_CLOUD_PROJECT"],
                    location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
                )
            else:
                # Bill/quota TTS to the same project as Imagen/Gemini (not the
                # ADC default), so everything stays on one project.
                client = tts.TextToSpeechClient(
                    client_options=ClientOptions(
                        quota_project_id=os.environ["GOOGLE_CLOUD_PROJECT"]
                    )
                )
        except Exception as exc:  # noqa: BLE001
            client_error = str(exc)

    img_config = types.GenerateImagesConfig(
        number_of_images=1,
        aspect_ratio=IMAGE_ASPECT_RATIO,
        output_mime_type="image/png",
        person_generation="allow_all",
        # Least aggressive generally-available content filter, to cut false
        # positives. (Responsible-AI blocks — real people, children, etc. — are
        # separate and reported via rai_filtered_reason.)
        safety_filter_level="BLOCK_ONLY_HIGH",
    )
    voice = tts.VoiceSelectionParams(language_code=TTS_LANGUAGE, name=TTS_VOICE)
    audio_config = tts.AudioConfig(audio_encoding=tts.AudioEncoding.MP3)

    results, generated, skipped = {}, 0, 0
    for i, section in enumerate(secs, start=1):
        out_path = out_dir / f"section-{i:02d}.{ext}"
        exists = out_path.is_file()

        if i not in regen:
            results[i] = {"path": _rel(out_path) if exists else None,
                          "status": "exists (kept)" if exists else "missing"}
            if exists:
                skipped += 1
            continue

        description = str(section.get("imageDescription", "")).strip()
        text = str(section.get("text", "")).strip()
        try:
            if client_error:
                raise RuntimeError(client_error)
            if is_image:
                if not description:
                    raise RuntimeError("no imageDescription")
                if fake:
                    out_path.write_bytes(_FAKE_PNG)
                else:
                    resp = client.models.generate_images(
                        model=IMAGE_MODEL,
                        prompt=_build_image_prompt(description, style),
                        config=img_config,
                    )
                    gen = resp.generated_images[0] if resp.generated_images else None
                    img_bytes = gen.image.image_bytes if (gen and gen.image) else None
                    if img_bytes is None:
                        # Imagen returned no usable image -> it was filtered. Surface
                        # the real reason instead of crashing on write_bytes(None).
                        reason = (gen.rai_filtered_reason if gen else None) or getattr(
                            resp, "positive_prompt_safety_attributes", None
                        )
                        raise RuntimeError(
                            "image blocked by Imagen safety filter"
                            + (f": {reason}" if reason else
                               " (no image returned; try rephrasing this scene's description)")
                        )
                    out_path.write_bytes(img_bytes)
            else:
                if not text:
                    raise RuntimeError("no text to narrate")
                words_path = out_dir / f"section-{i:02d}.words.json"
                if fake:
                    out_path.write_bytes(_FAKE_MP3)
                    words_path.unlink(missing_ok=True)  # no real timing in fake mode
                else:
                    ssml, tokens = _build_tts_ssml(text)
                    resp = client.synthesize_speech(
                        request=tts.SynthesizeSpeechRequest(
                            input=tts.SynthesisInput(ssml=ssml),
                            voice=voice,
                            audio_config=audio_config,
                            enable_time_pointing=[
                                tts.SynthesizeSpeechRequest.TimepointType.SSML_MARK
                            ],
                        )
                    )
                    out_path.write_bytes(resp.audio_content)
                    # Per-word timings for karaoke captions (best effort): if the
                    # voice returns no timepoints, drop the file and the renderer
                    # falls back to evenly-split chunks.
                    if resp.timepoints:
                        words_path.write_text(
                            json.dumps(
                                {"text": text,
                                 "words": _words_from_timepoints(tokens, resp.timepoints)},
                                ensure_ascii=False, indent=2,
                            ),
                            encoding="utf-8",
                        )
                    else:
                        words_path.unlink(missing_ok=True)
            results[i] = {"path": _rel(out_path), "status": "ok"}
            generated += 1
        except Exception as exc:  # noqa: BLE001 - keep going if one section fails
            results[i] = {"path": None, "status": f"error: {exc}"}

    return {"results": results, "generated": generated, "skipped": skipped}


# --------------------------------------------------------------------------- #
# Tool: list projects (for the resume-or-new decision)
# --------------------------------------------------------------------------- #
def list_projects() -> dict:
    """List saved video projects in the workdir so you can resume one.

    Call this at the very START of every conversation. Returns existing projects
    (most recent first) with their progress, plus `current` (the most recent
    project's slug, or "" if none).

    Returns:
        {"status", "count", "current",
         "projects": [{"slug","title","topic","sections","images_done",
                       "audio_done","status"}]}.
    """
    projects = []
    for s in _list_slugs():
        try:
            projects.append(_status(s))
        except Exception:  # noqa: BLE001
            continue
    return {
        "status": "success",
        "count": len(projects),
        "current": projects[0]["slug"] if projects else "",
        "projects": projects,
    }


# --------------------------------------------------------------------------- #
# Tool: load a project (to resume without regenerating the story)
# --------------------------------------------------------------------------- #
def load_project(slug: str = "") -> dict:
    """Load a saved project's full state so you can RESUME it.

    Use this when the user chooses to continue a project — never rewrite the
    story or descriptions, reuse what this returns. If slug is empty, the most
    recent project is loaded.

    Returns:
        {"status","slug","title","topic","story","style","sections" (each with
         text/imageDescription/imagePath/audioPath), "images_done","audio_done",
         "images_total","resume" ("ready_for_media" or "complete")}.
    """
    resolved = _resolve_slug(slug)
    if resolved is None:
        return {"status": "error", "message": "No saved project found to load."}
    p = _paths(resolved)
    data = json.loads(p["script"].read_text(encoding="utf-8"))
    secs = _sections_of(data)
    states = _media_states(resolved, data, _load_manifest(resolved))
    enriched, imgs_done, aud_done = [], 0, 0
    stale_images, stale_audio = [], []
    for i, sec in enumerate(secs, start=1):
        ip = p["img"] / f"section-{i:02d}.png"
        ap = p["audio"] / f"section-{i:02d}.mp3"
        st = states[i]
        has_i, has_a = st["image_exists"], st["audio_exists"]
        imgs_done += 1 if has_i else 0
        aud_done += 1 if has_a else 0
        if st["image_stale"]:
            stale_images.append(i)
        if st["audio_stale"]:
            stale_audio.append(i)
        enriched.append({
            "text": sec.get("text", ""),
            "imageDescription": sec.get("imageDescription", ""),
            "imagePath": _rel(ip) if has_i else None,
            "audioPath": _rel(ap) if has_a else None,
            "imageStale": st["image_stale"],
            "audioStale": st["audio_stale"],
        })
    total = len(secs)
    has_stale = bool(stale_images or stale_audio)
    complete = bool(total and imgs_done >= total and aud_done >= total)
    has_video = (p["base"] / "video.mp4").is_file()
    # A scene edited but not yet regenerated counts as "needs media" again, even if
    # every file is present — so the agent regenerates it before re-rendering.
    if complete and not has_stale and has_video:
        resume = "rendered"
    elif complete and not has_stale:
        resume = "ready_to_render"
    else:
        resume = "ready_for_media"
    return {
        "status": "success",
        "slug": resolved,
        "title": data.get("title", ""),
        "topic": data.get("topic", ""),
        "story": data.get("story", ""),
        "style": data.get("style", ""),
        "sections": enriched,
        "images_done": imgs_done,
        "audio_done": aud_done,
        "images_total": total,
        "stale_images": stale_images,
        "stale_audio": stale_audio,
        "resume": resume,
        "has_video": has_video,
        "video": _rel(p["base"] / "video.mp4") if has_video else None,
        "script_path": _rel(p["script"]),
    }


# --------------------------------------------------------------------------- #
# Tool: save the planned script (story + matched image descriptions)
# --------------------------------------------------------------------------- #
def save_video_script(
    title: str, topic: str, story: str, images_json: str, style: str
) -> dict:
    """Save a planned video script to workdir/<slug>/script.json.

    Call this once you've written the full script (story + matched scenes), to save
    it before showing the user the Style + Scenes summary to confirm. Returns
    `json_text` (the saved JSON) for reference — but do NOT dump it to the user;
    show the clean Style + Scenes summary instead.

    Args:
        title: The title you created for the video.
        topic: The topic the user gave.
        story: The approved story text, verbatim (one continuous paragraph).
        images_json: A JSON array (string) of section objects, each with "text"
            (1-2 matched story sentences, in order) and "imageDescription".
        style: The single shared visual style used across every section.

    Returns:
        {"status","slug","script_path","section_count","json_text"} or
        {"status":"error","message":...}.
    """
    try:
        images = _parse_images(images_json)
    except ValueError as exc:
        return {
            "status": "error",
            "message": (
                f"{exc} Provide images_json as a JSON array of "
                '{"text": "...", "imageDescription": "..."} objects.'
            ),
        }

    if not isinstance(images, list) or not images:
        return {
            "status": "error",
            "message": 'images_json must be a non-empty JSON array of {"text","imageDescription"} objects.',
        }

    normalized = []
    for i, item in enumerate(images):
        if not isinstance(item, dict):
            return {"status": "error", "message": f'Item {i} is not an object.'}
        text = str(item.get("text", "")).strip()
        desc = str(item.get("imageDescription", "")).strip()
        if not text or not desc:
            return {
                "status": "error",
                "message": f'Item {i} is missing a non-empty "text" or "imageDescription".',
            }
        normalized.append({"text": text, "imageDescription": desc})

    if not (SECTIONS_MIN <= len(normalized) <= SECTIONS_MAX):
        return {
            "status": "error",
            "message": (
                f"A video must have between {SECTIONS_MIN} and {SECTIONS_MAX} "
                f"sections, but got {len(normalized)}. Regenerate the image "
                f"descriptions with a section count in that range."
            ),
        }

    record = {
        "title": (title or "Untitled").strip(),
        "topic": topic.strip(),
        "story": story.strip(),
        "style": (style or "").strip(),
        "sections": normalized,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    slug = _slugify(record["title"])
    p = _paths(slug)
    json_text = json.dumps(record, ensure_ascii=False, indent=2)
    _write_json_atomic(p["script"], record)

    return {
        "status": "success",
        "slug": slug,
        "script_path": _rel(p["script"]),
        "section_count": len(normalized),
        "json_text": json_text,
    }


# --------------------------------------------------------------------------- #
# Tool: edit ONE scene (narration text and/or image description) in place
# --------------------------------------------------------------------------- #
def edit_scene(
    slug: str = "", section: int = 0, text: str = "", image_description: str = ""
) -> dict:
    """Change a SINGLE scene's narration text and/or image description in script.json.

    Edits only the one section you name and leaves every other scene — and all of
    their already-generated media — completely untouched. No media is regenerated
    or deleted here; this just updates the script and tells you which of that
    scene's files are now out of date, so you can regenerate ONLY that scene next
    (cheap: it never re-pays for the other scenes' photos). Works the same on a
    freshly planned project or one resumed from a previous session.

    What goes stale:
      - Changing `text` makes that scene's narration audio out of date, and — per
        the user's preference — its photo should be refreshed too (regenerate
        `kinds="both"`).
      - Changing `image_description` makes only that scene's photo out of date
        (regenerate `kinds="image"`).

    Args:
        slug: Which project. Empty = most recent project.
        section: 1-based scene number to edit (e.g. 1 for the first scene).
        text: New narration text for the scene (the words spoken / captioned).
            Leave empty to keep the current text.
        image_description: New, detailed description of what the scene's photo
            should show. Leave empty to keep the current description.

    Returns:
        {"status","slug","section","title","changed" (list of edited fields),
         "regenerate" (which of "audio"/"image" to redo for THIS scene),
         "billed" (true if regenerate includes the photo — a paid image call),
         "text","imageDescription","next"} or {"status":"error","message":...}.
    """
    resolved = _resolve_slug(slug)
    if resolved is None:
        return {"status": "error", "message": "No project found. Save a script first, or pass a valid slug."}

    p = _paths(resolved)
    try:
        data = json.loads(p["script"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "message": f"Could not read script.json: {exc}"}

    secs = _sections_of(data)
    total = len(secs)
    try:
        section = int(section)
    except (TypeError, ValueError):
        section = 0
    if not (1 <= section <= total):
        return {
            "status": "error",
            "message": f"section must be between 1 and {total} (got {section}).",
        }

    new_text = (text or "").strip()
    new_desc = (image_description or "").strip()
    if not new_text and not new_desc:
        return {
            "status": "error",
            "message": "Nothing to change — provide a new `text` and/or `image_description`.",
        }

    sec = secs[section - 1]
    old_text = str(sec.get("text", "")).strip()
    old_desc = str(sec.get("imageDescription", "")).strip()
    text_changed = bool(new_text) and new_text != old_text
    desc_changed = bool(new_desc) and new_desc != old_desc

    if not text_changed and not desc_changed:
        return {
            "status": "success",
            "slug": resolved,
            "section": section,
            "title": data.get("title", ""),
            "changed": [],
            "regenerate": [],
            "billed": False,
            "text": old_text,
            "imageDescription": old_desc,
            "next": "No change — the new value(s) matched the current scene.",
        }

    changed = []
    if text_changed:
        sec["text"] = new_text
        changed.append("text")
    if desc_changed:
        sec["imageDescription"] = new_desc
        changed.append("imageDescription")
    secs[section - 1] = sec
    data["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json_atomic(p["script"], data)

    # Narration change -> audio is stale AND (user preference) refresh the photo too.
    # Image-description change -> only the photo is stale.
    regenerate = []
    if text_changed:
        regenerate.append("audio")
    if desc_changed or text_changed:
        regenerate.append("image")
    billed = "image" in regenerate

    bits = []
    if "audio" in regenerate:
        bits.append("narration audio")
    if "image" in regenerate:
        bits.append("photo (billed image call)")
    next_hint = (
        f"Regenerate ONLY scene {section}: "
        + " + ".join(bits)
        + f". Call the media agent with sections={section} and "
        + ("kinds=both" if len(regenerate) == 2 else f"kinds={regenerate[0]}")
        + ", then re-render. Every other scene is left as-is."
    )

    return {
        "status": "success",
        "slug": resolved,
        "section": section,
        "title": data.get("title", ""),
        "changed": changed,
        "regenerate": regenerate,
        "billed": billed,
        "text": sec.get("text", ""),
        "imageDescription": sec.get("imageDescription", ""),
        "next": next_hint,
    }


# --------------------------------------------------------------------------- #
# Tool: generate per-scene media (image + audio) in parallel
# --------------------------------------------------------------------------- #
def generate_scene_media(
    slug: str = "", sections: str = "", force: bool = False, kinds: str = "both"
) -> dict:
    """Generate per-scene media for a project: an image AND an audio clip per section.

    Images come from Imagen 4 Fast (portrait 9:16); audio from Google Cloud TTS
    (Standard voice, MP3) narrating each section's `text`. Images and audio are
    generated IN PARALLEL. Files are saved to workdir/<slug>/img/section-NN.png
    and workdir/<slug>/audio/section-NN.mp3, and a manifest workdir/<slug>/media.json
    lists every path.

    Resume-aware: fresh files that already exist are SKIPPED, so this only fills in
    what is MISSING or STALE (a scene whose text/description changed since its file
    was made) — unless `force`, or unless `sections` targets specific ones.

    Args:
        slug: Which project. Empty = most recent project.
        sections: Comma-separated section numbers to (re)generate ONLY, e.g. "6"
            or "3,6". Leaves every other file untouched. Use it to redo a single
            failed/unwanted scene, or to re-roll specific photos. Takes precedence
            over `force`.
        force: Regenerate everything, even files that already exist.
        kinds: "both" (default), "image" (images only), or "audio" (audio only) —
            so you can retry just one kind for a section.

    Returns:
        {"status","slug","manifest","images_generated","audio_generated","total",
         "img_dir","audio_dir","missing_images","missing_audio","errors"}.
        `errors` is a list of {"section","kind","message"} for every image/audio
        that failed — show these messages to the user.
    """
    resolved = _resolve_slug(slug)
    if resolved is None:
        return {"status": "error", "message": "No project found. Save a script first, or pass a valid slug."}

    p = _paths(resolved)
    try:
        data = json.loads(p["script"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "message": f"Could not read script.json: {exc}"}

    secs = data.get("sections") or data.get("images") or []
    total = len(secs)
    if not secs:
        return {"status": "error", "message": f"No sections in {_rel(p['script'])}."}

    targets = _parse_targets(sections, total)
    style = data.get("style", "")
    fake = _fake()
    kinds = (kinds or "both").lower()
    do_image = kinds in ("both", "image", "images")
    do_audio = kinds in ("both", "audio")

    # Decide which sections to (re)generate for each kind:
    #   - explicit `sections` win: regenerate exactly those (and nothing else);
    #   - else `force`: regenerate everything;
    #   - else the default fills in only what is MISSING or STALE (its script
    #     text/description changed since the file was made) and leaves fresh files
    #     untouched — so editing one scene only ever costs that one scene.
    manifest_prev = _load_manifest(resolved)
    states = _media_states(resolved, data, manifest_prev)
    all_secs = set(range(1, total + 1))
    if targets:
        img_regen = set(targets) if do_image else set()
        aud_regen = set(targets) if do_audio else set()
    elif force:
        img_regen = set(all_secs) if do_image else set()
        aud_regen = set(all_secs) if do_audio else set()
    else:
        img_regen = {i for i in all_secs
                     if do_image and (not states[i]["image_exists"] or states[i]["image_stale"])}
        aud_regen = {i for i in all_secs
                     if do_audio and (not states[i]["audio_exists"] or states[i]["audio_stale"])}

    # Run image and audio generation concurrently.
    jobs = {}
    with ThreadPoolExecutor(max_workers=2) as ex:
        futs = {}
        if do_image:
            futs["image"] = ex.submit(_generate_kind, "image", secs, p, style, img_regen, fake)
        if do_audio:
            futs["audio"] = ex.submit(_generate_kind, "audio", secs, p, style, aud_regen, fake)
        for k, f in futs.items():
            jobs[k] = f.result()

    def _existing(kind: str, i: int):
        f = (p["img"] / f"section-{i:02d}.png") if kind == "image" else (p["audio"] / f"section-{i:02d}.mp3")
        return ({"path": _rel(f), "status": "exists"} if f.is_file()
                else {"path": None, "status": "missing"})

    built_prev = _as_built(manifest_prev)
    sections_out = []
    for i, section in enumerate(secs, start=1):
        ir = jobs["image"]["results"][i] if "image" in jobs else _existing("image", i)
        ar = jobs["audio"]["results"][i] if "audio" in jobs else _existing("audio", i)
        # media.json is an AS-BUILT record: store the text / description each file
        # was actually generated from. Update to the live script value only for a
        # file (re)generated successfully this call ("ok"); otherwise carry the
        # prior value forward — so a scene left stale stays detectably stale even
        # when some OTHER scene was just regenerated.
        prev_text, prev_desc = built_prev.get(i, (None, None))
        rec_desc = (section.get("imageDescription", "") if ir["status"] == "ok"
                    else prev_desc if prev_desc is not None
                    else section.get("imageDescription", ""))
        rec_text = (section.get("text", "") if ar["status"] == "ok"
                    else prev_text if prev_text is not None
                    else section.get("text", ""))
        sections_out.append({
            "text": rec_text,
            "imageDescription": rec_desc,
            "imagePath": ir["path"], "imageStatus": ir["status"],
            "audioPath": ar["path"], "audioStatus": ar["status"],
        })

    manifest = {
        "title": data.get("title", ""),
        "topic": data.get("topic", ""),
        "story": data.get("story", ""),
        "style": style,
        "image_model": "fake" if fake else IMAGE_MODEL,
        "aspect_ratio": IMAGE_ASPECT_RATIO,
        "tts_voice": "fake" if fake else TTS_VOICE,
        "tts_language": TTS_LANGUAGE,
        "audio_encoding": "mp3",
        "sections": sections_out,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json_atomic(p["manifest"], manifest)

    missing_images = [i for i, s in enumerate(sections_out, 1) if not s["imagePath"]]
    missing_audio = [i for i, s in enumerate(sections_out, 1) if not s["audioPath"]]
    errors = []
    for i, s in enumerate(sections_out, 1):
        if str(s["imageStatus"]).startswith("error"):
            errors.append({"section": i, "kind": "image", "message": s["imageStatus"]})
        if str(s["audioStatus"]).startswith("error"):
            errors.append({"section": i, "kind": "audio", "message": s["audioStatus"]})
    return {
        "status": "success",
        "slug": resolved,
        "manifest": _rel(p["manifest"]),
        "images_generated": jobs.get("image", {}).get("generated", 0),
        "audio_generated": jobs.get("audio", {}).get("generated", 0),
        "total": total,
        "img_dir": _rel(p["img"]),
        "audio_dir": _rel(p["audio"]),
        "missing_images": missing_images,
        "missing_audio": missing_audio,
        "errors": errors,
    }


# --------------------------------------------------------------------------- #
# Tool: render the final video with Remotion (image + audio + captions)
# --------------------------------------------------------------------------- #
async def render_video(slug: str = "", tool_context: ToolContext = None) -> dict:
    """Render the final portrait MP4 from a project's scenes using Remotion.

    Each scene shows its image as a full-screen background while its narration
    audio plays, with the scene text shown as on-screen captions. Requires every
    scene to already have BOTH an image and an audio file. The MP4 is saved to
    workdir/<slug>/video.mp4. If VIDEO_BUCKET is set (the Docker image / Cloud Run),
    the MP4 is uploaded there and a public download link is returned; otherwise it
    is attached as an artifact so it plays inline in the `adk web` dev UI.

    Args:
        slug: Which project. Empty = most recent project.

    Returns:
        On success: {"status","slug","video","artifact","download_url",
        "download_error","scenes","size_mb"}. With VIDEO_BUCKET set, `download_url`
        is a public link to the uploaded MP4 and `artifact` is null; otherwise the
        MP4 is inlined as `artifact`. On failure: {"status":"error","message":...}.
    """
    resolved = _resolve_slug(slug)
    if resolved is None:
        return {"status": "error", "message": "No project found. Save a script first, or pass a valid slug."}

    p = _paths(resolved)
    try:
        data = json.loads(p["script"].read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "error", "message": f"Could not read script.json: {exc}"}

    secs = data.get("sections") or data.get("images") or []
    if not secs:
        return {"status": "error", "message": f"No sections in {_rel(p['script'])}."}

    # Every scene must have an image AND audio, and neither may be STALE (its scene
    # was edited after the file was generated) — otherwise the render would silently
    # use out-of-date media. List every problem so the user can regenerate just
    # those scenes.
    states = _media_states(resolved, data)
    scenes, problems = [], []
    for i, sec in enumerate(secs, start=1):
        st = states[i]
        if not st["image_exists"]:
            problems.append(f"scene {i}: image missing")
        elif st["image_stale"]:
            problems.append(f"scene {i}: image is out of date (its description changed) — regenerate it")
        if not st["audio_exists"]:
            problems.append(f"scene {i}: audio missing")
        elif st["audio_stale"]:
            problems.append(f"scene {i}: audio is out of date (its narration changed) — regenerate it")
        scene = {
            "image": f"img/section-{i:02d}.png",
            "audio": f"audio/section-{i:02d}.mp3",
            "text": sec.get("text", ""),
        }
        # Per-word timings (if generated) drive karaoke-style caption highlighting.
        words_file = p["audio"] / f"section-{i:02d}.words.json"
        if words_file.is_file():
            try:
                wj = json.loads(words_file.read_text(encoding="utf-8"))
                if wj.get("words"):
                    scene["words"] = wj["words"]
            except (OSError, json.JSONDecodeError):
                pass
        scenes.append(scene)
    if problems:
        return {
            "status": "error",
            "message": "Cannot render — fix these scenes first:\n- " + "\n- ".join(problems)
            + "\nRegenerate ONLY the affected scene(s) with the media agent "
            "(e.g. sections=2 kinds=image), then render again.",
        }

    if not (REMOTION_DIR / "node_modules").is_dir():
        return {
            "status": "error",
            "message": f"Remotion is not installed. Run `npm install` once in {_rel(REMOTION_DIR)}.",
        }

    props_path = p["base"] / "render-props.json"
    _write_json_atomic(props_path, {"title": data.get("title", ""), "scenes": scenes})
    out_path = p["base"] / "video.mp4"
    cmd = [
        "npx", "remotion", "render", "src/index.ts", "VideoComposition",
        str(out_path),
        f"--props={props_path}",
        f"--public-dir={p['base']}",
        "--log=error",
    ]
    try:
        proc = await asyncio.to_thread(
            subprocess.run, cmd,
            cwd=str(REMOTION_DIR), capture_output=True, text=True, timeout=1800,
        )
    except subprocess.TimeoutExpired:
        return {"status": "error", "message": "Render timed out (30 min)."}
    except FileNotFoundError:
        return {"status": "error", "message": "`npx` not found — Node.js is required to render."}

    if proc.returncode != 0 or not out_path.is_file():
        tail = (proc.stderr or proc.stdout or "").strip()
        return {"status": "error", "message": "Remotion render failed:\n" + tail[-1500:]}

    size = out_path.stat().st_size
    # How the user gets the finished video out of the dev UI:
    #   - VIDEO_BUCKET set (the Docker image / Cloud Run): upload the MP4 to that GCS
    #     bucket and return a public download link. We do NOT also attach it as an
    #     artifact, because the dev UI fetches the artifact as one base64 JSON blob
    #     and Cloud Run rejects responses over 32 MiB ("response size too large" ->
    #     500). The link downloads straight from GCS, at any size.
    #   - Otherwise (plain local `adk web`): attach it as an artifact so it still
    #     plays inline, as before.
    bucket = os.environ.get("VIDEO_BUCKET", "").strip()
    artifact_name = None
    download_url = None
    download_error = None
    if bucket:
        try:
            download_url = await asyncio.to_thread(
                _upload_to_gcs, bucket, out_path, f"{resolved}.mp4"
            )
        except Exception as exc:  # noqa: BLE001 - file is still on disk; report why
            download_error = str(exc)
    elif tool_context is not None and size <= 64_000_000:
        try:
            await tool_context.save_artifact(
                f"{resolved}.mp4",
                types.Part.from_bytes(data=out_path.read_bytes(), mime_type="video/mp4"),
            )
            artifact_name = f"{resolved}.mp4"
        except Exception:  # noqa: BLE001 - embedding is best-effort; file is on disk
            artifact_name = None

    return {
        "status": "success",
        "slug": resolved,
        "video": _rel(out_path),
        "artifact": artifact_name,
        "download_url": download_url,
        "download_error": download_error,
        "scenes": len(scenes),
        "size_mb": round(size / 1_000_000, 2),
    }


# --------------------------------------------------------------------------- #
# Agents
# --------------------------------------------------------------------------- #
media_generator_agent = Agent(
    name="media_generator_agent",
    model="gemini-2.5-flash",
    description=(
        "Generates per-scene media for a video project: one portrait (9:16) photo "
        "(Imagen 4 Fast) and one narration audio clip (Google Cloud TTS) per "
        "section, in parallel. Skips media that already exists so it can resume."
    ),
    instruction=(
        "You generate the per-scene media (images + narration audio) for an "
        "already-planned video project.\n"
        "From the request, work out: the project slug; optional specific section "
        "numbers (e.g. 'section 6' or 'sections=3,6'); which kinds to do "
        "('both' (default), 'image', or 'audio'); and whether to 'force' redo "
        "everything.\n"
        "Then call `generate_scene_media` exactly once:\n"
        "- normal / fill in missing + refresh out-of-date: generate_scene_media(slug)\n"
        "    (regenerates only media that is missing or stale; fresh files are kept)\n"
        "- redo one scene fully: generate_scene_media(slug, sections=\"6\")\n"
        "- redo only the image of a scene: generate_scene_media(slug, sections=\"6\", kinds=\"image\")\n"
        "- redo only the audio of a scene: generate_scene_media(slug, sections=\"6\", kinds=\"audio\")\n"
        "- redo several scenes' photos only: generate_scene_media(slug, sections=\"2,5\", kinds=\"image\")\n"
        "- redo everything: generate_scene_media(slug, force=true)\n"
        "When the caller names specific sections, regenerate ONLY those — never the\n"
        "whole set.\n"
        "If no slug is in the request, pass an empty string (most recent project).\n"
        "When it returns, report concisely: images_generated and audio_generated "
        "out of total; the manifest path; the image and audio folders. If "
        "`missing_images` or `missing_audio` is non-empty, list those section "
        "numbers as still missing.\n"
        "IMPORTANT: if `errors` is non-empty, show the user EACH entry with its "
        "exact message verbatim, e.g. 'Section 6 image failed: <message>'. Never "
        "hide or paraphrase the error text.\n"
        "Only report what the tool returns — never invent paths or results."
    ),
    tools=[generate_scene_media],
)


video_render_agent = Agent(
    name="video_render_agent",
    model="gemini-2.5-flash",
    description=(
        "Renders the final portrait MP4 from a prepared project's images, audio, "
        "and captions using Remotion."
    ),
    instruction=(
        "You render the final video for a prepared video project.\n"
        "Work out the project slug from the request and call `render_video(slug)` "
        "exactly once (empty string = most recent project). Rendering can take a "
        "few minutes — just wait for the tool to return.\n"
        "On success, report based on what the tool returns:\n"
        "- If `download_url` is set, give the user a clickable Markdown DOWNLOAD "
        "link to it, e.g. \"[⬇ Download the video](<download_url>)\", plus the saved "
        "video path, the number of scenes, and the size in MB.\n"
        "- Else if `download_error` is set, tell the user the render SUCCEEDED (give "
        "the saved path + size) but uploading the download link failed, and show the "
        "`download_error` text verbatim.\n"
        "- Otherwise `artifact` is set: the video is attached and plays inside the "
        "dev UI — tell the user it is shown above (also in the Artifacts panel), and "
        "give the saved video path, the number of scenes, and the size in MB.\n"
        "If it returns an error, show the user the EXACT error message verbatim.\n"
        "Only report what the tool returns."
    ),
    tools=[render_video],
)


# --------------------------------------------------------------------------- #
# Root agent (coordinator) — instruction + wiring
# --------------------------------------------------------------------------- #
# Resume-or-continue preamble (STEP 0). Spliced into the instruction ONLY when
# LOAD_LAST_PROJECT is True; with it False the agent never lists or reopens old
# projects and every run starts a brand-new video.
_RESUME_STEP = """\
STEP 0 — Resume or start new (do this FIRST).
Call `list_projects`. If `count` is 0, go to STEP 1. Otherwise briefly tell the
user about the current project (the first one returned) — its title and progress
(e.g. "7 sections, 5/7 images, 0/7 audio") — and ask: "Continue this project, or
start a new one?" (If more than one project exists, they can name another to
continue.)
  * Continue -> call `load_project` (chosen slug, or empty for the most recent).
    REUSE its story, style, and sections verbatim — never rewrite them. If
    `stale_images` or `stale_audio` is non-empty, tell the user EXACTLY which
    scenes are out of date and offer to regenerate just those (STEP 5). Then act on
    `resume`:
      - "ready_for_media": go to STEP 3 (generate the remaining / out-of-date media).
      - "ready_to_render": go to STEP 4 (offer to render the final video).
      - "rendered": all done — tell them where the video and files are and offer to
        re-render, revise a scene (STEP 5), or start a new project.
  * Start new -> go to STEP 1.

"""

_INSTRUCTION_TEMPLATE = """\
You are a short-form video maker. You turn a TITLE (and an optional short
description) into a professional, vertical 9:16 video: a punchy story split into
scenes, one shared visual style, a background photo + narration audio per scene,
and finally a rendered MP4 (photo background + narration + on-screen captions, via
Remotion). Every project is saved under workdir/<slug>/ so it can be resumed. Work
through the steps in order, one at a time.

<<RESUME_STEP>>STEP 1 — Get the idea (NEW projects).
If the user has not already said what they want, ask exactly ONE short question and
nothing else: "What should the video be about? (e.g. 'benefits of wearing
sunscreen'). You can also be more detailed and say how it should be structured —
e.g. 'top 3 countries with most tourists — 4 scenes: an intro plus one per country,
ordered from third to first place'." Their answer is the TOPIC/idea; any extra
detail (structure, ordering, scene count) is guidance you MUST follow. Invent a
short, catchy TITLE yourself from the idea — never ask the user for a title.
Decide the scene count N: default to 5; use a different number if the user names
one, OR if the idea clearly implies a structure (e.g. a "top 3 …" ranking -> 4
scenes: an intro + one per item). Always keep N between 3 and 12; do not ask about
the count otherwise.

STEP 2 — Write the FULL script (story + scenes) in one go, then SHOW it.
This is a vertical short (TikTok / Reels / YouTube Shorts): it must hook hard in
the first second and stay engaging to the very end. Create the whole script at once
(N = the scene count from STEP 1):
  1. STORY — a short story of N-2N sentences that splits cleanly into exactly N
     scenes of 1-2 sentences each. The FIRST sentence is the HOOK: one short, punchy
     line (5-10 words, never more than ~12) that instantly creates curiosity,
     surprise, or stakes. After the hook keep every line tight, build curiosity and
     pay it off, and end on a satisfying or surprising punch. Conversational spoken
     English, no filler, no intros like "in this video", no title — one continuous
     paragraph. Examples may come from anywhere in the world; the text is in English.
  2. STYLE — a SINGLE consistent visual style for the WHOLE video (medium/render,
     lighting, color palette/grade, mood, lens/finish), captured in one short string.
  3. SCENES — exactly N scenes as a JSON array, in story order, each with:
       - "text": the 1-2 matched story sentences, verbatim and in order. Scene 1's
         text is the HOOK sentence ALONE (do not merge it with the next), so the
         video opens on the hook. This text is also what gets narrated.
       - "imageDescription": a very detailed 9:16 portrait background description
         that BAKES IN the shared style. Make scene 1 especially bold and
         scroll-stopping. Show the story content in a visually engaging way, not
         just characters.
Then call `save_video_script` with: the TITLE you invented; the TOPIC (what the
user said the video should be about); the story verbatim; the scene array as a JSON
string in `images_json`; and your style string.
After it succeeds, do NOT print the raw JSON. Present the script as a clean,
readable summary in Markdown — NO code block, NO JSON, and WITHOUT the title,
topic, or story. Show ONLY the shared visual style and the scenes, exactly like
this (fill in the real content; keep each scene's narration verbatim):

  **Style:** <the one-line visual style>

  **Scene 1**
  - Narration: <scene 1 text>
  - Image: <scene 1 image description>

  **Scene 2**
  - Narration: …
  - Image: …

…continuing for every scene. On error from the tool, fix the JSON and call it again.

STEP 3 — Confirm or change (single checkpoint), then generate the media.
Ask the user to confirm the script, or tell you what to change:
  * Change -> apply their feedback. To rework the whole script, regenerate it per
    STEP 2 and save again; for a single scene prefer STEP 5 (cheaper). Show the
    updated script in the same clean Style + Scenes format and ask again — repeat
    until they clearly approve.
  * Confirm -> tell them this now generates, per scene, a 9:16 photo (Imagen 4 Fast)
    AND a narration clip (Google Cloud TTS) — real, billed calls; files that already
    exist are skipped — then call the `media_generator_agent` tool with just the
    project slug as the request. When it returns, relay images_generated and
    audio_generated out of the total, the manifest path, and the image/audio
    folders. If `errors` is non-empty, show the user EACH entry with its exact
    message verbatim (e.g. "Section 6 image failed: <message>"). If `missing_images`
    or `missing_audio` is non-empty, say EXACTLY which scenes are missing (image,
    audio, or both) and offer to retry only those (e.g. "<slug> sections=6", or
    "<slug> sections=6 kinds=audio"). Never regenerate everything unless the user
    explicitly asks.

STEP 4 — Render the final video (only when EVERY scene has BOTH an image and audio).
Offer to render the portrait MP4 with Remotion: each scene's photo is the
background while its narration plays, with the scene text shown as captions. Ask
whether to render; if the user agrees, call the `video_render_agent` tool with the
project slug. Rendering can take a few minutes. The MP4 is attached as an artifact
and plays right here in the dev UI. Relay that the video is shown above (and in the
Artifacts panel), plus the saved path, scene count, and size in MB. On error, show
the EXACT error message verbatim.

STEP 5 — Revise ONE scene at a time (cheap; after media and after a render<<RESUME_S5>>).
Only the named scene's media is regenerated, so you never re-pay for the other
scenes' photos. Use it whenever the user tweaks a specific scene ("change scene 3's
narration to …", "make scene 2's photo show …", "give scene 4 a different photo",
"redo the photos for scenes 2 and 5").
A) Changing a scene's WORDS and/or what its photo shows -> `edit_scene(slug,
   section=N, text="…" and/or image_description="…")`. It updates only scene N in
   script.json and returns `regenerate` (which of audio/image is now out of date)
   and `billed` (true when a photo must be remade). If `billed` is true, first tell
   the user it will remake scene N's photo (a billed Imagen call — a narration
   change also refreshes that scene's photo) and ask to proceed; if they decline
   the photo, regenerate just the audio. Then call the media agent for that ONE
   scene, e.g. "<slug> sections=N kinds=both" (narration + photo), "<slug>
   sections=N kinds=audio" (narration only), or "<slug> sections=N kinds=image"
   (photo only).
B) Re-rolling a photo WITHOUT changing its description -> skip edit_scene and call
   the media agent naming those sections with kinds=image, e.g. "<slug>
   sections=2,5 kinds=image". Confirm first (billed); one or several scenes at once,
   every other scene left untouched.
After regenerating, offer to re-render (STEP 4) — re-rendering rebuilds the whole
MP4 from the current files and is local/free; only regenerated photos cost anything.

Rules:
- Do the steps in order. Wait for the user before generating media (STEP 3) and
  before rendering (STEP 4) — both do real, billed work.
- Keep ONE shared visual style across all scenes. Pass story/text to tools verbatim.
- For single-scene tweaks use STEP 5 — never re-plan scenes the user didn't ask to
  change.<<RESUME_RULE>>
- Default to 5 scenes (or infer from the idea — e.g. a "top 3 …" ranking -> 4:
  intro + one per item); keep any count between 3 and 12. Be concise.
"""

INSTRUCTION = (
    _INSTRUCTION_TEMPLATE
    .replace("<<RESUME_STEP>>", _RESUME_STEP if LOAD_LAST_PROJECT else "")
    .replace(
        "<<RESUME_S5>>",
        ", and when resuming an old project" if LOAD_LAST_PROJECT else "",
    )
    .replace(
        "<<RESUME_RULE>>",
        "\n- When resuming, NEVER rewrite an existing story or descriptions."
        if LOAD_LAST_PROJECT
        else "",
    )
)


# list_projects / load_project only matter when resuming is enabled, so they are
# registered as tools only then — with LOAD_LAST_PROJECT False the agent has no way
# to list or reopen old projects and always starts fresh.
_root_tools = []
if LOAD_LAST_PROJECT:
    _root_tools += [list_projects, load_project]
_root_tools += [
    save_video_script,
    edit_scene,
    AgentTool(agent=media_generator_agent),
    AgentTool(agent=video_render_agent),
]


root_agent = Agent(
    name="video_script_agent",
    # NOTE: use an enterprise-supported model name for your project.
    model="gemini-2.5-flash",
    description=(
        "Turns a title (+ optional description) into a portrait short-form video: "
        "writes the full script (story + per-scene image prompts) in one shared "
        "style, saves each project under workdir/ (resumable), prints the script "
        "JSON, and hands off to generate an image + narration per scene and render "
        "the final MP4."
    ),
    instruction=INSTRUCTION,
    tools=_root_tools,
)
