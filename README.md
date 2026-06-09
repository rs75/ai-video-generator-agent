# Video Script Agent — ADK + Gemini + Imagen 4 Fast + Cloud TTS + Remotion

Turn a **topic** into a portrait (9:16) short-form video. A multi-agent
[Agent Development Kit (ADK)](https://google.github.io/adk-docs/) pipeline writes
a short story, picks a consistent visual style with one image description per
scene, generates a background photo **and** narration audio for each scene (in
parallel), and renders a final **MP4** — photo background, narration, and
karaoke-style captions — with [Remotion](https://www.remotion.dev/).

## Demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/88_q5Mz1No0/hqdefault.jpg)](https://www.youtube.com/shorts/88_q5Mz1No0)

▶️ [Watch a generated short on YouTube](https://www.youtube.com/shorts/88_q5Mz1No0)

## Quick start (Docker)

The whole stack — the Python agent **and** the Remotion renderer (Node +
headless Chrome) — runs in one container. It authenticates to Google Cloud by
**mounting your local credentials** at run time; no keys or project ids are
baked into the image.

**Prerequisites**

- [Docker](https://docs.docker.com/get-docker/)
- The [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- A Google Cloud project with **billing enabled** and these APIs turned on:
  ```bash
  gcloud services enable aiplatform.googleapis.com texttospeech.googleapis.com
  ```
  (Vertex AI = Gemini + Imagen; Text-to-Speech = narration.)

**Run it**

```bash
# 1. Authenticate once on your host (opens a browser).
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# 2. Build the image.
docker build -t video-agent .

# 3. Run it — mounts your gcloud credentials read-only.
docker run --rm -p 8080:8080 \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  video-agent
```

Open **http://localhost:8080**, select the agent, and give it a topic — for
example **`5 biggest cities`**. The agent asks how many scenes (3–12), writes the
story, generates each scene's photo + narration, and renders the final MP4
(which also plays inline in the dev UI).

> The container reads your project from the mounted gcloud config automatically.
> If it can't, pass it explicitly by adding
> `-e GOOGLE_CLOUD_PROJECT=your-project-id` to the `docker run` command.

### Try the full flow free (no API cost)

Image and narration are real, **billed** Google Cloud calls (~$0.15 for a
7-scene video, mostly Imagen). To exercise the entire pipeline with **zero** API
cost — it writes tiny placeholder media instead of calling Imagen/TTS — add this
to the `docker run` command:

```bash
  -e VIDEO_AGENT_FAKE_MEDIA=1
```

## How it works

1. **Scene planner** — `video_script_agent` (root). Asks for the topic and scene
   count, invents a title, and writes a short, viral-friendly script — the first
   scene is always a short, punchy hook — then picks **one** consistent visual
   style and writes one image description per scene. Saves `script.json`.
2. **Media generator** — `media_generator_agent`. Per scene, in parallel:
   - **Image** — Imagen 4 Fast (`imagen-4.0-fast-generate-001`), 9:16.
   - **Audio** — Cloud TTS (Standard voice `en-US-Standard-C`, MP3) narrating the
     scene. SSML `<mark>` tags make the response also return per-word timepoints
     for karaoke captions (the `<mark>` tags aren't billed).
3. **Video render** — `video_render_agent`. Remotion renders `video.mp4`: each
   scene's photo is the background (subtle Ken Burns zoom) under its narration,
   with CapCut-style captions shown a few words at a time and the currently
   spoken word highlighted from the TTS timepoints. Portrait 1080×1920 @ 30 fps;
   each scene lasts as long as its audio clip.

Every project is saved under `workdir/<slug>/`, so work survives restarts and can
be resumed — media generation skips files that already exist.

## Outputs

Per project, under `workdir/<slug>/`:

- `script.json` — title, topic, story, style, and sections (each `text` +
  `imageDescription`).
- `media.json` — manifest: per scene `imagePath` + `audioPath` (+ statuses) and
  the models/voice used.
- `img/section-NN.png` — portrait 9:16 background photos.
- `audio/section-NN.mp3` — narration, one clip per scene.
- `audio/section-NN.words.json` — per-word timings for karaoke captions.
- `video.mp4` — the final short (1080×1920, H.264 + AAC).

## Resume, retry & revise

State lives in those files, so you can stop and restart anytime:

- On startup the agent offers to **continue** the most recent project or start new.
- Media generation **skips** any photo/audio that already exists, filling in only
  what's missing or out of date.
- If a scene fails, the agent says which one and can retry **just that scene** (or
  just its image, or just its audio).
- You can revise a **single scene** after the fact — change its narration or what
  its photo shows — and only that scene's media is regenerated. The renderer
  refuses to build while any scene is stale, and names the offending scenes, so a
  re-render never uses out-of-date media. Re-rendering itself is local and free.

## Cost notes

- **Imagen 4 Fast** — billed per generated image (~$0.02 each on Vertex AI).
- **Cloud TTS Standard voice** — the cheapest tier (~$4 per 1M characters; first
  4M characters/month free). The `<mark>` tags used for karaoke aren't billed.
- **Remotion render** — runs locally in the container (CPU + headless Chrome); no
  cloud cost.

## Local development (without Docker)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install --prefix remotion      # Node 18+; first render downloads a headless Chrome

# Configure auth + project:
cp create_video_agent/.env.example create_video_agent/.env   # then set GOOGLE_CLOUD_PROJECT
gcloud auth application-default login

# Run the agent (pass the folder so ADK runs in single-agent mode):
adk web create_video_agent --port 8000     # web UI
adk run create_video_agent                 # terminal chat
```

## Configuration

In `create_video_agent/agent.py`:

- `IMAGE_MODEL = "imagen-4.0-fast-generate-001"`, `IMAGE_ASPECT_RATIO = "9:16"`.
- `TTS_VOICE = "en-US-Standard-C"`, `TTS_LANGUAGE = "en-US"`.
- Both agents use `gemini-2.5-flash`.

Auth and project come from the environment (`.env` for local dev, or the mounted
gcloud config / `-e` for Docker) — nothing is hardcoded.

> Note: `adk web` is ADK's local dev UI, not a production server. For a hosted
> deployment, put a custom front-end in front of `adk api_server`, or deploy to
> Cloud Run.
