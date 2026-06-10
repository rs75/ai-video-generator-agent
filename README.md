# Faceless Video Agent

**Give it a topic, get a ready-to-post short video about a minute later.**

Making short-form, faceless videos for YouTube Shorts, TikTok, or Reels usually
takes a lot of time, effort, and editing skill. This project automates the whole
thing. You hand an AI agent a topic — say *"Benefits of learning a new
language"* — and it writes a viral-friendly script, generates the visuals and
voiceover, and renders a finished portrait (9:16) MP4 you can post as-is.

It's built on Google's [Agent Development Kit (ADK)](https://google.github.io/adk-docs/),
runs locally, and is designed to be **fast, cheap, and interactive** — you can
tweak any part of the video by just asking, and only that piece gets remade.

## Demo

[![Watch the demo on YouTube](https://img.youtube.com/vi/88_qzmCygsPAr0/hqdefault.jpg)](https://www.youtube.com/shorts/qzmCygsPAr0)

▶️ [Watch a generated short on YouTube](https://www.youtube.com/shorts/88_q5Mz1No0)

## How it works

You chat with one main agent. It plans the video, then hands the heavy lifting to
specialized sub-agents:

1. **Scene planner** (`video_script_agent`, the agent you talk to) — asks for your
   topic and how many scenes you want (3–12), comes up with a title, and writes a
   short, punchy script. The first scene is always a hook to stop the scroll. It
   then picks **one** consistent visual style and writes an image description for
   each scene. Saved as `script.json`.
2. **Media generator** (`media_generator_agent`) — for every scene, in parallel:
   - **Image** — a 9:16 background photo from Imagen 4 Fast.
   - **Voiceover** — narration from Google Cloud Text-to-Speech. It also returns
     per-word timings, which is what powers the karaoke-style captions later.
3. **Video render** (`video_render_agent`) — [Remotion](https://www.remotion.dev/)
   stitches everything into the final `video.mp4`: each scene's photo as the
   background (with a subtle Ken Burns zoom so it feels alive), the narration on
   top, and CapCut-style captions that highlight each word as it's spoken.
   Portrait 1080×1920 at 30 fps; each scene lasts exactly as long as its audio.

Everything is saved to disk as the agent works, so nothing is lost if you stop
and come back later.

## It's interactive

This is the part that makes it actually usable. The agent saves every asset
locally as it goes, so you're never stuck with a take you don't like:

- Don't like one image or voiceover? Just ask the agent to change it — it
  regenerates **only that piece**, not the whole video.
- Want to rewrite a single scene's narration or change what its photo shows? Same
  deal — only that scene's media is remade.
- If a scene fails, the agent tells you which one and can retry just that scene
  (or just its image, or just its audio).

The renderer refuses to build while any scene is out of date and names the stale
scenes, so a re-render never ships old media. Re-rendering is local and free.

## Quick start (Docker)

The whole stack — the Python agent **and** the Remotion renderer (Node + headless
Chrome) — runs in one container. It signs in to Google Cloud by **mounting your
local credentials** at run time, so no keys or project IDs are ever baked into the
image.

**You'll need:**

- [Docker](https://docs.docker.com/get-docker/)
- The [gcloud CLI](https://cloud.google.com/sdk/docs/install)
- A Google Cloud project with **billing enabled** and these two APIs turned on:
  ```bash
  gcloud services enable aiplatform.googleapis.com texttospeech.googleapis.com
  ```
  (Vertex AI powers Gemini + Imagen; Text-to-Speech powers the narration.)

**Then run:**

```bash
# 1. Sign in once on your machine (opens a browser).
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# 2. Build the image.
docker build -t video-agent .

# 3. Run it — mounts your gcloud credentials read-only.
docker run --rm -p 8080:8080 \
  -v "$HOME/.config/gcloud:/root/.config/gcloud:ro" \
  video-agent
```

Open **http://localhost:8080**, pick the agent, and give it a topic — for example
**`5 biggest cities`**. It asks how many scenes you want, writes the script,
generates each scene's photo + narration, and renders the final MP4 (which also
plays right inside the dev UI).

> The container reads your project from the mounted gcloud config automatically.
> If it can't find it, pass it explicitly by adding
> `-e GOOGLE_CLOUD_PROJECT=your-project-id` to the `docker run` command.

## What you get

Every project is saved under `workdir/<slug>/`:

- `script.json` — title, topic, story, style, and each scene's text +
  image description.
- `media.json` — a manifest of each scene's image and audio paths (and statuses),
  plus the models and voice used.
- `img/section-NN.png` — the 9:16 background photos.
- `audio/section-NN.mp3` — one narration clip per scene.
- `audio/section-NN.words.json` — per-word timings for the karaoke captions.
- `video.mp4` — the final short (1080×1920, H.264 + AAC).

Because all the state lives in these files, you can stop anytime and pick up where
you left off. On startup the agent offers to **continue** your most recent project
or start fresh, and media generation **skips** anything that already exists —
filling in only what's missing or out of date.

## Tech stack

| Tool | What it does |
|------|--------------|
| **[Google ADK](https://google.github.io/adk-docs/)** | Runs the multi-agent pipeline (scene planner → media generator → video render). |
| **Gemini 2.5 Flash** (via Vertex AI) | Writes the scripts and drives the agents. |
| **Imagen 4 Fast** | Generates the 9:16 background images. |
| **Google Cloud TTS** | Generates narration and per-word timings (via SSML) for the captions. |
| **[Remotion](https://www.remotion.dev/)** | Stitches images, audio, and captions into the final MP4. |
| **Docker** | Packages the Python agent and Node renderer into one container. |

The agent leans on Gemini's built-in knowledge to write accurate, engaging
scripts from your prompt — no external data sources to wire up.

## Why static images instead of AI video?

Our biggest design decision was balancing quality against cost and speed. We
first considered AI-generated video models (like Veo 3) for the backgrounds, but
they're expensive and slow to run at scale. We found that high-quality static
images (Imagen 4 Fast) + a dynamic Ken Burns zoom + engaging karaoke captions get
you a highly watchable result at a **fraction of the cost** — and in about a
minute instead of many.

That trade-off shows up in the numbers:

- **Imagen 4 Fast** — billed per image (~$0.02 each on Vertex AI).
- **Cloud TTS Standard voice** — the cheapest tier (~$4 per 1M characters; first
  4M characters/month are free). The timing tags used for captions aren't billed.
- **Remotion render** — runs locally in the container; no cloud cost at all.

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

---

This project was built to generate promotional videos for our apps [Attractiveness Test](https://attractivenesstest.com) and [Attractiveness AI](https://attractivenessai.com).
