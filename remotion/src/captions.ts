// Turn a scene's narration into short, CapCut-style caption chunks. When the TTS
// step produced per-word timepoints (SSML <mark>), we time each chunk — and each
// word inside it — to the real audio, which drives karaoke-style highlighting.
// Without timings we fall back to spreading evenly-split chunks across the audio.

export type Word = {word: string; start: number; end: number}; // seconds

export type CaptionWord = {
  text: string;
  fromFrame: number; // scene-relative
  durationInFrames: number;
};

export type Caption = {
  text: string;
  fromFrame: number; // scene-relative
  durationInFrames: number;
  words?: CaptionWord[]; // present when per-word timings exist (karaoke)
};

// Keep captions short and glanceable: a few words on one short line.
const MAX_WORDS = 4;
const MAX_CHARS = 24;

// Group word tokens into caption-sized chunks (<= MAX_WORDS / MAX_CHARS), breaking
// early after sentence-ending punctuation. Returns arrays of word indices.
function groupWords(words: string[]): number[][] {
  const groups: number[][] = [];
  let cur: number[] = [];
  let curLen = 0;
  const flush = () => {
    if (cur.length) {
      groups.push(cur);
      cur = [];
      curLen = 0;
    }
  };
  for (let i = 0; i < words.length; i++) {
    const w = words[i];
    const extra = (curLen === 0 ? 0 : 1) + w.length;
    if (cur.length >= MAX_WORDS || (cur.length && curLen + extra > MAX_CHARS)) {
      flush();
    }
    cur.push(i);
    curLen += (cur.length === 1 ? 0 : 1) + w.length;
    if (/[.!?…]["')\]]?$/.test(w)) {
      flush();
    }
  }
  flush();
  return groups;
}

export function splitIntoChunks(text: string): string[] {
  const words = (text || '').trim().split(/\s+/).filter(Boolean);
  return groupWords(words).map((g) => g.map((i) => words[i]).join(' '));
}

// Fallback: distribute `windowFrames` across chunks, weighted by character count
// (a rough proxy for speech pace — no per-word timing available). No highlighting.
export function buildCaptions(text: string, windowFrames: number): Caption[] {
  const chunks = splitIntoChunks(text);
  if (!chunks.length || windowFrames <= 0) return [];
  const weights = chunks.map((c) => Math.max(1, c.replace(/\s+/g, '').length));
  const totalW = weights.reduce((a, b) => a + b, 0);
  const caps: Caption[] = [];
  let cursor = 0;
  let acc = 0;
  for (let i = 0; i < chunks.length; i++) {
    acc += weights[i];
    const end =
      i === chunks.length - 1
        ? windowFrames
        : Math.round((acc / totalW) * windowFrames);
    const dur = Math.max(1, end - cursor);
    caps.push({text: chunks[i], fromFrame: cursor, durationInFrames: dur});
    cursor += dur;
  }
  return caps;
}

// Exact timing from TTS word timepoints. Each chunk runs from its first word's
// start to the next chunk's first word start (so chunks tile with no gaps), and
// carries per-word frame ranges for karaoke highlighting. All frames are
// scene-relative; `sceneFrames` clamps the final chunk/word.
export function buildCaptionsFromWords(
  words: Word[],
  fps: number,
  sceneFrames: number,
): Caption[] {
  if (!words.length) return [];
  const groups = groupWords(words.map((w) => w.word));
  const f = (s: number) => Math.round(s * fps);
  const caps: Caption[] = [];

  for (let gi = 0; gi < groups.length; gi++) {
    const g = groups[gi];
    const from = Math.max(0, f(words[g[0]].start));
    let end =
      gi < groups.length - 1
        ? f(words[groups[gi + 1][0]].start)
        : f(words[g[g.length - 1]].end);
    if (!isFinite(end) || end <= from) end = sceneFrames;
    end = Math.min(Math.max(from + 1, end), Math.max(from + 1, sceneFrames));

    const capWords: CaptionWord[] = g.map((wi, k) => {
      const wFrom = Math.min(Math.max(from, f(words[wi].start)), end - 1);
      const wEndRaw = k < g.length - 1 ? f(words[g[k + 1]].start) : end;
      const wEnd = Math.max(wFrom + 1, Math.min(wEndRaw, end));
      return {text: words[wi].word, fromFrame: wFrom, durationInFrames: wEnd - wFrom};
    });

    caps.push({
      text: g.map((i) => words[i].word).join(' '),
      fromFrame: from,
      durationInFrames: Math.max(1, end - from),
      words: capWords,
    });
  }
  return caps;
}
