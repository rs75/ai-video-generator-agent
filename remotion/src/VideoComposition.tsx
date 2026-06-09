import React from 'react';
import {
  AbsoluteFill,
  Audio,
  Img,
  interpolate,
  Series,
  spring,
  staticFile,
  useCurrentFrame,
  useVideoConfig,
} from 'remotion';
import {buildCaptions, Caption, Word} from './captions';

export type Scene = {
  image: string; // path relative to the public dir, e.g. "img/section-01.png"
  audio: string; // e.g. "audio/section-01.mp3"
  text: string; // full narration text for the scene
  words?: Word[]; // per-word TTS timepoints (seconds) -> karaoke highlighting
  durationInFrames?: number;
  captions?: Caption[]; // timed chunks, precomputed in Root.calculateMetadata
};

export const VideoComposition: React.FC<{title: string; scenes: Scene[]}> = ({
  scenes,
}) => {
  return (
    <AbsoluteFill style={{backgroundColor: 'black'}}>
      <Series>
        {scenes.map((scene, i) => (
          <Series.Sequence
            key={i}
            durationInFrames={Math.max(1, scene.durationInFrames ?? 120)}
          >
            <SceneView scene={scene} />
          </Series.Sequence>
        ))}
      </Series>
    </AbsoluteFill>
  );
};

const SceneView: React.FC<{scene: Scene}> = ({scene}) => {
  const frame = useCurrentFrame();
  const dur = Math.max(1, scene.durationInFrames ?? 120);

  // Gentle Ken Burns zoom so static photos feel alive.
  const scale = interpolate(frame, [0, dur], [1.05, 1.14], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  // Short, time-synced caption chunks (CapCut style). Prefer the ones precomputed
  // in Root (timed to real audio / word timepoints); fall back to even chunks.
  const captions =
    scene.captions && scene.captions.length
      ? scene.captions
      : buildCaptions(scene.text, dur);

  // Which chunk is on screen now (the last one whose start we've reached).
  let active: Caption | undefined;
  for (const c of captions) {
    if (frame >= c.fromFrame) active = c;
  }
  if (!active && captions.length) active = captions[0];

  return (
    <AbsoluteFill style={{backgroundColor: 'black'}}>
      <AbsoluteFill style={{transform: `scale(${scale})`}}>
        <Img
          src={staticFile(scene.image)}
          style={{width: '100%', height: '100%', objectFit: 'cover'}}
        />
      </AbsoluteFill>

      {/* Narration audio for this scene. */}
      <Audio src={staticFile(scene.audio)} />

      {active ? <CaptionView caption={active} /> : null}
    </AbsoluteFill>
  );
};

// Per-word styling. A thick dark outline keeps the text legible on ANY background
// (paint-order draws the stroke behind the fill so the letters stay crisp); the
// active word turns accent-yellow and pops slightly for the karaoke effect.
const WORD_BASE: React.CSSProperties = {
  fontFamily: 'Inter, "Helvetica Neue", Helvetica, Arial, sans-serif',
  fontSize: 78,
  fontWeight: 800,
  lineHeight: 1.12,
  letterSpacing: '0.5px',
  margin: 0,
  WebkitTextStroke: '10px #000000',
  paintOrder: 'stroke fill',
  textShadow: '0 6px 18px rgba(0,0,0,0.45)',
  display: 'inline-block',
};
const ACCENT = '#FFD83A';

const CaptionView: React.FC<{caption: Caption}> = ({caption}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const local = frame - caption.fromFrame;

  // Whole chunk pops in (scale overshoot + fade), restarted for each new caption.
  const enter = spring({
    frame: local,
    fps,
    config: {damping: 12, stiffness: 170, mass: 0.6},
  });
  const popScale = interpolate(enter, [0, 1], [0.7, 1]);
  const opacity = interpolate(local, [0, 3], [0, 1], {
    extrapolateLeft: 'clamp',
    extrapolateRight: 'clamp',
  });

  const words = caption.words;

  // The word being spoken right now (last word whose start we've reached).
  let activeIdx = -1;
  if (words) {
    for (let i = 0; i < words.length; i++) {
      if (frame >= words[i].fromFrame) activeIdx = i;
    }
  }

  return (
    <AbsoluteFill
      style={{
        justifyContent: 'flex-end',
        alignItems: 'center',
        padding: '0 60px 340px 60px',
      }}
    >
      {words ? (
        <div
          style={{
            display: 'flex',
            flexWrap: 'wrap',
            justifyContent: 'center',
            alignItems: 'flex-end',
            columnGap: 18,
            rowGap: 6,
            maxWidth: '88%',
            transform: `scale(${popScale})`,
            opacity,
          }}
        >
          {words.map((w, i) => {
            const isActive = i === activeIdx;
            return (
              <span
                key={i}
                style={{
                  ...WORD_BASE,
                  color: isActive ? ACCENT : '#ffffff',
                  transform: isActive ? 'scale(1.06)' : 'scale(1)',
                }}
              >
                {w.text}
              </span>
            );
          })}
        </div>
      ) : (
        <span
          style={{
            ...WORD_BASE,
            color: '#ffffff',
            textAlign: 'center',
            maxWidth: '88%',
            transform: `scale(${popScale})`,
            opacity,
          }}
        >
          {caption.text}
        </span>
      )}
    </AbsoluteFill>
  );
};
