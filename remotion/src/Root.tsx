import React from 'react';
import {Composition, staticFile} from 'remotion';
import {getAudioDurationInSeconds} from '@remotion/media-utils';
import {VideoComposition, Scene} from './VideoComposition';
import {buildCaptions, buildCaptionsFromWords} from './captions';

// Portrait 9:16 for phone / TikTok / YouTube Shorts.
const FPS = 30;
const WIDTH = 1080;
const HEIGHT = 1920;
const FALLBACK_SECONDS = 4; // used if an audio duration can't be read

type Props = {
  title: string;
  scenes: Scene[];
};

export const RemotionRoot: React.FC = () => {
  return (
    <Composition
      id="VideoComposition"
      component={VideoComposition as React.FC<Props>}
      durationInFrames={FPS * FALLBACK_SECONDS}
      fps={FPS}
      width={WIDTH}
      height={HEIGHT}
      defaultProps={{title: '', scenes: []} as Props}
      // Compute each scene's length from its audio file, and the total duration.
      calculateMetadata={async ({props}) => {
        const scenes: Scene[] = [];
        let total = 0;
        for (const s of props.scenes ?? []) {
          let audioSeconds = FALLBACK_SECONDS;
          try {
            const dur = await getAudioDurationInSeconds(staticFile(s.audio));
            if (dur && isFinite(dur)) {
              audioSeconds = dur;
            }
          } catch (e) {
            // keep fallback duration
          }
          // Captions are timed to the spoken audio window; the scene runs a small
          // tail longer so the last caption / image don't cut off the instant
          // the audio ends.
          const audioFrames = Math.max(1, Math.round(audioSeconds * FPS));
          const frames = Math.max(1, Math.round((audioSeconds + 0.4) * FPS));
          // Real per-word timings -> karaoke captions; otherwise even chunks.
          const captions =
            s.words && s.words.length
              ? buildCaptionsFromWords(s.words, FPS, audioFrames)
              : buildCaptions(s.text, audioFrames);
          scenes.push({...s, durationInFrames: frames, captions});
          total += frames;
        }
        return {
          durationInFrames: Math.max(1, total),
          fps: FPS,
          width: WIDTH,
          height: HEIGHT,
          props: {...props, scenes},
        };
      }}
    />
  );
};
