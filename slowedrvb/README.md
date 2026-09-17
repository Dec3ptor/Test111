# slowedrvb — high quality slow + reverb

A rebuilt version of the browser slowed-and-reverb tool, focused on playback
and editing quality. Everything runs client side: `index.html`, `index.js`
and `stick.svg` are the whole app, so it drops straight into a static host.

## Why the old reverb sounded robotic

The metallic, robotic character came from the reverb being built out of Web
Audio nodes. Three things go wrong there, and all three are fixed here:

1. **A static impulse response rings.** A `ConvolverNode` fed decaying white
   noise puts every reflection on the same fixed comb, which is heard as a
   metallic tone sitting on the tail.
2. **`BiquadFilterNode` reads `Q` in decibels for `lowpass`/`highpass`, not as
   a linear Q.** So `Q = 0.5` is *+0.5 dB of resonance*, not a gentle
   roll-off. Any peak inside a feedback loop beats the decay gain, and the
   tank grows instead of decaying. Butterworth is `Q = -3.01`.
3. **Feedback through the node graph is quantised to 128 samples**, and
   Chrome's cycle handling is not reproducible — the same settings rendered
   twice measured up to 6.6 dB apart, so an export never matched what you
   heard.

The reverb is now real DSP inside an `AudioWorklet`: pre-delay, four series
allpass diffusers per side, then an eight line feedback delay network with a
lossless Householder mixing matrix, one-pole damping (which cannot resonate)
in every feedback path, and a slow independent LFO on each delay tap so the
comb never sits still. Renders are now bit-identical run to run.

## Other quality work

- **Playback** runs through a worklet with Hermite interpolation rather than
  `playbackRate`, and the rate is glided rather than jumped, so slider moves
  do not click.
- **Keep pitch** does WSOLA time-stretching (correlation-aligned grains), so
  you can slow a track down without pitching it down. Verified: a 220 Hz tone
  at 0.5x stays at 221 Hz with keep-pitch on and drops to 110 Hz with it off.
- **Equal-power dry/wet.** A linear crossfade dips ~3 dB in the middle, which
  is exactly where most people leave the mix slider.
- **Every parameter uses `setTargetAtTime`.** Writing `.value` on each `input`
  event is what makes a slider sweep sound like a stepper motor.
- **Bass boost** is a shelf plus a sub bell plus a small 330 Hz scoop, with
  automatic input trim so it does not simply slam the limiter.
- **8D** uses an HRTF `PannerNode` on a circular orbit with the bass kept
  centred below 200 Hz, which is what stops 8D renders sounding thin.
- **Master** ends in a limiter and a safety clipper that is exactly unity
  below its knee.
- **Export** renders offline through the identical chain and writes 16-bit WAV
  (dithered) or 320 kbps MP3.
- Library moved from `localStorage` to IndexedDB, so real audio files fit.
- Waveform scrubbing, drag-to-set loop regions, and keyboard transport.

## Configuration: the YouTube endpoint

Loading by URL needs a server-side downloader — a browser cannot fetch YouTube
audio directly. The client calls:

```
GET <endpoint>?url=<encoded youtube url>
```

and accepts either the audio bytes directly, or JSON containing a link in
`url`, `audioUrl`, `link` or `downloadUrl`. It defaults to `/api/youtube`.
Point it at your existing backend by setting the global before `index.js`
loads:

```html
<script>window.SRVB_YT_ENDPOINT = '/api/your-downloader';</script>
<script src="index.js"></script>
```

Everything else — file loading, drag and drop, tab capture, all effects and
both export formats — works with no backend at all.

## Browser support

Needs `AudioWorklet` (Chrome/Edge 66+, Firefox 76+, Safari 14.1+). Without it
the app falls back to plain `playbackRate` playback and a convolution reverb,
and says so in a banner; time-stretch, tab capture and export are unavailable.
