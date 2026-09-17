# slowedrvb — high quality slow + reverb

A browser slowed-and-reverb tool, focused on playback and editing quality.
Everything runs client side, so it is a plain static site:

```
index.html   markup + styles
index.js     the whole audio engine and UI
stick.svg    tonearm
favicon.svg
.nojekyll    tells github pages to serve the files as-is
```

## Hosting on GitHub Pages

The app lives at the repository root, so Pages serves it with no build step:
**Settings -> Pages -> Source: Deploy from a branch**, then `main` and
`/ (root)`. All asset paths are relative, so it works both at
`user.github.io/repo/` and at a custom domain. Leave **Enforce HTTPS** on:
AudioWorklet and tab capture both need a secure context.

Worth knowing:

- **Pages is static, so there is no YouTube downloader.** Pasting a youtube
  link offers tab capture instead, which needs no server. See below.
- `<link rel="canonical">`, the Open Graph tags and the JSON-LD block in
  `index.html` point at `https://dec3ptor.github.io/Test111/`. Update all four
  if you move to a custom domain (and add a `CNAME` file for it).
- `og:image` is an SVG, which X and Facebook do not render in link previews.
  Swap in a PNG if link cards matter.

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

## YouTube links

A page cannot fetch YouTube audio by itself. Two things stop it, and neither
is fixable in the client:

- `googlevideo.com` serves no `Access-Control-Allow-Origin` header, so the
  browser blocks the request whatever URL you hand it;
- getting that URL at all means running YouTube's player code to solve the
  signature cipher.

So it needs a server, and a static host has nowhere to run one.

### What works with no backend: tab capture

Paste a link and the app offers this instead. Open the video in a tab, come
back, hit **capture tab**, and pick that tab with **share tab audio** ticked.

Let the track play through, then press **stop & keep as track**. The captured
audio is pulled out of the engine's ring buffer, silence either end is
trimmed, and it becomes an ordinary loaded track — waveform, seeking, loop
regions and offline export to WAV or MP3, exactly like a file. So a link ends
up at the same place a download would, without a downloader.

The ring is sized as large as the device allows (7 minutes at 48kHz on
desktop, halved as needed until the allocation succeeds), and it keeps the
most recent audio, so a capture longer than the ring keeps the end rather
than failing. Chrome and Edge support tab audio capture; Firefox and Safari
do not.

### Wiring up a downloader anyway

Host one somewhere that can run code (a small serverless function is plenty,
with CORS allowing your Pages origin). Note that downloading YouTube audio is
against YouTube's Terms of Service; tab capture is not. The client calls:

```
GET <endpoint>?url=<encoded youtube url>
```

and accepts either the audio bytes directly, or JSON containing a link in
`url`, `audioUrl`, `link` or `downloadUrl`. It defaults to `/api/youtube`.
Point it at your backend by setting the global before `index.js` loads:

```html
<script>window.SRVB_YT_ENDPOINT = '/api/your-downloader';</script>
<script src="index.js"></script>
```


## Browser support

Needs `AudioWorklet` (Chrome/Edge 66+, Firefox 76+, Safari 14.1+). Without it
the app falls back to plain `playbackRate` playback and a convolution reverb,
and says so in a banner; time-stretch, tab capture and export are unavailable.
