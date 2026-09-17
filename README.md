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

- **Loading by url goes through a downloader you run**, since Pages is
  static and a browser cannot pull audio off youtube itself. With nothing
  running, pasting a youtube link offers tab capture instead, which needs no
  server. See below.
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

### Wiring up a downloader

Loading by url defaults to a local [yt-audio-api][yt-audio-api] instance at
`http://127.0.0.1:5000/`, which you run yourself — it needs Python, `yt-dlp`
and FFmpeg:

```bash
git clone https://github.com/alperensumeroglu/yt-audio-api
cd yt-audio-api
pip install -r requirements.txt
python3 main.py
```

**It will not work unmodified from the Pages site.** It sends no
`Access-Control-Allow-Origin`, so the browser blocks every response; see
[CORS](#cors-since-the-endpoint-is-on-a-different-origin-to-pages) below.
Note too that downloading YouTube audio is against YouTube's Terms of
Service; tab capture is not.

[yt-audio-api]: https://github.com/alperensumeroglu/yt-audio-api

To point at a different downloader, uncomment the line in `index.html`:

```html
<script>window.SRVB_YT_ENDPOINT = 'https://your-server.example.com/api/youtube';</script>
```

It is called as `GET <endpoint>?url=<encoded youtube url>` and any of these
shapes works:

- **audio bytes directly** — any non-JSON content type. Send
  `content-length` (or `estimated-content-length` if the length is not known
  until the stream ends) so the progress bar moves, and
  `content-disposition: attachment; filename="..."` to name the track. Both
  the RFC 5987 `filename*=UTF-8''...` form and the plain quoted form are read.
- **a token** — `{"token": "..."}`, what yt-audio-api returns. The bytes are
  then fetched from `download?token=...` resolved against the endpoint url,
  so an endpoint at `http://127.0.0.1:5000/` is paired with
  `http://127.0.0.1:5000/download`. Tokens are one-shot and expire five
  minutes after conversion finishes; `401` and `408` on that second request
  are reported as an expired link rather than a bare status.
- **a media url** — `{"mediaInfo": {"title": "...", "audioUrl": "https://..."}}`,
  also read from `mediaInfo.audio_url` or a top-level `audioUrl`,
  `audio_url`, `url`, `link`, `downloadUrl` or `download_url`. The browser
  fetches that url itself, so *it* needs CORS too. Streaming the bytes
  through your own endpoint avoids that entirely.
- **audio bytes directly** — see above.

`title`, where an endpoint sends one, names the track: anything illegal in a
filename is replaced and `.mp3` appended if it carries no extension. With no
title the name comes from the video id (`youtube dQw4w9WgXcQ.mp3`), matching
how a tab capture is named — yt-audio-api names every file after a uuid, and
a bare uuid is ignored in favour of the id. `{"error": "..."}` or
`{"success": false, "message": "..."}` is reported verbatim.

On failure, reply with JSON `{"error": "..."}` and the message is shown to
the user verbatim instead of a bare status code.

#### CORS, since the endpoint is on a different origin to Pages

Your endpoint **must** send:

```
Access-Control-Allow-Origin: https://dec3ptor.github.io
Access-Control-Expose-Headers: Content-Length, Estimated-Content-Length, Content-Disposition
```

yt-audio-api sends neither, so add them to its `main.py` before it will work
from anywhere but a page opened off the same origin:

```python
from flask_cors import CORS   # pip install flask-cors

app = Flask(__name__)
CORS(app, expose_headers=['Content-Length', 'Content-Disposition'])
```

Without the first, the browser blocks the response and the app cannot tell
that apart from the server being down — it reports the endpoint as
unreachable and offers tab capture. Without the second, the download still
works but arrives unnamed and with no progress bar, because cross-origin
JavaScript cannot read headers that are not explicitly exposed.

`404` and `405` are treated as "no downloader here" and also fall back to
tab capture, so an endpoint behind a typo degrades gracefully rather than
looking broken.

## Browser support

Needs `AudioWorklet` (Chrome/Edge 66+, Firefox 76+, Safari 14.1+). Without it
the app falls back to plain `playbackRate` playback and a convolution reverb,
and says so in a banner; time-stretch, tab capture and export are unavailable.
