/*
 * slowedrvb — high quality slow + reverb engine
 *
 * Everything runs client side. The signal path is:
 *
 *   sr-player worklet (variable rate / WSOLA time-stretch)
 *     -> tone stage (low shelf, air shelf, gain compensation)
 *       -> equal power dry/wet split
 *         -> algorithmic reverb (diffusers + modulated 8 line FDN)
 *       -> spatialiser (HRTF orbit for 8D, stereo panner otherwise)
 *         -> limiter -> soft clip -> master
 */

'use strict';

/* ------------------------------------------------------------------ *
 * 1. worklet: sample accurate variable rate player + WSOLA stretcher
 * ------------------------------------------------------------------ */

const WORKLET_SRC = `
'use strict';

function hermite(y0, y1, y2, y3, t) {
  var c0 = y1;
  var c1 = 0.5 * (y2 - y0);
  var c2 = y0 - 2.5 * y1 + 2 * y2 - 0.5 * y3;
  var c3 = 0.5 * (y3 - y0) + 1.5 * (y1 - y2);
  return ((c3 * t + c2) * t + c1) * t + c0;
}

class SRPlayer extends AudioWorkletProcessor {
  constructor(options) {
    super();

    this.mode = 'idle';        // 'file' | 'stream'
    this.chans = [];           // always two Float32Arrays
    this.frames = 0;           // valid frames (file) / ring length (stream)
    this.capacity = 0;
    this.writePos = 0;         // absolute write cursor, stream only
    this.readPos = 0;          // absolute fractional read cursor
    this.srcRate = sampleRate;

    this.playing = false;
    this.gain = 0;             // declick envelope
    this.gainCoef = Math.exp(-1 / (0.004 * sampleRate));

    this.rate = 1;
    this.rateTarget = 1;
    this.readRate = 1;
    this.preservePitch = false;

    this.loop = false;
    this.loopStart = 0;
    this.loopEnd = 0;

    // WSOLA state. 46ms grains with 50% overlap is a good compromise
    // between transient smearing and comb tone on sustained material.
    var g = Math.max(512, Math.round(0.046 * sampleRate));
    this.grain = g + (g % 2);
    this.hop = this.grain >> 1;
    this.win = new Float32Array(this.grain);
    for (var i = 0; i < this.grain; i++) {
      this.win[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / this.grain);
    }
    this.accL = new Float32Array(this.grain);
    this.accR = new Float32Array(this.grain);
    this.tpl = new Float32Array(this.hop);
    this.hasTpl = false;
    this.search = Math.min(this.hop - 8, Math.round(0.011 * sampleRate));
    this.fifoL = new Float32Array(this.grain * 4);
    this.fifoR = new Float32Array(this.grain * 4);
    this.fifoLen = 0;

    this.posCounter = 0;
    this.ended = false;

    this.port.onmessage = (e) => this.onMsg(e.data);

    // An OfflineAudioContext can finish rendering before a postMessage is
    // ever delivered to this thread, so everything needed to make sound
    // arrives through processorOptions instead.
    var init = options && options.processorOptions;
    if (init) {
      if (init.rate !== undefined) { this.rateTarget = init.rate; this.rate = init.rate; }
      if (init.preservePitch !== undefined) this.preservePitch = init.preservePitch;
      if (init.t) this.onMsg(init);
      if (init.autoplay) this.playing = true;
    }
  }

  resetOla() {
    this.accL.fill(0);
    this.accR.fill(0);
    this.fifoLen = 0;
    this.hasTpl = false;
  }

  onMsg(m) {
    switch (m.t) {
      case 'file': {
        this.mode = 'file';
        var chans = m.chans.map(function (b) {
          return ArrayBuffer.isView(b) ? b : new Float32Array(b);
        });
        if (chans.length === 1) chans.push(chans[0]);
        this.chans = chans;
        this.frames = m.frames;
        this.capacity = m.frames;
        this.srcRate = m.srcRate;
        this.writePos = m.frames;
        this.readPos = 0;
        this.ended = false;
        this.resetOla();
        this.port.postMessage({ t: 'loaded', duration: this.frames / this.srcRate });
        break;
      }
      case 'stream': {
        this.mode = 'stream';
        this.srcRate = sampleRate;
        // the ring doubles as the recording, so ask for as much as this
        // device will give us and halve until the allocation succeeds
        var cap = Math.max(sampleRate * 4, Math.round(m.seconds * sampleRate));
        var got = false;
        while (!got) {
          try {
            this.chans = [new Float32Array(cap), new Float32Array(cap)];
            got = true;
          } catch (e) {
            if (cap <= sampleRate * 20) throw e;
            cap = Math.floor(cap / 2);
          }
        }
        this.capacity = cap;
        this.frames = cap;
        this.writePos = 0;
        this.readPos = 0;
        this.ended = false;
        this.resetOla();
        this.port.postMessage({ t: 'streamReady', seconds: cap / sampleRate });
        break;
      }

      // hand back everything captured so far, oldest first, so the main
      // thread can turn it into an ordinary track
      case 'dump': {
        if (this.mode !== 'stream') { this.port.postMessage({ t: 'dump', frames: 0 }); break; }
        var total = Math.min(this.writePos, this.capacity);
        var from = this.writePos - total;
        var dl = new Float32Array(total), dr = new Float32Array(total);
        for (var i = 0; i < total; i++) {
          var k = (from + i) % this.capacity;
          dl[i] = this.chans[0][k];
          dr[i] = this.chans[1][k];
        }
        this.port.postMessage(
          { t: 'dump', chans: [dl.buffer, dr.buffer], frames: total, srcRate: this.srcRate },
          [dl.buffer, dr.buffer]);
        break;
      }
      case 'play':
        if (this.mode === 'file' && this.readPos >= this.frames - 1) this.readPos = 0;
        this.playing = true;
        this.ended = false;
        break;
      case 'pause':
        this.playing = false;
        break;
      case 'seek':
        this.readPos = Math.max(0, Math.min(this.frames - 1, m.time * this.srcRate));
        this.ended = false;
        this.resetOla();
        break;
      case 'live':
        this.readPos = Math.max(0, this.writePos - 0.2 * this.srcRate);
        this.resetOla();
        break;
      case 'params':
        if (m.rate !== undefined) this.rateTarget = Math.max(0.05, m.rate);
        if (m.preservePitch !== undefined && m.preservePitch !== this.preservePitch) {
          this.preservePitch = m.preservePitch;
          this.resetOla();
        }
        if (m.loop !== undefined) this.loop = m.loop;
        if (m.loopStart !== undefined) this.loopStart = m.loopStart * this.srcRate;
        if (m.loopEnd !== undefined) this.loopEnd = m.loopEnd * this.srcRate;
        break;
      case 'snapRate':
        this.rateTarget = Math.max(0.05, m.rate);
        this.rate = this.rateTarget;
        break;
      case 'clear':
        this.mode = 'idle';
        this.playing = false;
        this.chans = [];
        this.frames = 0;
        this.resetOla();
        break;
    }
  }

  readAt(ch, i) {
    if (this.mode === 'stream') {
      if (i < 0) return 0;
      var k = i % this.capacity;
      return ch[k];
    }
    if (i < 0 || i >= this.frames) return 0;
    return ch[i];
  }

  sampleAt(ch, pos) {
    var i = Math.floor(pos);
    var t = pos - i;
    return hermite(this.readAt(ch, i - 1), this.readAt(ch, i),
                   this.readAt(ch, i + 1), this.readAt(ch, i + 2), t);
  }

  /* keeps readPos inside the track / loop. false => nothing left to play */
  guard() {
    if (this.mode === 'stream') return true;
    if (this.loop && this.loopEnd > this.loopStart + 0.01 * this.srcRate) {
      if (this.readPos >= this.loopEnd) {
        var over = this.readPos - this.loopEnd;
        this.readPos = this.loopStart + (over % Math.max(1, this.loopEnd - this.loopStart));
        this.hasTpl = false;
      }
      return true;
    }
    if (this.readPos >= this.frames) {
      this.readPos = this.frames;
      if (!this.ended) {
        this.ended = true;
        this.playing = false;
        this.port.postMessage({ t: 'ended' });
      }
      return false;
    }
    return true;
  }

  renderResample(outL, outR, n) {
    var L = this.chans[0], R = this.chans[1];
    var rr = this.readRate;
    for (var i = 0; i < n; i++) {
      if (!this.guard()) { outL[i] = 0; outR[i] = 0; continue; }
      outL[i] = this.sampleAt(L, this.readPos);
      outR[i] = this.sampleAt(R, this.readPos);
      this.readPos += rr;
    }
  }

  /* one WSOLA grain -> hop samples pushed into the fifo */
  genGrain() {
    if (!this.guard()) return false;

    var L = this.chans[0], R = this.chans[1];
    var G = this.grain, H = this.hop, w = this.win;
    var base = Math.round(this.readPos);
    var chosen = base;

    // find the shift that lines the new grain up with what the output
    // is already expecting. this is what stops the phasey / robotic
    // warble a naive overlap-add stretcher produces.
    if (this.hasTpl && Math.abs(this.readRate - 1) > 0.002) {
      var best = -Infinity, bi = 0;
      for (var d = -this.search; d <= this.search; d += 2) {
        var acc = 0, en = 1e-9;
        var p = base + d;
        for (var j = 0; j < H; j += 4) {
          var s = this.readAt(L, p + j) + this.readAt(R, p + j);
          acc += s * this.tpl[j];
          en += s * s;
        }
        var score = acc / Math.sqrt(en);
        if (score > best) { best = score; bi = d; }
      }
      chosen = base + bi;
    }

    for (var k = 0; k < G; k++) {
      var gw = w[k];
      this.accL[k] += gw * this.readAt(L, chosen + k);
      this.accR[k] += gw * this.readAt(R, chosen + k);
    }
    for (var m = 0; m < H; m += 4) {
      this.tpl[m] = this.readAt(L, chosen + H + m) + this.readAt(R, chosen + H + m);
    }
    this.hasTpl = true;

    var o = this.fifoLen;
    for (var q = 0; q < H; q++) {
      this.fifoL[o + q] = this.accL[q];
      this.fifoR[o + q] = this.accR[q];
    }
    this.fifoLen = o + H;

    this.accL.copyWithin(0, H);
    this.accR.copyWithin(0, H);
    this.accL.fill(0, H);
    this.accR.fill(0, H);

    this.readPos = base + H * this.readRate;
    return true;
  }

  renderStretch(outL, outR, n) {
    while (this.fifoLen < n) {
      if (this.mode === 'stream' &&
          this.writePos - this.readPos < this.grain + this.search + 16) break;
      if (!this.genGrain()) break;
    }
    var avail = Math.min(n, this.fifoLen);
    for (var i = 0; i < avail; i++) { outL[i] = this.fifoL[i]; outR[i] = this.fifoR[i]; }
    for (var j = avail; j < n; j++) { outL[j] = 0; outR[j] = 0; }
    if (avail > 0) {
      this.fifoL.copyWithin(0, avail, this.fifoLen);
      this.fifoR.copyWithin(0, avail, this.fifoLen);
      this.fifoLen -= avail;
    }
  }

  process(inputs, outputs) {
    var out = outputs[0];
    var outL = out[0], outR = out.length > 1 ? out[1] : out[0];
    var n = outL.length;

    if (this.mode === 'stream') {
      var inp = inputs[0];
      if (inp && inp.length) {
        var il = inp[0], ir = inp.length > 1 ? inp[1] : inp[0];
        for (var i = 0; i < n; i++) {
          var k = (this.writePos + i) % this.capacity;
          this.chans[0][k] = il[i];
          this.chans[1][k] = ir[i];
        }
        this.writePos += n;
        // never let the read head fall out of the ring
        var lag = this.writePos - this.readPos;
        if (lag > this.capacity - 8192) this.readPos = this.writePos - this.capacity + 8192;
        if (this.readPos > this.writePos) this.readPos = this.writePos;
      }
    }

    // glide the rate so slider moves do not click or zipper
    this.rate += (this.rateTarget - this.rate) * 0.08;
    if (Math.abs(this.rateTarget - this.rate) < 1e-5) this.rate = this.rateTarget;
    this.readRate = this.rate * (this.srcRate / sampleRate);

    var active = this.playing || this.gain > 0.0005;
    if (this.mode === 'idle' || !this.chans.length || !active) {
      outL.fill(0); outR.fill(0);
      this.gain = this.playing ? this.gain : 0;
      return true;
    }

    if (this.preservePitch) this.renderStretch(outL, outR, n);
    else this.renderResample(outL, outR, n);

    var target = this.playing ? 1 : 0;
    var c = this.gainCoef;
    for (var s = 0; s < n; s++) {
      this.gain = target + (this.gain - target) * c;
      outL[s] *= this.gain;
      outR[s] *= this.gain;
    }

    this.posCounter += n;
    if (this.posCounter >= 2048) {
      this.posCounter = 0;
      this.port.postMessage({
        t: 'pos',
        pos: this.readPos / this.srcRate,
        write: this.writePos / this.srcRate,
        rate: this.rate
      });
    }
    return true;
  }
}

registerProcessor('sr-player', SRPlayer);
`;

// (the reverb processor is appended to this module further down)


/* ------------------------------------------------------------------ *
 * 2. reverb — an FDN implemented as real DSP inside a worklet
 *
 * The obvious browser approach (a ConvolverNode fed decaying white
 * noise, or a tank built from DelayNodes and BiquadFilterNodes) has
 * three problems that all read as "robotic":
 *
 *   - a ConvolverNode IR is static, so every reflection lands on the
 *     same fixed comb and the tail rings like a metal pipe;
 *   - lowpass/highpass BiquadFilterNode Q is specified in DECIBELS, so
 *     the "gentle" Q values you would reach for are actually resonant,
 *     and any peak inside a feedback loop beats the decay and the tank
 *     runs away;
 *   - every feedback path through the node graph is quantised to 128
 *     samples and Chrome's cycle handling is not reproducible, so the
 *     same export rendered twice came out up to 6.6dB apart.
 *
 * Writing the tank as a sample loop removes all three. It is a
 * pre-delay, four series allpass diffusers per side, then an eight line
 * FDN with a lossless Householder mixing matrix, one pole damping in
 * every feedback path, and a slow independent LFO on each delay tap so
 * the comb never sits still — that last part is what takes the metallic
 * edge off the tail.
 * ------------------------------------------------------------------ */

const REVERB_SRC = `
'use strict';

const RV_AP_L = [0.0083, 0.0137, 0.0191, 0.0273];
const RV_AP_R = [0.0097, 0.0151, 0.0213, 0.0299];
const RV_AP_G = [0.72, 0.70, 0.62, 0.58];
// mutually prime-ish lengths so the modal density never collapses
const RV_FDN  = [0.0437, 0.0493, 0.0561, 0.0619, 0.0683, 0.0747, 0.0811, 0.0893];
const RV_MOD  = [0.071, 0.093, 0.117, 0.134, 0.152, 0.171, 0.193, 0.211];
const RV_MAXSCALE = 1.95;

class SRReverb extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      { name: 'predelay', defaultValue: 0.028, minValue: 0,    maxValue: 0.25, automationRate: 'k-rate' },
      { name: 'decay',    defaultValue: 3.2,   minValue: 0.1,  maxValue: 20,   automationRate: 'k-rate' },
      { name: 'damp',     defaultValue: 0.45,  minValue: 0,    maxValue: 1,    automationRate: 'k-rate' },
      { name: 'size',     defaultValue: 0.7,   minValue: 0.05, maxValue: 1,    automationRate: 'k-rate' },
      { name: 'width',    defaultValue: 1,     minValue: 0,    maxValue: 1.4,  automationRate: 'k-rate' }
    ];
  }

  constructor() {
    super();
    var sr = sampleRate;

    this.pdLen = Math.ceil(0.3 * sr) + 4;
    this.pdL = new Float32Array(this.pdLen);
    this.pdR = new Float32Array(this.pdLen);
    this.pdW = 0;
    this.pdCur = 0.028 * sr;

    this.apL = RV_AP_L.map((t, i) => this.mkAp(t * sr, RV_AP_G[i]));
    this.apR = RV_AP_R.map((t, i) => this.mkAp(t * sr, RV_AP_G[i]));

    this.N = RV_FDN.length;
    this.maxLen = Math.ceil(RV_FDN[this.N - 1] * RV_MAXSCALE * sr) + 8;
    this.lines = RV_FDN.map((base, i) => ({
      buf: new Float32Array(this.maxLen),
      w: 0,
      base: base * sr,
      len: base * 1.465 * sr,
      lenT: base * 1.465 * sr,
      g: 0.9,
      gT: 0.9,
      lp: 0,
      hp: 0,
      phase: i / RV_FDN.length,
      inc: RV_MOD[i] / sr,
      depth: (0.00055 + 0.00035 * (i % 3)) * sr
    }));

    this.inHp = [0, 0];
    this.hpCoef = 1 - Math.exp(-2 * Math.PI * 130 / sr);
    this.lineHpCoef = 1 - Math.exp(-2 * Math.PI * 190 / sr);
    this.lpCoef = 0.3;
    this.width = 1;
    // tuned so a fully wet render sits at about the same level as a dry
    // one, which is what makes the mix slider behave like a crossfade
    this.inTrim = 1.15;
  }

  mkAp(samples, g) {
    var n = Math.max(2, Math.round(samples));
    return { buf: new Float32Array(n), w: 0, g: g };
  }

  ap(a, x) {
    var d = a.buf[a.w];
    var v = x + a.g * d;
    a.buf[a.w] = v;
    a.w = a.w + 1 < a.buf.length ? a.w + 1 : 0;
    return d - a.g * v;
  }

  readLine(ln, delay) {
    var L = ln.buf.length;
    var pos = ln.w - delay;
    while (pos < 0) pos += L;
    while (pos >= L) pos -= L;
    var i0 = Math.floor(pos);
    var frac = pos - i0;
    var b = ln.buf;
    var im1 = i0 > 0 ? i0 - 1 : L - 1;
    var i1 = i0 + 1 < L ? i0 + 1 : 0;
    var i2 = i1 + 1 < L ? i1 + 1 : 0;
    return hermite(b[im1], b[i0], b[i1], b[i2], frac);
  }

  process(inputs, outputs, params) {
    var out = outputs[0];
    var outL = out[0], outR = out.length > 1 ? out[1] : out[0];
    var n = outL.length;

    var inp = inputs[0];
    var inL = inp && inp.length ? inp[0] : null;
    var inR = inp && inp.length > 1 ? inp[1] : inL;

    var sr = sampleRate;
    var predelay = params.predelay[0] * sr;
    var decay = params.decay[0];
    var damp = params.damp[0];
    var size = params.size[0];
    this.width = params.width[0];

    // per block target update, glided per sample so slider moves do not
    // click and a size change reads as a tape style glide
    var scale = 0.45 + size * 1.45;
    var rt60 = Math.max(0.15, decay * (1 + damp));
    for (var li = 0; li < this.N; li++) {
      var ln = this.lines[li];
      ln.lenT = Math.min(this.maxLen - 4, ln.base * scale);
      ln.gT = Math.min(0.995, Math.pow(10, (-3 * (ln.lenT / sr)) / rt60));
    }
    var fc = Math.max(700, 18000 * Math.pow(0.055, damp));
    var lpTarget = 1 - Math.exp(-2 * Math.PI * Math.min(fc, sr * 0.45) / sr);

    var glide = 1 - Math.exp(-1 / (0.05 * sr));
    var N = this.N, hh = -2 / N;
    var v = this.scratch || (this.scratch = new Float32Array(16));

    for (var s = 0; s < n; s++) {
      var xl = inL ? inL[s] : 0;
      var xr = inR ? inR[s] : 0;

      // keep rumble out of the tail
      this.inHp[0] += this.hpCoef * (xl - this.inHp[0]);
      this.inHp[1] += this.hpCoef * (xr - this.inHp[1]);
      xl -= this.inHp[0];
      xr -= this.inHp[1];

      // pre-delay
      this.pdL[this.pdW] = xl;
      this.pdR[this.pdW] = xr;
      this.pdCur += (predelay - this.pdCur) * glide;
      var pp = this.pdW - this.pdCur;
      while (pp < 0) pp += this.pdLen;
      var p0 = Math.floor(pp), pf = pp - p0;
      var p1 = p0 + 1 < this.pdLen ? p0 + 1 : 0;
      var dl = this.pdL[p0] + (this.pdL[p1] - this.pdL[p0]) * pf;
      var dr = this.pdR[p0] + (this.pdR[p1] - this.pdR[p0]) * pf;
      this.pdW = this.pdW + 1 < this.pdLen ? this.pdW + 1 : 0;

      // input diffusion, decorrelated per side
      for (var a = 0; a < this.apL.length; a++) {
        dl = this.ap(this.apL[a], dl);
        dr = this.ap(this.apR[a], dr);
      }
      dl *= this.inTrim;
      dr *= this.inTrim;

      this.lpCoef += (lpTarget - this.lpCoef) * glide;

      var sum = 0;
      for (var i = 0; i < N; i++) {
        var l = this.lines[i];
        l.len += (l.lenT - l.len) * glide;
        l.g += (l.gT - l.g) * glide;
        l.phase += l.inc;
        if (l.phase >= 1) l.phase -= 1;
        var d = l.len + l.depth * Math.sin(2 * Math.PI * l.phase);
        if (d < 2) d = 2;
        var y = this.readLine(l, d);
        // one pole damping: cannot resonate, unlike a biquad
        l.lp += this.lpCoef * (y - l.lp);
        y = l.lp;
        l.hp += this.lineHpCoef * (y - l.hp);
        y -= l.hp;
        v[i] = y;
        sum += l.g * y;
      }

      var h = hh * sum;
      for (var k = 0; k < N; k++) {
        var lk = this.lines[k];
        // Householder: I - (2/N)*ones, lossless, so the only loss in the
        // loop is the decay gain and the damping
        lk.buf[lk.w] = (k % 2 === 0 ? dl : dr) + lk.g * v[k] + h;
        lk.w = lk.w + 1 < lk.buf.length ? lk.w + 1 : 0;
      }

      var wl = 0.5 * (v[0] - v[1] + v[2] - v[3]);
      var wr = 0.5 * (v[4] + v[5] - v[6] - v[7]);
      var mid = (wl + wr) * 0.5;
      var side = (wl - wr) * 0.5 * this.width;
      outL[s] = mid + side;
      outR[s] = mid - side;
    }
    return true;
  }
}

registerProcessor('sr-reverb', SRReverb);
`;

/* Fallback for browsers with no AudioWorklet. A generated impulse
 * response cannot modulate, so the tail is flatter than the real tank,
 * but it beats having no reverb at all. */
class ConvolverReverb {
  constructor(ctx) {
    this.ctx = ctx;
    this.decay = 3.2;
    this.damp = 0.45;
    this._pending = 0;
    this.input = ctx.createGain();
    this.output = ctx.createGain();
    this.pre = ctx.createDelay(0.5);
    this.pre.delayTime.value = 0.028;
    this.conv = ctx.createConvolver();
    this.conv.normalize = true;
    this.input.connect(this.pre).connect(this.conv).connect(this.output);
    this.build();
  }

  build() {
    const ctx = this.ctx, sr = ctx.sampleRate;
    const len = Math.max(1, Math.round(Math.min(10, this.decay * (1 + this.damp)) * sr));
    const ir = ctx.createBuffer(2, len, sr);
    const fc = Math.max(700, 18000 * Math.pow(0.055, this.damp));
    const a = 1 - Math.exp(-2 * Math.PI * Math.min(fc, sr * 0.45) / sr);
    for (let c = 0; c < 2; c++) {
      const d = ir.getChannelData(c);
      let lp = 0, seed = c ? 8191 : 4093;
      const rnd = () => { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 2147483648 - 1; };
      for (let i = 0; i < len; i++) {
        lp += a * (rnd() - lp);
        d[i] = lp * Math.exp(-6.9 * i / len);
      }
    }
    this.conv.buffer = ir;
  }

  rebuild() {
    clearTimeout(this._pending);
    this._pending = setTimeout(() => this.build(), 180);
  }

  setDecay(sec)       { this.decay = sec; this.rebuild(); }
  setDamping(norm)    { this.damp = norm; this.rebuild(); }
  setSize()           { /* baked into the impulse response */ }
  setWidth()          { /* not available on a convolver */ }
  setPreDelay(ms, tc) { setParam(this.ctx, this.pre.delayTime, ms / 1000, tc); }
  get tailSeconds()   { return this.decay * (1 + this.damp) + 0.4; }
}

/* main thread handle for the reverb worklet */
class Reverb {
  constructor(ctx) {
    this.ctx = ctx;
    this.decay = 3.2;
    this.damp = 0.45;
    this.node = new AudioWorkletNode(ctx, 'sr-reverb', {
      numberOfInputs: 1,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      channelCount: 2,
      channelCountMode: 'explicit',
      channelInterpretation: 'speakers'
    });
    this.input = this.node;
    this.output = this.node;
  }

  param(name) { return this.node.parameters.get(name); }

  setDecay(sec, tc)   { this.decay = sec; setParam(this.ctx, this.param('decay'), sec, tc); }
  setSize(norm, tc)   { setParam(this.ctx, this.param('size'), norm, tc ? Math.max(tc, 0.2) : 0); }
  setPreDelay(ms, tc) { setParam(this.ctx, this.param('predelay'), ms / 1000, tc); }
  setWidth(norm, tc)  { setParam(this.ctx, this.param('width'), norm, tc); }
  setDamping(norm, tc) { this.damp = norm; setParam(this.ctx, this.param('damp'), norm, tc); }

  get tailSeconds() { return this.decay * (1 + this.damp) + 0.4; }
}

/* setTargetAtTime everywhere: direct .value writes are what make a
 * slider sweep sound like a stepper motor */
function setParam(ctx, param, value, tc) {
  if (!isFinite(value)) return;
  if (!tc) {
    param.cancelScheduledValues(ctx.currentTime);
    param.setValueAtTime(value, ctx.currentTime);
  } else {
    param.setTargetAtTime(value, ctx.currentTime, tc);
  }
}

const dbToGain = db => Math.pow(10, db / 20);


/* ------------------------------------------------------------------ *
 * 3. the rest of the signal chain
 * ------------------------------------------------------------------ */

const WORKLET_URL = URL.createObjectURL(
  new Blob([WORKLET_SRC, REVERB_SRC], { type: 'text/javascript' }));

// For lowpass/highpass BiquadFilterNodes the Web Audio spec reads Q in
// DECIBELS, not as a linear Q, so Q = 0 is already slightly resonant.
// Butterworth, which is what a crossover wants, is -3.01dB.
const Q_BUTTER = -3.0103;

const DEFAULTS = {
  rate: 1,
  preservePitch: false,
  mix: 0,
  decay: 3.2,
  preDelay: 28,
  damp: 0.45,
  size: 0.7,
  width: 1,
  bass: 0,
  air: 0,
  eightD: false,
  orbit: 10,
  pan: 0,
  limiter: true,
  volume: 1
};

/* Safety clipper: exactly unity below the knee, smoothly saturating
 * above it. tanh(d*x)/tanh(d) looks like a soft clipper but has a
 * small signal slope of d/tanh(d) — at d=1.6 that is +4.8dB of gain and
 * harmonic distortion applied to the whole mix, dry path included. */
function softClipCurve(n = 4096, knee = 0.7) {
  const c = new Float32Array(n);
  const span = 1 - knee;
  for (let i = 0; i < n; i++) {
    const x = (i / (n - 1)) * 2 - 1;
    const a = Math.abs(x);
    c[i] = a <= knee ? x : Math.sign(x) * (knee + span * Math.tanh((a - knee) / span));
  }
  return c;
}

class Chain {
  constructor(ctx) {
    this.ctx = ctx;
    this.input = ctx.createGain();

    // ---- tone ------------------------------------------------------
    this.trim = ctx.createGain();

    this.lowShelf = ctx.createBiquadFilter();
    this.lowShelf.type = 'lowshelf';
    this.lowShelf.frequency.value = 120;

    this.subPeak = ctx.createBiquadFilter();
    this.subPeak.type = 'peaking';
    this.subPeak.frequency.value = 62;
    this.subPeak.Q.value = 0.9;

    // scooping a little mud back out is what keeps a big bass boost
    // from swallowing the vocal
    this.mudCut = ctx.createBiquadFilter();
    this.mudCut.type = 'peaking';
    this.mudCut.frequency.value = 330;
    this.mudCut.Q.value = 0.8;

    this.airShelf = ctx.createBiquadFilter();
    this.airShelf.type = 'highshelf';
    this.airShelf.frequency.value = 7200;

    this.preFx = ctx.createGain();

    this.input
      .connect(this.trim)
      .connect(this.lowShelf)
      .connect(this.subPeak)
      .connect(this.mudCut)
      .connect(this.airShelf)
      .connect(this.preFx);

    // ---- dry / wet -------------------------------------------------
    try {
      this.reverb = new Reverb(ctx);
    } catch (err) {
      this.reverb = new ConvolverReverb(ctx);
    }
    this.dryGain = ctx.createGain();
    this.wetGain = ctx.createGain();
    this.dryGain.gain.value = 1;
    this.wetGain.gain.value = 0;

    this.spatialIn = ctx.createGain();
    this.preFx.connect(this.dryGain).connect(this.spatialIn);
    this.preFx.connect(this.reverb.input);
    this.reverb.output.connect(this.wetGain).connect(this.spatialIn);

    // ---- spatialiser ------------------------------------------------
    this.postSpatial = ctx.createGain();

    this.stereoBus = ctx.createGain();
    this.stereoBus.gain.value = 1;
    this.stereoPan = ctx.createStereoPanner();
    this.spatialIn.connect(this.stereoBus).connect(this.stereoPan).connect(this.postSpatial);

    this.orbitBus = ctx.createGain();
    this.orbitBus.gain.value = 0;
    this.spatialIn.connect(this.orbitBus);

    // Linkwitz-Riley style split: the orbit only moves the top end.
    // Spinning the bass around the head is what makes most 8D renders
    // sound thin and seasick.
    const lp1 = ctx.createBiquadFilter(), lp2 = ctx.createBiquadFilter();
    const hp1 = ctx.createBiquadFilter(), hp2 = ctx.createBiquadFilter();
    for (const f of [lp1, lp2]) { f.type = 'lowpass';  f.frequency.value = 200; f.Q.value = Q_BUTTER; }
    for (const f of [hp1, hp2]) { f.type = 'highpass'; f.frequency.value = 200; f.Q.value = Q_BUTTER; }
    this.orbitBus.connect(lp1).connect(lp2).connect(this.postSpatial);
    this.orbitBus.connect(hp1).connect(hp2);

    this.panner = ctx.createPanner();
    this.hasPositionParams = !!(this.panner.positionX && this.panner.positionX.setValueAtTime);
    if (this.hasPositionParams) {
      this.panner.panningModel = 'HRTF';
      this.panner.distanceModel = 'inverse';
      // matching refDistance to the orbit radius keeps the distance gain
      // flat all the way round, so 8D does not quieten the track
      this.panner.refDistance = 1.7;
      this.panner.maxDistance = 12;
      this.panner.rolloffFactor = 0.55;
      this.panner.positionY.value = 0;
      this.panner.positionZ.value = 1.6;
      hp2.connect(this.panner).connect(this.postSpatial);

      const sinW = ctx.createPeriodicWave(new Float32Array([0, 0]), new Float32Array([0, 1]),
                                          { disableNormalization: true });
      const cosW = ctx.createPeriodicWave(new Float32Array([0, 1]), new Float32Array([0, 0]),
                                          { disableNormalization: true });
      this.oscX = ctx.createOscillator();
      this.oscZ = ctx.createOscillator();
      this.oscX.setPeriodicWave(sinW);
      this.oscZ.setPeriodicWave(cosW);
      this.radX = ctx.createGain(); this.radX.gain.value = 1.7;
      this.radZ = ctx.createGain(); this.radZ.gain.value = 1.7;
      this.oscX.connect(this.radX).connect(this.panner.positionX);
      this.oscZ.connect(this.radZ).connect(this.panner.positionZ);
      const t0 = ctx.currentTime;
      this.oscX.frequency.value = 0.1;
      this.oscZ.frequency.value = 0.1;
      this.oscX.start(t0);
      this.oscZ.start(t0);

      const L = ctx.listener;
      if (L.positionX) {
        L.positionX.value = 0; L.positionY.value = 0; L.positionZ.value = 0;
        L.forwardX.value = 0; L.forwardY.value = 0; L.forwardZ.value = -1;
        L.upX.value = 0; L.upY.value = 1; L.upZ.value = 0;
      }
    } else {
      // older safari: no connectable position params, sweep a stereo panner
      this.orbitPan = ctx.createStereoPanner();
      hp2.connect(this.orbitPan).connect(this.postSpatial);
      this.oscX = ctx.createOscillator();
      this.oscX.type = 'sine';
      this.oscX.frequency.value = 0.1;
      this.radX = ctx.createGain();
      this.radX.gain.value = 0.92;
      this.oscX.connect(this.radX).connect(this.orbitPan.pan);
      this.oscX.start(ctx.currentTime);
    }

    // ---- master ----------------------------------------------------
    this.limiter = ctx.createDynamicsCompressor();
    this.limiter.threshold.value = -1.2;
    this.limiter.knee.value = 0;
    this.limiter.ratio.value = 20;
    this.limiter.attack.value = 0.002;
    this.limiter.release.value = 0.18;

    this.limitOn  = ctx.createGain(); this.limitOn.gain.value = 1;
    this.limitOff = ctx.createGain(); this.limitOff.gain.value = 0;

    this.shaper = ctx.createWaveShaper();
    this.shaper.curve = softClipCurve();
    this.shaper.oversample = '4x';

    this.master = ctx.createGain();
    this.analyser = ctx.createAnalyser();
    this.analyser.fftSize = 2048;
    this.analyser.smoothingTimeConstant = 0.6;

    this.postSpatial.connect(this.limiter).connect(this.limitOn).connect(this.shaper);
    this.postSpatial.connect(this.limitOff).connect(this.shaper);
    this.shaper.connect(this.master);
    this.master.connect(this.analyser);
    this.output = this.master;

    this.params = Object.assign({}, DEFAULTS);
    this.applyAll(this.params, 0);
  }

  applyAll(p, tc) {
    this.setTone(p, tc);
    this.setMix(p.mix, tc);
    this.reverb.setDecay(p.decay, tc);
    this.reverb.setPreDelay(p.preDelay, tc);
    this.reverb.setDamping(p.damp, tc);
    this.reverb.setSize(p.size, tc);
    this.reverb.setWidth(p.width, tc);
    this.setSpatial(p, tc);
    this.setLimiter(p.limiter, tc);
    setParam(this.ctx, this.master.gain, Math.pow(p.volume, 1.7), tc);
    this.params = Object.assign({}, p);
  }

  setTone(p, tc) {
    const ctx = this.ctx;
    setParam(ctx, this.lowShelf.gain, p.bass * 0.70, tc);
    setParam(ctx, this.subPeak.gain,  p.bass * 0.45, tc);
    setParam(ctx, this.mudCut.gain,  -p.bass * 0.22, tc);
    setParam(ctx, this.airShelf.gain, p.air, tc);
    // pull the input down as the bass goes up so the limiter is not
    // doing all the work
    setParam(ctx, this.trim.gain, dbToGain(-p.bass * 0.45 - Math.max(0, p.air) * 0.3), tc);
  }

  setMix(mix, tc) {
    const m = Math.max(0, Math.min(1, mix));
    // equal power: a linear crossfade dips ~3dB in the middle, which is
    // exactly where most people leave the slider
    setParam(this.ctx, this.dryGain.gain, Math.cos(m * Math.PI / 2), tc);
    setParam(this.ctx, this.wetGain.gain, Math.sin(m * Math.PI / 2) * 0.92, tc);
  }

  setSpatial(p, tc) {
    const ctx = this.ctx;
    const f = 1 / Math.max(0.5, p.orbit);
    if (this.oscX) setParam(ctx, this.oscX.frequency, f, tc ? 0.05 : 0);
    if (this.oscZ) setParam(ctx, this.oscZ.frequency, f, tc ? 0.05 : 0);
    setParam(ctx, this.orbitBus.gain,  p.eightD ? 1 : 0, tc ? 0.06 : 0);
    setParam(ctx, this.stereoBus.gain, p.eightD ? 0 : 1, tc ? 0.06 : 0);
    setParam(ctx, this.stereoPan.pan, Math.max(-1, Math.min(1, p.pan)), tc);
  }

  setLimiter(on, tc) {
    setParam(this.ctx, this.limitOn.gain,  on ? 1 : 0, tc ? 0.03 : 0);
    setParam(this.ctx, this.limitOff.gain, on ? 0 : 1, tc ? 0.03 : 0);
  }

  get tailSeconds() { return this.params.mix > 0.001 ? this.reverb.tailSeconds : 0.25; }
}


/* ------------------------------------------------------------------ *
 * 4. engine — owns the context, the source and the parameter state
 * ------------------------------------------------------------------ */

Object.assign(DEFAULTS, { loop: false, loopStart: 0, loopEnd: 0 });

/* used only if AudioWorklet is missing (very old safari): plain
 * playbackRate, no time-stretch, no live streaming */
class BufferFallback {
  constructor(ctx, dest) {
    this.ctx = ctx; this.dest = dest;
    this.buf = null; this.src = null;
    this.rate = 1; this.offset = 0; this.startedAt = 0; this.playing = false;
    this.onEnded = null;
  }
  load(b) { this.buf = b; this.offset = 0; }
  currentTime() {
    if (!this.playing) return this.offset;
    return this.offset + (this.ctx.currentTime - this.startedAt) * this.rate;
  }
  setRate(r) {
    this.rate = r;
    if (this.src) setParam(this.ctx, this.src.playbackRate, r, 0.03);
  }
  play(at) {
    if (!this.buf) return;
    this.stopSource();
    this.offset = Math.max(0, Math.min(this.buf.duration - 0.01, at));
    this.src = this.ctx.createBufferSource();
    this.src.buffer = this.buf;
    this.src.playbackRate.value = this.rate;
    this.src.connect(this.dest);
    this.src.onended = () => { if (this.playing && this.onEnded) this.onEnded(); };
    this.startedAt = this.ctx.currentTime;
    this.playing = true;
    this.src.start(0, this.offset);
  }
  pause() { this.offset = this.currentTime(); this.playing = false; this.stopSource(); }
  seek(t) { const was = this.playing; this.offset = t; if (was) this.play(t); }
  stopSource() {
    if (!this.src) return;
    try { this.src.onended = null; this.src.stop(); } catch (e) { /* already stopped */ }
    this.src.disconnect();
    this.src = null;
  }
}

class Engine {
  constructor() {
    this.ctx = null;
    this.chain = null;
    this.node = null;
    this.fallback = null;
    this.mode = 'idle';
    this.buffer = null;
    this.duration = 0;
    this.position = 0;
    this.writeHead = 0;
    this.playing = false;
    this.workletOK = false;
    this.streamSeconds = 420;   // the capture ring is also the recording
    this.streamCapacity = 0;
    this._dumpWaiters = [];
    this.params = Object.assign({}, DEFAULTS);
    this.onPos = null;
    this.onEnded = null;
    this._stream = null;
    this._streamSrc = null;
    this._raf = 0;
    this._starting = null;
  }

  async ensure() {
    if (this.ctx) {
      if (this.ctx.state === 'suspended') await this.ctx.resume();
      return;
    }
    // a click on play while a file is still loading would otherwise build
    // a second context and leave the first one orphaned
    if (this._starting) {
      await this._starting;
      if (this.ctx && this.ctx.state === 'suspended') await this.ctx.resume();
      return;
    }
    this._starting = this._start();
    try { await this._starting; } finally { this._starting = null; }
  }

  async _start() {
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) throw new Error('this browser has no web audio support');
    this.ctx = new AC({ latencyHint: 'playback' });
    try {
      await this.ctx.audioWorklet.addModule(WORKLET_URL);
      this.workletOK = true;
    } catch (err) {
      console.warn('AudioWorklet unavailable, falling back', err);
      this.workletOK = false;
    }
    this.chain = new Chain(this.ctx);
    this.chain.output.connect(this.ctx.destination);
    this.chain.applyAll(this.params, 0);
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    this._tick();
  }

  _tick() {
    const step = () => {
      if (this.fallback && this.playing) {
        this.position = this.fallback.currentTime();
        if (this.position >= this.duration) { this.playing = false; if (this.onEnded) this.onEnded(); }
      }
      if (this.onPos) this.onPos(this.position, this.writeHead);
      this._raf = requestAnimationFrame(step);
    };
    cancelAnimationFrame(this._raf);
    this._raf = requestAnimationFrame(step);
  }

  _makeNode(inputs, init) {
    this._dropNode();
    this.node = new AudioWorkletNode(this.ctx, 'sr-player', {
      numberOfInputs: inputs,
      numberOfOutputs: 1,
      outputChannelCount: [2],
      processorOptions: init || {}
    });
    this.node.port.onmessage = (e) => {
      const m = e.data;
      if (m.t === 'pos') { this.position = m.pos; this.writeHead = m.write; }
      else if (m.t === 'ended') { this.playing = false; if (this.onEnded) this.onEnded(); }
      else if (m.t === 'streamReady') { this.streamCapacity = m.seconds; }
      else if (m.t === 'dump') {
        const w = this._dumpWaiters.shift();
        if (w) w(m);
      }
    };
    this.node.connect(this.chain.input);
  }

  _dropNode() {
    if (!this.node) return;
    this.node.port.onmessage = null;
    try { this.node.port.postMessage({ t: 'clear' }); } catch (e) { /* closing */ }
    try { this.node.disconnect(); } catch (e) { /* already gone */ }
    this.node = null;
  }

  releaseSource() {
    this.playing = false;
    this._dropNode();
    if (this.fallback) { this.fallback.pause(); this.fallback = null; }
    if (this._streamSrc) { try { this._streamSrc.disconnect(); } catch (e) { /* gone */ } this._streamSrc = null; }
    if (this._stream) { this._stream.getTracks().forEach(t => t.stop()); this._stream = null; }
  }

  async loadBuffer(buffer) {
    await this.ensure();
    this.releaseSource();
    this.buffer = buffer;
    this.duration = buffer.duration;
    this.mode = 'file';
    this.position = 0;
    this.writeHead = buffer.duration;

    if (this.workletOK) {
      const chans = [];
      const n = Math.min(2, buffer.numberOfChannels);
      for (let c = 0; c < n; c++) chans.push(buffer.getChannelData(c));
      this._makeNode(0, {
        t: 'file',
        chans,
        frames: buffer.length,
        srcRate: buffer.sampleRate,
        rate: this.params.rate,
        preservePitch: this.params.preservePitch
      });
      this.pushParams(true);
    } else {
      this.fallback = new BufferFallback(this.ctx, this.chain.input);
      this.fallback.load(buffer);
      this.fallback.setRate(this.params.rate);
      this.fallback.onEnded = () => { this.playing = false; if (this.onEnded) this.onEnded(); };
    }
  }

  async loadStream(stream) {
    await this.ensure();
    this.releaseSource();
    this.buffer = null;
    this.mode = 'stream';
    this.duration = 0;
    this.position = 0;
    this._stream = stream;
    this._streamSrc = this.ctx.createMediaStreamSource(stream);

    if (!this.workletOK) {
      this._streamSrc.connect(this.chain.input);
      this.playing = true;
      return;
    }
    this._makeNode(1, {
      t: 'stream',
      seconds: this.streamSeconds,
      rate: this.params.rate,
      preservePitch: this.params.preservePitch
    });
    this._streamSrc.connect(this.node);
    this.pushParams(true);
    this.play();
  }

  play() {
    if (this.mode === 'idle') return;
    if (this.ctx && this.ctx.state === 'suspended') this.ctx.resume();
    this.playing = true;
    if (this.node) this.node.port.postMessage({ t: 'play' });
    else if (this.fallback) this.fallback.play(this.position);
  }

  pause() {
    this.playing = false;
    if (this.node) this.node.port.postMessage({ t: 'pause' });
    else if (this.fallback) this.fallback.pause();
  }

  seek(time) {
    const t = Math.max(0, Math.min(this.duration || time, time));
    this.position = t;
    if (this.node) this.node.port.postMessage({ t: 'seek', time: t });
    else if (this.fallback) this.fallback.seek(t);
  }

  jumpLive() { if (this.node) this.node.port.postMessage({ t: 'live' }); }

  /* pull the captured audio back out of the worklet's ring */
  dumpRecording() {
    return new Promise((resolve, reject) => {
      if (!this.node || this.mode !== 'stream') return reject(new Error('nothing is being captured'));
      const timer = setTimeout(() => reject(new Error('capture read timed out')), 15000);
      this._dumpWaiters.push(m => { clearTimeout(timer); resolve(m); });
      this.node.port.postMessage({ t: 'dump' });
    });
  }

  setParams(patch, smooth = true) {
    Object.assign(this.params, patch);
    if (this.chain) this.chain.applyAll(this.params, smooth ? 0.03 : 0);
    if (this.fallback && patch.rate !== undefined) this.fallback.setRate(patch.rate);
    this.pushParams(false);
  }

  pushParams(snap) {
    if (!this.node) return;
    const p = this.params;
    this.node.port.postMessage({
      t: 'params',
      rate: p.rate,
      preservePitch: p.preservePitch,
      loop: p.loop,
      loopStart: p.loopStart,
      loopEnd: p.loopEnd
    });
    if (snap) this.node.port.postMessage({ t: 'snapRate', rate: p.rate });
  }
}


/* ------------------------------------------------------------------ *
 * 5. offline render + encoders
 * ------------------------------------------------------------------ */

async function renderOffline(buffer, params) {
  const sr = Math.max(44100, buffer.sampleRate);
  // must match Reverb.tailSeconds or long, damped tails get cut off
  const tail = params.mix > 0.001
    ? params.decay * (1 + params.damp) + params.preDelay / 1000 + 0.4
    : 0.25;
  const total = buffer.duration / params.rate + tail + 0.15;
  const OC = window.OfflineAudioContext || window.webkitOfflineAudioContext;
  const off = new OC(2, Math.ceil(total * sr), sr);
  if (!off.audioWorklet) throw new Error('offline rendering needs AudioWorklet support');
  await off.audioWorklet.addModule(WORKLET_URL);

  const chans = [];
  const n = Math.min(2, buffer.numberOfChannels);
  for (let c = 0; c < n; c++) chans.push(buffer.getChannelData(c));

  const node = new AudioWorkletNode(off, 'sr-player', {
    numberOfInputs: 0,
    numberOfOutputs: 1,
    outputChannelCount: [2],
    processorOptions: {
      t: 'file',
      chans,
      frames: buffer.length,
      srcRate: buffer.sampleRate,
      rate: params.rate,
      preservePitch: params.preservePitch,
      autoplay: true
    }
  });
  const chain = new Chain(off);
  node.connect(chain.input);
  chain.output.connect(off.destination);
  chain.applyAll(Object.assign({}, params, { loop: false }), 0);

  const rendered = await off.startRendering();
  return trimAndNormalize(rendered);
}

/* scale back only if we would clip; never pump quiet mixes up */
function trimAndNormalize(buf) {
  let peak = 0;
  for (let c = 0; c < buf.numberOfChannels; c++) {
    const d = buf.getChannelData(c);
    for (let i = 0; i < d.length; i++) { const a = Math.abs(d[i]); if (a > peak) peak = a; }
  }
  if (peak > 0.999) {
    const g = 0.995 / peak;
    for (let c = 0; c < buf.numberOfChannels; c++) {
      const d = buf.getChannelData(c);
      for (let i = 0; i < d.length; i++) d[i] *= g;
    }
  }
  return buf;
}

function encodeWav(buf) {
  const chCount = Math.min(2, buf.numberOfChannels);
  const frames = buf.length;
  const bytes = frames * chCount * 2;
  const ab = new ArrayBuffer(44 + bytes);
  const view = new DataView(ab);
  const str = (o, s) => { for (let i = 0; i < s.length; i++) view.setUint8(o + i, s.charCodeAt(i)); };

  str(0, 'RIFF');  view.setUint32(4, 36 + bytes, true);
  str(8, 'WAVE');  str(12, 'fmt ');
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, chCount, true);
  view.setUint32(24, buf.sampleRate, true);
  view.setUint32(28, buf.sampleRate * chCount * 2, true);
  view.setUint16(32, chCount * 2, true);
  view.setUint16(34, 16, true);
  str(36, 'data'); view.setUint32(40, bytes, true);

  const chans = [];
  for (let c = 0; c < chCount; c++) chans.push(buf.getChannelData(c));

  let o = 44;
  for (let i = 0; i < frames; i++) {
    for (let c = 0; c < chCount; c++) {
      // triangular dither at the 16 bit LSB keeps long reverb tails from
      // granulating as they fade out
      const dither = (Math.random() + Math.random() - 1) / 32768;
      let s = chans[c][i] + dither;
      s = s < -1 ? -1 : s > 1 ? 1 : s;
      view.setInt16(o, s < 0 ? s * 0x8000 : s * 0x7fff, true);
      o += 2;
    }
  }
  return new Blob([ab], { type: 'audio/wav' });
}

let lamePromise = null;
function loadLame() {
  if (window.lamejs) return Promise.resolve(window.lamejs);
  if (!lamePromise) {
    lamePromise = import('https://cdn.jsdelivr.net/npm/@breezystack/lamejs@1.2.7/+esm')
      .then(m => { window.lamejs = m.default || m; return window.lamejs; });
  }
  return lamePromise;
}

async function encodeMp3(buf, kbps, onProgress) {
  const lame = await loadLame();
  const Enc = lame.Mp3Encoder || (lame.default && lame.default.Mp3Encoder);
  const chCount = Math.min(2, buf.numberOfChannels);
  const enc = new Enc(chCount, buf.sampleRate, kbps);

  const l = buf.getChannelData(0);
  const r = chCount > 1 ? buf.getChannelData(1) : l;
  const block = 1152;
  const out = [];
  const li = new Int16Array(block);
  const ri = new Int16Array(block);

  for (let i = 0; i < buf.length; i += block) {
    const n = Math.min(block, buf.length - i);
    for (let j = 0; j < n; j++) {
      let a = l[i + j]; a = a < -1 ? -1 : a > 1 ? 1 : a;
      let b = r[i + j]; b = b < -1 ? -1 : b > 1 ? 1 : b;
      li[j] = a < 0 ? a * 0x8000 : a * 0x7fff;
      ri[j] = b < 0 ? b * 0x8000 : b * 0x7fff;
    }
    const chunk = chCount > 1
      ? enc.encodeBuffer(li.subarray(0, n), ri.subarray(0, n))
      : enc.encodeBuffer(li.subarray(0, n));
    if (chunk.length) out.push(chunk);
    if (onProgress && (i / block) % 400 === 0) {
      onProgress(i / buf.length);
      await new Promise(res => setTimeout(res, 0));
    }
  }
  const last = enc.flush();
  if (last.length) out.push(last);
  return new Blob(out, { type: 'audio/mpeg' });
}


/* ------------------------------------------------------------------ *
 * 6. local library (IndexedDB — survives bigger files than localStorage)
 * ------------------------------------------------------------------ */

const DB_NAME = 'slowedrvb';
const STORE = 'tracks';
const MAX_SAVED = 24;
const MAX_SAVE_BYTES = 80 * 1024 * 1024;

function openDb() {
  return new Promise((res, rej) => {
    if (!window.indexedDB) return rej(new Error('no indexeddb'));
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) db.createObjectStore(STORE, { keyPath: 'id' });
    };
    req.onsuccess = () => res(req.result);
    req.onerror = () => rej(req.error);
  });
}

function tx(db, mode, fn) {
  return new Promise((res, rej) => {
    const t = db.transaction(STORE, mode);
    const req = fn(t.objectStore(STORE));
    t.oncomplete = () => res(req && req.result);
    t.onerror = () => rej(t.error);
    t.onabort = () => rej(t.error);
  });
}

const library = {
  async list() {
    try {
      const db = await openDb();
      const all = await tx(db, 'readonly', s => s.getAll());
      return (all || []).sort((a, b) => b.addedAt - a.addedAt);
    } catch (e) { return []; }
  },
  async save(rec) {
    try {
      if (rec.blob.size > MAX_SAVE_BYTES) return;
      const db = await openDb();
      await tx(db, 'readwrite', s => s.put(rec));
      const all = await this.list();
      if (all.length > MAX_SAVED) {
        const drop = all.slice(MAX_SAVED);
        const db2 = await openDb();
        await tx(db2, 'readwrite', s => { drop.forEach(d => s.delete(d.id)); });
      }
    } catch (e) { console.warn('save failed', e); }
  },
  async get(id) {
    try {
      const db = await openDb();
      return await tx(db, 'readonly', s => s.get(id));
    } catch (e) { return null; }
  },
  async remove(id) {
    try {
      const db = await openDb();
      await tx(db, 'readwrite', s => s.delete(id));
    } catch (e) { /* ignore */ }
  },
  async clear() {
    try {
      const db = await openDb();
      await tx(db, 'readwrite', s => s.clear());
    } catch (e) { /* ignore */ }
  }
};


/* ------------------------------------------------------------------ *
 * 7. waveform
 * ------------------------------------------------------------------ */

function computePeaks(buffer, buckets) {
  const chCount = Math.min(2, buffer.numberOfChannels);
  const chans = [];
  for (let c = 0; c < chCount; c++) chans.push(buffer.getChannelData(c));
  const len = buffer.length;
  const per = len / buckets;
  const mins = new Float32Array(buckets);
  const maxs = new Float32Array(buckets);

  for (let b = 0; b < buckets; b++) {
    const s = Math.floor(b * per);
    const e = Math.min(len, Math.floor((b + 1) * per));
    let mn = 0, mx = 0;
    for (let i = s; i < e; i++) {
      for (let c = 0; c < chCount; c++) {
        const v = chans[c][i];
        if (v < mn) mn = v;
        if (v > mx) mx = v;
      }
    }
    mins[b] = mn; maxs[b] = mx;
  }
  return { mins, maxs };
}

class Waveform {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.peaks = null;
    this.live = null;        // rolling peaks for tab streaming
    this.liveHead = 0;
    this.dpr = 1;
    this.buckets = 0;
    this.resize();
    window.addEventListener('resize', () => { this.resize(); this.rebuild(); });
  }

  resize() {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const w = Math.max(240, this.canvas.clientWidth || 480);
    const h = this.canvas.clientHeight || 84;
    this.dpr = dpr;
    this.canvas.width = Math.round(w * dpr);
    this.canvas.height = Math.round(h * dpr);
    this.buckets = Math.round(w);
  }

  setBuffer(buffer) {
    this.source = buffer;
    this.live = null;
    this.rebuild();
  }

  setLive() {
    this.source = null;
    this.peaks = null;
    this.live = new Float32Array(this.buckets);
    this.liveHead = 0;
  }

  rebuild() {
    if (this.source) this.peaks = computePeaks(this.source, this.buckets);
    else if (this.live && this.live.length !== this.buckets) {
      this.live = new Float32Array(this.buckets);
      this.liveHead = 0;
    }
  }

  pushLive(level) {
    if (!this.live) return;
    this.live[this.liveHead] = level;
    this.liveHead = (this.liveHead + 1) % this.live.length;
  }

  draw(opts) {
    const c = this.ctx;
    const W = this.canvas.width, H = this.canvas.height;
    const mid = H / 2;
    const css = getComputedStyle(document.documentElement);
    const colSub = css.getPropertyValue('--sub').trim() || '#646669';
    const colMain = css.getPropertyValue('--main').trim() || '#e2b714';
    const colBg = css.getPropertyValue('--bg').trim() || '#323437';

    c.clearRect(0, 0, W, H);
    c.fillStyle = colBg;
    c.fillRect(0, 0, W, H);

    const prog = opts.duration > 0 ? Math.max(0, Math.min(1, opts.position / opts.duration)) : 0;

    // loop shading
    if (opts.loopEnd > opts.loopStart && opts.duration > 0) {
      const a = (opts.loopStart / opts.duration) * W;
      const b = (opts.loopEnd / opts.duration) * W;
      c.fillStyle = 'rgba(226,183,20,0.12)';
      c.fillRect(a, 0, b - a, H);
      c.fillStyle = colMain;
      c.fillRect(a, 0, 1.5 * this.dpr, H);
      c.fillRect(b - 1.5 * this.dpr, 0, 1.5 * this.dpr, H);
    }

    const step = W / this.buckets;

    if (this.peaks) {
      const { mins, maxs } = this.peaks;
      for (let b = 0; b < this.buckets; b++) {
        const x = b * step;
        const played = (b / this.buckets) <= prog;
        c.fillStyle = played ? colMain : colSub;
        const top = mid - maxs[b] * mid * 0.94;
        const bot = mid - mins[b] * mid * 0.94;
        c.fillRect(x, top, Math.max(1, step - 0.6 * this.dpr), Math.max(1.2 * this.dpr, bot - top));
      }
    } else if (this.live) {
      for (let i = 0; i < this.live.length; i++) {
        const idx = (this.liveHead + i) % this.live.length;
        const v = this.live[idx];
        const x = i * step;
        const h = Math.max(1.2 * this.dpr, v * mid * 1.85);
        c.fillStyle = i > this.live.length - 4 ? colMain : colSub;
        c.fillRect(x, mid - h / 2, Math.max(1, step - 0.6 * this.dpr), h);
      }
    } else {
      c.fillStyle = colSub;
      c.fillRect(0, mid - 1, W, 2);
    }

    if (this.peaks) {
      c.fillStyle = colMain;
      c.fillRect(prog * W - this.dpr, 0, 2 * this.dpr, H);
    }
  }
}


/* ------------------------------------------------------------------ *
 * 8. app
 * ------------------------------------------------------------------ */

const $ = sel => document.querySelector(sel);

const PLAY_SVG  = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M8 5v14l11-7z"/></svg>';
const PAUSE_SVG = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M6 5h4v14H6zM14 5h4v14h-4z"/></svg>';

const PRESETS = {
  clean:      { rate: 1.00, mix: 0.00, decay: 3.2, damp: 0.45, size: 0.70, width: 1.00, bass: 0.0, air:  0.0, preDelay: 28, eightD: false, orbit: 10, preservePitch: false },
  slowed:     { rate: 0.85, mix: 0.36, decay: 3.6, damp: 0.50, size: 0.78, width: 1.15, bass: 4.0, air: -1.0, preDelay: 34, eightD: false, orbit: 10, preservePitch: false },
  screw:      { rate: 0.68, mix: 0.28, decay: 2.8, damp: 0.62, size: 0.62, width: 1.00, bass: 7.0, air: -3.0, preDelay: 22, eightD: false, orbit: 10, preservePitch: false },
  vapor:      { rate: 0.76, mix: 0.48, decay: 5.0, damp: 0.40, size: 0.86, width: 1.25, bass: 3.0, air:  1.5, preDelay: 45, eightD: false, orbit: 12, preservePitch: false },
  underwater: { rate: 0.80, mix: 0.62, decay: 6.5, damp: 0.86, size: 0.92, width: 1.30, bass: 6.0, air: -9.0, preDelay: 60, eightD: true,  orbit: 14, preservePitch: false },
  nightcore:  { rate: 1.28, mix: 0.14, decay: 1.8, damp: 0.30, size: 0.45, width: 1.00, bass: 1.5, air:  2.0, preDelay: 14, eightD: false, orbit: 10, preservePitch: false }
};

const panLabel = v => (Math.round(v) === 0 ? 'C' : (v < 0 ? 'L' + Math.round(-v) : 'R' + Math.round(v)));

const SLIDERS = {
  rate:     { el: '#playback-rate-control',    out: '#playback-rate-value',    fmt: v => v.toFixed(2),  get: v => v,     set: p => p },
  mix:      { el: '#reverb-mix-control',       out: '#reverb-mix-value',       fmt: v => Math.round(v), get: v => v / 100, set: p => p * 100 },
  bass:     { el: '#bass-boost-control',       out: '#bass-boost-value',       fmt: v => v.toFixed(1),  get: v => v,     set: p => p },
  decay:    { el: '#reverb-decay-control',     out: '#reverb-decay-value',     fmt: v => v.toFixed(1),  get: v => v,     set: p => p },
  preDelay: { el: '#reverb-predelay-control',  out: '#reverb-predelay-value',  fmt: v => Math.round(v), get: v => v,     set: p => p },
  damp:     { el: '#reverb-damp-control',      out: '#reverb-damp-value',      fmt: v => Math.round(v), get: v => v / 100, set: p => p * 100 },
  size:     { el: '#reverb-size-control',      out: '#reverb-size-value',      fmt: v => Math.round(v), get: v => v / 100, set: p => p * 100 },
  width:    { el: '#reverb-width-control',     out: '#reverb-width-value',     fmt: v => Math.round(v), get: v => v / 100, set: p => p * 100 },
  air:      { el: '#air-control',              out: '#air-value',              fmt: v => v.toFixed(1),  get: v => v,     set: p => p },
  orbit:    { el: '#eightd-period-control',    out: '#eightd-period-value',    fmt: v => v.toFixed(1),  get: v => v,     set: p => p },
  pan:      { el: '#pan-control',              out: '#pan-value',              fmt: panLabel,           get: v => v / 100, set: p => p * 100 }
};

const engine = new Engine();
let wave = null;
let trackName = '';
let activePreset = 'clean';
let ytAbort = null;
let analyserBuf = null;
let captureName = '';

/* ---- small helpers ------------------------------------------------ */

function fmtTime(s, precise) {
  if (!isFinite(s) || s < 0) s = 0;
  const m = Math.floor(s / 60);
  const r = s % 60;
  if (precise) return m + ':' + (r < 10 ? '0' : '') + r.toFixed(1);
  return m + ':' + String(Math.floor(r)).padStart(2, '0');
}

function setStatus(msg, cls) {
  const el = $('#status');
  el.textContent = msg;
  el.className = 'status' + (cls ? ' ' + cls : '');
}

function showBusy(text) { $('#busy-text').textContent = text; $('#busy').hidden = false; }
function setBusy(text) { $('#busy-text').textContent = text; }
function hideBusy() { $('#busy').hidden = true; }

function downloadBlob(blob, name) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 30000);
}

function baseName(n) { return (n || 'track').replace(/\.[a-z0-9]{1,5}$/i, ''); }

/* ---- controls ----------------------------------------------------- */

function refreshSlider(key) {
  const s = SLIDERS[key];
  const v = s.set(engine.params[key]);
  s.node.value = v;
  if (s.outNode) s.outNode.textContent = s.fmt(v);
}

function paintPresets() {
  document.querySelectorAll('.preset-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.preset === activePreset);
  });
}

function markManual() {
  if (activePreset !== null) { activePreset = null; paintPresets(); }
}

function updateSpin() {
  const r = Math.max(0.05, engine.params.rate);
  document.documentElement.style.setProperty('--rotation-duration', (1.8 / r).toFixed(3) + 's');
  $('#disc').classList.toggle('spinning', engine.playing);
  $('#btn-play').innerHTML = engine.playing ? PAUSE_SVG : PLAY_SVG;
}

function setToggle(btn, on, onText, offText) {
  btn.classList.toggle('active', on);
  btn.setAttribute('aria-pressed', String(on));
  btn.textContent = on ? onText : offText;
}

function syncToggles() {
  setToggle($('#btn-keep-pitch'), engine.params.preservePitch, 'on', 'off');
  setToggle($('#btn-8d'), engine.params.eightD, 'on', 'off');
  setToggle($('#btn-loop'), engine.params.loop, 'on', 'off');
  setToggle($('#btn-limiter'), engine.params.limiter, 'on', 'off');
  $('#eightd-speed').hidden = !engine.params.eightD;
  $('#pan-section').classList.toggle('disabled', engine.params.eightD);
}

function applyPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  engine.setParams(p);
  Object.keys(SLIDERS).forEach(refreshSlider);
  syncToggles();
  updateSpin();
  activePreset = name;
  paintPresets();
}

function initControls() {
  for (const key of Object.keys(SLIDERS)) {
    const s = SLIDERS[key];
    s.node = $(s.el);
    s.outNode = s.out ? $(s.out) : null;
    s.node.addEventListener('input', () => {
      markManual();
      const v = parseFloat(s.node.value);
      if (s.outNode) s.outNode.textContent = s.fmt(v);
      engine.setParams({ [key]: s.get(v) });
      if (key === 'rate') updateSpin();
    });
    refreshSlider(key);
  }

  document.querySelectorAll('.preset-btn').forEach(b => {
    b.addEventListener('click', () => applyPreset(b.dataset.preset));
  });
  paintPresets();

  $('#btn-keep-pitch').addEventListener('click', () => {
    markManual();
    engine.setParams({ preservePitch: !engine.params.preservePitch });
    syncToggles();
  });
  $('#btn-8d').addEventListener('click', () => {
    markManual();
    engine.setParams({ eightD: !engine.params.eightD });
    syncToggles();
  });
  $('#btn-loop').addEventListener('click', () => toggleLoop());
  $('#btn-limiter').addEventListener('click', () => {
    engine.setParams({ limiter: !engine.params.limiter });
    syncToggles();
  });

  const autosave = $('#btn-autosave');
  autosave.addEventListener('click', () => {
    const on = autosave.classList.toggle('active');
    autosave.setAttribute('aria-pressed', String(on));
    autosave.textContent = on ? 'on' : 'off';
    try { localStorage.setItem('srvb.autosave', on ? '1' : '0'); } catch (e) { /* private mode */ }
  });
  try {
    if (localStorage.getItem('srvb.autosave') === '0') {
      autosave.classList.remove('active');
      autosave.setAttribute('aria-pressed', 'false');
      autosave.textContent = 'off';
    }
  } catch (e) { /* private mode */ }

  const vol = $('#volume-control');
  const applyVol = () => {
    const v = parseInt(vol.value, 10) / 100;
    vol.style.setProperty('--volume-progress', vol.value + '%');
    engine.setParams({ volume: v });
  };
  vol.addEventListener('input', applyVol);
  applyVol();

  $('#btn-advanced').addEventListener('click', () => {
    const adv = $('#advanced');
    const open = adv.hidden;
    adv.hidden = !open;
    $('#btn-advanced').setAttribute('aria-expanded', String(open));
    $('#btn-advanced').innerHTML = (open ? 'hide advanced &#9652;' : 'show advanced &#9662;');
    if (open) wave.resize(), wave.rebuild();
  });

  syncToggles();
}

function toggleLoop() {
  const p = engine.params;
  if (!p.loop && p.loopEnd <= p.loopStart) {
    setStatus('drag across the waveform to pick a loop first.');
    return;
  }
  engine.setParams({ loop: !p.loop });
  syncToggles();
  renderLoopInfo();
}

function renderLoopInfo() {
  const p = engine.params;
  const el = $('#loop-info');
  if (p.loopEnd > p.loopStart) {
    const fine = p.loopEnd - p.loopStart < 10;
    el.textContent = (p.loop ? 'loop ' : 'sel ') +
      fmtTime(p.loopStart, fine) + '–' + fmtTime(p.loopEnd, fine);
  } else {
    el.textContent = '';
  }
}

/* ---- transport ----------------------------------------------------- */

function togglePlay() {
  if (engine.mode === 'idle') { setStatus('load a track first.'); return; }
  if (engine.playing) engine.pause(); else engine.play();
  updateSpin();
}

function onFrame(pos, write) {
  if (engine.mode === 'stream') {
    if (engine.chain && engine.chain.analyser) {
      if (!analyserBuf) analyserBuf = new Float32Array(engine.chain.analyser.fftSize);
      engine.chain.analyser.getFloatTimeDomainData(analyserBuf);
      let peak = 0;
      for (let i = 0; i < analyserBuf.length; i++) {
        const a = Math.abs(analyserBuf[i]);
        if (a > peak) peak = a;
      }
      wave.pushLive(peak);
    }
    const behind = Math.max(0, write - pos);
    $('#time-current').textContent = '-' + fmtTime(behind);
    $('#time-total').textContent = 'live';
  } else {
    $('#time-current').textContent = fmtTime(pos);
    $('#time-total').textContent = fmtTime(engine.duration);
  }
  wave.draw({
    position: pos,
    duration: engine.duration,
    loopStart: engine.params.loopStart,
    loopEnd: engine.params.loopEnd
  });
}

/* ---- loading -------------------------------------------------------- */

async function loadFromBlob(blob, name, autoplay) {
  setStatus('decoding ' + name + '…');
  try {
    await engine.ensure();
    const ab = await blob.arrayBuffer();
    const buffer = await engine.ctx.decodeAudioData(ab);
    await engine.loadBuffer(buffer);

    trackName = name;
    hideYtFallback();
    $('#track-name').textContent = baseName(name);
    document.title = baseName(name) + ' — slow playback audio';
    wave.setBuffer(buffer);
    engine.setParams({ loop: false, loopStart: 0, loopEnd: 0 });
    syncToggles();
    renderLoopInfo();
    $('#btn-play').disabled = false;
    $('#btn-export-wav').disabled = false;
    $('#btn-export-mp3').disabled = false;
    $('#capture-row').hidden = true;
    setStatus(baseName(name) + ' · ' + fmtTime(buffer.duration) + ' · ready', 'ready');
    if (autoplay) { engine.play(); }
    updateSpin();
    onFrame(0, 0);
    return true;
  } catch (err) {
    console.error(err);
    setStatus('could not decode that file: ' + err.message, 'error');
    return false;
  }
}

async function handleFile(file, fromLibrary) {
  if (!file) return;
  const ok = await loadFromBlob(file, file.name || 'track', true);
  if (ok && !fromLibrary && $('#btn-autosave').classList.contains('active')) {
    library.save({
      id: 'trk_' + Date.now() + '_' + Math.random().toString(36).slice(2, 8),
      name: file.name || 'track',
      blob: file,
      size: file.size,
      addedAt: Date.now()
    }).then(() => { if (!$('#saved-songs-list').hidden) renderLibrary(); });
  }
}

/* The processing constraints matter — echo cancellation on a music capture
 * sounds like a phone call. But an engine that cannot share tab audio at all
 * rejects the whole constraint object rather than ignoring it, so ask plainly
 * on the second attempt before blaming the browser. */
async function requestTabStream() {
  try {
    return await navigator.mediaDevices.getDisplayMedia({
      video: true,
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false }
    });
  } catch (err) {
    // a cancelled picker must not reopen the picker
    if (err && (err.name === 'NotAllowedError' || err.name === 'AbortError')) throw err;
    return navigator.mediaDevices.getDisplayMedia({ video: true, audio: true });
  }
}

async function streamTab() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getDisplayMedia) {
    setStatus('tab capture is not supported in this browser.', 'error');
    return;
  }
  try {
    const stream = await requestTabStream();
    if (!stream.getAudioTracks().length) {
      stream.getTracks().forEach(t => t.stop());
      setStatus('no audio in that capture — tick "share tab audio".', 'error');
      return;
    }
    stream.getVideoTracks().forEach(t => t.stop());
    await engine.loadStream(stream);
    hideYtFallback();
    // name the capture after the video id if a link was pasted
    const pasted = ($('#youtube-url-input').value || '').trim();
    const id = pasted.match(/(?:v=|youtu\.be\/|shorts\/|live\/|embed\/)([\w-]{6,})/);
    captureName = id ? 'youtube ' + id[1] : 'tab capture';
    trackName = 'live tab';
    $('#track-name').textContent = 'live tab';
    wave.setLive();
    $('#btn-play').disabled = false;
    $('#capture-row').hidden = false;
    $('#btn-export-wav').disabled = true;
    $('#btn-export-mp3').disabled = true;
    const cap = engine.streamCapacity ? Math.floor(engine.streamCapacity / 60) + ' min' : 'a few minutes';
    setStatus('capturing — let it play, then "stop & keep as track" to export. holds ' + cap + '.',
              'streaming');
    updateSpin();
    stream.getAudioTracks()[0].addEventListener('ended', () => {
      setStatus('tab capture ended.');
      engine.releaseSource();
      engine.mode = 'idle';
      updateSpin();
    });
  } catch (err) {
    const name = err && err.name;
    if (name === 'NotAllowedError' || name === 'AbortError') setStatus('capture cancelled.');
    // safari and firefox have no display-capture audio track to give
    else if (name === 'NotFoundError' || name === 'NotSupportedError' || name === 'OverconstrainedError')
      setStatus('this browser will not share tab audio — chrome or edge can.', 'error');
    else setStatus('capture failed: ' + (err && err.message || 'unknown error'), 'error');
  }
}

/* Capture starts before the video does, so drop the dead air either end
 * rather than making the user scrub past it. */
function trimSilence(l, r, sr, thresh = 0.0012) {
  let s = 0, e = l.length - 1;
  while (s < e && Math.abs(l[s]) < thresh && Math.abs(r[s]) < thresh) s++;
  while (e > s && Math.abs(l[e]) < thresh && Math.abs(r[e]) < thresh) e--;
  s = Math.max(0, s - Math.round(0.03 * sr));          // keep the attack
  e = Math.min(l.length - 1, e + Math.round(0.25 * sr)); // and the fade
  if (e - s < sr * 0.2) return { l, r };                // all quiet, keep as is
  return { l: l.subarray(s, e + 1), r: r.subarray(s, e + 1) };
}

/* Turn whatever has been captured into an ordinary track, so it gets the
 * waveform, seeking, loops and offline export like a loaded file. */
async function captureToTrack() {
  if (engine.mode !== 'stream') { setStatus('nothing is being captured.'); return; }
  showBusy('collecting captured audio…');
  try {
    const dump = await engine.dumpRecording();
    if (!dump.frames) throw new Error('nothing was captured yet');

    const sr = dump.srcRate;
    const cut = trimSilence(new Float32Array(dump.chans[0]), new Float32Array(dump.chans[1]), sr);
    if (cut.l.length < sr * 0.2) throw new Error('the captured audio was silent — was "share tab audio" ticked?');

    const buf = engine.ctx.createBuffer(2, cut.l.length, sr);
    buf.copyToChannel(cut.l, 0);
    buf.copyToChannel(cut.r, 1);

    await engine.loadBuffer(buf);   // this also stops the capture

    const name = (captureName || 'tab capture') + '.wav';
    trackName = name;
    hideYtFallback();
    $('#track-name').textContent = baseName(name);
    wave.setBuffer(buf);
    engine.setParams({ loop: false, loopStart: 0, loopEnd: 0 });
    syncToggles();
    renderLoopInfo();
    $('#capture-row').hidden = true;
    $('#btn-play').disabled = false;
    $('#btn-export-wav').disabled = false;
    $('#btn-export-mp3').disabled = false;
    updateSpin();
    onFrame(0, 0);
    setStatus('kept ' + fmtTime(buf.duration) + ' — now exportable.', 'ready');
  } catch (err) {
    setStatus(err.message, 'error');
  } finally {
    hideBusy();
  }
}

/* ---- youtube -------------------------------------------------------- */

/* A browser cannot fetch youtube audio itself, so loading by url goes
 * through a downloader. The default is a local yt-audio-api instance;
 * SRVB_YT_ENDPOINT points at another (see README). When the endpoint cannot
 * be reached the app falls back to capturing the tab, which needs no
 * server. */
const YT_ENDPOINT = window.SRVB_YT_ENDPOINT || 'http://127.0.0.1:5000/';
const YT_CONFIGURED = true;
const NO_BACKEND_MSG = 'youtube downloader unavailable — capture the tab instead:';

const YT_RE = /^(?:https?:\/\/)?(?:www\.|m\.|music\.)?(?:youtube\.com\/(?:watch|shorts|live|embed)|youtu\.be\/)/i;
const isYoutubeUrl = u => YT_RE.test((u || '').trim());

/* Offer the tab-capture route. A page cannot fetch youtube audio itself
 * (googlevideo.com sends no CORS headers), so capturing the tab while it
 * plays is what works when the downloader is out. */
function showYtFallback(url) {
  const box = document.getElementById('yt-fallback');
  if (!box) return;
  box.hidden = false;
  box.dataset.url = url || '';
  const openBtn = document.getElementById('btn-yt-open');
  if (openBtn) openBtn.disabled = !url;
}

function hideYtFallback() {
  const box = document.getElementById('yt-fallback');
  if (box) box.hidden = true;
}

/* content-disposition, both the RFC 5987 form and the plain quoted one */
function filenameFromDisposition(res) {
  const name = (res.headers.get('content-disposition') || '')
    .match(/filename\*?=(?:UTF-8'')?"?([^";]+)/i)?.[1];
  if (!name) return '';
  try { return decodeURIComponent(name); }
  catch (e) { return name; }
}

/* a downloader that fails usually explains why in its body; a bare
 * status code tells the user nothing they can act on */
async function messageFromError(res) {
  const ct = res.headers.get('content-type') || '';
  try {
    if (ct.includes('application/json')) {
      const j = await res.json();
      const m = j.error || j.message;
      if (m) return typeof m === 'string' ? m : (m.code || JSON.stringify(m));
    } else {
      const t = (await res.text()).trim();
      if (t && t.length < 200) return t;
    }
  } catch (e) { /* unreadable body, fall back to the status */ }
  return 'server returned ' + res.status;
}

async function readWithProgress(res, onProgress) {
  // a proxied stream often has no content-length, so honour the
  // estimated-content-length some downloaders send instead
  const total = parseInt(
    res.headers.get('content-length') ||
    res.headers.get('estimated-content-length') || '0', 10);

  if (!res.body || !total || !isFinite(total)) return res.blob();

  const reader = res.body.getReader();
  const parts = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;
    parts.push(value);
    got += value.byteLength;
    if (onProgress) onProgress(Math.min(got / total, 1));
  }
  return new Blob(parts, { type: res.headers.get('content-type') || 'audio/mpeg' });
}

/* yt-audio-api serves the converted file from /download, alongside the
 * endpoint that handed out the token */
function ytDownloadUrl(token) {
  return new URL('download?token=' + encodeURIComponent(token),
                 new URL(YT_ENDPOINT, location.href)).href;
}

/* it also names every file after a uuid, which says nothing about the
 * track; the video id at least matches how a tab capture is named */
const UUID_FILENAME_RE = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\.\w+$/i;

function nameFromYoutubeUrl(url) {
  const id = (url || '').match(/(?:v=|youtu\.be\/|shorts\/|live\/|embed\/)([\w-]{6,})/);
  return id ? 'youtube ' + id[1] + '.mp3' : 'youtube audio.mp3';
}

/* a title straight off the api can carry path separators and control
 * characters, neither of which belong in a download name */
function safeYoutubeFilename(title) {
  if (!title) return 'youtube audio.mp3';

  let name = String(title)
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
    .replace(/\s+/g, ' ')
    .trim();

  if (!name) name = 'youtube audio';
  if (!/\.[a-z0-9]{2,5}$/i.test(name)) name += '.mp3';
  return name;
}

async function downloadYoutube() {
  const url = $('#youtube-url-input').value.trim();
  if (!url) { setStatus('paste a youtube link first.'); return; }

  if (!isYoutubeUrl(url)) {
    setStatus('that does not look like a youtube link.', 'error');
    return;
  }

  // no endpoint configured: do not fire a request that can only 404
  if (!YT_CONFIGURED) {
    setStatus(NO_BACKEND_MSG, 'error');
    showYtFallback(url);
    return;
  }

  hideYtFallback();
  if (ytAbort) ytAbort.abort();
  ytAbort = new AbortController();
  const mine = ytAbort;

  $('#btn-youtube-cancel').hidden = false;
  $('#download-progress').hidden = false;
  $('#download-progress').firstElementChild.style.width = '0%';
  setStatus('converting on the server — this can take a while…');

  const progress = f => {
    const pct = Math.round(Math.min(Math.max(f, 0), 1) * 100);
    $('#download-progress').firstElementChild.style.width = pct + '%';
    setStatus('downloading… ' + pct + '%');
  };

  try {
    let res;
    try {
      res = await fetch(YT_ENDPOINT + '?url=' + encodeURIComponent(url), {
        method: 'GET',
        headers: { Accept: 'application/json' },
        signal: mine.signal
      });
    } catch (netErr) {
      if (netErr.name === 'AbortError') throw netErr;
      // cross-origin failures land here too: a missing
      // Access-Control-Allow-Origin looks exactly like an outage
      const e = new Error('could not reach youtube downloader');
      e.noBackend = true;
      throw e;
    }

    if (res.status === 404 || res.status === 405) {
      const e = new Error('youtube downloader is unavailable');
      e.noBackend = true;
      throw e;
    }
    if (!res.ok) throw new Error(await messageFromError(res));

    const contentType = res.headers.get('content-type') || '';
    let blob;
    let youtubeFilename = 'youtube audio.mp3';

    if (contentType.includes('application/json')) {
      /* yt-audio-api answers with { token }, and other downloaders with a
       * direct media url, so take whichever shape came back */
      const j = await res.json();

      if (j.success === false || j.error) {
        throw new Error(j.message || j.error || 'youtube downloader returned an error');
      }

      const link = j.token ? ytDownloadUrl(j.token) : (
        j.mediaInfo?.audioUrl ||
        j.mediaInfo?.audio_url ||
        j.audioUrl ||
        j.audio_url ||
        j.url ||
        j.link ||
        j.downloadUrl ||
        j.download_url);

      if (!link) {
        console.error('YouTube API response:', j);
        throw new Error('no audio url in the response');
      }

      const title = j.mediaInfo?.title || j.title || '';
      youtubeFilename = title ? safeYoutubeFilename(title) : nameFromYoutubeUrl(url);
      setStatus('audio found — downloading…');

      let r2;
      try {
        r2 = await fetch(link, { method: 'GET', signal: mine.signal });
      } catch (audioErr) {
        if (audioErr.name === 'AbortError') throw audioErr;
        throw new Error('could not download the returned audio file');
      }
      // yt-audio-api tokens are one-shot and expire after five minutes
      if (r2.status === 401 || r2.status === 408) {
        throw new Error('the download link expired before the fetch started — try again');
      }
      if (!r2.ok) throw new Error(await messageFromError(r2));

      blob = await readWithProgress(r2, progress);

      // only fall back to the media url's own name when the api gave no
      // title, and never to a bare uuid
      const dispositionName = filenameFromDisposition(r2);
      if (dispositionName && !title && !UUID_FILENAME_RE.test(dispositionName)) {
        youtubeFilename = dispositionName;
      }
    } else {
      // endpoints that hand back the bytes rather than a link
      blob = await readWithProgress(res, progress);
      const headerName = filenameFromDisposition(res);
      if (headerName) youtubeFilename = headerName;
    }

    if (!blob || !blob.size) throw new Error('downloaded audio file was empty');
    progress(1);
    setStatus('loading audio…');

    await handleFile(new File([blob], youtubeFilename, { type: blob.type || 'audio/mpeg' }), false);
    setStatus('youtube audio loaded.');
  } catch (err) {
    if (err.name === 'AbortError') setStatus('download cancelled.');
    else if (err.noBackend) { setStatus(NO_BACKEND_MSG, 'error'); showYtFallback(url); }
    else {
      console.error('YouTube download failed:', err);
      setStatus('download failed: ' + (err.message || 'unknown error'), 'error');
    }
  } finally {
    if (ytAbort === mine) ytAbort = null;
    $('#btn-youtube-cancel').hidden = true;
    $('#download-progress').hidden = true;
  }
}

/* ---- library UI ------------------------------------------------------ */

async function renderLibrary() {
  const list = $('#saved-songs-list');
  const items = await library.list();
  list.innerHTML = '';
  if (!items.length) {
    const li = document.createElement('li');
    li.className = 'empty';
    li.textContent = 'nothing saved yet';
    list.appendChild(li);
    $('#btn-clear-songs').hidden = true;
    return;
  }
  for (const it of items) {
    const li = document.createElement('li');

    const name = document.createElement('button');
    name.className = 'song-name';
    name.textContent = baseName(it.name);
    name.addEventListener('click', () => handleFile(new File([it.blob], it.name, { type: it.blob.type }), true));

    const meta = document.createElement('span');
    meta.className = 'song-meta';
    meta.textContent = (it.size / 1048576).toFixed(1) + 'mb';

    const del = document.createElement('button');
    del.className = 'song-del';
    del.textContent = '×';
    del.setAttribute('aria-label', 'delete ' + baseName(it.name));
    del.addEventListener('click', async () => { await library.remove(it.id); renderLibrary(); });

    li.append(name, meta, del);
    list.appendChild(li);
  }
  $('#btn-clear-songs').hidden = false;
}

/* ---- export ----------------------------------------------------------- */

async function doExport(kind) {
  if (!engine.buffer) { setStatus('load a track before exporting.'); return; }
  showBusy('rendering…');
  try {
    const rendered = await renderOffline(engine.buffer, engine.params);
    let blob;
    if (kind === 'wav') {
      setBusy('writing wav…');
      blob = encodeWav(rendered);
    } else {
      setBusy('encoding mp3…');
      blob = await encodeMp3(rendered, 320, f => setBusy('encoding mp3… ' + Math.round(f * 100) + '%'));
    }
    const p = engine.params;
    const tag = [];
    if (Math.abs(p.rate - 1) > 0.005) tag.push(p.rate < 1 ? 'slowed' : 'sped up');
    if (p.mix > 0.01) tag.push('reverb');
    if (p.eightD) tag.push('8d');
    const suffix = tag.length ? ' (' + tag.join(' + ') + ')' : '';
    downloadBlob(blob, baseName(trackName) + suffix + '.' + kind);
    setStatus('exported ' + kind + '.', 'ready');
  } catch (err) {
    console.error(err);
    setStatus('export failed: ' + err.message, 'error');
  } finally {
    hideBusy();
  }
}

/* ---- waveform interaction --------------------------------------------- */

function initWaveInteraction(canvas) {
  let down = false, dragged = false, startX = 0;
  const posFor = ev => {
    const r = canvas.getBoundingClientRect();
    return Math.max(0, Math.min(1, (ev.clientX - r.left) / r.width));
  };

  canvas.addEventListener('pointerdown', ev => {
    if (!engine.duration) return;
    canvas.setPointerCapture(ev.pointerId);
    down = true; dragged = false;
    startX = posFor(ev);
    if (ev.shiftKey) {
      engine.setParams({ loop: false, loopStart: 0, loopEnd: 0 });
      syncToggles(); renderLoopInfo();
      down = false;
    }
  });

  canvas.addEventListener('pointermove', ev => {
    if (!down) return;
    const x = posFor(ev);
    if (Math.abs(x - startX) > 0.005) {
      dragged = true;
      const a = Math.min(startX, x) * engine.duration;
      const b = Math.max(startX, x) * engine.duration;
      engine.setParams({ loopStart: a, loopEnd: b });
      renderLoopInfo();
    }
  });

  const finish = ev => {
    if (!down) return;
    down = false;
    if (!dragged) engine.seek(posFor(ev) * engine.duration);
    else if (!engine.params.loop) { engine.setParams({ loop: true }); syncToggles(); renderLoopInfo(); }
  };
  canvas.addEventListener('pointerup', finish);
  canvas.addEventListener('pointercancel', () => { down = false; });
}

/* ---- keyboard --------------------------------------------------------- */

function initKeyboard() {
  window.addEventListener('keydown', ev => {
    const t = ev.target;
    if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
    switch (ev.key) {
      case ' ': ev.preventDefault(); togglePlay(); break;
      case 'ArrowLeft':  ev.preventDefault(); engine.seek(engine.position - (ev.shiftKey ? 30 : 5)); break;
      case 'ArrowRight': ev.preventDefault(); engine.seek(engine.position + (ev.shiftKey ? 30 : 5)); break;
      case 'l': case 'L': toggleLoop(); break;
      case 'p': case 'P':
        engine.setParams({ preservePitch: !engine.params.preservePitch });
        syncToggles(); markManual();
        break;
    }
  });
}

/* ---- boot -------------------------------------------------------------- */

function init() {
  wave = new Waveform($('#buffer-canvas'));
  initControls();
  initWaveInteraction($('#buffer-canvas'));
  initKeyboard();

  engine.onPos = onFrame;
  engine.onEnded = () => { updateSpin(); setStatus('finished.', 'ready'); };

  $('#btn-play').innerHTML = PLAY_SVG;
  $('#btn-play').addEventListener('click', togglePlay);
  $('#btn-jump-live').addEventListener('click', () => engine.jumpLive());
  $('#btn-capture-done').addEventListener('click', captureToTrack);
  $('#btn-stream').addEventListener('click', streamTab);
  $('#btn-choose').addEventListener('click', () => $('#file-input').click());
  $('#btn-export-wav').addEventListener('click', () => doExport('wav'));
  $('#btn-export-mp3').addEventListener('click', () => doExport('mp3'));
  $('#file-input').addEventListener('change', e => {
    handleFile(e.target.files[0], false);
    e.target.value = '';
  });

  $('#btn-youtube-download').addEventListener('click', downloadYoutube);
  $('#youtube-url-input').addEventListener('keydown', e => { if (e.key === 'Enter') downloadYoutube(); });
  $('#youtube-url-input').addEventListener('input', e => {
    const v = e.target.value;
    // with no downloader there is nothing to wait for, so surface the
    // capture route as soon as a youtube link appears
    if (!YT_CONFIGURED && isYoutubeUrl(v)) showYtFallback(v.trim());
    else hideYtFallback();
  });
  $('#btn-yt-open').addEventListener('click', () => {
    const u = $('#yt-fallback').dataset.url;
    if (!u) return;
    window.open(/^https?:\/\//i.test(u) ? u : 'https://' + u, '_blank', 'noopener,noreferrer');
    setStatus('press play over there, then come back and hit "capture tab".');
  });
  $('#btn-yt-capture').addEventListener('click', streamTab);
  $('#btn-youtube-cancel').addEventListener('click', () => { if (ytAbort) ytAbort.abort(); });
  $('#btn-yt-paste').addEventListener('click', async () => {
    try {
      const txt = await navigator.clipboard.readText();
      if (txt) { $('#youtube-url-input').value = txt.trim(); downloadYoutube(); }
    } catch (e) { setStatus('clipboard blocked — paste the link manually.'); }
  });

  $('#btn-saved-songs').addEventListener('click', async () => {
    const list = $('#saved-songs-list');
    const open = list.hidden;
    list.hidden = !open;
    $('#btn-saved-songs').setAttribute('aria-expanded', String(open));
    if (open) await renderLibrary();
    else $('#btn-clear-songs').hidden = true;
  });
  $('#btn-clear-songs').addEventListener('click', async () => {
    await library.clear();
    renderLibrary();
  });

  /* An AudioContext built mid-await is built without user activation, which
   * browsers refuse to start: the click that began a youtube download has
   * expired long before the converted bytes land. Build it on the first
   * gesture instead, whatever that gesture happens to be. */
  const PRIME_EVENTS = ['pointerdown', 'keydown', 'touchstart'];
  const primeAudio = () => {
    PRIME_EVENTS.forEach(t => window.removeEventListener(t, primeAudio, true));
    // a real load reports its own failure; this is only a head start
    engine.ensure().catch(() => {});
  };
  PRIME_EVENTS.forEach(t => window.addEventListener(t, primeAudio, true));

  // drag and drop onto the record
  const disc = $('#disc');
  ['dragenter', 'dragover'].forEach(t => disc.addEventListener(t, e => {
    e.preventDefault(); disc.classList.add('dragover');
  }));
  ['dragleave', 'drop'].forEach(t => disc.addEventListener(t, e => {
    e.preventDefault(); disc.classList.remove('dragover');
  }));
  disc.addEventListener('drop', e => {
    const f = e.dataTransfer && e.dataTransfer.files[0];
    if (f) handleFile(f, false);
  });
  disc.addEventListener('click', () => { if (engine.mode !== 'idle') togglePlay(); });
  window.addEventListener('dragover', e => e.preventDefault());
  window.addEventListener('drop', e => e.preventDefault());

  // warn once if the browser cannot run the good engine
  engine.ensure().then(() => {
    if (!engine.workletOK) {
      $('#engine-notice-text').innerHTML =
        '<strong>limited mode.</strong> this browser has no AudioWorklet, so ' +
        'time-stretch, tab streaming and export are unavailable.';
      $('#engine-notice').hidden = false;
    }
  }).catch(err => {
    $('#engine-notice-text').innerHTML = '<strong>audio unavailable.</strong> ' + err.message;
    $('#engine-notice').hidden = false;
  });

  onFrame(0, 0);
}

if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', init);
else init();
