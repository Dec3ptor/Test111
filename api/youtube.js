/* Serverless audio fetch for the youtube panel.
 *
 * A browser cannot do this itself. googlevideo.com sends no CORS headers, so
 * a page can never read the media stream even once it holds the url — which
 * is why the client-only versions of this feature do not exist. Server-side
 * that rule does not apply, so this resolves the stream and pipes the bytes
 * back from the same origin as the page, and the browser sees an ordinary
 * same-origin request.
 *
 * Nothing is transcoded: youtube's audio-only streams are already m4a or
 * webm/opus, both of which decodeAudioData reads directly. That keeps FFmpeg
 * — which does not fit in a serverless function — out of the picture. */

import ytdl from '@distube/ytdl-core';

export const config = { maxDuration: 60 };

/* a title is free text: strip what cannot go in a filename, then send both
 * the ascii form and the utf-8 one, since a header cannot carry raw utf-8 */
export function contentDisposition(title, ext) {
  const clean = (String(title || '')
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
    .replace(/\s+/g, ' ')
    .trim() || 'youtube audio') + ext;
  const ascii = clean.replace(/[^\x20-\x7E]/g, '_');
  return 'attachment; filename="' + ascii + '"; ' +
         "filename*=UTF-8''" + encodeURIComponent(clean);
}

/* m4a decodes in every browser; webm/opus does not decode in safari, so it
 * is the fallback rather than the pick. Highest bitrate within whichever
 * container wins. */
export function pickFormat(formats) {
  const audioOnly = (formats || []).filter(f => f.hasAudio && !f.hasVideo);
  if (!audioOnly.length) return null;
  const mp4 = audioOnly.filter(f => f.container === 'mp4');
  const pool = mp4.length ? mp4 : audioOnly;
  return pool.slice().sort((a, b) => (b.audioBitrate || 0) - (a.audioBitrate || 0))[0];
}

export default async function handler(req, res) {
  // same-origin needs no CORS; this is only for a page hosted elsewhere,
  // such as the github pages copy pointed at this deployment
  const allowed = process.env.ALLOWED_ORIGIN;
  if (allowed) {
    res.setHeader('access-control-allow-origin', allowed);
    res.setHeader('access-control-expose-headers', 'Content-Length, Content-Disposition');
  }
  if (req.method === 'OPTIONS') return res.status(204).end();

  const url = req.query.url;
  if (!url) return res.status(400).json({ error: "missing 'url' parameter" });
  if (!ytdl.validateURL(url)) return res.status(400).json({ error: 'that is not a youtube url' });

  let info;
  try {
    info = await ytdl.getInfo(url);
  } catch (err) {
    // youtube blocks datacenter ips in waves, and that is not a bug in the
    // request — say which it is rather than returning a bare 500
    const detail = String((err && err.message) || err);
    // 403 is what the ip block usually looks like from here; the wordier
    // variants show up when youtube would rather ask for a login
    const blocked = /sign in|bot|403|429|consent|captcha/i.test(detail);
    return res.status(502).json({
      error: blocked
        ? 'youtube refused this server — it blocks datacenter addresses in waves'
        : 'could not read that video',
      detail
    });
  }

  const format = pickFormat(info.formats);
  if (!format) return res.status(415).json({ error: 'no audio-only stream for that video' });
  const ext = format.container === 'webm' ? '.webm' : '.m4a';

  res.setHeader('content-type', (format.mimeType || 'audio/mp4').split(';')[0]);
  res.setHeader('content-disposition', contentDisposition(info.videoDetails.title, ext));
  res.setHeader('cache-control', 'no-store');
  // the only length youtube gives up front, and what the progress bar reads
  if (format.contentLength) res.setHeader('content-length', format.contentLength);

  const stream = ytdl.downloadFromInfo(info, { format });
  stream.on('error', err => {
    console.error('youtube stream failed:', err);
    // headers are already out by now, so the client sees a truncated body
    // rather than a status it can read
    res.destroy(err);
  });
  req.on('close', () => stream.destroy());
  stream.pipe(res);
}
