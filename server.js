/* Local host for the whole app: the page and /api/youtube on one origin.
 *
 * This exists because youtube blocks datacenter addresses, which is every
 * free host there is — the deployed function works until it doesn't, and
 * nothing in the client can fix that. A home connection is a residential
 * address, so running the same function here is the setup that keeps
 * working. Serving the page from it too means same origin, so there is no
 * CORS to configure and nothing to keep in sync. */

import http from 'node:http';
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import handler from './api/youtube.js';

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 3000;

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.json': 'application/json',
  '.ico': 'image/x-icon'
};

/* the function is written against vercel's request and response, which put
 * req.query and a chainable res.status().json() on top of node's own */
function vercelShim(req, res, url) {
  req.query = Object.fromEntries(url.searchParams);
  res.status = code => { res.statusCode = code; return res; };
  res.json = body => {
    res.setHeader('content-type', 'application/json');
    res.end(JSON.stringify(body));
    return res;
  };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, 'http://' + (req.headers.host || 'localhost'));

  if (url.pathname === '/api/youtube') {
    vercelShim(req, res, url);
    try {
      await handler(req, res);
    } catch (err) {
      console.error('handler threw:', err);
      // a throw after the pipe has started leaves the headers gone already
      if (!res.headersSent) {
        res.statusCode = 500;
        res.setHeader('content-type', 'application/json');
        res.end(JSON.stringify({ error: 'the downloader failed', detail: String((err && err.message) || err) }));
      } else {
        res.destroy(err);
      }
    }
    return;
  }

  const file = path.resolve(ROOT, url.pathname === '/' ? 'index.html' : '.' + url.pathname);
  // a path is a claim, not a promise: never serve outside the repo
  if (!file.startsWith(ROOT + path.sep)) {
    res.statusCode = 403;
    return res.end('forbidden');
  }
  try {
    const body = await readFile(file);
    res.setHeader('content-type', TYPES[path.extname(file).toLowerCase()] || 'application/octet-stream');
    res.end(body);
  } catch (err) {
    res.statusCode = 404;
    res.end('not found');
  }
});

server.listen(PORT, () => {
  console.log('slowed playback audio → http://localhost:' + PORT);
});
