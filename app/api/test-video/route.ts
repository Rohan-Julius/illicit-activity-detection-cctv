import fs from 'node:fs';
import path from 'node:path';

export const runtime = 'nodejs';

function getVideoPath() {
  // `test.mp4` is in the project root (not `/public`).
  return path.join(process.cwd(), 'test.mp4');
}

export async function GET(req: Request) {
  const filePath = getVideoPath();
  if (!fs.existsSync(filePath)) {
    return new Response('test.mp4 not found', { status: 404 });
  }

  const stat = fs.statSync(filePath);
  const size = stat.size;
  const range = req.headers.get('range');

  // Support byte-range requests (required for smooth <video> playback).
  if (range) {
    const match = /bytes=(\d+)-(\d*)/.exec(range);
    if (!match) {
      return new Response('Malformed range', { status: 416 });
    }

    const start = Number(match[1]);
    const end = match[2] ? Number(match[2]) : Math.min(start + 1024 * 1024 - 1, size - 1);
    if (Number.isNaN(start) || Number.isNaN(end) || start >= size || end >= size) {
      return new Response('Range not satisfiable', {
        status: 416,
        headers: { 'Content-Range': `bytes */${size}` },
      });
    }

    const stream = fs.createReadStream(filePath, { start, end });
    return new Response(stream as any, {
      status: 206,
      headers: {
        'Content-Type': 'video/mp4',
        'Content-Length': String(end - start + 1),
        'Content-Range': `bytes ${start}-${end}/${size}`,
        'Accept-Ranges': 'bytes',
        'Cache-Control': 'no-store',
      },
    });
  }

  const stream = fs.createReadStream(filePath);
  return new Response(stream as any, {
    headers: {
      'Content-Type': 'video/mp4',
      'Content-Length': String(size),
      'Accept-Ranges': 'bytes',
      'Cache-Control': 'no-store',
    },
  });
}

