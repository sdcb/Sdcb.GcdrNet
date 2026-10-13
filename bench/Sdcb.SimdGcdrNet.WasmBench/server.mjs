// Local harness for the browser-wasm bench (after SimdPaddleOCR's WasmBench):
// serves the published wwwroot + tests/testdata with COOP/COEP (SharedArrayBuffer for
// the .NET Web Worker threads), launches headless Edge/Chrome on index.html, relays the
// browser console, saves the JSON result under artifacts/ and exits with the bench's code.
//
//   node server.mjs --args "--corpus cpu" [--out artifacts/wasm/corpus.json] [--env K=V]... [--open] [--no-launch] [--port N] [--timeout SEC]
import http from 'http';
import fs from 'fs';
import os from 'os';
import path from 'path';
import { fileURLToPath } from 'url';
import { spawn } from 'child_process';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, '..', '..');
const wwwroot = path.join(__dirname, 'bin', 'Release', 'net10.0', 'publish', 'wwwroot');
const testdata = path.join(repoRoot, 'tests', 'testdata');
const artifacts = path.join(repoRoot, 'artifacts');

const argv = process.argv.slice(2);
const getArg = (name, def) => {
    const i = argv.indexOf(name);
    return i !== -1 && i + 1 < argv.length ? argv[i + 1] : def;
};
const benchArgs = getArg('--args', '--corpus cpu');
const out = getArg('--out', '');
const port = parseInt(getArg('--port', '0'), 10);
const headless = !argv.includes('--open');
const noLaunch = argv.includes('--no-launch');
const timeoutSec = parseInt(getArg('--timeout', '7200'), 10);
// --env KEY=VALUE (repeatable): environment of the .NET runtime (e.g. GCDRNET_THREADS=4, GCDRNET_PROF=1)
const envPairs = argv.flatMap((a, i) => a === '--env' && i + 1 < argv.length ? [argv[i + 1]] : []);

if (!fs.existsSync(path.join(wwwroot, 'index.html'))) {
    console.error(`[SERVER] ${wwwroot} not found: dotnet publish bench/Sdcb.SimdGcdrNet.WasmBench -c Release first (or use run.ps1).`);
    process.exit(2);
}

const mimeTypes = {
    '.html': 'text/html', '.js': 'application/javascript', '.mjs': 'application/javascript',
    '.wasm': 'application/wasm', '.json': 'application/json', '.dat': 'application/octet-stream',
    '.webp': 'image/webp', '.png': 'image/png', '.jpg': 'image/jpeg', '.css': 'text/css',
};

/** Resolves a URL-relative path under root; null when it escapes root or is not a file. */
function safeFile(root, rel) {
    let p;
    try { p = path.resolve(root, '.' + path.sep + decodeURIComponent(rel)); } catch { return null; }
    if (p !== root && !p.startsWith(root + path.sep)) return null;
    return fs.existsSync(p) && fs.statSync(p).isFile() ? p : null;
}

function sendFile(res, file) {
    res.writeHead(200, { 'Content-Type': mimeTypes[path.extname(file).toLowerCase()] || 'application/octet-stream' });
    fs.createReadStream(file).on('error', () => res.destroy()).pipe(res);
}

const MaxBody = 16 << 20;

/** Request body as utf8, or null when it exceeds MaxBody. */
function readBody(req) {
    return new Promise(resolve => {
        const chunks = [];
        let size = 0;
        req.on('data', c => {
            size += c.length;
            if (size > MaxBody) { resolve(null); req.destroy(); return; }
            chunks.push(c);
        });
        req.on('end', () => resolve(Buffer.concat(chunks).toString('utf8')));
    });
}

let browser = null, profile = null;

function shutdown(code) {
    if (browser?.pid) {
        // Edge/Chrome fork children: kill the whole tree, not just the launcher
        if (process.platform === 'win32') spawn('taskkill', ['/PID', String(browser.pid), '/T', '/F'], { stdio: 'ignore' });
        else browser.kill();
    }
    server.close();
    setTimeout(() => {
        if (profile) try { fs.rmSync(profile, { recursive: true, force: true }); } catch { }
        process.exit(code);
    }, 500);
}

// /api/* only from our own page: a cross-origin page cannot add the custom header without a
// CORS preflight (never answered), and the Host check stops DNS-rebinding
function apiAllowed(req) {
    return req.headers['x-wasmbench'] === '1' && req.headers.host === `127.0.0.1:${server.address().port}`;
}

const server = http.createServer(async (req, res) => {
    // cross-origin isolation: required for SharedArrayBuffer / Web Worker threads
    res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
    res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
    const url = new URL(req.url, 'http://localhost');

    if (url.pathname.startsWith('/api/') && !apiAllowed(req)) {
        res.writeHead(403); res.end();
        return;
    }
    if (url.pathname === '/api/log' && req.method === 'POST') {
        const body = await readBody(req);
        if (body) console.log(body);
        res.writeHead(204); res.end();
        return;
    }
    if (url.pathname === '/api/save-benchmark' && req.method === 'POST') {
        // results only land under <repo>/artifacts
        const rel = url.searchParams.get('out') || 'wasm/result.json';
        const target = path.resolve(artifacts, rel.replace(/^artifacts[\\/]/, ''));
        if (!target.startsWith(artifacts + path.sep) || path.extname(target).toLowerCase() !== '.json') {
            res.writeHead(400); res.end('out must be a .json file under artifacts/');
            return;
        }
        const body = await readBody(req);
        if (body === null) { res.writeHead(413); res.end(); return; }
        fs.mkdirSync(path.dirname(target), { recursive: true });
        fs.writeFileSync(target, body, 'utf8');
        console.log(`[SERVER] saved ${target}`);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ ok: true }));
        return;
    }
    if (url.pathname === '/api/done') {
        const exitCode = parseInt(url.searchParams.get('exitCode') || '0', 10);
        const err = url.searchParams.get('error');
        if (err) console.error('[SERVER] browser reported error:\n' + err);
        console.log(`[SERVER] done, exit code ${exitCode}`);
        res.writeHead(200); res.end('OK');
        if (headless) shutdown(exitCode);
        return;
    }
    if (url.pathname.startsWith('/testdata/')) {
        const f = safeFile(testdata, url.pathname.substring('/testdata/'.length));
        if (!f) { res.writeHead(404); res.end(); return; }
        sendFile(res, f);
        return;
    }
    const f = safeFile(wwwroot, url.pathname === '/' ? 'index.html' : url.pathname.substring(1));
    if (!f) { res.writeHead(404); res.end('not found: ' + url.pathname); return; }
    sendFile(res, f);
});

// long benches leave connections idle for seconds; outlive the browser's socket reuse
server.keepAliveTimeout = 120_000;
server.headersTimeout = 125_000;
server.on('clientError', (err, socket) => {
    // the browser tree is killed with sockets still open
    if (err.code !== 'ECONNRESET') console.error(`[SERVER] client error: ${err.message}`);
    socket.destroy();
});

server.listen(port, '127.0.0.1', () => {
    const q = new URLSearchParams({ args: benchArgs, autorun: '1' });
    if (out) q.set('out', out);
    for (const pair of envPairs) q.append('env', pair);
    const target = `http://127.0.0.1:${server.address().port}/index.html?${q}`;
    console.log(`[SERVER] ${target}`);
    if (noLaunch) {
        console.log('[SERVER] --no-launch: open the URL above in a browser.');
        return;
    }
    const candidates = [
        'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe',
        'C:\\Program Files\\Microsoft\\Edge\\Application\\msedge.exe',
        'C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe',
        '/usr/bin/microsoft-edge', '/usr/bin/google-chrome', '/usr/bin/chromium',
        '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    ];
    const exe = candidates.find(p => fs.existsSync(p)) || 'msedge';
    // fresh profile per run: a running (or leftover) Edge on a shared profile would swallow the launch
    profile = fs.mkdtempSync(path.join(os.tmpdir(), 'gcdr-wasmbench-'));
    const args = [`--user-data-dir=${profile}`, '--no-first-run', '--no-default-browser-check'];
    if (headless) args.push('--headless=new', '--disable-gpu', '--log-level=3');
    args.push(target);
    console.log(`[SERVER] launching ${path.basename(exe)} (${headless ? 'headless' : 'windowed'})`);
    browser = spawn(exe, args, { stdio: 'ignore' });
    browser.on('error', e => console.error(`[SERVER] cannot launch browser (${e.message}); open the URL above manually.`));
    if (headless && timeoutSec > 0) setTimeout(() => {
        console.error(`[SERVER] no result after ${timeoutSec} s, giving up`);
        shutdown(124);
    }, timeoutSec * 1000).unref();
});
