// Minimal static server for the built demo, so it can be shared over a LAN or Tailscale
// without keeping Trunk's file watcher alive.
//
// Usage, from examples/web after `trunk build --release`:
//   node scripts/serve.cjs            # http://0.0.0.0:8080
//   PORT=9000 node scripts/serve.cjs  # custom port
const http = require("node:http");
const fs = require("node:fs");
const path = require("node:path");

const root = path.resolve(__dirname, "..", "dist");
const port = Number(process.env.PORT || 8080);
const host = process.env.HOST || "0.0.0.0";

const types = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".wasm": "application/wasm",
  ".bpk": "application/octet-stream",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

const server = http.createServer((request, response) => {
  const url = new URL(request.url, `http://${request.headers.host}`);
  const pathname = decodeURIComponent(url.pathname);
  const relative = pathname === "/" ? "/index.html" : pathname;
  const file = path.resolve(root, `.${relative}`);

  if (file !== root && !file.startsWith(`${root}${path.sep}`)) {
    response.writeHead(403);
    response.end("forbidden");
    return;
  }

  fs.readFile(file, (error, data) => {
    if (error) {
      response.writeHead(404);
      response.end("not found");
      return;
    }
    response.writeHead(200, {
      "content-type": types[path.extname(file)] || "application/octet-stream",
      "cache-control": "no-cache",
      "content-length": data.length,
    });
    response.end(data);
  });
});

server.listen(port, host, () => {
  console.log(`serving ${root} at http://${host}:${port}`);
});
