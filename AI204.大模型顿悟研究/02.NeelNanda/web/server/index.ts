/**
 * Grokking 论文中文站点 —— 后端服务器（可直接运行 .ts 源码）
 *
 * 运行方式（Node.js >= 24，无需编译、无需 npm install）：
 *   cd web/server
 *   npm start          # 等效于 node index.ts
 *   或
 *   node index.ts
 *
 * 提供的能力：
 *   GET /             -> public/index.html（首页）
 *   GET /paper-zh.html-> public/paper-zh.html（论文中文翻译页）
 *   GET /paper.pdf    -> docs/ 下的原始英文 PDF
 *   GET /api/meta     -> JSON 元信息（论文信息、Node 版本、运行时长…）
 *   其余路径          -> public/ 下的静态资源（css / js / images）
 */
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { readFile, readdir, stat } from "node:fs/promises";
import { extname, isAbsolute, join, normalize, relative, resolve } from "node:path";
import { fileURLToPath } from "node:url";

// ---------------------------------------------------------------------------
// 目录布局（以 web 为站点根）
//   server/index.ts           本文件
//   public/                   前端静态资源
//   上级目录/docs/             原始论文 PDF
// ---------------------------------------------------------------------------
const __dirname = fileURLToPath(new URL(".", import.meta.url));
const WEB_ROOT = resolve(__dirname, "..");           // .../web
const PUBLIC_DIR = join(WEB_ROOT, "public");
const DOCS_DIR = resolve(WEB_ROOT, "..", "docs");    // .../docs

const PORT: number = Number(process.env.PORT ?? 3000);
const HOST: string = process.env.HOST ?? "127.0.0.1";
const startedAt = Date.now();

const MIME: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".mjs": "text/javascript; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg",
  ".gif": "image/gif",
  ".webp": "image/webp",
  ".svg": "image/svg+xml",
  ".pdf": "application/pdf",
  ".ico": "image/x-icon",
  ".woff2": "font/woff2",
  ".txt": "text/plain; charset=utf-8",
};

function contentType(path: string): string {
  return MIME[extname(path).toLowerCase()] ?? "application/octet-stream";
}

function send(res: ServerResponse, code: number, body: string | Buffer, type?: string): void {
  res.writeHead(code, { "Content-Type": type ?? "text/plain; charset=utf-8" });
  res.end(body);
}

function sendJson(res: ServerResponse, code: number, obj: unknown): void {
  send(res, code, JSON.stringify(obj, null, 2), "application/json; charset=utf-8");
}

async function readIfExists(filePath: string): Promise<Buffer | null> {
  try {
    const info = await stat(filePath);
    if (!info.isFile()) return null;
    return await readFile(filePath);
  } catch {
    return null;
  }
}

async function serveFile(res: ServerResponse, filePath: string): Promise<void> {
  const data = await readIfExists(filePath);
  if (data === null) {
    send(res, 404, "404 Not Found");
    return;
  }
  res.writeHead(200, { "Content-Type": contentType(filePath) });
  res.end(data);
}

/** 防止路径穿越：把解码后的请求路径规整后确保落在目标根目录内 */
function safeJoin(root: string, urlPath: string): string | null {
  const decoded = decodeURIComponent(urlPath).replace(/^\/+/, "");
  const target = normalize(join(root, decoded));
  const rel = relative(root, target);
  if (rel === "" || isAbsolute(rel) || rel.startsWith("..")) return null;
  return target;
}

async function findFirstPdf(): Promise<string | null> {
  try {
    const files = await readdir(DOCS_DIR);
    const pdf = files.find((f) => f.toLowerCase().endsWith(".pdf"));
    return pdf ? join(DOCS_DIR, pdf) : null;
  } catch {
    return null;
  }
}

async function handleApiMeta(res: ServerResponse): Promise<void> {
  const pdfPath = await findFirstPdf();
  const info = await (pdfPath ? stat(pdfPath) : null);
  const html = await readIfExists(join(PUBLIC_DIR, "paper-zh.html"));
  sendJson(res, 200, {
    ok: true,
    server: {
      node: process.version,
      startedAt: new Date(startedAt).toISOString(),
      uptimeSeconds: Math.round((Date.now() - startedAt) / 1000),
    },
    paper: {
      file: pdfPath ? pdfPath.split(/[\\/]/).pop() : null,
      bytes: info?.size ?? 0,
      chineseTranslationAvailable: html !== null,
      originalTitle: "Progress measures for grokking via mechanistic interpretability",
      originalTitleZh: "通过机械可解释性度量「顿悟」现象(grokking)的进展(progress measures)",
    },
    folders: { docs: DOCS_DIR, public: PUBLIC_DIR, server: __dirname },
  });
}

const server = createServer(async (req: IncomingMessage, res: ServerResponse) => {
  try {
    const method = req.method ?? "GET";
    const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "localhost"}`);
    let pathname = url.pathname;

    if (method === "GET" && pathname === "/") pathname = "/index.html";

    if (method === "GET" && pathname === "/api/meta") {
      await handleApiMeta(res);
      return;
    }

    if (method === "GET" && pathname === "/paper.pdf") {
      const pdf = await findFirstPdf();
      if (pdf) {
        const data = await readIfExists(pdf);
        if (data) {
          res.writeHead(200, { "Content-Type": "application/pdf" });
          res.end(data);
          return;
        }
      }
      send(res, 404, "paper.pdf not found in docs/");
      return;
    }

    // 其余全部按 public/ 静态资源处理
    const target = safeJoin(PUBLIC_DIR, pathname);
    if (!target) {
      send(res, 403, "Forbidden");
      return;
    }
    await serveFile(res, target);
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    send(res, 500, `Internal Server Error: ${msg}`);
  }
});

server.listen(PORT, HOST, () => {
  console.log(`[grokking-web] 服务已启动  http://${HOST}:${PORT}`);
  console.log(`[grokking-web] public = ${PUBLIC_DIR}`);
  console.log(`[grokking-web] docs   = ${DOCS_DIR}`);
});
