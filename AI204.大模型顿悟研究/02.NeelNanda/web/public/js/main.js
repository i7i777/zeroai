/* Grokking 论文中文站 —— 前端脚本 */
(() => {
  "use strict";

  /** 首页：从后端 /api/meta 拉取状态并写入页脚 */
  async function loadServerMeta() {
    const line = document.getElementById("api-line");
    if (!line) return;
    try {
      const res = await fetch("/api/meta");
      const meta = await res.json();
      const pdf = meta.paper.file ? meta.paper.file : "（未找到 PDF）";
      line.textContent = `后端 Node ${meta.server.node} 已在线 · 运行 ${meta.server.uptimeSeconds}s · 论文：${pdf}`;
      const paperOk = document.getElementById("api-paper-file");
      if (paperOk) paperOk.textContent = pdf;
    } catch {
      line.textContent = "后端 API 不可用（请确认已通过 web/server 目录启动服务）";
    }
  }

  /** 论文页：根据视口滚动高亮当前章节（目录） */
  function initToc() {
    const sideNav = document.getElementById("paper-toc");
    const headings = Array.from(document.querySelectorAll("article.paper h2[id], article.paper h3[id]"));
    if (!sideNav || headings.length === 0) return;

    const links = new Map();
    headings.forEach((h) => {
      if (!h.id) return;
      const link = sideNav.querySelector(`a[href="#${CSS.escape(h.id)}"]`);
      if (link) links.set(h, link);
    });
    if (links.size === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            const link = links.get(entry.target);
            if (!link) return;
            sideNav.querySelectorAll("a.active").forEach((a) => a.classList.remove("active"));
            link.classList.add("active");
            link.scrollIntoView({ block: "nearest" });
          }
        });
      },
      { rootMargin: "-15% 0px -75% 0px" }
    );
    links.forEach((_l, h) => observer.observe(h));
  }

  /** 阅读进度条 + 回到顶部 */
  function initProgress() {
    const bar = document.getElementById("progress");
    const toTop = document.getElementById("toTop");
    if (!bar && !toTop) return;

    const onScroll = () => {
      const doc = document.documentElement;
      const total = doc.scrollHeight - window.innerHeight;
      const ratio = total > 0 ? window.scrollY / total : 0;
      if (bar) bar.style.width = `${ratio * 100}%`;
      if (toTop) toTop.classList.toggle("show", window.scrollY > 600);
    };
    window.addEventListener("scroll", onScroll, { passive: true });
    onScroll();
    if (toTop) {
      toTop.addEventListener("click", () => window.scrollTo({ top: 0, behavior: "smooth" }));
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    loadServerMeta();
    initToc();
    initProgress();
  });
})();
