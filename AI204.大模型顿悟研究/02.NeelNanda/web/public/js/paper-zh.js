/* 段落中英对照：给每个段落 / 图注 / 标题的末尾加一个「T」小按钮，点一下就把这一段换成中文译文。
 *
 * 译文来自 /assets/paper-zh.json（由 tools/translate_paper.py 用本地 MT 模型生成），
 * **按 DOM 顺序对齐**：这里用下面的选择器顺序取下标（段落 p.ltx_p、图注 figcaption.ltx_caption、
 * 标题 h1…h6.ltx_title），与 Python 侧 iter_blocks() 的文档顺序一致，所以不依赖元素 id。
 *
 * 译文里公式（<math>）、引用（<cite>）、图表编号等都是原样保留的 HTML，直接整段替换即可；
 * 再点一次切回英文原文（原文在首次切换时缓存下来）。
 * 第一次点击时才去取 JSON；取到后会把「没有译文」的那些按钮去掉，避免留下点了没反应的图标。
 */
(function () {
  "use strict";

  var SEL = ["p.ltx_p", "figcaption.ltx_caption",
             "h1.ltx_title", "h2.ltx_title", "h3.ltx_title",
             "h4.ltx_title", "h5.ltx_title", "h6.ltx_title"].join(",");
  var blocks = null;                 // 译文数组（按 DOM 顺序）
  var loading = null;                // 取数中的 Promise，避免重复请求
  var originals = new WeakMap();     // 元素 → 英文原文 HTML

  function loadBlocks() {
    if (!loading) {
      loading = fetch("/assets/paper-zh.json", { cache: "no-cache" })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (d) {
          blocks = (d && d.blocks) || [];
          pruneButtons();
          return blocks;
        })
        .catch(function (e) {
          loading = null;                                  // 允许下次点击重试
          throw e;
        });
    }
    return loading;
  }

  /** 没有译文的段落，把按钮去掉 */
  function pruneButtons() {
    document.querySelectorAll(".zh-tog").forEach(function (btn) {
      var i = Number(btn.dataset.idx);
      if (!blocks[i]) btn.remove();
    });
  }

  function buttonsOf(el) {
    return el.querySelector(":scope > .zh-tog");
  }

  function toggle(el, btn) {
    var i = Number(btn.dataset.idx);

    if (btn.classList.contains("on")) {                   // —— 切回英文 ——
      var en = originals.get(el) || "";
      el.innerHTML = en;
      el.classList.remove("zh-on");
      el.removeAttribute("lang");
      btn.classList.remove("on");
      btn.title = "显示这一段的中文翻译";
      el.appendChild(btn);
      return;
    }

    originals.set(el, el.innerHTML.replace(/<button[^>]*class="zh-tog"[^>]*>.*?<\/button>/, ""));
    btn.disabled = true;
    btn.textContent = "…";
    loadBlocks().then(function (list) {
      var zh = list[i];
      if (!zh) throw new Error("这一段还没有译文");
      el.innerHTML = zh;
      el.lang = "zh-CN";
      el.classList.add("zh-on");
      btn.classList.add("on");
      btn.title = "显示英文原文";
      el.appendChild(btn);
    }).catch(function (e) {
      alert("取中文译文失败：" + e.message + "\n（先跑 tools/translate_paper.py 生成 assets/paper-zh.json）");
    }).then(function () {                                  // finally（IE 兼容写法无所谓）
      btn.disabled = false;
      btn.textContent = "T";
    });
  }

  function decorate() {
    var nodes = document.querySelectorAll(SEL);
    for (var i = 0; i < nodes.length; i++) {
      var el = nodes[i];
      if (buttonsOf(el)) continue;                         // 已经有按钮
      var btn = document.createElement("button");
      btn.type = "button";
      btn.className = "zh-tog";
      btn.dataset.idx = String(i);
      btn.textContent = "T";
      btn.title = "显示这一段的中文翻译";
      btn.addEventListener("click", (function (node, b) {
        return function (ev) { ev.preventDefault(); ev.stopPropagation(); toggle(node, b); };
      })(el, btn));
      el.appendChild(btn);
    }
  }

  function start() {
    decorate();
    // 页面空闲时先把译文取回来（不阻塞阅读），顺便把没有译文的按钮清掉
    var go = function () { loadBlocks().catch(function () { /* 静默：点按钮时再报 */ }); };
    if (window.requestIdleCallback) window.requestIdleCallback(go, { timeout: 2500 });
    else setTimeout(go, 800);
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
