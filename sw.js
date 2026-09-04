/* AI 应用工程师面试备考 Service Worker v1 —— 推广自标杆仓库 architect-exam-learning（OPT-01/OPT-02）
   策略（同标杆 docs/perf/report.md 实测方案）：
   1. 安装期仅预缓存页面外壳，内容页首次访问时运行时缓存；
   2. HTML 导航一律 network-first：刷新即最新，离线才回退缓存；
   3. 同源静态资源 cache-first + 后台更新（stale-while-revalidate）。
   与标杆差异：本站无音频，省略音频分支；版本化 query 资源纳入 SWR（标杆对 query 直连）。 */
var SHELL_CACHE = 'llm-notes-shell-v1';
var PAGE_CACHE = 'llm-notes-pages-v1';
var SHELL = ["./", "./index.html"];

self.addEventListener('install', function (e) {
  e.waitUntil(caches.open(SHELL_CACHE).then(function (c) { return c.addAll(SHELL); }).then(function () { return self.skipWaiting(); }));
});

self.addEventListener('activate', function (e) {
  e.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) {
        return k !== SHELL_CACHE && k !== PAGE_CACHE;
      }).map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

/* 缓存键规范化：同页面不同 query 视为同一资源（标杆 sw.js pageKey 同款） */
function pageKey(req) {
  var url = new URL(req.url);
  url.search = '';
  return url.href;
}

/* OPT-02：导航 network-first，静态资源 cache-first + SWR（标杆 sw.js:97-152 同款） */
self.addEventListener('fetch', function (e) {
  var req = e.request;
  if (req.method !== 'GET') return;
  var url = new URL(req.url);
  if (url.origin !== location.origin) return;

  if (req.mode === 'navigate') {
    var key = pageKey(req);
    e.respondWith(
      fetch(req).then(function (resp) {
        var clone = resp.clone();
        caches.open(PAGE_CACHE).then(function (c) { c.put(key, clone); });
        return resp;
      }).catch(function () {
        return caches.match(key).then(function (hit) {
          return hit || caches.match('./index.html');
        });
      })
    );
    return;
  }

  e.respondWith(
    caches.match(req).then(function (hit) {
      var fetching = fetch(req).then(function (resp) {
        if (resp && resp.ok) {
          var clone = resp.clone();
          caches.open(SHELL_CACHE).then(function (c) { c.put(req, clone); });
        }
        return resp;
      }).catch(function () { return hit; });
      return hit || fetching;
    })
  );
});
