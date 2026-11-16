// Стежка: the plan list and the plan page. No build step, no framework.
// Every piece of data goes into the page as TEXT (textContent / text nodes), never as HTML: names come from
// OpenStreetMap and from a model, so they must not be able to inject markup. tests/test_web.py checks that
// this file never uses innerHTML / outerHTML / insertAdjacentHTML / document.write.

const app = document.getElementById("app");
const live = document.getElementById("live");
const SVG = "http://www.w3.org/2000/svg";

/* ---------- tiny DOM helpers ---------- */
function h(tag, props = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "text") el.textContent = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) if (kid !== null && kid !== undefined && kid !== false)
    el.append(kid instanceof Node ? kid : String(kid));
  return el;
}
function s(tag, attrs = {}, text) {
  const el = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}
const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const store = {
  get(key, fallback) { try { const v = localStorage.getItem(key); return v === null ? fallback : JSON.parse(v); } catch { return fallback; } },
  set(key, value) { try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* private mode: fine */ } },
};
const say = text => { live.textContent = ""; setTimeout(() => { live.textContent = text; }, 30); };

/* ---------- theme ---------- */
const root = document.documentElement;
const savedTheme = store.get("theme", null);
root.dataset.theme = savedTheme || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
document.getElementById("theme").addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  store.set("theme", root.dataset.theme);
  if (current.redraw) current.redraw();
});

/* ---------- router: / and /plan/<id> ---------- */
const current = { redraw: null, map: null };
async function getJSON(url) {
  const r = await fetch(url, { headers: { accept: "application/json" } });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.error || "Не вдалося завантажити дані.");
  return body;
}
async function route() {
  if (current.map) { current.map.remove(); current.map = null; }
  current.redraw = null;
  const m = location.pathname.match(/^\/plan\/([0-9a-f-]+)$/);
  app.replaceChildren(h("p", { class: "loading", text: "Завантажую…" }));
  try {
    if (m) renderPlan(await getJSON(`/api/plans/${m[1]}`));
    else renderList(await getJSON("/api/plans"));
  } catch (err) {
    app.replaceChildren(h("div", { class: "card error" },
      h("h1", { text: "Цього плану не знайдено" }), h("p", { text: err.message }),
      h("a", { class: "btn primary", href: "/", "data-nav": true, text: "← Усі плани" })));
    document.title = "Стежка";
  }
  app.focus({ preventScroll: true });
  window.scrollTo(0, 0);
}
document.addEventListener("click", e => {
  const a = e.target.closest("a[data-nav]");
  if (!a || e.metaKey || e.ctrlKey || e.shiftKey) return;
  e.preventDefault();
  history.pushState(null, "", a.getAttribute("href"));
  route();
});
window.addEventListener("popstate", route);

/* ---------- the list of saved plans ---------- */
function renderList(data) {
  document.title = "Стежка — плани мандрівок";
  const page = h("div", { class: "list-page" },
    h("section", { class: "hero" }, h("h1", { text: "Плани мандрівок" }),
      h("p", { text: "Кожен прогін агента зберігає план тут. Відкрийте його, щоб побачити карту, профіль висот, погоду й запаси." })));
  if (!data.plans.length) {
    page.append(h("div", { class: "card empty" },
      h("h2", { text: "Ще немає жодного плану" }),
      h("p", {}, "Запустіть агента в терміналі, наприклад ", h("code", { text: "uv run plast-agent --chat" }),
        ", і план з'явиться тут після оновлення сторінки.")));
  } else {
    page.append(h("div", { class: "plans" }, data.plans.map(p => h("a", { class: "card plan-row", href: `/plan/${p.id}`, "data-nav": true },
      h("div", {}, h("span", { class: `badge ${p.status.kind}`, text: p.status.label }), h("h3", { text: p.title }),
        h("div", { class: "sub", text: [p.request && `запит «${p.request}»`, ...p.facts, p.created].filter(Boolean).join(" · ") })),
      h("span", { class: "go", "aria-hidden": "true", text: "→" })))));
  }
  app.replaceChildren(page);
}

/* ---------- the plan page ---------- */
function renderPlan(v) {
  document.title = `${v.title} — Стежка`;
  const L = v.links;
  const page = h("article", { class: "plan" });
  page.append(h("nav", { class: "crumbs" }, h("a", { href: "/", "data-nav": true, text: "← Усі плани" })));

  page.append(h("header", { class: "head" },
    h("div", {}, h("span", { class: `badge ${v.status.kind}`, text: v.status.label }), h("h1", { text: v.title }),
      h("div", { class: "meta", text: v.meta })),
    h("span", { class: "spacer" }),
    h("div", { class: "actions" },
      L.mapy_url && h("a", { class: "btn", href: L.mapy_url, target: "_blank", rel: "noopener noreferrer", text: "Mapy.com ↗" }),
      L.gpx_url && h("a", { class: "btn", href: L.gpx_url, download: true, text: "⬇ Трек GPX" }),
      L.obsidian_url && h("a", { class: "btn", href: L.obsidian_url, text: "Відкрити в Obsidian" }),
      h("button", { class: "btn", type: "button", onclick: () => window.print(), text: "🖨 Друк" }))));

  page.append(h("section", { class: "card conclusion", "aria-label": "Висновок" },
    h("h2", { class: "h", text: "Висновок" }), h("p", { text: v.conclusion }),
    v.notes.length ? h("div", { class: "notes" }, v.notes.map(n => callout("info", "ℹ️", n))) : null));

  if (v.map) {
    const mapBox = h("div", { class: "card mapwrap" });
    page.append(h("div", { class: "grid" }, mapBox, h("div", { class: "side" },
      h("section", { class: "card" }, h("h2", { class: "h", text: "Коротко" }),
        h("div", { class: "kv" }, v.stats.map(x => h("div", {}, h("b", { class: "num", text: x.value }), h("span", { text: x.label })))),
        h("div", { class: "legend" }, v.days.map(d => h("span", {}, h("i", { class: `c${d.color}` }), `День ${d.n}`)),
          v.map.marks.some(m => m.kind === "place") ? h("span", {}, h("i", { class: "place" }), "Бажане місце") : null)),
      safetyCard(v))));
    page.append(mapLinks(L));
    const profile = h("section", { class: "section card profile", "aria-label": "Профіль висот" });
    page.append(profile);
    app.replaceChildren(page);   // the map and the profile need their real size
    const map = drawMap(mapBox, v);
    drawProfile(profile, v, map);
  } else {
    page.append(safetyCard(v));
  }

  if (v.days.length) page.append(h("section", { class: "section" }, h("h2", { class: "h", text: "По днях" }),
    h("div", { class: "days" }, v.days.map(dayCard))));
  page.append(weatherSection(v.weather));

  if (v.supplies || v.documents.length) {
    page.append(h("section", { class: "section two" },
      v.supplies ? h("div", { class: "card box" }, h("h2", { class: "h", text: "Запаси й спорядження" }),
        h("div", { class: "tiles" }, v.supplies.tiles.map(t => h("div", {}, h("b", { class: "num", text: t.value }), h("span", { text: t.label })))),
        h("ul", { class: "lines" }, v.supplies.lines.map(t => h("li", { text: t }))),
        v.supplies.warnings.map(t => callout("warn", "⚠️", t)),
        v.supplies.gear.length ? [h("div", { class: "sub-h", text: "Додати до спорядження" }), checklist(`${v.id}-gear`, v.supplies.gear)] : null) : null,
      v.documents.length ? h("div", { class: "card box" }, h("h2", { class: "h", text: "Документи" }), checklist(`${v.id}-docs`, v.documents)) : null));
  }

  if (v.understood.length) page.append(h("section", { class: "section card box" },
    h("h2", { class: "h", text: "Що агент зрозумів із запиту" }),
    h("div", { class: "facts" }, v.understood.map(f => h("div", { class: "fact" }, h("b", { text: f.label }), h("span", { text: f.value }))))));

  if (v.alternatives.length) page.append(h("section", { class: "section" },
    h("h2", { class: "h", text: "Інші варіанти маршруту" }),
    h("div", { class: "alts" }, v.alternatives.map(a => h("div", { class: "card alt" },
      h("b", {}, h("a", { href: a.osm_url, target: "_blank", rel: "noopener noreferrer", text: a.name })),
      h("span", { class: "num", text: a.facts }), a.places ? h("span", { text: a.places }) : null))),
    h("p", { class: "alt-note", text: "Щоб спланувати інший варіант, допишіть його назву в запит і запустіть агента знову." })));

  const dev = h("div", { class: "dev", hidden: true, text: techText(v.tech) });
  page.append(h("details", { class: "section card how" }, h("summary", { text: "Як агент дійшов до цього плану" }),
    h("ol", { class: "timeline" }, v.how.map(t => h("li", { text: t }))),
    h("label", { class: "toggle" }, h("input", { type: "checkbox", onchange: e => { dev.hidden = !e.target.checked; } }),
      "Показати технічні деталі (виклики MCP, модель)"), dev));
  page.append(h("p", { class: "foot", text: v.sources }));
  if (!v.map) app.replaceChildren(page);
}

function callout(kind, icon, text, small) {
  return h("div", { class: `callout ${kind}` }, h("span", { class: "ic", "aria-hidden": "true", text: icon }),
    h("span", {}, text, small ? h("small", { text: small }) : null));
}
function safetyCard(v) {
  const items = v.safety.length ? v.safety.map(x => callout(x.level === "danger" ? "danger" : "warn", x.level === "danger" ? "⛔" : "⚠️", x.text, x.advice))
    : [callout("info", "✅", "Зауважень до безпеки немає.")];
  return h("section", { class: "card" }, h("h2", { class: "h", text: "Головне для безпеки" }), h("div", { class: "stack" }, items));
}
function mapLinks(L) {
  const links = [];
  L.mapy_days.forEach(d => links.push(h("a", { href: d.url, target: "_blank", rel: "noopener noreferrer", text: `${d.label} на Mapy.com ↗` })));
  L.side_trips.forEach(d => links.push(h("a", { href: d.url, target: "_blank", rel: "noopener noreferrer", text: `⭐ ${d.label} ↗` })));
  if (L.osm_url) links.push(h("a", { href: L.osm_url, target: "_blank", rel: "noopener noreferrer", text: "Маршрут на OpenStreetMap ↗" }));
  return links.length ? h("nav", { class: "maplinks", "aria-label": "Маршрут на інших картах" }, links) : h("span");
}
function checklist(key, items) {
  const saved = store.get(`ticks-${key}`, {});
  return h("div", {}, items.map((t, n) => {
    const row = h("label", { class: `check ${saved[n] ? "done" : ""}` });
    row.append(h("input", { type: "checkbox", checked: !!saved[n], onchange: e => {
      saved[n] = e.target.checked; row.classList.toggle("done", e.target.checked); store.set(`ticks-${key}`, saved);
    } }), h("span", { text: t }));
    return row;
  }));
}
const pct = mins => `${Math.min(100, Math.max(0, (mins - 360) / (1080 - 360) * 100)).toFixed(1)}%`;   // 06:00..18:00
function dayCard(d) {
  const card = h("section", { class: `card day c${d.color}`, "aria-label": `День ${d.n}` },
    h("h3", { text: `День ${d.n}: ${d.title}` }),
    h("div", { class: "sub", text: [d.date, d.night && `ночівля: ${d.night}`].filter(Boolean).join(" · ") }),
    h("div", { class: "nums num" }, d.numbers.map(x => h("div", {}, h("b", { text: x.value }), h("span", { text: x.label })))));
  if (d.start_min !== null && d.arrival_min !== null && d.sunset_min !== null) {
    const walk = h("div", { class: `walk ${d.late ? "late" : ""}` });
    walk.style.left = pct(d.start_min);
    walk.style.width = `calc(${pct(d.arrival_min)} - ${pct(d.start_min)})`;
    const sun = h("div", { class: "sun" }); sun.style.left = pct(d.sunset_min);
    const sunLab = h("span", { class: "lab above", text: `захід ${d.sunset}` }); sunLab.style.left = pct(d.sunset_min);
    const arrLab = h("span", { class: "lab below", text: `прихід ${d.arrival}` }); arrLab.style.left = pct(d.arrival_min);
    card.append(h("div", { class: "daylight", role: "img",
      "aria-label": `Вихід о ${d.start}, прихід о ${d.arrival}, захід сонця о ${d.sunset}` },
      h("div", { class: "track" }), walk, sun, sunLab, arrLab),
      h("div", { class: "scale", "aria-hidden": "true" }, h("span", { text: "06:00" }), h("span", { text: "12:00" }), h("span", { text: "18:00" })));
    if (d.late) card.append(h("div", { class: "daynote", text: "Прихід пізніше, ніж за 3 год до заходу сонця." }));
  }
  return card;
}
const weatherIcon = t => /гроз/.test(t) ? "⛈" : /сніг/.test(t) ? "🌨" : /дощ|злива|мряк/.test(t) ? "🌧" : /туман/.test(t) ? "🌫" : /ясно/.test(t) ? "☀️" : "⛅";
function weatherSection(w) {
  if (w.state === "none") return h("span");
  const sec = h("section", { class: "section" }, h("h2", { class: "h" }, "Погода ", w.risk ? h("span", { class: `badge ${w.risk.kind}`, text: w.risk.label }) : null));
  if (w.state === "missing") { sec.append(callout("info", "🗓", w.note)); return sec; }
  sec.append(h("div", { class: "weather" }, w.days.map(d => h("div", { class: "card wday" },
    h("div", { class: "big", "aria-hidden": "true", text: weatherIcon(d.text) }),
    h("div", { class: "t num", text: d.temp }),
    h("div", { class: "d", text: `${d.date} · ${d.text} · ${d.details}` }),
    h("div", { class: "advice" }, h("span", {}, h("span", { class: `badge ${d.risk.kind}`, text: d.risk.label })),
      d.advice.map(a => h("span", { text: a })))))));
  sec.append(h("p", { class: "wnote", text: w.note }));
  return sec;
}
function techText(t) {
  return [`Модель: ${t.model}`, ...t.switches.map(x => `↪ ${x}`), "", "Виклики MCP:",
    ...(t.calls.length ? t.calls.map(c => `→ ${c}`) : ["(не записано)"]), "", "Спроби плану:", ...t.attempts].join("\n");
}

/* ---------- map (Leaflet + OpenTopoMap); without internet: the route picture instead ---------- */
function drawMap(box, v) {
  const fallback = note => {
    box.replaceChildren(h("img", { class: "fallback", src: v.links.picture_url, alt: `Малюнок маршруту ${v.title}` }),
      h("div", { class: "mapnote", text: note }));
    return null;
  };
  if (!window.L) return fallback("Карта не завантажилась (немає інтернету?): показую малюнок маршруту.");
  const el = h("div", { class: "map" });
  box.append(el);
  const map = L.map(el, { scrollWheelZoom: false });
  current.map = map;
  let loaded = 0, failed = 0;
  L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
    subdomains: "abc", maxZoom: 17,
    attribution: "© OpenStreetMap contributors, SRTM · стиль карти © OpenTopoMap (CC-BY-SA)",
  }).on("tileload", () => loaded++).on("tileerror", () => {
    if (++failed >= 6 && !loaded) { map.remove(); current.map = null; fallback("Тайли карти недоступні: показую малюнок маршруту."); }
  }).addTo(map);
  const colors = [css("--d0"), css("--d1"), css("--d2")];
  const all = [];
  v.map.tracks.forEach((t, i) => {
    all.push(...t);
    L.polyline(t, { color: "#fff", weight: 8, opacity: .85 }).addTo(map);
    L.polyline(t, { color: colors[i % 3], weight: 5 }).addTo(map);
  });
  const glyph = { start: "▶", finish: "⚑", night: "⛺", place: "★" };
  v.map.marks.forEach(m => {
    const icon = L.divIcon({ className: "", iconSize: [28, 28], iconAnchor: [14, 14], html: h("div", { class: `pin ${m.kind}`, text: glyph[m.kind] }) });
    L.marker([m.lat, m.lon], { icon, keyboard: true, title: m.label, alt: m.label }).addTo(map)
      .bindTooltip(h("span", { text: m.label }), { permanent: m.kind === "place", direction: "right" });
    all.push([m.lat, m.lon]);
  });
  const bounds = L.latLngBounds(all).pad(0.1);
  map.fitBounds(bounds);
  setTimeout(() => { map.invalidateSize(); map.fitBounds(bounds); }, 80);
  return map;
}

/* ---------- elevation profile: hover or arrow keys show km, height and slope, and the point on the map ---------- */
function drawProfile(box, v, map) {
  const pts = v.map.profile;   // [km, lat, lon, elevation, colour index]
  const svg = s("svg", { role: "img", tabindex: "0", "aria-label": "Профіль висот. Стрілки ліворуч і праворуч рухають точку." });
  const tip = h("div", { class: "tip", "aria-hidden": "true" });
  box.replaceChildren(h("h2", { class: "h" }, "Профіль висот ", h("span", { class: "hint", text: "· наведіть або фокус і стрілки: місце на карті" })),
    svg, tip, h("div", { class: "legend" },
      h("span", {}, h("i", { class: "g1" }), "до 8 %"), h("span", {}, h("i", { class: "g2" }), "8–15 %"),
      h("span", {}, h("i", { class: "g3" }), "понад 15 % — дуже круто")));
  const hover = window.L && map ? L.circleMarker([0, 0], { radius: 7, color: "#fff", weight: 3, fillColor: "#111", fillOpacity: 1 }) : null;
  let X, Y, W, H, at = -1;

  function draw() {
    W = Math.max(320, Math.round(svg.getBoundingClientRect().width)); H = 220;
    const L0 = 46, R0 = 12, T0 = 16, B0 = 26;
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    const k0 = pts[0][0], k1 = pts[pts.length - 1][0];
    const e0 = Math.floor(Math.min(...pts.map(q => q[3])) / 100) * 100 - 50;
    const e1 = Math.ceil(Math.max(...pts.map(q => q[3])) / 100) * 100 + 50;
    X = k => L0 + (k - k0) / Math.max(0.001, k1 - k0) * (W - L0 - R0);
    Y = e => T0 + (1 - (e - e0) / (e1 - e0)) * (H - T0 - B0);
    const colors = [css("--d0"), css("--d1"), css("--d2")];
    const kids = [];
    const stepE = e1 - e0 > 1200 ? 400 : 200;
    for (let e = Math.ceil(e0 / stepE) * stepE; e <= e1; e += stepE)
      kids.push(s("line", { class: "grid-line", x1: L0, x2: W - R0, y1: Y(e), y2: Y(e) }), s("text", { x: L0 - 8, y: Y(e) + 4, "text-anchor": "end" }, `${e}`));
    const stepK = Math.max(1, Math.round((k1 - k0) / (W / 90)));
    for (let k = Math.ceil(k0); k <= k1; k += stepK) kids.push(s("text", { x: X(k), y: H - 6, "text-anchor": "middle" }, `${k} км`));
    v.days.forEach((d, i) => {
      const mine = pts.filter((q, n) => dayOf(n) === i);
      if (!mine.length) return;
      const poly = `${X(mine[0][0])},${Y(e0)} ${mine.map(q => `${X(q[0])},${Y(q[3])}`).join(" ")} ${X(mine[mine.length - 1][0])},${Y(e0)}`;
      kids.push(s("polygon", { points: poly, fill: colors[d.color], opacity: ".12" }),
        s("text", { class: "day-label", x: (X(mine[0][0]) + X(mine[mine.length - 1][0])) / 2, y: Y(e0) - 8, "text-anchor": "middle", fill: colors[d.color] }, `день ${d.n}`));
    });
    for (let j = 1; j < pts.length; j++) {
      const a = pts[j - 1], b = pts[j];
      if (b[0] <= a[0]) continue;
      const g = Math.abs(b[3] - a[3]) / ((b[0] - a[0]) * 1000);
      kids.push(s("line", { x1: X(a[0]), y1: Y(a[3]), x2: X(b[0]), y2: Y(b[3]), stroke: g < .08 ? "#16a34a" : g < .15 ? "#ea580c" : "#dc2626", "stroke-width": 3, "stroke-linecap": "round" }));
    }
    v.map.marks.filter(m => m.kind === "night").forEach(m => kids.push(
      s("line", { class: "night-line", x1: X(m.km), x2: X(m.km), y1: T0, y2: Y(e0) }),
      s("text", { x: X(m.km), y: T0 - 3, "text-anchor": "middle" }, "⛺ ночівля")));
    kids.push(s("line", { class: "cursor", id: "cursor", x1: -10, x2: -10, y1: T0, y2: Y(e0) }));
    svg.replaceChildren(...kids);
    if (at >= 0) show(at);
  }
  // which day a profile point belongs to: points are in order, a new day starts where the colour changes
  const dayStarts = [0];
  for (let n = 1; n < pts.length; n++) if (pts[n][4] !== pts[n - 1][4]) dayStarts.push(n);
  function dayOf(n) { let d = 0; dayStarts.forEach((st, i) => { if (n >= st) d = i; }); return d; }

  function show(n) {
    at = Math.max(0, Math.min(pts.length - 1, n));
    const q = pts[at], a = pts[Math.max(0, at - 1)], b = pts[Math.min(pts.length - 1, at + 1)];
    const slope = Math.round((b[3] - a[3]) / Math.max(1, (b[0] - a[0]) * 1000) * 100);
    const cur = svg.querySelector("#cursor");
    cur.setAttribute("x1", X(q[0])); cur.setAttribute("x2", X(q[0]));
    const r = svg.getBoundingClientRect();
    tip.style.left = `${X(q[0]) / W * r.width + 14}px`;
    tip.style.top = `${Y(q[3]) / H * r.height + svg.offsetTop}px`;
    const text = `${q[0].toFixed(1)} км · ${Math.round(q[3])} м · ${slope > 0 ? "↗" : slope < 0 ? "↘" : "→"} ${Math.abs(slope)} %`;
    tip.textContent = text; tip.classList.add("on");
    if (hover && current.map) hover.setLatLng([q[1], q[2]]).addTo(current.map);
    return text;
  }
  function hide() { tip.classList.remove("on"); if (hover && current.map) current.map.removeLayer(hover); const c = svg.querySelector("#cursor"); if (c) { c.setAttribute("x1", -10); c.setAttribute("x2", -10); } }
  function nearest(clientX) {
    const r = svg.getBoundingClientRect();
    const x = (clientX - r.left) / r.width * W;
    let best = 0;
    pts.forEach((q, n) => { if (Math.abs(X(q[0]) - x) < Math.abs(X(pts[best][0]) - x)) best = n; });
    return best;
  }
  svg.addEventListener("pointermove", e => show(nearest(e.clientX)));
  svg.addEventListener("pointerleave", hide);
  svg.addEventListener("blur", hide);
  svg.addEventListener("keydown", e => {
    const step = e.shiftKey ? 10 : 1;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      say(show((at < 0 ? 0 : at) + (e.key === "ArrowRight" ? step : -step)));
    }
  });
  draw();
  let timer;
  current.redraw = draw;
  window.addEventListener("resize", () => { clearTimeout(timer); timer = setTimeout(() => current.redraw === draw && draw(), 150); });
}

route();
