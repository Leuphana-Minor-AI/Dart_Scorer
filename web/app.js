/* Dart Scorer Frontend: WebSocket-Zustand rendern, Aktionen per REST senden. */
(() => {
  "use strict";

  const SECTORS = [20, 1, 18, 4, 13, 6, 10, 15, 2, 17, 3, 19, 7, 16, 8, 11, 14, 9, 12, 5];
  const R = { bullseye: 6.35, bull: 15.9, trebleIn: 99, trebleOut: 107, doubleIn: 162, doubleOut: 170 };
  const SCALE = 200 / 170; // mm -> SVG-Einheiten

  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => Array.from(document.querySelectorAll(sel));

  let state = null;
  let ws = null;
  let cameraOn = localStorage.getItem("dart.camera") !== "0";
  let correctIndex = null;

  // ---------------------------------------------------------------- Board-SVG
  function polar(rMm, deg) {
    const a = (deg - 90) * Math.PI / 180;
    return [Math.cos(a) * rMm * SCALE, Math.sin(a) * rMm * SCALE];
  }
  function wedge(r1, r2, a1, a2) {
    const [x1, y1] = polar(r2, a1), [x2, y2] = polar(r2, a2);
    const [x3, y3] = polar(r1, a2), [x4, y4] = polar(r1, a1);
    return `M${x1},${y1} A${r2 * SCALE},${r2 * SCALE} 0 0 1 ${x2},${y2} L${x3},${y3} A${r1 * SCALE},${r1 * SCALE} 0 0 0 ${x4},${y4} Z`;
  }
  function buildBoard(svg, clickable) {
    const ns = "http://www.w3.org/2000/svg";
    svg.innerHTML = "";
    const bg = document.createElementNS(ns, "circle");
    bg.setAttribute("r", 228); bg.setAttribute("fill", "#0b0d11");
    svg.appendChild(bg);
    SECTORS.forEach((sec, i) => {
      const a1 = i * 18 - 9, a2 = i * 18 + 9;
      const dark = i % 2 === 0;
      const rings = [
        [R.bull, R.trebleIn, dark ? "#1b1b1b" : "#e9dcc2", 1],
        [R.trebleIn, R.trebleOut, dark ? "#d3232a" : "#1f9d55", 3],
        [R.trebleOut, R.doubleIn, dark ? "#1b1b1b" : "#e9dcc2", 1],
        [R.doubleIn, R.doubleOut, dark ? "#d3232a" : "#1f9d55", 2],
      ];
      rings.forEach(([r1, r2, color, mult]) => {
        const p = document.createElementNS(ns, "path");
        p.setAttribute("d", wedge(r1, r2, a1, a2));
        p.setAttribute("fill", color);
        p.setAttribute("class", "seg");
        p.dataset.field = `${"SDT"[mult - 1]}${sec}`;
        svg.appendChild(p);
      });
      const [nx, ny] = polar(190, i * 18);
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", nx); t.setAttribute("y", ny); t.setAttribute("class", "num");
      t.textContent = sec;
      svg.appendChild(t);
    });
    const bull = document.createElementNS(ns, "circle");
    bull.setAttribute("r", R.bull * SCALE); bull.setAttribute("fill", "#1f9d55"); bull.setAttribute("class", "seg");
    bull.dataset.field = "BULL";
    svg.appendChild(bull);
    const eye = document.createElementNS(ns, "circle");
    eye.setAttribute("r", R.bullseye * SCALE); eye.setAttribute("fill", "#d3232a"); eye.setAttribute("class", "seg");
    eye.dataset.field = "BULLSEYE";
    svg.appendChild(eye);
    const markers = document.createElementNS(ns, "g");
    markers.setAttribute("id", svg.id + "-markers");
    svg.appendChild(markers);
    if (clickable) {
      svg.addEventListener("click", (ev) => {
        const field = ev.target && ev.target.dataset && ev.target.dataset.field;
        if (field) submitCorrection(field);
      });
    }
  }
  function drawMarkers(svg, darts) {
    const g = svg.querySelector("g[id$='-markers']");
    if (!g) return;
    g.innerHTML = "";
    const ns = "http://www.w3.org/2000/svg";
    darts.forEach((d, i) => {
      if (d.x_mm == null || d.y_mm == null) return;
      const c = document.createElementNS(ns, "circle");
      c.setAttribute("cx", d.x_mm * SCALE); c.setAttribute("cy", d.y_mm * SCALE); c.setAttribute("r", 7);
      c.setAttribute("class", "marker" + (d.uncertain ? " uncertain" : ""));
      g.appendChild(c);
      const t = document.createElementNS(ns, "text");
      t.setAttribute("x", d.x_mm * SCALE); t.setAttribute("y", d.y_mm * SCALE + 1);
      t.setAttribute("class", "num"); t.setAttribute("style", "font-size:10px;fill:#000");
      t.textContent = i + 1;
      g.appendChild(t);
    });
  }

  // ---------------------------------------------------------------- API
  async function post(url, body) {
    try {
      const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body || {}) });
      return await r.json();
    } catch (e) {
      console.error(e);
      return { ok: false, error: String(e) };
    }
  }

  // ---------------------------------------------------------------- Setup
  function playerRows() { return $$("#player-list .player-row input").map((i) => i.value.trim()).filter(Boolean); }
  function addPlayerRow(name) {
    const list = $("#player-list");
    if (list.children.length >= 4) return;
    const row = document.createElement("div");
    row.className = "player-row";
    row.innerHTML = `<input type="text" maxlength="20" placeholder="Name"><button class="btn ghost" title="Entfernen">✕</button>`;
    row.querySelector("input").value = name || "";
    row.querySelector("button").addEventListener("click", () => { if (list.children.length > 1) row.remove(); });
    list.appendChild(row);
  }
  function segValue(id) { const b = $(`#${id} .active`); return b ? b.dataset.value : null; }
  function initSegmented() {
    $$(".segmented").forEach((seg) => seg.addEventListener("click", (ev) => {
      const b = ev.target.closest("button"); if (!b) return;
      seg.querySelectorAll("button").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
    }));
  }
  function applySettings(s) {
    if (!s) return;
    const list = $("#player-list");
    if (!list.children.length) (s.players || ["Spieler 1", "Spieler 2"]).forEach(addPlayerRow);
    const setSeg = (id, v) => { $$(`#${id} button`).forEach((b) => b.classList.toggle("active", b.dataset.value == v)); };
    setSeg("seg-score", s.start_score || 501);
    setSeg("seg-double", s.double_out === false ? 0 : 1);
    setSeg("seg-legs", s.legs_to_win || 1);
  }
  async function startGame() {
    const players = playerRows();
    if (!players.length) { $("#setup-hint").textContent = "Mindestens einen Spielernamen eingeben."; return; }
    const res = await post("/api/game/new", {
      players, start_score: Number(segValue("seg-score")), double_out: segValue("seg-double") === "1",
      legs_to_win: Number(segValue("seg-legs")),
    });
    if (!res.ok) $("#setup-hint").textContent = res.error || "Fehler";
  }

  // ---------------------------------------------------------------- Rendering
  function pill(el, text, cls) { el.textContent = text; el.className = "pill " + (cls || ""); }
  function renderDetection(d) {
    if (!d) return;
    const cam = d.camera || {};
    pill($("#pill-camera"), cam.connected ? `Kamera ${cam.index}` : (cam.source === "test_images" ? "Testbilder" : "Keine Kamera"), cam.connected ? "ok" : "bad");
    let calText = d.locked ? "Kalibriert ✓" : (d.searching ? "Board suchen …" : "Kalibriere …");
    if (d.learning_empty) calText = "Leeres Board lernen …";
    else if (d.locked && d.artifacts) calText += ` · ${d.artifacts} Störstelle${d.artifacts > 1 ? "n" : ""} ausgeblendet`;
    pill($("#pill-cal"), calText, d.locked ? "ok" : "warn");
    pill($("#pill-fps"), d.fps ? `${d.fps} FPS` : "", "");
    $("#setup-cal").textContent = calText;
  }
  function renderPlayers(g) {
    const box = $("#players");
    box.innerHTML = "";
    g.players.forEach((p, i) => {
      const el = document.createElement("div");
      el.className = "player" + (i === g.current && !g.over ? " active" : "");
      const st = p.stats || {};
      const shown = i === g.current && !g.over ? g.remaining : p.score;
      const pct = Math.max(0, Math.min(100, 100 * (1 - shown / g.start_score)));
      el.innerHTML = `
        <div class="name"><span>${escapeHtml(p.name)}</span><span class="legs">${"●".repeat(p.legs_won)}${"○".repeat(Math.max(0, g.legs_to_win - p.legs_won))}</span></div>
        <div class="score">${shown}</div>
        <div class="bar"><i style="width:${pct}%"></i></div>
        <div class="meta"><span>Ø <b>${st.average ?? 0}</b></span><span>180er <b>${st.count_180 ?? 0}</b></span><span>Pfeile <b>${p.leg_darts}</b></span></div>`;
      box.appendChild(el);
    });
  }
  function renderTurn(g) {
    const t = g.turn;
    const cur = g.players[g.current];
    $("#turn-player").textContent = g.over ? "Spiel beendet" : `${cur.name} wirft`;
    $("#turn-remaining").textContent = g.remaining;
    $$("#darts .dart-slot").forEach((slot, i) => {
      const d = t.darts[i];
      slot.className = "dart-slot" + (d ? " filled" : "") + (d && d.uncertain ? " uncertain" : "") + (d && d.manual ? " manual" : "");
      slot.querySelector(".slot-field").textContent = d ? d.field : "–";
      slot.querySelector(".slot-points").textContent = d ? `${d.points} Punkte` : "";
      slot.disabled = !d;
    });
    $("#turn-sum").textContent = t.points;
    $("#checkout").textContent = g.checkout ? `Checkout ${g.checkout}` : "";
    const banner = $("#banner");
    if (t.bust) { banner.hidden = false; banner.className = "banner bust"; banner.textContent = "BUST"; }
    else if (t.finished) { banner.hidden = false; banner.className = "banner finish"; banner.textContent = "GAME SHOT!"; }
    else if (t.complete) { banner.hidden = false; banner.className = "banner pull"; banner.textContent = "Pfeile ziehen oder „Weiter“"; }
    else banner.hidden = true;
    drawMarkers($("#board"), t.darts);
  }
  function renderEvents(events) {
    const box = $("#events");
    box.innerHTML = (events || []).slice().reverse().map((e) => `<div>${escapeHtml(e.text)}</div>`).join("");
  }
  function renderWinner(g) {
    const modal = $("#modal-winner");
    if (!g.over) { modal.hidden = true; return; }
    const w = g.players[g.winner];
    $("#winner-name").textContent = `${w.name} gewinnt!`;
    const st = w.stats || {};
    $("#winner-stats").textContent = `Ø ${st.average} · First 9: ${st.first9_average} · Höchste Aufnahme: ${st.highest_turn} · Bestes Leg: ${st.best_leg_darts ?? "–"} Pfeile`;
    modal.hidden = false;
  }
  function render() {
    if (!state) return;
    renderDetection(state.detection);
    applySettings(state.settings);
    const addr = state.addresses || [];
    $("#addresses").innerHTML = addr.length
      ? "Auf dem Handy (gleiches Netz): " + addr.map(([iface, url]) => `<b>${escapeHtml(url)}</b> <span style="opacity:.6">(${escapeHtml(iface)})</span>`).join(" · ")
      : "";
    const g = state.game;
    $("#view-setup").hidden = !!g;
    $("#view-game").hidden = !g;
    if (g) {
      renderPlayers(g);
      renderTurn(g);
      renderEvents(state.events);
      renderWinner(g);
    }
    $("#camera-card").hidden = !cameraOn;
  }
  function escapeHtml(s) { return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c])); }

  // ---------------------------------------------------------------- Korrektur
  function openCorrection(index) {
    if (!state || !state.game || !state.game.turn.darts[index]) return;
    correctIndex = index;
    $("#correct-title").innerHTML = `Pfeil <span id="correct-index">${index + 1}</span> korrigieren`;
    drawMarkers($("#board-correct"), [state.game.turn.darts[index]]);
    $("#modal-correct").hidden = false;
  }
  function openAddDart() {
    if (!state || !state.game || state.game.turn.complete) return;
    correctIndex = "add";
    $("#correct-title").textContent = `Pfeil ${state.game.turn.darts.length + 1} nachtragen`;
    drawMarkers($("#board-correct"), state.game.turn.darts);
    $("#modal-correct").hidden = false;
  }
  async function submitCorrection(field) {
    if (correctIndex == null) return;
    if (correctIndex === "add") await post("/api/turn/add", { field });
    else await post("/api/turn/correct", { index: correctIndex, field });
    $("#modal-correct").hidden = true;
    correctIndex = null;
  }

  // ---------------------------------------------------------------- WebSocket
  function connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    ws = new WebSocket(`${proto}://${location.host}/ws`);
    ws.onopen = () => pill($("#pill-ws"), "Verbunden", "ok");
    ws.onmessage = (ev) => { state = JSON.parse(ev.data); render(); };
    ws.onclose = () => { pill($("#pill-ws"), "Getrennt – verbinde …", "bad"); setTimeout(connect, 1500); };
    ws.onerror = () => ws.close();
  }
  setInterval(() => { if (ws && ws.readyState === 1) ws.send("ping"); }, 5000);

  // ---------------------------------------------------------------- Events
  function init() {
    buildBoard($("#board"), false);
    buildBoard($("#board-correct"), true);
    initSegmented();
    $("#btn-add-player").addEventListener("click", () => addPlayerRow(""));
    $("#btn-start").addEventListener("click", startGame);
    $$("[data-turn]").forEach((b) => b.addEventListener("click", () => post(`/api/turn/${b.dataset.turn}`)));
    $$("[data-detect]").forEach((b) => b.addEventListener("click", () => { post(`/api/detection/${b.dataset.detect}`); $("#menu").removeAttribute("open"); }));
    document.addEventListener("click", (ev) => { const m = $("#menu"); if (m.open && !m.contains(ev.target)) m.removeAttribute("open"); });
    $$("#darts .dart-slot").forEach((s) => s.addEventListener("click", () => openCorrection(Number(s.dataset.index))));
    $("#btn-add-dart").addEventListener("click", openAddDart);
    $$("#modal-correct [data-field]").forEach((b) => b.addEventListener("click", () => submitCorrection(b.dataset.field)));
    $("#btn-correct-close").addEventListener("click", () => { $("#modal-correct").hidden = true; correctIndex = null; });
    $("#btn-end-game").addEventListener("click", async () => { $("#menu").removeAttribute("open"); if (state && state.game && confirm("Spiel beenden?")) await post("/api/game/end"); });
    $("#btn-to-setup").addEventListener("click", () => post("/api/game/end"));
    $("#btn-rematch").addEventListener("click", () => {
      const s = state.settings;
      post("/api/game/new", { players: s.players, start_score: s.start_score, double_out: s.double_out, legs_to_win: s.legs_to_win });
    });
    $("#btn-camera-toggle").addEventListener("click", () => { cameraOn = !cameraOn; localStorage.setItem("dart.camera", cameraOn ? "1" : "0"); $("#menu").removeAttribute("open"); render(); });
    document.addEventListener("keydown", (ev) => {
      if (ev.target.tagName === "INPUT") return;
      if (!state || !state.game) return;
      if (ev.key === "u" || ev.key === "U") post("/api/turn/undo");
      if (ev.key === " " || ev.key === "Enter") { ev.preventDefault(); post("/api/turn/next"); }
      if (ev.key === "m" || ev.key === "M") post("/api/turn/miss");
      if (ev.key === "+" || ev.key === "a" || ev.key === "A") openAddDart();
      if (ev.key === "Escape") { $("#modal-correct").hidden = true; correctIndex = null; }
      if (["1", "2", "3"].includes(ev.key)) openCorrection(Number(ev.key) - 1);
    });
    connect();
  }
  document.addEventListener("DOMContentLoaded", init);
})();
