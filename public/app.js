"use strict";

/* Track row layout (see scripts/build_data.py) */
const ID = 0, NAME = 1, ARTIST = 2, ALBUM = 3, POP = 4, GENRE = 5, EXPL = 7;
const DANCE = 8, ENERGY = 9, VALENCE = 10, ACOUSTIC = 11, TEMPO = 15;
const FEAT_COLS = [8, 9, 10, 11, 12, 13, 14, 15, 16]; // dance..loudness
const FEAT_W = [1.0, 1.2, 1.0, 1.0, 0.8, 0.6, 0.3, 0.5, 0.6];
const PLAYLIST_LEN = 20;
const DECK_POOL_MIN_POP = 62;

const $ = (s) => document.querySelector(s);
const player = $("#player");

const state = {
  data: null,
  deckSize: 12,
  deck: [],          // [{t, preview, art}]
  deckDone: false,   // finished resolving candidates
  deckWaiters: [],
  idx: 0,
  swipes: [],        // [{t, dir: 'like'|'nope'|'love'}]
  seen: new Set(),   // track indices shown in any round
  z: null,           // standardized feature matrix
  playingRow: null,
};

/* ------------------------------------------------------------------ utils */
const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [a[i], a[j]] = [a[j], a[i]]; } return a; };
const norm = (s) => (s || "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^\p{L}\p{N} ]/gu, " ").replace(/\s+/g, " ").trim();
const cleanName = (s) => s.replace(/\s*[([].*?[)\]]/g, "").split(" - ")[0].trim() || s;
const firstArtist = (s) => s.split(", ")[0];
const prettyGenre = (g) => g.replace(/-n-/g, " & ").replace(/^r & b$/, "R&B").replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const spotifyUrl = (t) => `https://open.spotify.com/track/${state.data.tracks[t][ID]}`;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.remove("show"), 2200);
}

function hashHue(str) { let h = 0; for (const c of str) h = (h * 31 + c.charCodeAt(0)) | 0; return Math.abs(h) % 360; }
function fallbackArt(row) {
  const h = hashHue(row[ID]);
  return `linear-gradient(150deg, hsl(${h} 75% 55%), hsl(${(h + 60) % 360} 70% 35%))`;
}

/* ---------------------------------------------------------- preview lookup */
const previewCache = new Map();

async function lookupITunes(row) {
  const name = cleanName(row[NAME]);
  const artist = firstArtist(row[ARTIST]);
  const url = `https://itunes.apple.com/search?media=music&entity=song&limit=15&term=${encodeURIComponent(`${artist} ${name}`)}`;
  const r = await fetch(url);
  if (!r.ok) throw new Error("itunes " + r.status);
  const { results = [] } = await r.json();
  const na = norm(artist), nn = norm(name);
  const withPrev = results.filter((x) => x.previewUrl);
  const hit =
    withPrev.find((x) => norm(x.artistName).includes(na) && norm(cleanName(x.trackName)) === nn) ||
    withPrev.find((x) => norm(x.artistName).includes(na) && norm(x.trackName).startsWith(nn)) ||
    withPrev.find((x) => norm(x.artistName).includes(na) && norm(x.trackName).includes(nn)) ||
    withPrev.find((x) => norm(x.artistName).includes(na) && nn.startsWith(norm(cleanName(x.trackName))));
  if (!hit) return null;
  return { preview: hit.previewUrl, art: hit.artworkUrl100?.replace("100x100bb", "600x600bb") || null };
}

async function lookupDeezer(row) {
  const r = await fetch(`/api/preview?artist=${encodeURIComponent(row[ARTIST])}&track=${encodeURIComponent(cleanName(row[NAME]))}`);
  if (!r.ok) return null;
  return r.json();
}

async function lookupSpotifyArt(row) {
  try {
    const r = await fetch(`https://open.spotify.com/oembed?url=https://open.spotify.com/track/${row[ID]}`);
    if (!r.ok) return null;
    return (await r.json()).thumbnail_url || null;
  } catch { return null; }
}

function getPreview(t) {
  if (previewCache.has(t)) return previewCache.get(t);
  const row = state.data.tracks[t];
  const p = (async () => {
    let res = null;
    try { res = await lookupITunes(row); } catch { /* rate limited or offline: try Deezer */ }
    if (!res) { try { res = await lookupDeezer(row); } catch { res = null; } }
    if (res && !res.art) res.art = await lookupSpotifyArt(row);
    return res;
  })();
  previewCache.set(t, p);
  return p;
}

/* ------------------------------------------------------------ deck build */
function pickCandidates(count) {
  const { tracks, genreFamily, families } = state.data;
  const other = families.indexOf("Other");
  const byFam = families.map(() => []);
  tracks.forEach((row, t) => {
    if (row[POP] >= DECK_POOL_MIN_POP && !state.seen.has(t)) byFam[genreFamily[row[GENRE]]].push(t);
  });
  byFam.forEach(shuffle);
  const famOrder = shuffle(families.map((_, i) => i).filter((i) => i !== other && byFam[i].length));
  const usedGenres = new Set(), usedArtists = new Set(), out = [];
  for (let round = 0; out.length < count && round < 40; round++) {
    for (const f of famOrder) {
      if (out.length >= count) break;
      const list = byFam[f];
      const k = list.findIndex((t) => !usedArtists.has(firstArtist(tracks[t][ARTIST])) && (round > 1 || !usedGenres.has(tracks[t][GENRE])));
      if (k < 0) continue;
      const t = list.splice(k, 1)[0];
      usedArtists.add(firstArtist(tracks[t][ARTIST]));
      usedGenres.add(tracks[t][GENRE]);
      out.push(t);
    }
  }
  return out;
}

function notifyDeck() { state.deckWaiters.splice(0).forEach((fn) => fn()); updateStartBtn(); }

async function buildDeck() {
  const gen = (buildDeck.gen = (buildDeck.gen || 0) + 1);
  state.deck = [];
  state.deckDone = false;
  const target = 15; // build enough for the largest option
  const queue = pickCandidates(target + 12);
  const resolved = new Array(queue.length);
  let next = 0;

  // Resolve in parallel but publish in candidate order, so the deck stays diverse.
  let published = 0;
  const publish = () => {
    while (published < queue.length && resolved[published] !== undefined) {
      const r = resolved[published];
      if (r && state.deck.length < target) { state.deck.push({ t: queue[published], ...r }); preloadImg(r.art); }
      published++;
    }
    notifyDeck();
  };
  const worker = async () => {
    while (next < queue.length && state.deck.length < target) {
      const i = next++;
      resolved[i] = (await getPreview(queue[i])) || null;
      if (gen !== buildDeck.gen) return;
      publish();
      await sleep(150); // be polite to the iTunes API
    }
  };
  await Promise.all([worker(), worker(), worker()]);
  if (gen !== buildDeck.gen) return;
  state.deckDone = true;
  notifyDeck();
}

function preloadImg(src) { if (src) { const i = new Image(); i.src = src; } }

function deckReady(i) {
  return new Promise((res) => {
    const check = () => (state.deck[i] || state.deckDone ? res() : state.deckWaiters.push(check));
    check();
  });
}

/* --------------------------------------------------------------- screens */
function show(id) {
  for (const s of ["startScreen", "swipeScreen", "resultScreen"]) $("#" + s).hidden = s !== id;
  $("#counter").hidden = id !== "swipeScreen";
  window.scrollTo(0, 0);
}

function updateStartBtn() {
  const btn = $("#startBtn");
  if (!$("#startScreen").hidden) {
    const ready = !!state.deck[0];
    btn.disabled = !ready;
    btn.querySelector(".cta-text").textContent = !state.data ? "Loading songs…" : ready ? "Start swiping" : "Finding clips…";
  }
}

/* ----------------------------------------------------------------- cards */
function tagsFor(row) {
  const tags = [prettyGenre(state.data.genres[row[GENRE]])];
  if (row[ENERGY] > 0.78) tags.push("⚡ High energy");
  else if (row[ENERGY] < 0.35) tags.push("🌙 Mellow");
  if (row[DANCE] > 0.72) tags.push("💃 Danceable");
  if (row[VALENCE] > 0.7) tags.push("☀️ Feel-good");
  else if (row[VALENCE] < 0.25) tags.push("🌧 Moody");
  if (row[ACOUSTIC] > 0.7 && tags.length < 3) tags.push("🎸 Acoustic");
  return tags.slice(0, 3);
}

const ICON_PAUSE = '<svg viewBox="0 0 24 24"><path d="M7 4h3.5v16H7zM13.5 4H17v16h-3.5z"/></svg>';
const ICON_PLAY = '<svg viewBox="0 0 24 24"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/></svg>';
const ICON_REPLAY = '<svg viewBox="0 0 24 24"><path d="M12 5V2L7 6l5 4V7a6 6 0 1 1-6 6H4a8 8 0 1 0 8-8z"/></svg>';

function makeCard(item) {
  const row = state.data.tracks[item.t];
  const el = document.createElement("article");
  el.className = "card";
  el.setAttribute("aria-label", `${row[NAME]} by ${row[ARTIST]}`);
  const initial = esc(cleanName(row[NAME]).charAt(0).toUpperCase());
  el.innerHTML = `
    <div class="art art-fallback" style="background:${fallbackArt(row)}">
      ${item.art ? `<img alt="" draggable="false" src="${esc(item.art)}">` : initial}
    </div>
    <div class="shade"></div>
    <div class="clipbar"><i></i></div>
    ${item.preview ? `<div class="eq" aria-hidden="true"><b></b><b></b><b></b><b></b><em>0:30</em></div>
    <button class="playbtn" aria-label="Pause clip">${ICON_PAUSE}</button>` : `<div class="noclip">No clip available</div>`}
    <div class="stamp like">LIKE</div>
    <div class="stamp nope">NOPE</div>
    <div class="stamp love">LOVE<br>IT</div>
    <div class="info">
      <h2 class="title">${esc(row[NAME])}</h2>
      <p class="artist">${esc(row[ARTIST])}</p>
      <div class="meta">${row[EXPL] ? '<span class="tag e">E</span>' : ""}${tagsFor(row).map((t) => `<span class="tag">${esc(t)}</span>`).join("")}</div>
    </div>`;
  const img = el.querySelector("img");
  if (img) {
    const done = () => { img.classList.add("loaded"); img.parentElement.firstChild.textContent = ""; };
    img.complete && img.naturalWidth ? done() : img.addEventListener("load", done);
    img.addEventListener("error", () => img.remove());
  }
  return el;
}

async function renderDeck(enterFrom) {
  const deckEl = $("#deck");
  deckEl.innerHTML = "";
  $("#counter").textContent = `${Math.min(state.idx + 1, state.deckSize)} / ${state.deckSize}`;
  $("#rewindBtn").disabled = state.swipes.length === 0;

  if (state.idx >= state.deckSize || (state.deckDone && !state.deck[state.idx])) return finish();

  if (!state.deck[state.idx]) {
    deckEl.innerHTML = '<div class="deck-empty"><div><div class="spinner"></div>Finding the next clip…</div></div>';
    setActionsEnabled(false);
    await deckReady(state.idx);
    return renderDeck(enterFrom);
  }
  setActionsEnabled(true);

  const nextItem = state.idx + 1 < state.deckSize ? state.deck[state.idx + 1] : null;
  if (nextItem) { const b = makeCard(nextItem); b.classList.add("behind"); deckEl.appendChild(b); }

  const item = state.deck[state.idx];
  const card = makeCard(item);
  deckEl.appendChild(card);
  attachDrag(card);

  if (enterFrom) {
    card.style.transform = enterFrom;
    requestAnimationFrame(() => requestAnimationFrame(() => { card.classList.add("snap"); card.style.transform = ""; }));
    card.addEventListener("transitionend", () => card.classList.remove("snap"), { once: true });
  }
  playClip(item, card);
}

function setActionsEnabled(on) {
  for (const id of ["nopeBtn", "likeBtn", "loveBtn"]) $("#" + id).disabled = !on;
}

/* ----------------------------------------------------------------- audio */
let activeCard = null;

function playClip(item, card) {
  activeCard = card;
  state.playingRow = null;
  if (!item.preview) { player.pause(); return; }
  player.src = item.preview;
  player.currentTime = 0;
  player.play().catch(() => syncPlayUI());
  syncPlayUI();
}

function syncPlayUI() {
  if (activeCard) {
    const btn = activeCard.querySelector(".playbtn");
    const playing = !player.paused && !player.ended;
    activeCard.classList.toggle("playing", playing);
    if (btn) {
      btn.innerHTML = player.ended ? ICON_REPLAY : playing ? ICON_PAUSE : ICON_PLAY;
      btn.setAttribute("aria-label", player.ended ? "Replay clip" : playing ? "Pause clip" : "Play clip");
    }
  }
  document.querySelectorAll(".row").forEach((r) => {
    const on = state.playingRow === r && !player.paused;
    r.classList.toggle("playing", on);
    const b = r.querySelector(".thumb button");
    if (b) b.innerHTML = on ? ICON_PAUSE : ICON_PLAY;
  });
}

function togglePlay() {
  if (!player.src) return;
  if (player.paused || player.ended) { if (player.ended) player.currentTime = 0; player.play().catch(() => {}); }
  else player.pause();
}

player.addEventListener("play", syncPlayUI);
player.addEventListener("pause", syncPlayUI);
player.addEventListener("ended", syncPlayUI);
player.addEventListener("timeupdate", () => {
  if (!activeCard || state.playingRow) return;
  const d = player.duration || 30;
  const bar = activeCard.querySelector(".clipbar i");
  if (bar) bar.style.width = `${(player.currentTime / d) * 100}%`;
  const em = activeCard.querySelector(".eq em");
  if (em) { const left = Math.max(0, Math.ceil(d - player.currentTime)); em.textContent = `0:${String(left).padStart(2, "0")}`; }
});

/* ------------------------------------------------------------ drag/swipe */
const THRESH = 100;

function attachDrag(card) {
  let sx = 0, sy = 0, dx = 0, dy = 0, lastX = 0, lastT = 0, vx = 0, down = false, moved = false;
  const stamps = { like: card.querySelector(".stamp.like"), nope: card.querySelector(".stamp.nope"), love: card.querySelector(".stamp.love") };

  card.querySelector(".playbtn")?.addEventListener("click", (e) => { e.stopPropagation(); togglePlay(); });

  card.addEventListener("pointerdown", (e) => {
    if (e.target.closest(".playbtn") || card.dataset.gone) return;
    down = true; moved = false; sx = e.clientX; sy = e.clientY; dx = dy = 0; lastX = sx; lastT = e.timeStamp; vx = 0;
    card.setPointerCapture(e.pointerId);
    card.classList.add("dragging");
    card.classList.remove("snap");
  });
  card.addEventListener("pointermove", (e) => {
    if (!down) return;
    dx = e.clientX - sx; dy = e.clientY - sy;
    if (Math.abs(dx) + Math.abs(dy) > 6) moved = true;
    const dt = e.timeStamp - lastT;
    if (dt > 0) { vx = (e.clientX - lastX) / dt; lastX = e.clientX; lastT = e.timeStamp; }
    card.style.transform = `translate(${dx}px, ${dy}px) rotate(${dx * 0.06}deg)`;
    const up = -dy > Math.abs(dx) ? Math.min(1, -dy / THRESH) : 0;
    stamps.like.style.opacity = up ? 0 : Math.max(0, Math.min(1, dx / THRESH));
    stamps.nope.style.opacity = up ? 0 : Math.max(0, Math.min(1, -dx / THRESH));
    stamps.love.style.opacity = up;
    const behind = card.previousElementSibling;
    if (behind) { const p = Math.min(1, (Math.abs(dx) + Math.max(0, -dy)) / 200); behind.style.transform = `scale(${0.95 + 0.05 * p}) translateY(${10 - 10 * p}px)`; }
  });
  const release = () => {
    if (!down) return;
    down = false;
    card.classList.remove("dragging");
    if (!moved) { togglePlay(); return; }
    const fastX = Math.abs(vx) > 0.6 && Math.abs(dx) > 40;
    if (-dy > THRESH && -dy > Math.abs(dx)) return swipe("love", { dx, dy });
    if (dx > THRESH || (fastX && vx > 0)) return swipe("like", { dx, dy });
    if (dx < -THRESH || (fastX && vx < 0)) return swipe("nope", { dx, dy });
    card.classList.add("snap");
    card.style.transform = "";
    Object.values(stamps).forEach((s) => (s.style.opacity = 0));
    const behind = card.previousElementSibling;
    if (behind) behind.style.transform = "";
  };
  card.addEventListener("pointerup", release);
  card.addEventListener("pointercancel", release);
}

function swipe(dir, from) {
  const deckEl = $("#deck");
  const card = deckEl.lastElementChild;
  if (!card || !card.classList.contains("card") || card.dataset.gone || state.idx >= state.deckSize || !state.deck[state.idx]) return;
  card.dataset.gone = "1";

  const btn = { like: "#likeBtn", nope: "#nopeBtn", love: "#loveBtn" }[dir];
  $(btn).classList.add("pulse");
  setTimeout(() => $(btn).classList.remove("pulse"), 250);

  card.querySelector(".stamp." + dir).style.opacity = 1;
  const { dx = 0, dy = 0 } = from || {};
  const w = window.innerWidth;
  card.classList.add("fly");
  card.style.transform =
    dir === "love" ? `translate(${dx}px, ${-window.innerHeight}px) rotate(${dx * 0.06}deg)`
    : `translate(${(dir === "like" ? 1 : -1) * (w + 200)}px, ${dy + 40}px) rotate(${dir === "like" ? 35 : -35}deg)`;
  card.style.opacity = "0.9";
  const behind = card.previousElementSibling;
  if (behind) { behind.classList.remove("behind"); behind.style.transform = ""; behind.style.filter = ""; }

  const t = state.deck[state.idx].t;
  state.swipes.push({ t, dir });
  state.seen.add(t);
  state.idx++;
  player.pause();
  setTimeout(() => renderDeck(), 280);
}

function rewind() {
  const last = state.swipes.pop();
  if (!last) return;
  state.idx--;
  const off = last.dir === "love" ? "translate(0, -120%)" : `translate(${last.dir === "like" ? 120 : -120}%, 0) rotate(${last.dir === "like" ? 30 : -30}deg)`;
  renderDeck(off);
}

/* -------------------------------------------------------- recommendation */
function buildZ() {
  if (state.z) return;
  const { tracks } = state.data;
  const n = tracks.length, k = FEAT_COLS.length;
  const mean = new Float64Array(k), sd = new Float64Array(k);
  for (const row of tracks) FEAT_COLS.forEach((c, j) => (mean[j] += row[c]));
  mean.forEach((_, j) => (mean[j] /= n));
  for (const row of tracks) FEAT_COLS.forEach((c, j) => (sd[j] += (row[c] - mean[j]) ** 2));
  sd.forEach((_, j) => (sd[j] = Math.sqrt(sd[j] / n) || 1));
  const z = new Float32Array(n * k);
  tracks.forEach((row, i) => FEAT_COLS.forEach((c, j) => (z[i * k + j] = ((row[c] - mean[j]) / sd[j]) * FEAT_W[j])));
  state.z = z;
}

function recommend() {
  buildZ();
  const { tracks, genreFamily } = state.data;
  const k = FEAT_COLS.length, z = state.z;
  const likes = state.swipes.filter((s) => s.dir !== "nope");
  const nopes = state.swipes.filter((s) => s.dir === "nope");
  if (!likes.length) return null;

  const weight = (s) => (s.dir === "love" ? 2 : 1);
  const avg = (list, wfn) => {
    const v = new Float64Array(k); let tot = 0;
    for (const s of list) { const w = wfn(s); tot += w; for (let j = 0; j < k; j++) v[j] += z[s.t * k + j] * w; }
    return v.map((x) => x / tot);
  };
  const L = avg(likes, weight);
  const D = nopes.length ? avg(nopes, () => 1) : null;
  const normOf = (v) => Math.sqrt(v.reduce((a, b) => a + b * b, 0)) || 1;
  const nL = normOf(L), nD = D ? normOf(D) : 1;

  const gLike = {}, fLike = {}, fNope = {};
  let totW = 0;
  for (const s of likes) { const g = tracks[s.t][GENRE], w = weight(s); totW += w; gLike[g] = (gLike[g] || 0) + w; fLike[genreFamily[g]] = (fLike[genreFamily[g]] || 0) + w; }
  for (const s of nopes) { const f = genreFamily[tracks[s.t][GENRE]]; fNope[f] = (fNope[f] || 0) + 1; }

  const swiped = new Set(state.swipes.map((s) => s.t));
  const scored = [];
  for (let i = 0; i < tracks.length; i++) {
    if (swiped.has(i)) continue;
    const row = tracks[i], g = row[GENRE], f = genreFamily[g];
    let dotL = 0, dotD = 0, nx = 0;
    for (let j = 0; j < k; j++) { const x = z[i * k + j]; nx += x * x; dotL += x * L[j]; if (D) dotD += x * D[j]; }
    nx = Math.sqrt(nx) || 1;
    let s = dotL / (nx * nL);
    if (D) s -= 0.3 * (dotD / (nx * nD));
    s += 0.55 * ((gLike[g] || 0) / totW) + 0.35 * ((fLike[f] || 0) / totW);
    if (fNope[f] && !fLike[f]) s -= 0.25 * (fNope[f] / nopes.length);
    s += 0.25 * (row[POP] / 100);
    scored.push([s, i]);
  }
  scored.sort((a, b) => b[0] - a[0]);

  const perArtist = {}, names = new Set(), picks = [];
  for (const [s, i] of scored) {
    const row = tracks[i], a = firstArtist(row[ARTIST]), nm = norm(cleanName(row[NAME])) + "|" + a;
    if ((perArtist[a] || 0) >= 2 || names.has(nm)) continue;
    perArtist[a] = (perArtist[a] || 0) + 1; names.add(nm);
    picks.push({ t: i, s });
    if (picks.length === PLAYLIST_LEN) break;
  }

  // "Because you liked …": the closest liked song in feature space
  for (const p of picks) {
    let best = null, bd = Infinity;
    for (const l of likes) {
      let d = 0;
      for (let j = 0; j < k; j++) d += (z[p.t * k + j] - z[l.t * k + j]) ** 2;
      if (tracks[p.t][GENRE] === tracks[l.t][GENRE]) d *= 0.6;
      if (d < bd) { bd = d; best = l.t; }
    }
    p.because = best;
  }
  const hi = picks[0].s, lo = picks[picks.length - 1].s;
  picks.forEach((p) => (p.pct = Math.round(hi === lo ? 95 : 72 + 26 * ((p.s - lo) / (hi - lo)))));
  return { picks, likes, gLike };
}

/* --------------------------------------------------------------- results */
function finish() {
  player.pause();
  activeCard = null;
  const res = recommend();
  show("resultScreen");
  const { tracks, genres } = state.data;
  const nLike = state.swipes.filter((s) => s.dir === "like").length;
  const nLove = state.swipes.filter((s) => s.dir === "love").length;

  if (!res) {
    $("#matchBanner").hidden = false;
    $("#collage").innerHTML = "";
    $("#matchSub").textContent = "";
    $("#matchBanner .match-title").textContent = "No matches… yet";
    $("#taste").hidden = true;
    $(".pl-head").hidden = true;
    $("#copyBtn").hidden = true;
    $("#playlist").innerHTML = `<li class="empty-state">You passed on every song. Swipe again with a new set, and like at least one song so we have something to go on.</li>`;
    return;
  }
  $("#matchBanner .match-title").textContent = "Your mix is ready";
  $("#taste").hidden = false; $(".pl-head").hidden = false; $("#copyBtn").hidden = false;

  const artFor = (t) => state.deck.find((d) => d.t === t)?.art;
  const liked = [...res.likes].sort((a, b) => (b.dir === "love") - (a.dir === "love")).slice(0, 4);
  $("#collage").innerHTML = liked.map((l, i) => {
    const a = artFor(l.t);
    return `<div style="background-image:${a ? `url('${esc(a)}')` : fallbackArt(tracks[l.t])};animation-delay:${i * 90}ms"></div>`;
  }).join("");
  $("#matchSub").textContent = `You liked ${nLike + nLove} of ${state.swipes.length} songs${nLove ? ` and loved ${nLove}` : ""}. Here are ${res.picks.length} songs picked to match.`;

  // taste profile from liked songs
  const wavg = (c) => { let s = 0, w = 0; for (const l of res.likes) { const x = l.dir === "love" ? 2 : 1; s += tracks[l.t][c] * x; w += x; } return s / w; };
  const traits = [["Energy", wavg(ENERGY)], ["Danceability", wavg(DANCE)], ["Positivity", wavg(VALENCE)], ["Acoustic", wavg(ACOUSTIC)]];
  const topGenres = Object.entries(res.gLike).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([g]) => prettyGenre(genres[g]));
  $("#taste").innerHTML = `
    <h4>Your taste</h4>
    ${traits.map(([n, v]) => `<div class="trait"><span>${n}</span><div class="bar"><i style="width:${Math.round(v * 100)}%"></i></div><span>${Math.round(v * 100)}</span></div>`).join("")}
    <div class="trait"><span>Tempo</span><div></div><span>${Math.round(wavg(TEMPO))}</span></div>
    <div class="genres">${topGenres.map((g) => `<span>${esc(g)}</span>`).join("")}</div>`;

  const mins = Math.round(res.picks.reduce((a, p) => a + tracks[p.t][6], 0) / 60);
  $("#plMeta").textContent = `${res.picks.length} songs · ${mins} min`;
  state.lastPicks = res.picks;

  $("#playlist").innerHTML = res.picks.map((p, n) => {
    const row = tracks[p.t];
    return `<li class="row" data-t="${p.t}">
      <div class="thumb" style="background-image:${fallbackArt(row)}"><button aria-label="Play preview of ${esc(row[NAME])}">${ICON_PLAY}</button></div>
      <div class="txt">
        <div class="t">${n + 1}. ${esc(row[NAME])}</div>
        <div class="a">${esc(row[ARTIST])}</div>
        <div class="why">Because you liked <b>${esc(cleanName(tracks[p.because][NAME]))}</b></div>
      </div>
      <div class="right"><span class="pct">${p.pct}%</span><a class="sp" href="${spotifyUrl(p.t)}" target="_blank" rel="noopener">Spotify</a></div>
    </li>`;
  }).join("");

  // Fill in album art from Spotify (no preview lookups until a row is played)
  (async () => {
    for (const li of document.querySelectorAll(".row")) {
      const t = +li.dataset.t;
      const known = state.deck.find((d) => d.t === t)?.art;
      lookupSpotifyArt(tracks[t]).then((src) => {
        const img = known || src;
        if (img) li.querySelector(".thumb").style.backgroundImage = `url('${img}')`;
      });
      await sleep(40);
    }
  })();
}

async function playRow(li) {
  if (state.playingRow === li) { togglePlay(); return; }
  player.pause();
  state.playingRow = li;
  activeCard = null;
  syncPlayUI();
  const res = await getPreview(+li.dataset.t);
  if (state.playingRow !== li) return;
  if (!res?.preview) { toast("No preview for this one. Try it on Spotify."); state.playingRow = null; syncPlayUI(); return; }
  player.src = res.preview;
  player.play().catch(() => {});
}

async function copyLinks() {
  const text = (state.lastPicks || []).map((p) => spotifyUrl(p.t)).join("\n");
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch {}
    ta.remove();
  }
  toast("Copied! Paste into a Spotify playlist.");
}

/* ------------------------------------------------------------- startup */
function startSwiping() {
  if (!state.deck[0]) return;
  state.idx = 0;
  state.swipes = [];
  show("swipeScreen");
  // Calling play() inside this click unlocks audio for the rest of the session.
  renderDeck();
}

function newRound() {
  player.pause();
  state.playingRow = null;
  show("startScreen");
  buildDeck();
  updateStartBtn();
}

document.querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => {
  document.querySelectorAll(".chip").forEach((x) => { x.classList.toggle("on", x === c); x.setAttribute("aria-checked", x === c); });
  state.deckSize = +c.dataset.n;
}));
$("#startBtn").addEventListener("click", startSwiping);
$("#likeBtn").addEventListener("click", () => swipe("like"));
$("#nopeBtn").addEventListener("click", () => swipe("nope"));
$("#loveBtn").addEventListener("click", () => swipe("love"));
$("#rewindBtn").addEventListener("click", rewind);
$("#againBtn").addEventListener("click", newRound);
$("#copyBtn").addEventListener("click", copyLinks);
$("#homeBtn").addEventListener("click", () => { if ($("#startScreen").hidden) newRound(); });
$("#playlist").addEventListener("click", (e) => {
  const b = e.target.closest(".thumb button");
  if (b) playRow(b.closest(".row"));
});

document.addEventListener("keydown", (e) => {
  if ($("#swipeScreen").hidden || e.metaKey || e.ctrlKey || e.altKey) return;
  const map = { ArrowRight: () => swipe("like"), ArrowLeft: () => swipe("nope"), ArrowUp: () => swipe("love"), " ": togglePlay, Backspace: rewind };
  if (map[e.key]) { e.preventDefault(); map[e.key](); }
});

(async function init() {
  try {
    const r = await fetch("songs.json");
    state.data = await r.json();
  } catch {
    $("#startBtn .cta-text").textContent = "Couldn't load songs. Refresh to retry.";
    return;
  }
  updateStartBtn();
  buildDeck();
})();
