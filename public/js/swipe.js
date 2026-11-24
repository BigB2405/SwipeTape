/* Swipe tab: the Tinder-style deck (unchanged behaviour), then a playlist from the live model.
 * If the API can't be reached, the old in-browser formula makes an "offline" playlist instead. */
import { $, $$, store, row, esc, shuffle, cleanName, firstArtist, norm, sleep, toast, fallbackArt, tagsFor, spotifyUrl,
  ensureRow, tasteProfile, tasteHTML, indexOf, ID, NAME, ARTIST, POP, GENRE, EXPL, DUR, FEAT_COLS } from "./data.js";
import { player, now, getPreview, stopAudio, syncPlayButtons, ICON_PAUSE, ICON_PLAY, ICON_REPLAY } from "./preview.js";
import { api, capIds } from "./api.js";
import { account, saveSwipe, removeSwipe, toggleMyList, signedIn, onAccountChange } from "./auth.js";
import { rowHTML, listBtn, hydrate, similarBtn } from "./ui.js";
import { loadPrefs, langOk } from "./prefs.js";

const FEAT_W = [1.0, 1.2, 1.0, 1.0, 0.8, 0.6, 0.3, 0.5, 0.6];
const PLAYLIST_LEN = 20;
const DECK_POOL_MIN_POP = 62;

const state = {
  deckSize: 12,
  deck: [],          // [{t, preview, art}]
  deckDone: false,
  deckWaiters: [],
  idx: 0,
  swipes: [],        // this round: [{t, dir: 'like'|'nope'|'love'}]
  seen: new Set(),   // track indices shown in any round (plus the account's history)
  z: null,
  lastPicks: [],
  active: false,
};

/* ------------------------------------------------------------ deck build */
function pickCandidates(count) {
  const { tracks, genreFamily, families } = store.data;
  const prefs = loadPrefs();
  const chosen = prefs.families.map((f) => families.indexOf(f)).filter((i) => i >= 0);
  // "Other" never goes in the deck; "Kids & Sleep" only if the survey picked it (v2)
  const kids = families.indexOf("Kids & Sleep");
  const hidden = new Set([families.indexOf("Other"), ...(chosen.includes(kids) ? [] : [kids])]);
  const byFam = families.map(() => []);
  tracks.forEach((r, t) => {
    if (r[POP] >= DECK_POOL_MIN_POP && !state.seen.has(t) && !account.swipes.has(r[ID]) && langOk(t, prefs.languages)) byFam[genreFamily[r[GENRE]]].push(t);
  });
  byFam.forEach(shuffle);
  const usedGenres = new Set(), usedArtists = new Set(), out = [];
  const fill = (fams, upTo) => {
    const famOrder = shuffle(fams.filter((i) => !hidden.has(i) && byFam[i].length));
    for (let round = 0; out.length < upTo && round < 40; round++) {
      for (const f of famOrder) {
        if (out.length >= upTo) break;
        const list = byFam[f];
        const k = list.findIndex((t) => !usedArtists.has(firstArtist(tracks[t][ARTIST])) && (round > 1 || !usedGenres.has(tracks[t][GENRE])));
        if (k < 0) continue;
        const t = list.splice(k, 1)[0];
        usedArtists.add(firstArtist(tracks[t][ARTIST]));
        usedGenres.add(tracks[t][GENRE]);
        out.push(t);
      }
    }
  };
  // v2: about 80% from the survey's groups, the rest from everything (so new genres still show up)
  if (chosen.length) fill(chosen, Math.ceil(count * 0.8));
  fill(families.map((_, i) => i), count);
  return shuffle(out);
}

function notifyDeck() { state.deckWaiters.splice(0).forEach((fn) => fn()); updateStartBtn(); }

async function buildDeck() {
  const gen = (buildDeck.gen = (buildDeck.gen || 0) + 1);
  state.deck = [];
  state.deckDone = false;
  const target = 15;
  const queue = pickCandidates(target + 12);
  const resolved = new Array(queue.length);
  let next = 0, published = 0;
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
    btn.querySelector(".cta-text").textContent = !store.data ? "Loading songs…" : ready ? "Start swiping" : "Finding clips…";
  }
}

/* ----------------------------------------------------------------- cards */
function makeCard(item) {
  const r = row(item.t);
  const el = document.createElement("article");
  el.className = "card";
  el.setAttribute("aria-label", `${r[NAME]} by ${r[ARTIST]}`);
  const initial = esc(cleanName(r[NAME]).charAt(0).toUpperCase());
  el.innerHTML = `
    <div class="art art-fallback" style="background:${fallbackArt(item.t)}">
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
      <h2 class="title">${esc(r[NAME])}</h2>
      <p class="artist">${esc(r[ARTIST])}</p>
      <div class="meta">${r[EXPL] ? '<span class="tag e">E</span>' : ""}${tagsFor(item.t).map((x) => `<span class="tag">${esc(x)}</span>`).join("")}</div>
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
  if (state.active) playClip(item, card);
}

function setActionsEnabled(on) { for (const id of ["nopeBtn", "likeBtn", "loveBtn"]) $("#" + id).disabled = !on; }

/* ----------------------------------------------------------------- audio */
let activeCard = null;

function playClip(item, card) {
  activeCard = card;
  now.t = item.t; now.owner = "swipe";
  syncPlayButtons();
  if (!item.preview) { player.pause(); return; }
  player.src = item.preview;
  player.currentTime = 0;
  player.play().catch(() => syncCardUI());
  syncCardUI();
}

function syncCardUI() {
  if (!activeCard || now.owner !== "swipe") return;
  const btn = activeCard.querySelector(".playbtn");
  const playing = !player.paused && !player.ended;
  activeCard.classList.toggle("playing", playing);
  if (btn) {
    btn.innerHTML = player.ended ? ICON_REPLAY : playing ? ICON_PAUSE : ICON_PLAY;
    btn.setAttribute("aria-label", player.ended ? "Replay clip" : playing ? "Pause clip" : "Play clip");
  }
}

function toggleCard() {
  if (now.owner !== "swipe" || !player.src) {
    const item = state.deck[state.idx];
    if (item && activeCard) playClip(item, activeCard);
    return;
  }
  if (player.paused || player.ended) { if (player.ended) player.currentTime = 0; player.play().catch(() => {}); }
  else player.pause();
}

for (const ev of ["play", "pause", "ended"]) player.addEventListener(ev, syncCardUI);
player.addEventListener("timeupdate", () => {
  if (!activeCard || now.owner !== "swipe") return;
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

  card.querySelector(".playbtn")?.addEventListener("click", (e) => { e.stopPropagation(); toggleCard(); });

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
    if (!moved) { toggleCard(); return; }
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
  saveSwipe(row(t)[ID], dir);
  document.dispatchEvent(new CustomEvent("swipetape:swiped"));
  player.pause();
  setTimeout(() => renderDeck(), 280);
}

function rewind() {
  const last = state.swipes.pop();
  if (!last) return;
  state.idx--;
  removeSwipe(row(last.t)[ID]);
  document.dispatchEvent(new CustomEvent("swipetape:swiped"));
  const off = last.dir === "love" ? "translate(0, -120%)" : `translate(${last.dir === "like" ? 120 : -120}%, 0) rotate(${last.dir === "like" ? 30 : -30}deg)`;
  renderDeck(off);
}

/* -------------------------------------------- offline fallback (pre-D5 formula) */
function buildZ() {
  if (state.z) return;
  const tracks = store.data.tracks.slice(0, store.n0);   // songs.json rows only (API-only rows have no features)
  const n = tracks.length, k = FEAT_COLS.length;
  const mean = new Float64Array(k), sd = new Float64Array(k);
  for (const r of tracks) FEAT_COLS.forEach((c, j) => (mean[j] += r[c]));
  mean.forEach((_, j) => (mean[j] /= n));
  for (const r of tracks) FEAT_COLS.forEach((c, j) => (sd[j] += (r[c] - mean[j]) ** 2));
  sd.forEach((_, j) => (sd[j] = Math.sqrt(sd[j] / n) || 1));
  const z = new Float32Array(n * k);
  tracks.forEach((r, i) => FEAT_COLS.forEach((c, j) => (z[i * k + j] = ((r[c] - mean[j]) / sd[j]) * FEAT_W[j])));
  state.z = { z, n };
}

function localRecommend(swipes) {
  buildZ();
  const { tracks, genreFamily } = store.data;
  const k = FEAT_COLS.length, { z, n } = state.z;
  const likes = swipes.filter((s) => s.dir !== "nope");
  const nopes = swipes.filter((s) => s.dir === "nope");
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

  const swiped = new Set([...state.seen]);
  const scored = [];
  for (let i = 0; i < n; i++) {
    if (swiped.has(i) || account.swipes.has(tracks[i][ID])) continue;
    const r = tracks[i], g = r[GENRE], f = genreFamily[g];
    let dotL = 0, dotD = 0, nx = 0;
    for (let j = 0; j < k; j++) { const x = z[i * k + j]; nx += x * x; dotL += x * L[j]; if (D) dotD += x * D[j]; }
    nx = Math.sqrt(nx) || 1;
    let s = dotL / (nx * nL);
    if (D) s -= 0.3 * (dotD / (nx * nD));
    s += 0.55 * ((gLike[g] || 0) / totW) + 0.35 * ((fLike[f] || 0) / totW);
    if (fNope[f] && !fLike[f]) s -= 0.25 * (fNope[f] / nopes.length);
    s += 0.25 * (r[POP] / 100);
    scored.push([s, i]);
  }
  scored.sort((a, b) => b[0] - a[0]);

  const perArtist = {}, names = new Set(), picks = [];
  for (const [s, i] of scored) {
    const r = tracks[i], a = firstArtist(r[ARTIST]), nm = norm(cleanName(r[NAME])) + "|" + a;
    if ((perArtist[a] || 0) >= 2 || names.has(nm)) continue;
    perArtist[a] = (perArtist[a] || 0) + 1; names.add(nm);
    picks.push({ t: i, s });
    if (picks.length === PLAYLIST_LEN) break;
  }
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
  picks.forEach((p) => (p.label = `${Math.round(hi === lo ? 95 : 72 + 26 * ((p.s - lo) / (hi - lo)))}%`));
  return picks;
}

/** Live model: loves first (they matter most), then likes, at most 25 each way. */
async function modelRecommend(swipes, onSlow) {
  const liked = capIds([...swipes.filter((s) => s.dir === "love"), ...swipes.filter((s) => s.dir === "like")].map((s) => row(s.t)[ID]));
  const nopeIds = capIds(swipes.filter((s) => s.dir === "nope").map((s) => row(s.t)[ID]).filter((id) => !liked.includes(id)));
  const recs = await api.recommend(liked, nopeIds, 35, { onSlow });
  // The API already skips this round's swipes; also leave out songs this account swiped in earlier rounds.
  return recs.filter((r) => !account.swipes.has(r.track_id)).slice(0, PLAYLIST_LEN)
    .map((r) => ({ t: ensureRow(r), s: r.score, because: indexOf(r.because_you_liked), label: `${Math.round(r.score * 100)}%` }));
}

/* --------------------------------------------------------------- results */
async function finish() {
  player.pause();
  activeCard = null;
  now.owner = null;
  show("resultScreen");
  const nLike = state.swipes.filter((s) => s.dir === "like").length;
  const nLove = state.swipes.filter((s) => s.dir === "love").length;
  const likes = state.swipes.filter((s) => s.dir !== "nope");
  const gen = (finish.gen = (finish.gen || 0) + 1);

  const empty = !likes.length;
  $("#matchBanner .match-title").textContent = empty ? "No matches… yet" : "Your mix is ready";
  $("#taste").hidden = empty; $(".pl-head").hidden = empty; $("#resultActions").hidden = empty;
  $("#mixSource").hidden = true;
  if (empty) {
    $("#collage").innerHTML = "";
    $("#matchSub").textContent = "";
    $("#playlist").innerHTML = `<li class="empty-state">You passed on every song. Swipe again with a new set, and like at least one song so we have something to go on.</li>`;
    return;
  }

  const artFor = (t) => state.deck.find((d) => d.t === t)?.art;
  const top = [...likes].sort((a, b) => (b.dir === "love") - (a.dir === "love")).slice(0, 4);
  $("#collage").innerHTML = top.map((l, i) => {
    const a = artFor(l.t);
    return `<div style="background-image:${a ? `url('${esc(a)}')` : fallbackArt(l.t)};animation-delay:${i * 90}ms"></div>`;
  }).join("");
  $("#matchSub").textContent = `You liked ${nLike + nLove} of ${state.swipes.length} songs${nLove ? ` and loved ${nLove}` : ""}.`;
  $("#taste").innerHTML = tasteHTML(tasteProfile(likes));
  $("#plMeta").textContent = "";
  $("#playlist").innerHTML = `<li class="loading-state"><div class="spinner"></div><span id="mixWait">Asking the SwipeTape model for your mix…</span></li>`;

  let picks, source = "model", why = "";
  try {
    picks = await modelRecommend(state.swipes, () => { const w = $("#mixWait"); if (w) w.textContent = "Waking up the model. The first request can take up to 30 seconds…"; });
    if (!picks.length) throw new Error("The model returned no songs.");
  } catch (e) {
    why = e.message;
    source = "offline";
    picks = localRecommend(state.swipes);
    toast("The model API didn't answer, so these are offline picks.", 3500);
  }
  if (gen !== finish.gen) return;
  state.lastPicks = picks;

  $("#mixSource").hidden = false;
  $("#mixSource").className = "mix-source " + source;
  $("#mixSource").innerHTML = source === "model"
    ? "Picked by the SwipeTape model (song kNN, live API). % = how close each song is to the songs you liked (audio features + genre)."
    : `Offline picks: a simple in-browser formula, because the model API failed (${esc(why)}). <button class="linkbtn" id="retryMix">Try the model again</button>`;
  $("#matchSub").textContent += ` Here are ${picks.length} songs picked to match.`;
  const mins = Math.round(picks.reduce((a, p) => a + (row(p.t)[DUR] || 0), 0) / 60);
  $("#plMeta").textContent = `${picks.length} songs${mins ? ` · ${mins} min` : ""}`;
  $("#playlist").innerHTML = picks.map((p, n) => rowHTML(p.t, {
    n: n + 1,
    why: p.because !== undefined && p.because !== null ? `Because you liked <b>${esc(cleanName(row(p.because)[NAME]))}</b>` : "",
    right: `<span class="pct">${p.label}</span><span class="row-btns">${listBtn(p.t, "swipe")}<a class="sp" href="${spotifyUrl(p.t)}" target="_blank" rel="noopener">Spotify</a></span>`,
  })).join("");
  hydrate($("#resultScreen"));
}

async function copyLinks() {
  const text = state.lastPicks.map((p) => spotifyUrl(p.t)).join("\n");
  try { await navigator.clipboard.writeText(text); }
  catch {
    const ta = document.createElement("textarea");
    ta.value = text; document.body.appendChild(ta); ta.select();
    try { document.execCommand("copy"); } catch { /* ignore */ }
    ta.remove();
  }
  toast("Copied! Paste into a Spotify playlist.");
}

async function saveAll() {
  if (!signedIn()) { toast("Sign in on the Me tab to save songs to My List.", 3200); return; }
  let n = 0;
  for (const p of state.lastPicks) if (!account.myList.has(row(p.t)[ID])) { await toggleMyList(row(p.t)[ID], "swipe"); n++; }
  toast(n ? `Saved ${n} songs to My List.` : "They're all in My List already.");
}

/* ------------------------------------------------------------- startup */
function startSwiping() {
  if (!state.deck[0]) return;
  state.idx = 0;
  state.swipes = [];
  show("swipeScreen");
  renderDeck();   // play() inside this click unlocks audio for the rest of the session
}

function newRound() {
  player.pause();
  show("startScreen");
  buildDeck();
  updateStartBtn();
}

$$("#tab-swipe .chip").forEach((c) => c.addEventListener("click", () => {
  $$("#tab-swipe .chip").forEach((x) => { x.classList.toggle("on", x === c); x.setAttribute("aria-checked", x === c); });
  state.deckSize = +c.dataset.n;
}));
$("#startBtn").addEventListener("click", startSwiping);
$("#likeBtn").addEventListener("click", () => swipe("like"));
$("#nopeBtn").addEventListener("click", () => swipe("nope"));
$("#loveBtn").addEventListener("click", () => swipe("love"));
$("#rewindBtn").addEventListener("click", rewind);
$("#againBtn").addEventListener("click", newRound);
$("#copyBtn").addEventListener("click", copyLinks);
$("#saveAllBtn").addEventListener("click", saveAll);
$("#resultScreen").addEventListener("click", (e) => { if (e.target.id === "retryMix") finish(); });

document.addEventListener("keydown", (e) => {
  if (!state.active || $("#swipeScreen").hidden || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.target.closest?.("input, textarea")) return;
  const map = { ArrowRight: () => swipe("like"), ArrowLeft: () => swipe("nope"), ArrowUp: () => swipe("love"), " ": toggleCard, Backspace: rewind };
  if (map[e.key]) { e.preventDefault(); map[e.key](); }
});

// Signing in mid-round: rebuild the deck so it skips songs this account already swiped.
let lastUser = null;
onAccountChange(() => {
  const id = account.user?.id || null;
  if (id !== lastUser) { lastUser = id; if (store.data && !$("#startScreen").hidden) buildDeck(); }
});

let started = false;
export const swipeTab = {
  enter() {
    state.active = true;
    if (!started) { started = true; buildDeck(); updateStartBtn(); }
    if (!$("#swipeScreen").hidden) {          // came back mid-round: resume the current card
      const item = state.deck[state.idx];
      const card = $("#deck .card:last-child");
      if (item && card) playClip(item, card);
    }
  },
  leave() {
    state.active = false;
    if (now.owner === "swipe") { player.pause(); now.owner = null; }
  },
};
