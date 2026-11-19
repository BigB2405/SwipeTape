/* Find similar: pick one song, hear it, and the live kNN model returns the 5 closest songs. */
import { $, store, row, esc, shuffle, cleanName, fallbackArt, tagsFor, spotifyUrl, ensureRow, indexOf, topTracks,
  ID, NAME, ARTIST, POP, EXPL } from "./data.js";
import { togglePreview, now, player } from "./preview.js";
import { api } from "./api.js";
import { playBtn, listBtn, rowHTML, hydrate, ICON_SPARK } from "./ui.js";

const K = 5;
const searchEl = $("#simSearch");
let seed = null;            // track index of the current seed
let searchSeq = 0, recSeq = 0, timer = null;

/* ------------------------------------------------------------ search */
function renderQuickPicks() {
  const picks = shuffle(topTracks(() => true, 60)).slice(0, 8);
  $("#simQuick").innerHTML = picks.map((t) =>
    `<button class="qchip" data-seed="${row(t)[ID]}"><span class="qart" data-art="${t}" style="background-image:${fallbackArt(t)}"></span>${esc(cleanName(row(t)[NAME]))}<small>${esc(row(t)[ARTIST].split(", ")[0])}</small></button>`).join("");
  hydrate($("#simQuick"));
}

function setResults(html, show = true) { const el = $("#simResults"); el.innerHTML = html; el.hidden = !show; hydrate(el); }

async function runSearch(q) {
  const seq = ++searchSeq;
  setResults(`<li class="loading-state small"><div class="spinner sm"></div>Searching ${store.n0 ? "the catalog" : ""}…</li>`);
  try {
    const hits = await api.search(q, 25, { onSlow: () => { if (seq === searchSeq) setResults(`<li class="loading-state small"><div class="spinner sm"></div>Waking up the model API (first request can take ~30 s)…</li>`); } });
    if (seq !== searchSeq) return;
    if (!hits.length) return setResults(`<li class="empty-state small">No songs match “${esc(q)}”. Try just the artist, or a shorter title.</li>`);
    // Same match quality: more popular first (popularity is known for songs in songs.json).
    const pop = (h) => { const t = indexOf(h.track_id); return t === undefined ? -1 : row(t)[POP]; };
    hits.sort((a, b) => a.match - b.match || pop(b) - pop(a));
    setResults(hits.slice(0, 10).map((h) => {
      const t = ensureRow(h);
      return rowHTML(t, { right: `<button class="pick" data-seed="${h.track_id}">Pick</button>`, attrs: `data-seed-row="${h.track_id}"` });
    }).join(""));
  } catch (e) {
    if (seq !== searchSeq) return;
    setResults(`<li class="error-state small">${esc(e.message)} <button class="linkbtn" data-retry-search>Retry</button></li>`);
  }
}

searchEl.addEventListener("input", () => {
  clearTimeout(timer);
  const q = searchEl.value.trim();
  if (q.length < 2) { searchSeq++; setResults("", false); return; }
  timer = setTimeout(() => runSearch(q), 300);
});
searchEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); clearTimeout(timer); const q = searchEl.value.trim(); if (q.length >= 2) runSearch(q); }
  if (e.key === "Escape") { searchEl.value = ""; setResults("", false); }
});

/* ------------------------------------------------------------ seed + model */
function seedHTML(t) {
  const r = row(t);
  return `<div class="seed-art" data-art="${t}" style="background-image:${fallbackArt(t)}">${playBtn(t, "pbtn big")}</div>
    <div class="seed-info">
      <span class="kicker">You picked</span>
      <h2>${esc(r[NAME])}</h2>
      <p class="seed-artist">${esc(r[ARTIST])}</p>
      <div class="meta">${r[EXPL] ? '<span class="tag e">E</span>' : ""}${tagsFor(t).map((x) => `<span class="tag">${esc(x)}</span>`).join("")}</div>
      <div class="seed-actions">${listBtn(t, "similar", true)}<a class="sp" href="${spotifyUrl(t)}" target="_blank" rel="noopener">Spotify</a>
      <button class="linkbtn" id="simClear">Pick another song</button></div>
    </div>`;
}

function resultCard(r, n) {
  const t = ensureRow(r);
  const x = row(t);
  return `<article class="simcard" style="animation-delay:${n * 70}ms">
    <div class="simcard-art" data-art="${t}" data-open="${t}" style="background-image:${fallbackArt(t)}">
      <span class="rank">${n + 1}</span>${playBtn(t, "pbtn")}
    </div>
    <div class="simcard-body">
      <div class="simcard-t" title="${esc(x[NAME])}">${esc(x[NAME])}</div>
      <div class="simcard-a">${esc(x[ARTIST])}</div>
      <div class="simcard-score"><div class="bar"><i style="width:${Math.max(0, Math.min(100, r.score * 100))}%"></i></div><span>${Math.round(r.score * 100)}% similar</span></div>
      <div class="simcard-btns">${listBtn(t, "similar")}<a class="simbtn sm" href="#/similar/${x[ID]}" title="Find songs like this one">${ICON_SPARK}<span>More like this</span></a><a class="sp" href="${spotifyUrl(t)}" target="_blank" rel="noopener">Spotify</a></div>
    </div>
  </article>`;
}

async function loadSimilar(t) {
  const seq = ++recSeq;
  const box = $("#simOut");
  box.innerHTML = `<div class="sim-head"><h3>5 songs you'd also like</h3></div>
    <div class="simgrid">${Array.from({ length: K }, () => '<div class="simcard skel"><div class="simcard-art"></div><div class="simcard-body"><i></i><i></i></div></div>').join("")}</div>
    <p class="sim-status" id="simStatus"><span class="spinner sm"></span> Asking the model for the 5 closest songs…</p>`;
  try {
    const recs = await api.recommend([row(t)[ID]], [], K, {
      onSlow: () => { const s = $("#simStatus"); if (s && seq === recSeq) s.innerHTML = '<span class="spinner sm"></span> Waking up the model. The first request can take up to 30 seconds…'; },
    });
    if (seq !== recSeq) return;
    box.innerHTML = `<div class="sim-head"><h3>5 songs you'd also like</h3>
        <span class="sim-note">From the SwipeTape song kNN model (live API). Similarity combines audio features and genre.</span></div>
      <div class="simgrid">${recs.map(resultCard).join("")}</div>`;
    hydrate(box);
  } catch (e) {
    if (seq !== recSeq) return;
    box.innerHTML = `<div class="error-state"><b>Couldn't get recommendations.</b><br>${esc(e.message)}<br><button class="ghost-sm" id="simRetry">Try again</button></div>`;
  }
}

async function lookupById(trackId) {
  // Songs outside songs.json (e.g. a shared link): get the title from Spotify, then let the API decide.
  try {
    const r = await fetch(`https://open.spotify.com/oembed?url=https://open.spotify.com/track/${trackId}`);
    if (!r.ok) return undefined;
    const j = await r.json();
    return ensureRow({ track_id: trackId, track_name: j.title || "Unknown song", artist: "" });
  } catch { return undefined; }
}

async function setSeed(trackId, { autoplay = true } = {}) {
  let t = indexOf(trackId);
  if (t === undefined && /^[A-Za-z0-9]{22}$/.test(trackId)) t = await lookupById(trackId);
  if (t === undefined) {
    $("#simPick").hidden = false;
    $("#simSeed").hidden = true;
    $("#simOut").innerHTML = `<div class="error-state">We couldn't find that song. Search for it above.</div>`;
    return;
  }
  seed = t;
  searchEl.value = "";
  setResults("", false);
  $("#simPick").hidden = true;
  $("#simSeed").hidden = false;
  $("#simSeed").innerHTML = seedHTML(t);
  hydrate($("#simSeed"));
  if (autoplay && !(now.t === t && !player.paused)) togglePreview(t);   // "try it": the seed starts playing
  if (location.hash !== `#/similar/${trackId}`) history.replaceState(null, "", `#/similar/${trackId}`);
  loadSimilar(t);
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function clearSeed() {
  seed = null; recSeq++;
  $("#simSeed").hidden = true;
  $("#simOut").innerHTML = "";
  $("#simPick").hidden = false;
  history.replaceState(null, "", "#/similar");
  searchEl.focus();
}

document.addEventListener("click", (e) => {
  if (!e.target.closest("#tab-similar")) return;
  const s = e.target.closest("[data-seed]");
  if (s) { e.preventDefault(); setSeed(s.dataset.seed); return; }
  const r = e.target.closest("[data-seed-row]");
  if (r && !e.target.closest("[data-play]")) { setSeed(r.dataset.seedRow); return; }
  if (e.target.closest("#simClear")) clearSeed();
  if (e.target.closest("#simRetry") && seed !== null) loadSimilar(seed);
  if (e.target.closest("[data-retry-search]")) runSearch(searchEl.value.trim());
});

let built = false;
export const similar = {
  /** param: a track id from #/similar/<id>, or undefined */
  enter(param) {
    if (!built) { built = true; renderQuickPicks(); }
    if (param) {
      if (seed === null || row(seed)[ID] !== param) setSeed(param);
    } else if (seed === null) { $("#simPick").hidden = false; setTimeout(() => searchEl.focus(), 50); }
  },
  leave() {},
};
