/* Catalog (songs.json) and small shared helpers. Row layout: see scripts/build_data.py */
export const ID = 0, NAME = 1, ARTIST = 2, ALBUM = 3, POP = 4, GENRE = 5, DUR = 6, EXPL = 7;
export const DANCE = 8, ENERGY = 9, VALENCE = 10, ACOUSTIC = 11, TEMPO = 15;
export const FEAT_COLS = [8, 9, 10, 11, 12, 13, 14, 15, 16]; // dance..loudness

export const store = { data: null, byId: new Map(), n0: 0 };

export async function loadSongs() {
  const r = await fetch("songs.json");
  if (!r.ok) throw new Error("songs.json " + r.status);
  store.data = await r.json();
  store.data.tracks.forEach((row, i) => store.byId.set(row[ID], i));
  store.n0 = store.data.tracks.length;
  return store.data;
}

export const row = (t) => store.data.tracks[t];
export const indexOf = (id) => store.byId.get(id);

/** Songs returned by the API can be outside songs.json (the model catalog has every train+val song,
 *  songs.json only popularity >= 40). Add a minimal row so every view can show them. */
export function ensureRow({ track_id, track_name, artist }) {
  let t = store.byId.get(track_id);
  if (t === undefined) {
    t = store.data.tracks.length;
    store.data.tracks.push([track_id, track_name, artist, "", -1, -1, 0, 0]);
    store.byId.set(track_id, t);
  }
  return t;
}
export const hasFeatures = (t) => row(t)[GENRE] >= 0;

export const $ = (s, root = document) => root.querySelector(s);
export const $$ = (s, root = document) => [...root.querySelectorAll(s)];
export const shuffle = (a) => { for (let i = a.length - 1; i > 0; i--) { const j = (Math.random() * (i + 1)) | 0; [a[i], a[j]] = [a[j], a[i]]; } return a; };
export const norm = (s) => (s || "").toLowerCase().normalize("NFKD").replace(/[̀-ͯ]/g, "").replace(/[^\p{L}\p{N} ]/gu, " ").replace(/\s+/g, " ").trim();
export const cleanName = (s) => s.replace(/\s*[([].*?[)\]]/g, "").split(" - ")[0].trim() || s;
export const firstArtist = (s) => s.split(", ")[0];
export const prettyGenre = (g) => g.replace(/-n-/g, " & ").replace(/^r & b$/, "R&B").replace(/-/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
export const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
export const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
export const spotifyUrl = (t) => `https://open.spotify.com/track/${row(t)[ID]}`;
export const genreName = (t) => (hasFeatures(t) ? prettyGenre(store.data.genres[row(t)[GENRE]]) : "");
export const familyOf = (t) => (hasFeatures(t) ? store.data.genreFamily[row(t)[GENRE]] : -1);

export function toast(msg, ms = 2600) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toast.t);
  toast.t = setTimeout(() => el.classList.remove("show"), ms);
}

function hashHue(str) { let h = 0; for (const c of str) h = (h * 31 + c.charCodeAt(0)) | 0; return Math.abs(h) % 360; }
export function fallbackArt(t) {
  const h = hashHue(row(t)[ID]);
  return `linear-gradient(150deg, hsl(${h} 75% 55%), hsl(${(h + 60) % 360} 70% 35%))`;
}

export function tagsFor(t) {
  const r = row(t);
  if (!hasFeatures(t)) return [];
  const tags = [genreName(t)];
  if (r[ENERGY] > 0.78) tags.push("⚡ High energy");
  else if (r[ENERGY] < 0.35) tags.push("🌙 Mellow");
  if (r[DANCE] > 0.72) tags.push("💃 Danceable");
  if (r[VALENCE] > 0.7) tags.push("☀️ Feel-good");
  else if (r[VALENCE] < 0.25) tags.push("🌧 Moody");
  if (r[ACOUSTIC] > 0.7 && tags.length < 3) tags.push("🎸 Acoustic");
  return tags.slice(0, 3);
}

/** Pick the most popular tracks, at most `perArtist` per artist and no repeated titles. */
export function topTracks(filter, n, perArtist = 1) {
  const { tracks } = store.data;
  const idx = [];
  for (let i = 0; i < tracks.length; i++) if (tracks[i][POP] >= 0 && filter(i)) idx.push(i);
  idx.sort((a, b) => tracks[b][POP] - tracks[a][POP]);
  const per = {}, names = new Set(), out = [];
  for (const i of idx) {
    const a = firstArtist(tracks[i][ARTIST]), nm = norm(cleanName(tracks[i][NAME]));
    if ((per[a] || 0) >= perArtist || names.has(nm)) continue;
    per[a] = (per[a] || 0) + 1; names.add(nm); out.push(i);
    if (out.length === n) break;
  }
  return out;
}

/** Weighted averages of the audio traits over liked songs (love counts double). */
export function tasteProfile(entries) {
  const withF = entries.filter((e) => hasFeatures(e.t));
  if (!withF.length) return null;
  const wavg = (c) => { let s = 0, w = 0; for (const e of withF) { const x = e.dir === "love" ? 2 : 1; s += row(e.t)[c] * x; w += x; } return s / w; };
  const g = {};
  for (const e of withF) { const k = row(e.t)[GENRE]; g[k] = (g[k] || 0) + (e.dir === "love" ? 2 : 1); }
  return {
    traits: [["Energy", wavg(ENERGY)], ["Danceability", wavg(DANCE)], ["Positivity", wavg(VALENCE)], ["Acoustic", wavg(ACOUSTIC)]],
    tempo: wavg(TEMPO),
    genres: Object.entries(g).sort((a, b) => b[1] - a[1]).slice(0, 5).map(([k]) => prettyGenre(store.data.genres[k])),
  };
}

export function tasteHTML(p) {
  if (!p) return "";
  return `<h4>Your taste</h4>
    ${p.traits.map(([n, v]) => `<div class="trait"><span>${n}</span><div class="bar"><i style="width:${Math.round(v * 100)}%"></i></div><span>${Math.round(v * 100)}</span></div>`).join("")}
    <div class="trait"><span>Tempo</span><div></div><span>${Math.round(p.tempo)}</span></div>
    <div class="genres">${p.genres.map((g) => `<span>${esc(g)}</span>`).join("")}</div>`;
}
