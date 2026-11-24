/* 30-second previews (iTunes, then Deezer via /api/preview) and cover art (Spotify oEmbed), plus the one
 * shared <audio> player. Any element with data-play="<track index>" becomes a play/pause button for it. */
import { $, $$, row, cleanName, firstArtist, norm, esc, toast, ID, NAME, ARTIST } from "./data.js";

export const player = $("#player");
export const ICON_PAUSE = '<svg viewBox="0 0 24 24"><path d="M7 4h3.5v16H7zM13.5 4H17v16h-3.5z"/></svg>';
export const ICON_PLAY = '<svg viewBox="0 0 24 24"><path d="M7 4.5v15a1 1 0 0 0 1.5.86l12.5-7.5a1 1 0 0 0 0-1.72L8.5 3.64A1 1 0 0 0 7 4.5z"/></svg>';
export const ICON_REPLAY = '<svg viewBox="0 0 24 24"><path d="M12 5V2L7 6l5 4V7a6 6 0 1 1-6 6H4a8 8 0 1 0 8-8z"/></svg>';

/* ------------------------------------------------------------- lookups */
const previewCache = new Map();
const artCache = new Map();

async function lookupITunes(r) {
  const name = cleanName(r[NAME]);
  const artist = firstArtist(r[ARTIST]);
  const url = `https://itunes.apple.com/search?media=music&entity=song&limit=15&term=${encodeURIComponent(`${artist} ${name}`)}`;
  const res = await fetch(url);
  if (!res.ok) throw new Error("itunes " + res.status);
  const { results = [] } = await res.json();
  const na = norm(artist), nn = norm(name);
  const withPrev = results.filter((x) => x.previewUrl);
  const hit =
    withPrev.find((x) => norm(x.artistName).includes(na) && norm(cleanName(x.trackName)) === nn) ||
    withPrev.find((x) => norm(x.artistName).includes(na) && norm(x.trackName).startsWith(nn)) ||
    withPrev.find((x) => norm(x.artistName).includes(na) && norm(x.trackName).includes(nn)) ||
    withPrev.find((x) => norm(x.artistName).includes(na) && nn.startsWith(norm(cleanName(x.trackName))));
  if (!hit) return null;
  return { preview: hit.previewUrl, art: hit.artworkUrl100?.replace("100x100bb", "600x600bb") || null,
           apple: hit.trackViewUrl || null,     // v2: the song's Apple Music page
           genre: hit.primaryGenreName || null }; // v2: e.g. "Punjabi Pop", used for the language check
}

async function lookupDeezer(r) {
  const res = await fetch(`/api/preview?artist=${encodeURIComponent(r[ARTIST])}&track=${encodeURIComponent(cleanName(r[NAME]))}`);
  if (!res.ok) return null;
  return res.json();
}

export async function lookupSpotifyArt(t) {
  if (artCache.has(t)) return artCache.get(t);
  const p = (async () => {
    try {
      const res = await fetch(`https://open.spotify.com/oembed?url=https://open.spotify.com/track/${row(t)[ID]}`);
      if (!res.ok) return null;
      return (await res.json()).thumbnail_url || null;
    } catch { return null; }
  })();
  artCache.set(t, p);
  return p;
}

export function getPreview(t) {
  if (previewCache.has(t)) return previewCache.get(t);
  const r = row(t);
  const p = (async () => {
    let res = null;
    try { res = await lookupITunes(r); } catch { /* rate limited or offline: try Deezer */ }
    if (!res) { try { res = await lookupDeezer(r); } catch { res = null; } }
    if (res && !res.art) res.art = await lookupSpotifyArt(t);
    if (res?.art && !artCache.has(t)) artCache.set(t, Promise.resolve(res.art));
    return res;
  })();
  previewCache.set(t, p);
  return p;
}

/** v2: Apple Music page of a song (from the same iTunes lookup as the preview), or null. */
export async function getAppleUrl(t) { try { return (await getPreview(t))?.apple || null; } catch { return null; } }

/** Cover art for a tile: whatever is already known, else Spotify oEmbed. */
export function getArt(t) { return artCache.get(t) || lookupSpotifyArt(t); }

/* ------------------------------------------------------- shared player */
export const now = { t: null, owner: null };   // owner: "swipe" (deck card) or "list" (any data-play button)

export function stopAudio() {
  player.pause();
  now.t = null; now.owner = null;
  syncPlayButtons();
}

/** Play/pause the preview of track t from any list, tile or sheet. */
export async function togglePreview(t) {
  if (now.t === t && now.owner === "list") {
    if (player.paused || player.ended) { if (player.ended) player.currentTime = 0; player.play().catch(() => {}); }
    else player.pause();
    return;
  }
  player.pause();
  now.t = t; now.owner = "list";
  syncPlayButtons(true);
  const res = await getPreview(t);
  if (now.t !== t) return;
  if (!res?.preview) {
    toast("No preview for this one. Try it on Spotify.");
    now.t = null; now.owner = null; syncPlayButtons();
    return;
  }
  player.src = res.preview;
  player.currentTime = 0;
  player.play().catch(() => syncPlayButtons());
}

export function syncPlayButtons(loading = false) {
  const playing = !player.paused && !player.ended;
  for (const b of $$("[data-play]")) {
    const mine = now.owner === "list" && +b.dataset.play === now.t;
    const on = mine && (playing || loading);
    b.classList.toggle("on", on);
    b.classList.toggle("loading", mine && loading);
    const svg = on ? ICON_PAUSE : ICON_PLAY;
    if (b.dataset.icon !== svg) { b.dataset.icon = svg; const holder = b.querySelector(".ico") || b; holder.innerHTML = svg; }
    b.setAttribute("aria-label", (on ? "Pause preview of " : "Play preview of ") + (b.dataset.label || "song"));
  }
  const bar = $("#nowbar");
  if (now.owner === "list" && now.t !== null) {
    const r = row(now.t);
    bar.hidden = false;
    $("#nowTitle").textContent = r[NAME];
    $("#nowArtist").textContent = r[ARTIST];
    $("#nowBtn").innerHTML = playing || loading ? ICON_PAUSE : ICON_PLAY;
    getArt(now.t).then((src) => { if (now.t !== null && src) $("#nowArt").style.backgroundImage = `url('${esc(src)}')`; });
  } else bar.hidden = true;
}

for (const ev of ["play", "pause", "ended"]) player.addEventListener(ev, () => syncPlayButtons());
player.addEventListener("timeupdate", () => {
  if (now.owner !== "list") return;
  $("#nowProgress").style.width = `${(player.currentTime / (player.duration || 30)) * 100}%`;
});

document.addEventListener("click", (e) => {
  const b = e.target.closest("[data-play]");
  if (b) { e.preventDefault(); e.stopPropagation(); togglePreview(+b.dataset.play); }
});
$("#nowBtn").addEventListener("click", () => { if (now.t !== null) togglePreview(now.t); });
$("#nowClose").addEventListener("click", stopAudio);

/** Lazy-load cover art into every [data-art] element inside root as it scrolls into view. */
const io = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
  for (const en of entries) {
    if (!en.isIntersecting) continue;
    io.unobserve(en.target);
    paintArt(en.target);
  }
}, { rootMargin: "300px" }) : null;

function paintArt(el) {
  const t = +el.dataset.art;
  getArt(t).then((src) => {
    if (!src) return;
    const img = new Image();
    img.onload = () => { el.style.backgroundImage = `url('${esc(src)}')`; el.classList.add("has-art"); };
    img.src = src;
  });
}

export function observeArt(root) {
  for (const el of $$("[data-art]:not([data-art-seen])", root)) {
    el.dataset.artSeen = "1";
    io ? io.observe(el) : paintArt(el);
  }
}
