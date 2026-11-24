/* Home: Netflix-style browse. A featured hero, then horizontally scrolling rows ("shelves") of songs. */
import { $, $$, store, row, esc, shuffle, cleanName, firstArtist, topTracks, tagsFor, fallbackArt, ensureRow, familyOf, indexOf,
  NAME, ARTIST, POP, EXPL } from "./data.js";
import { ICON_PLAY } from "./preview.js";
import { tileHTML, hydrate, listBtn, similarBtn } from "./ui.js";
import { api, capIds } from "./api.js";
import { account, onAccountChange, likedEntries, signedIn } from "./auth.js";
import { loadPrefs, langOk } from "./prefs.js";

const ROW_LEN = 20;
let built = false;

function shelfHTML(title, tiles, { sub = "", id = "", cls = "" } = {}) {
  return `<section class="shelf ${cls}" ${id ? `id="${id}"` : ""}>
    <div class="shelf-head"><h3>${title}</h3>${sub ? `<span class="shelf-sub">${sub}</span>` : ""}</div>
    <div class="shelf-wrap">
      <button class="shelf-arrow l" aria-label="Scroll left">‹</button>
      <div class="shelf-row" role="list">${tiles}</div>
      <button class="shelf-arrow r" aria-label="Scroll right">›</button>
    </div>
  </section>`;
}
const skeleton = (n = 8) => Array.from({ length: n }, () => '<div class="tile skel"><div class="tile-art"></div><div class="tile-txt"><i></i><i></i></div></div>').join("");

function heroHTML(t) {
  const r = row(t);
  return `<div class="hero-bg" data-art="${t}" style="background-image:${fallbackArt(t)}"></div>
    <div class="hero-fade"></div>
    <div class="hero-content">
      <span class="hero-kicker">Featured song</span>
      <h1 class="hero-title">${esc(cleanName(r[NAME]))}</h1>
      <p class="hero-artist">${esc(r[ARTIST])}</p>
      <div class="meta">${r[EXPL] ? '<span class="tag e">E</span>' : ""}${tagsFor(t).map((x) => `<span class="tag">${esc(x)}</span>`).join("")}</div>
      <div class="hero-actions">
        <button class="cta cta-sm" data-play="${t}" data-label="${esc(r[NAME])}"><span class="ico">${ICON_PLAY}</span><span>Preview</span></button>
        ${similarBtn(t)}
        ${listBtn(t, "browse", true)}
        <button class="ghost-sm" data-open="${t}">More info</button>
      </div>
    </div>
    <div class="hero-poster" data-art="${t}" data-open="${t}" style="background-image:${fallbackArt(t)}"></div>`;
}

function buildStatic() {
  const { families } = store.data;
  const trending = topTracks((i) => langOk(i), ROW_LEN);
  const featured = shuffle(trending.slice(0, 12))[0];
  $("#homeHero").innerHTML = heroHTML(featured);

  // Mid-popularity songs, shuffled: a counterweight to rows that only show hits (popularity bias).
  const pool = shuffle(store.data.tracks.map((_, i) => i).filter((i) => row(i)[POP] >= 40 && row(i)[POP] <= 55 && langOk(i)));
  const seenA = new Set(), gems = [];
  for (const i of pool) { const a = firstArtist(row(i)[ARTIST]); if (seenA.has(a)) continue; seenA.add(a); gems.push(i); if (gems.length === ROW_LEN) break; }

  // v2: the survey's groups come first; "Kids & Sleep" only if picked; rows follow the chosen languages
  const prefs = loadPrefs();
  const fam = families.map((name, f) => ({ name, f }))
    .filter((x) => x.name !== "Other" && (x.name !== "Kids & Sleep" || prefs.families.includes(x.name)))
    .sort((a, b) => prefs.families.includes(b.name) - prefs.families.includes(a.name));
  $("#homeShelves").innerHTML =
    shelfHTML("Trending now", trending.map((t) => tileHTML(t)).join("")) +
    fam.slice(0, 3).map(({ name, f }) => shelfHTML(`Top ${esc(name)}`, topTracks((i) => familyOf(i) === f && langOk(i), ROW_LEN).map((t) => tileHTML(t)).join(""))).join("") +
    shelfHTML("Hidden gems", gems.map((t) => tileHTML(t)).join(""), { sub: "Less-played songs, picked at random" }) +
    fam.slice(3).map(({ name, f }) => shelfHTML(`Top ${esc(name)}`, topTracks((i) => familyOf(i) === f && langOk(i), ROW_LEN).map((t) => tileHTML(t)).join(""))).join("");
  hydrate($("#tab-home"));
  built = true;
}

/* Personal shelves: "Because you liked …" (from the live model) and My List. */
let personalGen = 0;
async function buildPersonal() {
  if (!built) return;
  const gen = ++personalGen;
  const box = $("#homePersonal");
  const likes = likedEntries();
  const list = [...account.myList].sort((a, b) => (b[1].at > a[1].at ? 1 : -1)).map(([id]) => indexOf(id)).filter((t) => t !== undefined);
  const seeds = [];
  for (const e of likes) { if (!seeds.some((s) => firstArtist(row(s.t)[ARTIST]) === firstArtist(row(e.t)[ARTIST]))) seeds.push(e); if (seeds.length === 2) break; }

  box.innerHTML =
    seeds.map((s, k) => shelfHTML(`Because you liked <em>${esc(cleanName(row(s.t)[NAME]))}</em>`, skeleton(), { id: `byl${k}`, sub: "From the SwipeTape model" })).join("") +
    (list.length ? shelfHTML("My List", list.map((t) => tileHTML(t)).join("")) : "") +
    (!seeds.length && !list.length ? `<div class="home-nudge">
        <p><b>Make Home yours.</b> Like a few songs on the <a href="#/swipe">Swipe</a> tab and we'll add rows picked by the model just for you${signedIn() ? "" : ` (<a href="#/me">sign in</a> to keep them)`}.</p></div>` : "");
  hydrate(box);

  const swiped = [...account.swipes.keys()];
  await Promise.all(seeds.map(async (s, k) => {
    const el = $(`#byl${k} .shelf-row`);
    try {
      const recs = await api.recommend([s.id], capIds(swiped.filter((x) => x !== s.id)), 15);
      if (gen !== personalGen) return;
      el.innerHTML = recs.map((r) => tileHTML(ensureRow(r))).join("");
      hydrate(el);
    } catch (e) {
      if (gen !== personalGen) return;
      el.innerHTML = `<div class="shelf-error">Couldn't load these picks: ${esc(e.message)} <button class="linkbtn" data-retry-home>Retry</button></div>`;
    }
  }));
}

// Personal rows are rebuilt when Home is shown after something changed (not on every swipe elsewhere).
let active = false, dirty = true, pending = null;
const markDirty = () => {
  dirty = true;
  if (active) { clearTimeout(pending); pending = setTimeout(() => { dirty = false; buildPersonal(); }, 150); }
};
onAccountChange(markDirty);
document.addEventListener("swipetape:swiped", () => { dirty = true; });
document.addEventListener("swipetape:prefs", () => { built = false; dirty = true; });   // survey redone: rebuild rows

document.addEventListener("click", (e) => {
  const a = e.target.closest(".shelf-arrow");
  if (a) { const r = a.parentElement.querySelector(".shelf-row"); r.scrollBy({ left: (a.classList.contains("l") ? -1 : 1) * r.clientWidth * 0.85, behavior: "smooth" }); }
  if (e.target.closest("[data-retry-home]")) buildPersonal();
});

export const home = {
  enter() {
    active = true;
    if (!built) buildStatic();
    if (dirty) { dirty = false; buildPersonal(); }
  },
  leave() { active = false; },
};
