/* Pieces shared by the tabs: poster tiles, list rows, My List buttons and the song detail sheet. */
import { $, $$, row, esc, fallbackArt, tagsFor, spotifyUrl, cleanName, NAME, ARTIST, ALBUM, EXPL, ID } from "./data.js";
import { ICON_PLAY, observeArt, syncPlayButtons } from "./preview.js";
import { inMyList, toggleMyList, onAccountChange } from "./auth.js";

const ICON_PLUS = '<svg viewBox="0 0 24 24"><path d="M11 5h2v6h6v2h-6v6h-2v-6H5v-2h6z"/></svg>';
const ICON_CHECK = '<svg viewBox="0 0 24 24"><path d="M9.5 16.2 5.3 12l-1.4 1.4 5.6 5.6L20.1 8.4 18.7 7z"/></svg>';
export const ICON_SPARK = '<svg viewBox="0 0 24 24"><path d="M12 2l1.9 5.6L19.5 9.5l-5.6 1.9L12 17l-1.9-5.6L4.5 9.5l5.6-1.9zM19 14l.9 2.6 2.6.9-2.6.9L19 21l-.9-2.6-2.6-.9 2.6-.9z"/></svg>';

const initial = (t) => esc(cleanName(row(t)[NAME]).charAt(0).toUpperCase());

export const playBtn = (t, cls = "pbtn") =>
  `<button class="${cls}" data-play="${t}" data-label="${esc(row(t)[NAME])}" aria-label="Play preview of ${esc(row(t)[NAME])}"><span class="ico">${ICON_PLAY}</span></button>`;

export const listBtn = (t, source, label = false) => {
  const on = inMyList(row(t)[ID]);
  return `<button class="listbtn${on ? " on" : ""}${label ? " wide" : ""}" data-list="${t}" data-source="${source}" aria-pressed="${on}" title="${on ? "Remove from My List" : "Add to My List"}">${on ? ICON_CHECK : ICON_PLUS}${label ? `<span>${on ? "In My List" : "My List"}</span>` : ""}</button>`;
};

export const similarBtn = (t, label = "Find similar") =>
  `<a class="simbtn" href="#/similar/${row(t)[ID]}">${ICON_SPARK}<span>${label}</span></a>`;

/** Netflix-style square poster for the browse rows. */
export function tileHTML(t, extra = "") {
  const r = row(t);
  return `<div class="tile" role="listitem">
    <button class="tile-art" data-art="${t}" data-open="${t}" style="background-image:${fallbackArt(t)}" aria-label="${esc(r[NAME])} by ${esc(r[ARTIST])}">
      <span class="tile-initial">${initial(t)}</span>${extra}
    </button>
    ${playBtn(t, "pbtn tile-play")}
    <div class="tile-txt"><div class="tile-t">${esc(r[NAME])}</div><div class="tile-a">${esc(r[ARTIST])}</div></div>
  </div>`;
}

/** Compact list row (playlist, search results, My List). `right` is extra HTML for the right column. */
export function rowHTML(t, { n, why = "", right = "", attrs = "" } = {}) {
  const r = row(t);
  return `<li class="row" data-t="${t}" ${attrs}>
    <div class="thumb" data-art="${t}" style="background-image:${fallbackArt(t)}">${playBtn(t, "thumb-play")}</div>
    <div class="txt">
      <div class="t">${n ? `${n}. ` : ""}${esc(r[NAME])}</div>
      <div class="a">${esc(r[ARTIST])}</div>
      ${why ? `<div class="why">${why}</div>` : ""}
    </div>
    <div class="right">${right}</div>
  </li>`;
}

export function hydrate(root) { observeArt(root); syncPlayButtons(); }

/* ---------------------------------------------------- My List buttons */
document.addEventListener("click", async (e) => {
  const b = e.target.closest("[data-list]");
  if (!b) return;
  e.preventDefault(); e.stopPropagation();
  await toggleMyList(row(+b.dataset.list)[ID], b.dataset.source || "browse");
});
function syncListButtons() {
  for (const b of $$("[data-list]")) {
    const t = +b.dataset.list, on = inMyList(row(t)[ID]);
    if (b.classList.contains("on") === on) continue;
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", on);
    b.title = on ? "Remove from My List" : "Add to My List";
    b.innerHTML = (on ? ICON_CHECK : ICON_PLUS) + (b.classList.contains("wide") ? `<span>${on ? "In My List" : "My List"}</span>` : "");
  }
}
onAccountChange(syncListButtons);

/* ---------------------------------------------------- detail sheet */
const sheet = $("#sheet");
export function openSheet(t) {
  const r = row(t);
  $("#sheetBody").innerHTML = `
    <div class="sheet-art" data-art="${t}" style="background-image:${fallbackArt(t)}"><span class="tile-initial">${initial(t)}</span></div>
    <div class="sheet-info">
      <h2>${esc(r[NAME])}</h2>
      <p class="sheet-artist">${esc(r[ARTIST])}</p>
      ${r[ALBUM] && r[ALBUM] !== r[NAME] ? `<p class="sheet-album">from <i>${esc(r[ALBUM])}</i></p>` : ""}
      <div class="meta">${r[EXPL] ? '<span class="tag e">E</span>' : ""}${tagsFor(t).map((x) => `<span class="tag">${esc(x)}</span>`).join("")}</div>
      <div class="sheet-actions">
        <button class="cta cta-sm" data-play="${t}" data-label="${esc(r[NAME])}"><span class="ico">${ICON_PLAY}</span><span>Preview</span></button>
        ${similarBtn(t)}
        ${listBtn(t, "browse", true)}
        <a class="sp" href="${spotifyUrl(t)}" target="_blank" rel="noopener">Open in Spotify</a>
      </div>
    </div>`;
  sheet.hidden = false;
  document.body.classList.add("noscroll");
  hydrate(sheet);
  $("#sheetClose").focus();
}
export function closeSheet() { sheet.hidden = true; document.body.classList.remove("noscroll"); }
$("#sheetClose").addEventListener("click", closeSheet);
sheet.addEventListener("click", (e) => { if (e.target === sheet || e.target.closest(".simbtn")) closeSheet(); });
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !sheet.hidden) closeSheet(); });
document.addEventListener("click", (e) => {
  const o = e.target.closest("[data-open]");
  if (o && !e.target.closest("[data-play],[data-list]")) openSheet(+o.dataset.open);
});
