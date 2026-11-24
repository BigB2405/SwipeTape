/* Taste survey: shown on a first visit (no swipes yet, survey not done), or from "Redo taste survey" on Me.
 *   1. languages   2. genre groups   3. tap songs you know   -> taps become liked songs (saveSwipe),
 * so Home's "Because you liked" rows and the Swipe deck start from real picks instead of guesses. */
import { $, $$, store, row, esc, fallbackArt, cleanName, firstArtist, ID, NAME, ARTIST, POP, GENRE } from "./data.js";
import { account, saveSwipe } from "./auth.js";
import { playBtn, hydrate } from "./ui.js";
import { stopAudio } from "./preview.js";
import { loadPrefs, savePrefs, LANG_NAMES, SURVEY_LANGS, langOk } from "./prefs.js";

const N_SONGS = 12, MIN_POP = 60, MAX_FAMS = 4;
const box = () => $("#onboard");
let st = null;           // { step, languages:Set, families:Set, picks:Set<track index>, shown:[t], round }

export const needsSurvey = () => !loadPrefs().done && account.swipes.size === 0;

export function openSurvey() {
  const p = loadPrefs();
  st = { step: 1, languages: new Set(p.languages), families: new Set(p.families), picks: new Set(), shown: [], round: 0 };
  box().hidden = false;
  document.body.classList.add("noscroll");
  render();
}

function close() {
  stopAudio();
  box().hidden = true;
  document.body.classList.remove("noscroll");
  st = null;
}

const famNames = () => store.data.families.filter((f) => f !== "Other");

/** Most popular songs from the chosen groups and languages, 1 per artist, round-robin over groups. */
function pickSongs() {
  const { tracks, genreFamily, families } = store.data;
  const want = [...st.families].map((f) => families.indexOf(f)).filter((i) => i >= 0);
  const langs = [...st.languages];
  const byFam = new Map(want.map((f) => [f, []]));
  const skip = new Set(st.shown);
  tracks.forEach((r, t) => {
    if (r[POP] < MIN_POP || r[GENRE] < 0 || skip.has(t) || account.swipes.has(r[ID])) return;
    const f = genreFamily[r[GENRE]];
    if (byFam.has(f) && langOk(t, langs)) byFam.get(f).push(t);
  });
  for (const list of byFam.values()) list.sort((a, b) => tracks[b][POP] - tracks[a][POP]);
  const out = [], artists = new Set(), titles = new Set(), lists = [...byFam.values()];
  for (let guard = 0; out.length < N_SONGS && guard < 200 && lists.some((l) => l.length); guard++) {
    for (const list of lists) {
      while (list.length) {
        const t = list.shift(), a = firstArtist(tracks[t][ARTIST]), n = cleanName(tracks[t][NAME]).toLowerCase();
        if (artists.has(a) || titles.has(n)) continue;
        artists.add(a); titles.add(n); out.push(t);
        break;
      }
      if (out.length === N_SONGS) break;
    }
  }
  return out;
}

const chip = (val, label, on, attr) =>
  `<button class="ob-chip${on ? " on" : ""}" ${attr}="${esc(val)}" aria-pressed="${on}">${esc(label)}</button>`;

function songCard(t) {
  const r = row(t), on = st.picks.has(t);
  return `<div class="ob-song${on ? " on" : ""}" data-pick="${t}" role="button" tabindex="0" aria-pressed="${on}" aria-label="${esc(r[NAME])} by ${esc(r[ARTIST])}">
    <div class="ob-art" data-art="${t}" style="background-image:${fallbackArt(t)}">${playBtn(t, "pbtn ob-play")}<span class="ob-check" aria-hidden="true">✓</span></div>
    <div class="ob-t">${esc(cleanName(r[NAME]))}</div><div class="ob-a">${esc(firstArtist(r[ARTIST]))}</div>
  </div>`;
}

function render() {
  const el = box();
  const dots = [1, 2, 3].map((i) => `<i class="${i <= st.step ? "on" : ""}"></i>`).join("");
  let body = "", next = "";
  if (st.step === 1) {
    const any = !st.languages.size;
    body = `<h2>What languages do you listen to?</h2>
      <p class="sub">We'll favor songs in these languages. Pick as many as you like.</p>
      <div class="ob-chips">${chip("any", "Any language", any, "data-lang")}${SURVEY_LANGS.map((c) => chip(c, LANG_NAMES[c], st.languages.has(c), "data-lang")).join("")}</div>
      <p class="fine left">Song languages are estimated from genre tags and titles, so a few may be off.</p>`;
    next = `<button class="cta" data-ob-next><span class="cta-text">Next</span></button>`;
  } else if (st.step === 2) {
    body = `<h2>Pick the music you're into</h2>
      <p class="sub">Choose 1 to ${MAX_FAMS} groups.</p>
      <div class="ob-chips">${famNames().map((f) => chip(f, f, st.families.has(f), "data-fam")).join("")}</div>`;
    next = `<button class="cta" data-ob-next ${st.families.size ? "" : "disabled"}><span class="cta-text">Next</span></button>`;
  } else {
    if (!st.shown.length) st.shown = pickSongs();
    const n = st.picks.size;
    body = `<h2>Tap songs you know and like</h2>
      <p class="sub">Tap the play button to hear a clip. Your picks start your recommendations.</p>
      ${st.shown.length ? `<div class="ob-grid">${st.shown.map(songCard).join("")}</div>
        <button class="linkbtn ob-more" data-ob-more>Show me different songs</button>`
      : `<p class="empty-state">No popular songs match those languages and groups. Go back and pick another language or group.</p>`}`;
    next = `<button class="cta" data-ob-done ${n ? "" : "disabled"}><span class="cta-text">${n ? `Done · ${n} song${n > 1 ? "s" : ""}` : "Pick at least one"}</span></button>`;
  }
  el.innerHTML = `<div class="ob-panel">
    <div class="ob-top">${st.step > 1 ? '<button class="linkbtn" data-ob-back>‹ Back</button>' : "<span></span>"}<div class="ob-dots" aria-label="Step ${st.step} of 3">${dots}</div><button class="linkbtn" data-ob-skip>Skip</button></div>
    <div class="ob-body">${body}</div>
    <div class="ob-foot">${next}</div>
  </div>`;
  hydrate(el);
}

async function finish(skipped) {
  const p = loadPrefs();
  const seeds = [...(st?.picks || [])].map((t) => row(t)[ID]);
  savePrefs({ ...p, done: true, languages: st ? [...st.languages] : p.languages, families: st ? [...st.families] : p.families,
              seeds: skipped ? p.seeds : [...new Set([...seeds, ...p.seeds])].slice(0, 50) });
  for (const id of seeds) await saveSwipe(id, "like");
  if (seeds.length) document.dispatchEvent(new CustomEvent("swipetape:swiped"));
  close();
  if (location.hash !== "#/home") location.hash = "#/home";
  else window.dispatchEvent(new HashChangeEvent("hashchange"));
}

/** Guests' swipes live only in memory, so put the survey picks back after a reload. */
export function restoreGuestSeeds() {
  if (account.user) return;
  for (const id of loadPrefs().seeds) if (!account.swipes.has(id)) account.swipes.set(id, { dir: "like", at: new Date(0).toISOString() });
}

document.addEventListener("click", (e) => {
  if (!st || !e.target.closest("#onboard")) return;
  const t = e.target;
  const lang = t.closest("[data-lang]");
  if (lang) {
    const c = lang.dataset.lang;
    if (c === "any") st.languages.clear();
    else st.languages.has(c) ? st.languages.delete(c) : st.languages.add(c);
    return render();
  }
  const fam = t.closest("[data-fam]");
  if (fam) {
    const f = fam.dataset.fam;
    if (st.families.has(f)) st.families.delete(f);
    else if (st.families.size < MAX_FAMS) st.families.add(f);
    return render();
  }
  if (t.closest("[data-play]")) return;                 // preview button inside a song card
  const pick = t.closest("[data-pick]");
  if (pick) { const i = +pick.dataset.pick; st.picks.has(i) ? st.picks.delete(i) : st.picks.add(i); return render(); }
  if (t.closest("[data-ob-more]")) { st.shown = [...st.shown.filter((x) => st.picks.has(x)), ...pickSongs()].slice(0, N_SONGS + st.picks.size); return render(); }
  if (t.closest("[data-ob-next]")) { if (st.step === 2) st.shown = []; st.step++; return render(); }
  if (t.closest("[data-ob-back]")) { st.step--; return render(); }
  if (t.closest("[data-ob-skip]")) return finish(true);
  if (t.closest("[data-ob-done]")) return finish(false);
});
document.addEventListener("keydown", (e) => {
  if (!st || (e.key !== "Enter" && e.key !== " ")) return;
  const pick = e.target.closest?.("[data-pick]");
  if (pick) { e.preventDefault(); pick.click(); }
});
