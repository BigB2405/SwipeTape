/* Taste survey answers, kept in this browser (localStorage).
 *   languages: estimated language codes ([] = any language)   families: genre group names
 *   seeds: track ids the user tapped in the survey             done: survey finished or skipped */
import { store, row } from "./data.js";

const KEY = "swipetape.prefs.v1";
const EMPTY = { done: false, languages: [], families: [], seeds: [] };

export const LANG_NAMES = {
  en: "English", es: "Spanish", pt: "Portuguese", hi: "Hindi / Punjabi", ko: "Korean", ja: "Japanese",
  zh: "Chinese", fr: "French", it: "Italian", de: "German", sv: "Swedish", tr: "Turkish",
};
export const SURVEY_LANGS = ["en", "es", "pt", "hi", "ko", "ja", "zh", "fr"];

export function loadPrefs() {
  try { return { ...EMPTY, ...JSON.parse(localStorage.getItem(KEY) || "{}") }; } catch { return { ...EMPTY }; }
}
export function savePrefs(p) {
  try { localStorage.setItem(KEY, JSON.stringify(p)); } catch { /* private mode: answers last this visit only */ }
  document.dispatchEvent(new CustomEvent("swipetape:prefs"));
}
export const myLanguages = () => loadPrefs().languages;

// iTunes genre -> language, checked by prefix (keep in step with pipeline_def.ITUNES_GENRE_LANG)
const ITUNES_GENRE_LANG = [
  [["punjabi", "regional indian", "indian", "bollywood", "tamil", "telugu", "malayalam", "kannada", "bengali",
    "ghazal", "sufi", "devotional & spiritual", "carnatic", "hindustani"], "hi"],
  [["latin", "urbano latino", "música mexicana", "musica mexicana", "regional mexicano", "reggaeton", "salsa",
    "pop latino", "flamenco", "tango", "cumbia", "bachata"], "es"],
  [["brazilian", "mpb", "sertanejo", "pagode", "samba", "funk brasileiro", "axé", "axe", "forró", "forro", "bossa nova"], "pt"],
  [["k-pop"], "ko"], [["j-pop", "anime", "enka", "kayokyoku", "j-rock"], "ja"],
  [["c-pop", "mandopop", "cantopop", "chinese", "taiwanese"], "zh"],
  [["french pop", "chanson", "variété française", "french"], "fr"], [["german pop", "schlager", "german"], "de"],
  [["turkish", "arabesque"], "tr"], [["arabic", "khaleeji"], "ar"], [["russian"], "ru"], [["italian"], "it"],
  [["worldwide"], "xx"],
];
export function itunesLang(genre) {
  const g = (genre || "").toLowerCase();
  for (const [prefixes, code] of ITUNES_GENRE_LANG) if (prefixes.some((p) => g.startsWith(p))) return code;
  return null;
}
/** Second check once a song's preview is looked up: its own iTunes genre (e.g. "Punjabi Pop") must fit
 *  the chosen languages. "Worldwide" (xx) is allowed only if a non-English language was chosen. */
export function previewLangOk(pv, languages = myLanguages()) {
  if (!languages.length || !pv) return true;
  const code = itunesLang(pv.genre);
  if (!code) return true;
  if (code === "xx") return languages.some((c) => c !== "en");
  return languages.includes(code);
}

const LANG_COL = 17;   // songs.json row: list of language indexes (scripts/build_data.py)
/** Estimated language codes of a catalog track ([] for songs added from API results). */
export function langsOf(t) {
  const r = row(t), L = store.data.langs || [];
  return Array.isArray(r[LANG_COL]) ? r[LANG_COL].map((i) => L[i]) : [];
}
/** True if the track matches the chosen languages (always true when none are chosen). */
export function langOk(t, languages = myLanguages()) {
  if (!languages.length) return true;
  const ls = langsOf(t);
  return !ls.length || ls.some((c) => languages.includes(c));
}
