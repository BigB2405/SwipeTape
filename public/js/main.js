/* Boot: load the catalog, start auth, warm up the API, and route between the four tabs.
 *   #/home   browse (Netflix-style)      #/swipe   the deck
 *   #/similar[/<track_id>]  kNN          #/me      account   */
import { $, $$, loadSongs } from "./data.js";
import { stopAudio, now } from "./preview.js";
import { initAuth, onAccountChange, account, displayName } from "./auth.js";
import { api } from "./api.js";
import { closeSheet } from "./ui.js";
import { home } from "./home.js";
import { swipeTab } from "./swipe.js";
import { similar } from "./similar.js";
import { profile } from "./profile.js";

const TABS = { home, swipe: swipeTab, similar, me: profile };
let current = null;

function parse() {
  const [, tab = "home", param] = (location.hash || "#/home").match(/^#\/?([a-z]*)\/?([A-Za-z0-9]*)/) || [];
  return { tab: TABS[tab] ? tab : "home", param: param || undefined };
}

function route() {
  const { tab, param } = parse();
  closeSheet();
  if (current !== tab) {
    if (current) TABS[current].leave();
    if (now.owner === "list" && tab === "swipe") stopAudio();   // the deck plays its own clips
    for (const k of Object.keys(TABS)) $("#tab-" + k).hidden = k !== tab;
    for (const a of $$("[data-tab]")) { const on = a.dataset.tab === tab; a.classList.toggle("on", on); on ? a.setAttribute("aria-current", "page") : a.removeAttribute("aria-current"); }
    document.body.dataset.tab = tab;
    current = tab;
    window.scrollTo(0, 0);
  }
  TABS[tab].enter(param);
}

onAccountChange(() => {
  const b = $("#accountBtn");
  b.textContent = account.user ? displayName().charAt(0).toUpperCase() : "";
  b.classList.toggle("signed-in", !!account.user);
  b.title = account.user ? `Signed in as ${displayName()}` : "Sign in";
  b.setAttribute("aria-label", b.title);
});

(async function init() {
  api.health().catch(() => {});        // wake the Modal container while the catalog loads
  try {
    await loadSongs();
  } catch {
    $("#bootMsg").innerHTML = "Couldn't load songs. <a href=''>Refresh to retry</a>.";
    return;
  }
  $("#boot").hidden = true;
  await initAuth();
  const back = sessionStorage.getItem("swipetape.afterLogin");    // returning from Google sign-in
  if (back && account.user) { sessionStorage.removeItem("swipetape.afterLogin"); history.replaceState(null, "", location.pathname + back); }
  window.addEventListener("hashchange", route);
  route();
})();
