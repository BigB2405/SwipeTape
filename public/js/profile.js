/* Me: sign in / create an account, then the user's stats, taste, My List and liked songs. */
import { CONFIG } from "../config.js";
import { $, esc, indexOf, tasteProfile, tasteHTML, spotifyUrl, toast } from "./data.js";
import { account, onAccountChange, signIn, signUp, signInWithGoogle, signOut, likedEntries, displayName } from "./auth.js";
import { rowHTML, listBtn, similarBtn, hydrate } from "./ui.js";

const GOOGLE = '<svg viewBox="0 0 24 24" aria-hidden="true"><path fill="#4285F4" d="M23.5 12.3c0-.8-.1-1.6-.2-2.3H12v4.4h6.5a5.6 5.6 0 0 1-2.4 3.6v3h3.9c2.2-2.1 3.5-5.1 3.5-8.7z"/><path fill="#34A853" d="M12 24c3.2 0 6-1.1 8-2.9l-3.9-3c-1.1.7-2.5 1.2-4.1 1.2-3.1 0-5.8-2.1-6.7-5H1.3v3.1A12 12 0 0 0 12 24z"/><path fill="#FBBC05" d="M5.3 14.3a7.2 7.2 0 0 1 0-4.6V6.6h-4a12 12 0 0 0 0 10.8z"/><path fill="#EA4335" d="M12 4.8c1.8 0 3.3.6 4.6 1.8l3.4-3.4A12 12 0 0 0 1.3 6.6l4 3.1c.9-2.9 3.6-4.9 6.7-4.9z"/></svg>';
let mode = "signin";   // or "signup"
let active = false;

function authFormHTML() {
  if (!account.enabled) {
    return `<div class="auth-card"><h2>Accounts are coming soon</h2>
      <p class="muted">This copy of SwipeTape isn't connected to the account service yet, so everything works in guest mode: your swipes last until you close the tab.</p></div>`;
  }
  const up = mode === "signup";
  return `<div class="auth-card">
    <h2>${up ? "Create your SwipeTape account" : "Welcome back"}</h2>
    <p class="muted">${up ? "Save your swipes and My List, and get Home rows picked for you." : "Sign in to see your swipes, My List and your picks."}</p>
    ${CONFIG.GOOGLE_LOGIN ? `<button class="google" id="googleBtn">${GOOGLE}<span>Continue with Google</span></button>
    <div class="or"><span>or</span></div>` : ""}
    <form id="authForm" novalidate>
      ${up ? '<label>Name<input name="name" autocomplete="name" maxlength="60" placeholder="What should we call you?"></label>' : ""}
      <label>Email<input name="email" type="email" autocomplete="email" required placeholder="you@example.com"></label>
      <label>Password<input name="password" type="password" autocomplete="${up ? "new-password" : "current-password"}" required minlength="6" placeholder="${up ? "At least 6 characters" : ""}"></label>
      <p class="form-msg" id="authMsg" role="alert"></p>
      <button class="cta" type="submit"><span class="cta-text">${up ? "Create account" : "Sign in"}</span></button>
    </form>
    <p class="switch">${up ? 'Already have an account? <button class="linkbtn" data-mode="signin">Sign in</button>' : 'New here? <button class="linkbtn" data-mode="signup">Create an account</button>'}</p>
  </div>`;
}

function meHTML() {
  const likes = likedEntries();
  const loves = likes.filter((e) => e.dir === "love").length;
  const nopes = [...account.swipes.values()].filter((v) => v.dir === "nope").length;
  const list = [...account.myList].sort((a, b) => (b[1].at > a[1].at ? 1 : -1)).map(([id]) => indexOf(id)).filter((t) => t !== undefined);
  const name = displayName();
  const taste = tasteProfile(likes);
  return `<div class="me-head">
      <div class="avatar">${esc(name.charAt(0).toUpperCase())}</div>
      <div><h2>${esc(name)}</h2><p class="muted">${esc(account.user.email || "")}</p></div>
      <button class="ghost-sm" id="signOutBtn">Sign out</button>
    </div>
    <div class="stats">
      <div><b>${account.swipes.size}</b><span>songs swiped</span></div>
      <div><b>${likes.length - loves}</b><span>liked</span></div>
      <div><b>${loves}</b><span>loved</span></div>
      <div><b>${list.length}</b><span>in My List</span></div>
    </div>
    ${taste ? `<div class="taste">${tasteHTML(taste)}</div>` : ""}
    <section class="me-sec">
      <div class="pl-head"><h3>My List</h3><span>${list.length} songs</span></div>
      ${list.length ? `<ol class="playlist">${list.map((t) => rowHTML(t, { right: `<span class="row-btns">${listBtn(t, "browse")}<a class="sp" href="${spotifyUrl(t)}" target="_blank" rel="noopener">Spotify</a></span>` })).join("")}</ol>
        <button class="ghost-sm" id="copyListBtn">Copy Spotify links</button>`
      : '<p class="empty-state small">Nothing saved yet. Tap <b>+</b> on any song to add it here.</p>'}
    </section>
    <section class="me-sec">
      <div class="pl-head"><h3>Songs you liked</h3><span>${likes.length} songs · ${nopes} passed</span></div>
      ${likes.length ? `<ol class="playlist">${likes.slice(0, 100).map((e) => rowHTML(e.t, { why: e.dir === "love" ? "★ Loved" : "", right: `<span class="row-btns">${similarBtn(e.t, "Similar")}${listBtn(e.t, "swipe")}</span>` })).join("")}</ol>`
      : '<p class="empty-state small">Head to <a href="#/swipe">Swipe</a> to start rating songs.</p>'}
    </section>`;
}

function render() {
  if (!active) return;
  const box = $("#meBody");
  box.innerHTML = account.user ? meHTML() : authFormHTML();
  hydrate(box);
}
onAccountChange(render);

document.addEventListener("click", async (e) => {
  if (!e.target.closest("#tab-me")) return;
  const m = e.target.closest("[data-mode]");
  if (m) { mode = m.dataset.mode; render(); return; }
  if (e.target.closest("#googleBtn")) {
    try { await signInWithGoogle(); } catch (err) { $("#authMsg").textContent = err.message; }
  }
  if (e.target.closest("#signOutBtn")) await signOut();
  if (e.target.closest("#copyListBtn")) {
    const text = [...account.myList.keys()].map((id) => `https://open.spotify.com/track/${id}`).join("\n");
    try { await navigator.clipboard.writeText(text); } catch { /* ignore */ }
    toast("Copied! Paste into a Spotify playlist.");
  }
});

document.addEventListener("submit", async (e) => {
  if (e.target.id !== "authForm") return;
  e.preventDefault();
  const f = new FormData(e.target);
  const email = String(f.get("email") || "").trim(), password = String(f.get("password") || "");
  const msg = $("#authMsg"), btn = e.target.querySelector("button[type=submit]");
  msg.className = "form-msg";
  if (!/^\S+@\S+\.\S+$/.test(email)) { msg.textContent = "Enter a valid email address."; return; }
  if (password.length < 6) { msg.textContent = "Passwords need at least 6 characters."; return; }
  btn.disabled = true; btn.querySelector(".cta-text").textContent = mode === "signup" ? "Creating account…" : "Signing in…";
  try {
    if (mode === "signup") {
      const { needsConfirm } = await signUp(email, password, String(f.get("name") || "").trim());
      if (needsConfirm) {
        mode = "signin";
        render();
        $("#authForm [name=email]").value = email;
        $("#authMsg").className = "form-msg ok";
        $("#authMsg").textContent = `Almost done: we sent a confirmation link to ${email}. Open it, then sign in here.`;
        return;
      }
    } else {
      await signIn(email, password);
    }
  } catch (err) {
    msg.textContent = /invalid login/i.test(err.message) ? "That email and password don't match an account." : err.message;
  } finally {
    if (document.body.contains(btn)) { btn.disabled = false; btn.querySelector(".cta-text").textContent = mode === "signup" ? "Create account" : "Sign in"; }
  }
});

export const profile = {
  enter() { active = true; render(); },
  leave() { active = false; },
};
