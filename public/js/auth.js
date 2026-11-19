/* Accounts (Supabase Auth) and the signed-in user's swipes and My List.
 * Guests can use every tab; their swipes live only in this browser session. */
import { CONFIG } from "../config.js";
import { indexOf, toast } from "./data.js";

const SUPABASE_JS = "https://esm.sh/@supabase/supabase-js@2.49.4";

export const account = {
  enabled: false,          // true when a publishable key is configured and supabase-js loaded
  client: null,
  user: null,
  profile: null,
  swipes: new Map(),       // track_id -> {dir, at}   (account history, or this session's for guests)
  myList: new Map(),       // track_id -> {source, at}
};
const listeners = new Set();
export const onAccountChange = (fn) => { listeners.add(fn); };
const emit = () => listeners.forEach((fn) => { try { fn(account); } catch (e) { console.error(e); } });

export async function initAuth() {
  if (!CONFIG.SUPABASE_PUBLISHABLE_KEY) { emit(); return; }
  try {
    const { createClient } = await import(SUPABASE_JS);
    account.client = createClient(CONFIG.SUPABASE_URL, CONFIG.SUPABASE_PUBLISHABLE_KEY, {
      auth: { persistSession: true, autoRefreshToken: true, detectSessionInUrl: true, flowType: "pkce" },
    });
    account.enabled = true;
    const { data: { session } } = await account.client.auth.getSession();
    await setUser(session?.user || null);
    account.client.auth.onAuthStateChange((_event, s) => {
      const u = s?.user || null;
      if ((u?.id || null) !== (account.user?.id || null)) setTimeout(() => setUser(u), 0);  // not inside the callback (supabase-js deadlock note)
    });
  } catch (e) {
    console.error("auth init failed", e);
    account.enabled = false;
    emit();
  }
}

async function setUser(user) {
  account.user = user;
  account.profile = null;
  if (!user) { account.swipes = new Map(); account.myList = new Map(); emit(); return; }
  const c = account.client;
  const [p, s, l] = await Promise.all([
    c.from("profiles").select("display_name").eq("id", user.id).maybeSingle(),
    c.from("swipes").select("track_id, dir, created_at").order("created_at", { ascending: true }).limit(5000),
    c.from("my_list").select("track_id, source, created_at").order("created_at", { ascending: true }).limit(5000),
  ]);
  if (p.error || s.error || l.error) {
    console.error(p.error || s.error || l.error);
    toast("Signed in, but your saved songs couldn't be loaded. Is sql/accounts.sql set up?", 4500);
  }
  account.profile = p.data || null;
  // Keep songs from this session that the guest swiped before signing in, then the account's history.
  const guest = account.swipes;
  account.swipes = new Map((s.data || []).map((r) => [r.track_id, { dir: r.dir, at: r.created_at }]));
  account.myList = new Map((l.data || []).map((r) => [r.track_id, { source: r.source, at: r.created_at }]));
  for (const [id, v] of guest) if (!account.swipes.has(id)) saveSwipe(id, v.dir);
  emit();
}

export const signedIn = () => !!account.user;
export const displayName = () =>
  account.profile?.display_name || account.user?.user_metadata?.full_name || account.user?.email?.split("@")[0] || "Guest";

/* ------------------------------------------------------------ sign in/out */
function needClient() { if (!account.enabled) throw new Error("Accounts aren't configured on this site yet."); return account.client; }
const redirectTo = () => location.origin + location.pathname;

export async function signUp(email, password, name) {
  const { data, error } = await needClient().auth.signUp({
    email, password, options: { data: { full_name: name || undefined }, emailRedirectTo: redirectTo() },
  });
  if (error) throw error;
  return { needsConfirm: !data.session };
}
export async function signIn(email, password) {
  const { error } = await needClient().auth.signInWithPassword({ email, password });
  if (error) throw error;
}
export async function signInWithGoogle() {
  sessionStorage.setItem("swipetape.afterLogin", location.hash || "#/home");
  const { error } = await needClient().auth.signInWithOAuth({ provider: "google", options: { redirectTo: redirectTo() } });
  if (error) throw error;
}
export async function signOut() { if (account.client) await account.client.auth.signOut(); }

/* ------------------------------------------------------------ swipes */
export async function saveSwipe(trackId, dir) {
  account.swipes.set(trackId, { dir, at: new Date().toISOString() });
  if (!account.user) return;
  const { error } = await account.client.from("swipes").upsert({ user_id: account.user.id, track_id: trackId, dir, created_at: new Date().toISOString() });
  if (error) console.error("saveSwipe", error);
}
export async function removeSwipe(trackId) {
  account.swipes.delete(trackId);
  if (!account.user) return;
  const { error } = await account.client.from("swipes").delete().eq("user_id", account.user.id).eq("track_id", trackId);
  if (error) console.error("removeSwipe", error);
}
/** Liked/loved songs as track indices, most recent first. */
export function likedEntries() {
  return [...account.swipes].filter(([, v]) => v.dir !== "nope")
    .sort((a, b) => (b[1].at > a[1].at ? 1 : -1))
    .map(([id, v]) => ({ t: indexOf(id), id, dir: v.dir }))
    .filter((e) => e.t !== undefined);
}

/* ------------------------------------------------------------ My List */
export const inMyList = (trackId) => account.myList.has(trackId);

/** Add or remove a song; returns the new state, or null if the user has to sign in first. */
export async function toggleMyList(trackId, source = "browse") {
  if (!account.user) {
    toast(account.enabled ? "Sign in on the Me tab to save songs to My List." : "Accounts aren't set up on this site yet.", 3200);
    return null;
  }
  const c = account.client;
  if (account.myList.has(trackId)) {
    account.myList.delete(trackId);
    emit();
    const { error } = await c.from("my_list").delete().eq("user_id", account.user.id).eq("track_id", trackId);
    if (error) { toast("Couldn't update My List."); console.error(error); }
    return false;
  }
  account.myList.set(trackId, { source, at: new Date().toISOString() });
  emit();
  const { error } = await c.from("my_list").upsert({ user_id: account.user.id, track_id: trackId, source });
  if (error) { toast("Couldn't save to My List."); console.error(error); account.myList.delete(trackId); emit(); }
  return true;
}
