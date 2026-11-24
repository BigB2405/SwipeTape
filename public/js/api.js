/* Calls to the live SwipeTape API on Modal (serve.py). Every failure becomes an ApiError with a message
 * a person can read; callers show it as an error state. */
import { CONFIG } from "../config.js";
import { myLanguages } from "./prefs.js";

export class ApiError extends Error {
  constructor(message, status = 0) { super(message); this.status = status; }
}

const SLOW_MS = 3000;        // after this, tell the user the model may be waking up (Modal cold start)
const TIMEOUT_MS = 60000;    // a cold start loads the ~100 MB model, so allow up to a minute

function describe(status, detail) {
  const d = typeof detail === "string" ? detail
    : Array.isArray(detail) ? detail.map((x) => x.msg).filter(Boolean).join("; ") : "";
  if (status === 422) return `The API couldn't use that request${d ? `: ${d}` : "."}`;
  if (status === 503) return "The recommendation model is unavailable right now (503). Please try again shortly.";
  if (status >= 500) return `The recommendation API had a problem (${status}). Please try again.`;
  return `Unexpected API response (${status}).`;
}

async function request(path, { method = "GET", body, onSlow } = {}) {
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), TIMEOUT_MS);
  const slow = onSlow ? setTimeout(onSlow, SLOW_MS) : null;
  let res;
  try {
    res = await fetch(CONFIG.API_URL + path, {
      method,
      headers: body ? { "Content-Type": "application/json" } : undefined,
      body: body ? JSON.stringify(body) : undefined,
      signal: ctl.signal,
    });
  } catch (e) {
    throw new ApiError(e.name === "AbortError"
      ? "The recommendation API took too long to answer. Please try again."
      : "Can't reach the recommendation API. Check your connection and try again.");
  } finally {
    clearTimeout(timer);
    if (slow) clearTimeout(slow);
  }
  if (!res.ok) {
    let detail = null;
    try { detail = (await res.json()).detail; } catch { /* not JSON */ }
    throw new ApiError(describe(res.status, detail), res.status);
  }
  return res.json();
}

export const api = {
  health: () => request("/health"),
  /** -> [{track_id, track_name, artist, match}] (match 0 = exact title ... 4 = words found anywhere) */
  search: (q, limit = 15, opts) => request(`/search?q=${encodeURIComponent(q)}&limit=${limit}`, opts).then((r) => r.tracks),
  /** -> [{track_id, track_name, artist, score, because_you_liked}]
   *  With the v2 API, the survey's languages are sent too (opts.anyLanguage = true leaves them out). */
  recommend: (liked_ids, disliked_ids = [], k = 10, opts = {}) => {
    const { anyLanguage, ...rest } = opts;
    const languages = CONFIG.API_V2 && !anyLanguage ? myLanguages() : [];
    const body = { liked_ids, disliked_ids, k, ...(languages.length ? { languages } : {}) };
    return request("/recommend", { method: "POST", body, ...rest }).then((r) => r.recommendations);
  },
};

/** The API limits each list to 25 unique ids. Keeps the first occurrence, in order. */
export function capIds(ids, n = 25) { return [...new Set(ids)].slice(0, n); }
