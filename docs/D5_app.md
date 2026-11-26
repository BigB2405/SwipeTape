# D5: Working application and tests

**DTSC 3602 · Group 10 · SoundSwipe** (project and repository name: SwipeTape)

| Item | Link |
|---|---|
| Website (Vercel) | https://swipetape-group10.vercel.app (runs **SoundSwipe v2** since Nov 25, see below) |
| Live API the website calls (Modal) | https://bigb2405--swipetape-api-v2-web.us-east.modal.run · [/docs](https://bigb2405--swipetape-api-v2-web.us-east.modal.run/docs) |
| D4 API (still live, unchanged) | https://bigb2405--swipetape-api-web.us-east.modal.run · [/docs](https://bigb2405--swipetape-api-web.us-east.modal.run/docs) |
| Frontend code | [`public/`](../public/): `index.html`, `style.css`, `config.js`, `js/*.js` |
| Postman collections | v2 API: [`postman/soundswipe_v2_api.postman_collection.json`](../postman/soundswipe_v2_api.postman_collection.json) (12 requests, 27 assertions) · D4/D5 API: [`postman/swipetape_api.postman_collection.json`](../postman/swipetape_api.postman_collection.json) (10 requests, 23 assertions) |
| Screenshots | [`docs/d5_screenshots/`](d5_screenshots/): v2 live site `v2_1_`–`v2_7_` (survey, Home, playlist, Find similar, error state, phone) and v2 Postman run `v2_8_`; the D5 version before v2 `1_`–`10_` |
| Branch / tag | `main` (v2) · branch `D5` and tag `d5-final` (the version before v2) |

## Update, Nov 25: the live site runs SoundSwipe v2

After the D5 build we renamed the site to **SoundSwipe** (instructor feedback on the name) and made v2 the live version. Everything below about tabs, loading and error states, accounts and tests still applies; v2 adds to it:

- **Taste survey** on the first visit: languages → genre groups → tap songs you know. The picks seed Home's "Because you liked …" rows and the swipe deck. It can be skipped or redone from the Me tab.
- **Better model (M6, chosen on validation only):** real genre tags in the song embedding and a small popularity boost. On validation, genre agreement went from 19% to 98% and kids' songs in mixes from 2.3% to 0.04%. *Gangsta's Paradise* now returns Drake and Snoop Dogg instead of a kids' song.
- **Languages:** the API takes an optional `languages` list, and Find similar has a "My languages / Any language" switch. Languages are *estimated* (the data has no language column), and the app says so.
- **Adaptive swipe deck:** after a like, "Because you liked …" cards from the model join the deck.
- **Apple Music links** next to Spotify; match scores never go over 100%.

The v2 API is a **separate** Modal app, so the D4 API, its URL and its outputs did not change.

**v2 checks (Nov 25, deployed URL):** Postman v2 collection **27/27 assertions passed** (screenshot `v2_8_*`; health within 15 s, because a sleeping Modal container took 5.7 s to wake in one run and the app shows "Waking up the model…" meanwhile; pipeline info, sample, valid recommend, three 422s, search, short search 422, Find similar k=5, `languages=["en"]` sorted and within 0–1, unknown language 422); `check_deployed.py` with the v2 bundle **ALL CHECKS PASSED**; pytest 34/34. The D4 API still passes its own collection 23/23 and `check_deployed.py`.

## What the app does

**SoundSwipe** (the website's public name since Nov 25; the repository, API and tag names still say SwipeTape) helps listeners find songs they like beyond the current hits. It has four tabs:

| Tab | What the user does | Live API call |
|---|---|---|
| **Home** | Browses a Netflix-style catalog: a featured song, Trending, top songs per genre family, Hidden gems (random less-played songs), and, once they like songs, **"Because you liked …"** rows and **My List** | `POST /recommend` with the latest liked song, `k=15`, for each "Because you liked" row |
| **Swipe** | Hears 30-second clips and swipes right (like), left (nope) or up (love), then gets a 20-song playlist | `POST /recommend` with loves first, then likes (max 25), and nopes (max 25), `k=35`, filtered to 20 |
| **Find similar** | Searches any catalog song or taps a quick pick; the song starts playing, and the model returns the **5 closest songs** | `GET /search?q=…&limit=25`, then `POST /recommend` with one liked id and `k=5` |
| **Me** | Signs up or in (email and password, or Google), and sees swipe stats, taste profile, My List and liked songs | none (Supabase Auth and tables) |

Guests can use Home, Swipe and Find similar without an account. Signing in saves swipes and My List, and the deck then skips songs the user already rated.

## Inputs and outputs a person can understand

- **Inputs** are actions, not ids: a swipe, a tap on a song, or a search by title or artist. The app turns them into Spotify track ids for the API.
- **Outputs** show the song title, artist, cover art and a 30-second preview, with a **% similarity** (the model's cosine score × 100), a "Because you liked …" reason, and a Spotify link. Each result explains where it came from ("From the SoundSwipe song kNN model (live API)").

## Loading and error states

All API calls go through [`public/js/api.js`](../public/js/api.js).

| Situation | What the user sees |
|---|---|
| Waiting for the API | Spinner and skeleton cards ("Asking the model for the 5 closest songs…", "Building your mix…") |
| Slow first request (Modal cold start, over 3 s) | "Waking up the model. The first request can take up to 30 seconds…" |
| No answer after 60 s | "The recommendation API took too long to answer. Please try again." with a **Try again** button |
| Network down or blocked | "Can't reach the recommendation API. Check your connection and try again." |
| **422** invalid input | "The API couldn't use that request: …" (with the API's own message) |
| **503** model unavailable | "The recommendation model is unavailable right now (503). Please try again shortly." |
| Swipe playlist when the API fails | Falls back to the old in-browser formula, labelled **"Offline picks"** with the reason and a "Try the model again" button |
| Search with no results | "No songs match “…”. Try just the artist, or a shorter title." |
| Song has no preview clip | "No preview for this one. Try it on Spotify." |
| Sign-up problems | Readable messages for a wrong password, an email already registered, and too many sign-ups at once |

We checked each state in a browser by stopping the API, sending bad input and searching for nonsense.

## API changes for D5

The fitted pipeline (`pipeline.joblib`) is the same one as in D4. It was not retrained and the D3 hyperparameters are unchanged.
- **CORS** (`serve.py`): the browser may call the API from `*.vercel.app` and localhost, plus any origin listed in `ALLOWED_ORIGINS`. Other origins are refused.
- **`GET /search?q=&limit=`**: finds catalog songs whose title and artist contain every word of `q`. Exact and prefix matches rank first (`match` 0–4), and the app breaks ties by popularity. A query under 2 characters returns **422**; a missing model returns **503**.
- `/health`, `/pipeline/info`, `/sample` and `/recommend` are unchanged. Version 0.5.0.

## Tests

1. **Postman** (deployed URL, 10 requests, all with assertions): health (200, model loaded, under 5 s), pipeline info (steps, versions, catalog size), sample (stores ids), valid recommend (200, k items, no swiped songs), empty `liked_ids` (422), `k` out of range (422), unknown track id (422), valid search (best match first), too-short search (422), and **Find similar** with one seed and `k=5` (exactly 5 songs, sorted by score, none is the seed). **Result (Postman web Collection Runner, Nov 18, 2026): 23/23 tests passed, 0 failed, average response 224 ms** (`docs/d5_screenshots/6_*`–`8_*`). **Re-run Nov 25, 2026 (newman CLI, same collection, deployed URL): 23/23 assertions passed, 10/10 requests, average response 86 ms.** An earlier run on a cold container passed 22/23: `/health` took 10.2 s (Modal loading the model) and failed the "under 5 s" check. The app handles this with its "Waking up the model…" message, and the frontend sends a warm-up `/health` request when it loads.
2. **`scripts/check_deployed.py`** against the Modal URL: the deployed API returns the same tracks, order and scores (within 1e-4) as the local pipeline, all invalid inputs return 422, search matches the local results, and a CORS preflight from a `vercel.app` origin is allowed. **ALL CHECKS PASSED** (Nov 18, re-checked Nov 25, 2026).
3. **pytest**: 23 tests (15 from D4, plus search, invalid search, 503 search and CORS allow/deny). All pass.
4. **Browser checks** (desktop 1440×900 and phone 390×844): Home rows, search, the Find similar 5 results, a full swipe round with the playlist coming from the API, the detail sheet, the tab bar on phones, and the error and offline states with the API stopped.

## Accounts and data protection

- Supabase Auth with email and password, or Google (Google Cloud OAuth client, published). Email confirmation is turned off because the free Supabase mailer sends only about 2 emails per hour, which blocked sign-ups.
- [`sql/accounts.sql`](../sql/accounts.sql) adds `profiles`, `swipes` and `my_list`. **Row level security** lets each signed-in user read and write only their own rows. We checked that the public key returns no rows and that anonymous inserts get 401.
- The browser only holds the **publishable** key, which is public by design. The secret key is not in the code. The Modal API receives only track ids, never names or emails. There is a [privacy page](https://swipetape-group10.vercel.app/privacy.html).

## Data integrity (for D6)

`public/songs.json` was built from the raw CSV and included **5,010 songs from the reserved test split**. It is now built from `data/processed/tracks_clean.parquet` (train + validation only; `scripts/build_data.py` asserts there is no test id). The app never shows a test-split song, so the D6 test evaluation stays untouched.

## Known limitations

- The model matches on audio features and genre mix, not on artist or culture. For example, "Blinding Lights" (The Weeknd) returns a metal and a German schlager song among its 5 picks. We will discuss this in D6.
- Home rows and the popularity ≥ 40 filter in `songs.json` favour hits (popularity bias). The Hidden gems row is a small counterweight.
- Previews depend on iTunes and Deezer search, so some songs have no clip.
- Google's sign-in screen names the Supabase domain rather than "SwipeTape". Showing our own name would need Google's brand verification or a custom domain.

## How to run it locally

```bash
uv run --frozen uvicorn serve:app --port 8000     # local API
cd public && python3 -m http.server 3000          # static site; open http://localhost:3000/?api=local
uv run --frozen --with pytest pytest -q tests     # 23 tests
uv run --frozen python scripts/check_deployed.py --url https://bigb2405--swipetape-api-web.us-east.modal.run
```

## Contributions and AI use

Britain Russell: app redesign (tabs, Home, Find similar, accounts), search endpoint and CORS, Supabase, Google and Vercel setup, tests and Postman. Claude Code drafted the frontend modules, the API changes, the tests and this note; Britain reviewed and ran them.
