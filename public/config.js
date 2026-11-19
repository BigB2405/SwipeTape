// Public settings for the browser. Only PUBLIC values belong here: this file is served to everyone.
// - API_URL: the D4/D5 FastAPI service on Modal.
// - SUPABASE_PUBLISHABLE_KEY: the "publishable" (anon) key from Supabase > Project Settings > API Keys.
//   It is designed to be public; row level security (sql/accounts.sql) keeps each user's data private.
//   NEVER put the service_role / secret key here. Leave the key empty to run the app in guest mode.
export const CONFIG = {
  API_URL: "https://bigb2405--swipetape-api-web.us-east.modal.run",
  SUPABASE_URL: "https://igwdtnmodzidwxnnwhtm.supabase.co",
  GOOGLE_LOGIN: true,    // set to true once the Google provider is enabled in Supabase (Authentication > Providers)
  SUPABASE_PUBLISHABLE_KEY: "sb_publishable_PyAVpd6tUm5y_huVnwtxJg_lG3RFyVz",
};

// Local development: `?api=local` talks to `uv run --frozen uvicorn serve:app` on port 8000.
if (new URLSearchParams(location.search).get("api") === "local") CONFIG.API_URL = "http://127.0.0.1:8000";
