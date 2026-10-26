// Fallback preview lookup via Deezer (the browser can't call Deezer directly: no CORS).
// GET /api/preview?artist=...&track=...  ->  { preview, art } or 404
export default async function handler(req, res) {
  const { artist = "", track = "" } = req.query;
  if (!artist || !track) return res.status(400).json({ error: "artist and track required" });

  const norm = (s) => s.toLowerCase().normalize("NFKD").replace(/[^\p{L}\p{N} ]/gu, "").trim();
  const want = norm(artist.split(",")[0]);

  try {
    const q = encodeURIComponent(`${artist.split(",")[0]} ${track}`);
    const r = await fetch(`https://api.deezer.com/search?q=${q}&limit=10`);
    const { data = [] } = await r.json();
    const hit = data.find((d) => d.preview && norm(d.artist?.name || "").includes(want));
    if (!hit) return res.status(404).json({ error: "no preview" });

    res.setHeader("Cache-Control", "public, s-maxage=3600");
    return res.json({ preview: hit.preview, art: hit.album?.cover_xl || hit.album?.cover_big || null });
  } catch (e) {
    return res.status(502).json({ error: "lookup failed" });
  }
}
