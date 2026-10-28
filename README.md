# SwipeTape

SwipeTape is a music discovery web app inspired by the classic Tinder-style swipe experience. Users listen to 30-second song previews, swipe left or right (or up to “love” a track), and the app builds a personalized playlist based on their taste.

The project combines a curated music dataset, client-side recommendation logic, and a lightweight preview lookup API to make music discovery feel interactive and fun.

## Features

- Tinder-style swipe interface for songs
- 30-second audio preview playback
- Personalized playlist generation from likes and dislikes
- Genre and mood-aware recommendation logic
- Responsive single-page frontend
- Vercel-ready API for preview lookup fallback
- Spotify link generation for final playlist

## Tech Stack

- Frontend: HTML, CSS, JavaScript
- Data processing: Python + pandas
- API: Vercel serverless function
- Deployment: Vercel
- Data source: processed music metadata dataset

## Repository Structure

```text
.
├── api/
│   └── preview.js            # Preview fallback API for Deezer lookup
├── public/
│   ├── app.js                # Swipe logic, audio playback, recommendation engine
│   ├── index.html            # App shell and UI structure
│   ├── songs.json            # Compiled song dataset
│   └── style.css             # App styling
├── scripts/
│   └── build_data.py         # Converts CSV music data into compact JSON
├── .gitignore
├── package.json
├── vercel.json               # Vercel configuration and cache headers
├── README.md
