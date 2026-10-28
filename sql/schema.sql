-- Run once in the Supabase SQL editor before scripts/load_supabase.py.
-- Only train + val rows are loaded; the reserved test split never leaves data/processed/.

create table if not exists tracks (
  track_id         text primary key,
  track_name       text not null,
  artists          text not null,
  primary_artist   text not null,
  album_name       text,
  popularity       smallint,
  pop_bucket       text,
  duration_ms      integer,
  duration_s       real,
  explicit         boolean,
  danceability     real,
  energy           real,
  key              smallint,
  loudness         real,
  mode             smallint,
  speechiness      real,
  acousticness     real,
  instrumentalness real,
  liveness         real,
  valence          real,
  tempo            real,
  time_signature   smallint,
  primary_genre    text not null,
  genres           text[] not null,
  n_genres         smallint,
  genre_family     text not null,
  split            text not null check (split in ('train', 'val'))
);

create table if not exists audit_steps (
  ord   smallint primary key,
  step  text not null,
  rows  integer not null,
  note  text
);

create table if not exists split_summary (
  split   text primary key,
  rows    integer not null,
  artists integer not null,
  genres  integer not null
);

-- Read-only public access for the dashboard (anon key); writes need the service-role key.
alter table tracks        enable row level security;
alter table audit_steps   enable row level security;
alter table split_summary enable row level security;
create policy "public read" on tracks        for select using (true);
create policy "public read" on audit_steps   for select using (true);
create policy "public read" on split_summary for select using (true);
