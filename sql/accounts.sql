-- D5: user accounts. Run once in the Supabase SQL editor (SwipeTape project), after sql/schema.sql.
-- Adds tables only; tracks / audit_steps / split_summary are not touched.
-- Every row belongs to one signed-in user, and row level security lets a user see and change only their own rows.
-- The browser uses the publishable key; the service_role key is never needed by the app.

-- One profile per auth user, created automatically at sign-up.
create table if not exists profiles (
  id           uuid primary key references auth.users (id) on delete cascade,
  display_name text,
  created_at   timestamptz not null default now()
);

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer set search_path = public
as $$
begin
  insert into public.profiles (id, display_name)
  values (new.id, coalesce(new.raw_user_meta_data ->> 'full_name',
                           new.raw_user_meta_data ->> 'name',
                           split_part(new.email, '@', 1)))
  on conflict (id) do nothing;
  return new;
end;
$$;

drop trigger if exists on_auth_user_created on auth.users;
create trigger on_auth_user_created
  after insert on auth.users
  for each row execute function public.handle_new_user();

-- The latest swipe on each song (swiping a song again replaces the old direction).
create table if not exists swipes (
  user_id    uuid not null references auth.users (id) on delete cascade default auth.uid(),
  track_id   text not null check (track_id ~ '^[A-Za-z0-9]{22}$'),
  dir        text not null check (dir in ('like', 'nope', 'love')),
  created_at timestamptz not null default now(),
  primary key (user_id, track_id)
);

-- Songs saved with "+ My List" from any tab.
create table if not exists my_list (
  user_id    uuid not null references auth.users (id) on delete cascade default auth.uid(),
  track_id   text not null check (track_id ~ '^[A-Za-z0-9]{22}$'),
  source     text not null default 'browse' check (source in ('browse', 'swipe', 'similar')),
  created_at timestamptz not null default now(),
  primary key (user_id, track_id)
);

create index if not exists swipes_user_time  on swipes  (user_id, created_at desc);
create index if not exists my_list_user_time on my_list (user_id, created_at desc);

alter table profiles enable row level security;
alter table swipes   enable row level security;
alter table my_list  enable row level security;

drop policy if exists "own profile read"   on profiles;
drop policy if exists "own profile update" on profiles;
create policy "own profile read"   on profiles for select to authenticated using (id = auth.uid());
create policy "own profile update" on profiles for update to authenticated using (id = auth.uid()) with check (id = auth.uid());

drop policy if exists "own swipes" on swipes;
create policy "own swipes" on swipes for all to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());

drop policy if exists "own list" on my_list;
create policy "own list" on my_list for all to authenticated
  using (user_id = auth.uid()) with check (user_id = auth.uid());
