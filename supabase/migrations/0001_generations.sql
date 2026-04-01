-- NovaBeat initial schema for generation tracking

create extension if not exists pgcrypto;

create table if not exists public.generations (
  id uuid primary key default gen_random_uuid(),
  provider text not null,
  prompt text not null default '',
  output_file text not null,
  created_at timestamptz not null default now(),
  extra jsonb not null default '{}'::jsonb
);

create index if not exists idx_generations_created_at on public.generations (created_at desc);
create index if not exists idx_generations_provider on public.generations (provider);

alter table public.generations enable row level security;

-- Read policy for authenticated users
create policy if not exists "generations_select_authenticated"
on public.generations
for select
to authenticated
using (true);

-- Insert policy for service role / authenticated users
create policy if not exists "generations_insert_authenticated"
on public.generations
for insert
to authenticated
with check (true);

comment on table public.generations is 'Metadata de generaciones de audio producidas por NovaBeat';
