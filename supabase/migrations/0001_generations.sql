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

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'generations'
      AND policyname = 'generations_select_authenticated'
  ) THEN
    CREATE POLICY "generations_select_authenticated"
    ON public.generations
    FOR SELECT
    TO authenticated
    USING (true);
  END IF;
END $$;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_policies
    WHERE schemaname = 'public'
      AND tablename = 'generations'
      AND policyname = 'generations_insert_authenticated'
  ) THEN
    CREATE POLICY "generations_insert_authenticated"
    ON public.generations
    FOR INSERT
    TO authenticated
    WITH CHECK (true);
  END IF;
END $$;

grant select, insert on public.generations to authenticated;

comment on table public.generations is 'Metadata de generaciones de audio producidas por NovaBeat';
