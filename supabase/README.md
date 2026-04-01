# Supabase setup

## 1) Crear entidades

Ejecuta la migración `migrations/0001_generations.sql` en SQL Editor de Supabase o con Supabase CLI.

## 2) Variables de entorno

Usa la URL del proyecto y una key con permisos de inserción:

- `SUPABASE_URL`
- `SUPABASE_KEY`

## 3) Uso desde NovaBeat

```bash
novabeat-generate --supabase-url "$SUPABASE_URL" --supabase-key "$SUPABASE_KEY" meta-generate \
  --prompt "ambient piano" \
  --output ./artifacts/out.wav
```
