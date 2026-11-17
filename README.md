# Sustav za detekciju bolesti i preporuku njege biljaka

Programski dio završnog rada (Fakultet informatike u Puli, mentor doc. dr. sc. Sandi Baressi Šegota).

Sustav iz fotografije sobne biljke procjenjuje je li biljka **zdrava** ili **nezdrava**, a zatim jezičnim
modelom sastavlja preporuku za njegu na hrvatskom. Ako klasifikator nije dovoljno siguran, sustav to kaže i
daje samo opće savjete.

## Sadržaj

| Mapa | Što sadrži |
| --- | --- |
| `src/` | priprema podataka (Python) |
| `web/` | sučelje (Vue + Vite) |
| `znanje/` | baza činjenica o njezi s izvorima (`cinjenice.json`) |
| `alati/` | alat za ručnu provjeru oznaka fotografija |

## Pokretanje sučelja

Potrebni su Node.js i pnpm. U `.env` (u korijenu repozitorija, nije u gitu) postavlja se ključ jezičnog modela:

```
CLAUDE_API_KEY=...
CLAUDE_MODEL=claude-haiku-5-5
```

```bash
pnpm --dir web install
pnpm --dir web dev
```

Ključ se čita samo u razvojnom načinu (`pnpm dev`) i ne ulazi u izgrađenu inačicu.

## Podaci

Slike se ne nalaze u repozitoriju (veličina i licence skupova). Preuzimaju se u `data/raw/`; koja mapa kojeg
skupa daje koju vrstu i oznaku zapisano je u `src/config.py`.

Potreban je [uv](https://docs.astral.sh/uv/):

```bash
uv sync
uv run src/k1_manifest.py            # popis slika, sažeci, smanjene kopije
```
