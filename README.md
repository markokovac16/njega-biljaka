# Sustav za detekciju bolesti i preporuku njege biljaka

Programski dio završnog rada (Fakultet informatike u Puli, mentor doc. dr. sc. Sandi Baressi Šegota).

Sustav iz fotografije sobne biljke procjenjuje je li biljka **zdrava** ili **nezdrava**, a zatim jezičnim
modelom sastavlja preporuku za njegu na hrvatskom. Ako klasifikator nije dovoljno siguran, sustav to kaže i
daje samo opće savjete.

## Sadržaj

| Mapa | Što sadrži |
| --- | --- |
| `src/` | priprema podataka, učenje i evaluacija klasifikatora (Python) |
| `rezultati/` | rezultati pokretanja (JSON) |
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
uv run src/k2_ciscenje_i_podjela.py  # uklanjanje duplikata; podjela po skupinama srodnih slika
uv run src/k2b_provjera_podjele.py   # što je od srodnih slika ipak ostalo s obje strane podjele
```

Učenje klasifikatora (oko 17 min po modelu na Apple M5); svaka inačica uči se s tri sjemena:

```bash
uv run src/k4_trening.py --naziv osnovni
uv run src/k4_trening.py --naziv osnovni_s1 --sjeme 1
uv run src/k4_trening.py --naziv osnovni_s2 --sjeme 2
uv run src/k4_trening.py --naziv jaka --augmentacija jaka         # i jaka_s1, jaka_s2
```
