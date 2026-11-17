"""Korak 2b: provjera podjele. Što je od srodnih slika ipak ostalo s obje strane?

Podjela u koraku 2 drži zajedno duplikate, zakrenute i zrcaljene kopije (do
PRAG_ZAKRENUTE bitova) i serije snimanja (razmak do RAZMAK_SERIJE sekundi). Ova
skripta mjeri što ta pravila ne pokrivaju, na gotovoj podjeli:
  (a) testne slike čija je zakrenuta ili zrcaljena inačica nekoj slici za učenje
      blizu, ali iznad praga grupiranja (do 6 bitova);
  (b) testne slike koje u skupu za učenje imaju uspravnu sliku udaljenu 7 do 10 bitova;
  (c) testne slike snimljene nedugo prije ili poslije neke slike za učenje iz istog
      izvora (do 1, 2 i 5 minuta), samo za izvore s vremenom u imenu datoteke.
Uz (a) i (b) broji se i koliko parova ima različitu oznaku: stvarne kopije imaju
istu oznaku, pa visok udio različitih znači da su to većinom različiti listovi.

Pokretanje:  uv run src/k2b_provjera_podjele.py
"""
import json

import numpy as np
import pandas as pd

import config as c
import slicnost as s
from k2_ciscenje_i_podjela import PRAG_DUPLIKAT, PRAG_SLICNE, PRAG_ZAKRENUTE, RAZMAK_SERIJE

SEKUNDE = [RAZMAK_SERIJE, 60, 120, 300]


def test_ucenje(parovi, dio):
    """Od parova (i, j, ...) ostavlja one kojima je jedna slika u testu, a druga u učenju; testna ide u prvi stupac."""
    i, j = parovi[:, 0], parovi[:, 1]
    a = (dio[i] == "test") & (dio[j] == "train")
    b = (dio[i] == "train") & (dio[j] == "test")
    return np.concatenate([parovi[a], parovi[b][:, [1, 0, 2, 3]]])


def sazetak(parovi, df):
    oznaka = df.oznaka.to_numpy()
    return dict(testnih_slika=int(len(set(parovi[:, 0]))), parova=int(len(parovi)),
                parova_razlicite_oznake=int((oznaka[parovi[:, 0]] != oznaka[parovi[:, 1]]).sum()),
                po_izvoru=df.izvor.iloc[sorted(set(parovi[:, 0]))].value_counts().to_dict())


def main():
    svi = pd.read_csv(c.PODACI / "manifest_sirovi.csv")
    podjela = pd.read_csv(c.PODACI / "manifest.csv").set_index("id").dio
    dio = svi.id.map(podjela).fillna("").to_numpy()      # uklonjene slike imaju prazan dio
    ph, dh = s.u_uint64(svi.phash), s.u_uint64(svi.dhash)
    rez = dict(test_slika=int((dio == "test").sum()), prag_zakrenute=PRAG_ZAKRENUTE, razmak_serije_s=RAZMAK_SERIJE)

    # (a) zakrenute i zrcaljene inačice
    ph_i, dh_i = s.sazeci_inacica(svi)
    zakrenuti = np.concatenate([
        s.parovi_ispod_praga(ph, dh, PRAG_DUPLIKAT, ph2=ph_i[:, k], dh2=dh_i[:, k]) for k in range(ph_i.shape[1])])
    zakrenuti = test_ucenje(zakrenuti, dio)
    udalj = np.maximum(zakrenuti[:, 2], zakrenuti[:, 3])
    rez["zakrenute_do_praga"] = sazetak(zakrenuti[udalj <= PRAG_ZAKRENUTE], svi)
    rez["zakrenute_iznad_praga"] = sazetak(zakrenuti[udalj > PRAG_ZAKRENUTE], svi)

    # (b) uspravne slične slike
    uspravni = test_ucenje(s.parovi_ispod_praga(ph, dh, PRAG_SLICNE), dio)
    udalj = np.maximum(uspravni[:, 2], uspravni[:, 3])
    rez["uspravne_do_praga"] = sazetak(uspravni[udalj <= PRAG_DUPLIKAT], svi)
    rez["uspravne_iznad_praga"] = sazetak(uspravni[udalj > PRAG_DUPLIKAT], svi)
    rez["slicne_test_id"] = sorted(int(x) for x in set(svi.id.to_numpy()[uspravni[udalj > PRAG_DUPLIKAT][:, 0]]))

    # (c) vremenski razmak do najbliže slike za učenje iz istog izvora
    v = svi.putanja.str.split("/").str[-1].map(s.vrijeme_snimanja)
    t = svi[v.notna()].assign(zapis=[x[0] for x in v.dropna()], t=[x[1] for x in v.dropna()], dio=dio[v.notna()])
    razmaci, po_izvoru = [], {}
    for (izvor, zapis), g in t.groupby(["izvor", "zapis"]):
        t_te, t_tr = g[g.dio == "test"].t.to_numpy(), np.sort(g[g.dio == "train"].t.to_numpy())
        if len(t_te) == 0 or len(t_tr) == 0:
            continue
        k = np.searchsorted(t_tr, t_te)
        r = np.minimum(np.abs(t_te - t_tr[np.clip(k - 1, 0, len(t_tr) - 1)]),
                       np.abs(t_te - t_tr[np.clip(k, 0, len(t_tr) - 1)]))
        razmaci.append(r)
        red = po_izvoru.setdefault(izvor, dict(test_s_vremenom=0, **{f"do_{x}_s": 0 for x in SEKUNDE}))
        red["test_s_vremenom"] += int(len(r))
        for x in SEKUNDE:
            red[f"do_{x}_s"] += int((r <= x).sum())
    r = np.concatenate(razmaci)
    rez["uzastopne"] = dict(test_s_vremenom=int(len(r)), medijan_sekundi=float(np.median(r)),
                            **{f"do_{x}_s": int((r <= x).sum()) for x in SEKUNDE}, po_izvoru=po_izvoru)

    (c.REZULTATI / "provjera_podjele.json").write_text(json.dumps(rez, indent=1, ensure_ascii=False))
    for k in ["zakrenute_do_praga", "zakrenute_iznad_praga", "uspravne_do_praga", "uspravne_iznad_praga"]:
        x = rez[k]
        print(f"{k:24s} testnih slika {x['testnih_slika']:4d}, parova {x['parova']:4d}, "
              f"različite oznake {x['parova_razlicite_oznake']:3d}; {x['po_izvoru']}")
    u = rez["uzastopne"]
    print(f"uzastopne snimke: od {u['test_s_vremenom']} testnih slika s vremenom u imenu, sliku za učenje istog izvora ima "
          + ", ".join(f"unutar {x} s: {u[f'do_{x}_s']}" for x in SEKUNDE) + f" (medijan razmaka {u['medijan_sekundi']:.0f} s)")
    for izvor, red in u["po_izvoru"].items():
        print(f"  {izvor:24s} {red}")


if __name__ == "__main__":
    main()
