import Anthropic from '@anthropic-ai/sdk'
import postavkeDemo from './postavke.json'
import cinjenice from '../../../znanje/cinjenice.json'
import sustav from './sustav.md?raw'

export function postavke() {
  if (!__KLJUC__) throw new Error('Nema API ključa: u projekt/.env treba postaviti CLAUDE_API_KEY i pokrenuti pnpm dev.')
  return postavkeDemo
}

const izmedu = (a, b) => a + Math.random() * (b - a)

export function nasumicanRezultat(prag) {
  const stanje = ['zdrava', 'nezdrava', 'nesigurno'][Math.floor(Math.random() * 3)]
  const pouzdanost = stanje === 'nesigurno' ? izmedu(0.5, prag) : izmedu(prag, 0.995)
  const nezdravaJeVeca = stanje === 'nezdrava' || (stanje === 'nesigurno' && Math.random() < 0.5)
  return { stanje, pouzdanost, vjerojatnost_nezdrave: nezdravaJeVeca ? pouzdanost : 1 - pouzdanost, prag }
}

const odaberiCinjenice = (vrsta) => cinjenice.filter((f) => f.tema === 'opcenito' || f.tema === vrsta)

function sloziUpit(vrsta, stanje, pouzdanost, uvjeti, odabrane) {
  const dijelovi = [
    'REZULTAT KLASIFIKATORA',
    `stanje: ${stanje}`,
    `pouzdanost: ${Math.round(pouzdanost * 100)}%`,
    '',
    'PODACI KORISNIKA',
    `vrsta biljke: ${vrsta || 'nepoznata'}`,
  ]
  for (const [naziv, vrijednost] of Object.entries(uvjeti)) if (vrijednost) dijelovi.push(`${naziv}: ${vrijednost}`)
  dijelovi.push('', 'ČINJENICE', ...odabrane.map((f) => `[${f.id}] (${f.tema}, ${f.kategorija}) ${f.cinjenica}`))
  return dijelovi.join('\n')
}

const klijent = new Anthropic({ apiKey: __KLJUC__, dangerouslyAllowBrowser: true })

export async function analiziraj(obrazac, prag) {
  const rezultat = nasumicanRezultat(prag)
  const { vrsta, ...ostalo } = obrazac
  const uvjeti = Object.fromEntries(Object.entries(ostalo).map(([k, v]) => [k, v.trim().slice(0, 200)]))
  const odabrane = odaberiCinjenice(vrsta)

  let odgovor
  try {
    odgovor = await klijent.messages.create({
      model: __MODEL__,
      max_tokens: 1500,
      thinking: { type: 'disabled' },
      system: sustav.trim(),
      messages: [{ role: 'user', content: sloziUpit(vrsta, rezultat.stanje, rezultat.pouzdanost, uvjeti, odabrane) }],
    })
  } catch (e) {
    return { ...rezultat, preporuka: null, izvori: [], greska: `Preporuka nije dostupna: ${e.status ?? e.name}` }
  }

  const tekst = odgovor.content.filter((b) => b.type === 'text').map((b) => b.text).join('')
  const poId = Object.fromEntries(odabrane.map((f) => [f.id, f]))
  const navedene = [...new Set(tekst.match(/C\d{3}/g) ?? [])].sort()
  const izvori = navedene.filter((id) => poId[id]).map((id) => {
    const { izvor, url, cinjenica } = poId[id]
    return { id, izvor, url, cinjenica }
  })
  return { ...rezultat, preporuka: tekst, izvori }
}
