// Yearly sky events, all computed client-side from lib/astro.js — no
// providers, no network, dates adapt to whatever year is on screen.
//
// Meteor showers peak when Earth crosses the debris stream, i.e. at a
// fixed *solar longitude* (values from the IMO working list), not a
// fixed calendar date — which is exactly why they belong on the orbit
// diagram: each one is a point on Earth's orbit.

import { nextSolarLongitude, nextMoonElongation, PERIHELION_LON, orbit } from './astro.js';

const norm = (deg) => ((deg % 360) + 360) % 360;

// { id, name, icon, lambdaSun, zhr } — zhr is the peak zenithal hourly
// rate, shown as a hint of how strong the shower is.
export const SHOWERS = [
  { id: 'qua', name: 'Quadrantids', lambdaSun: 283.16, zhr: 110 },
  { id: 'lyr', name: 'Lyrids', lambdaSun: 32.32, zhr: 18 },
  { id: 'eta', name: 'η-Aquariids', lambdaSun: 45.5, zhr: 50 },
  { id: 'sda', name: 'δ-Aquariids', lambdaSun: 127.0, zhr: 25 },
  { id: 'per', name: 'Perseids', lambdaSun: 140.0, zhr: 100 },
  { id: 'ori', name: 'Orionids', lambdaSun: 208.0, zhr: 20 },
  { id: 'leo', name: 'Leonids', lambdaSun: 235.27, zhr: 15 },
  { id: 'gem', name: 'Geminids', lambdaSun: 262.2, zhr: 150 },
  { id: 'urs', name: 'Ursids', lambdaSun: 270.7, zhr: 10 },
];

export const SEASONS = [
  { id: 'eq-mar', name: 'March equinox', icon: '🌱', lambdaSun: 0 },
  { id: 'sol-jun', name: 'June solstice', icon: '☀️', lambdaSun: 90 },
  { id: 'eq-sep', name: 'September equinox', icon: '🍂', lambdaSun: 180 },
  { id: 'sol-dec', name: 'December solstice', icon: '❄️', lambdaSun: 270 },
];

// Perihelion: Earth's heliocentric longitude equals PERIHELION_LON,
// i.e. the sun's geocentric longitude is that + 180.
export const APSIDES = [
  { id: 'peri', name: 'Perihelion', icon: '🌍', lambdaSun: norm(PERIHELION_LON + 180) },
  { id: 'aph', name: 'Aphelion', icon: '🌏', lambdaSun: PERIHELION_LON },
];

/** Earth's heliocentric longitude when the sun sits at `lambdaSun`. */
export const earthLambda = (lambdaSun) => norm(lambdaSun + 180);

/**
 * All events of the sliding year that starts at `from`, sorted by date:
 * meteor showers, equinoxes/solstices, apsides, next full + new moon.
 * Each entry: { id, name, icon, date, kind, detail?, lambdaSun? }.
 */
export function upcomingEvents(from) {
  const out = [];
  for (const s of SHOWERS) {
    out.push({
      id: s.id,
      name: s.name,
      icon: '☄️',
      kind: 'shower',
      detail: `ZHR ~${s.zhr}`,
      lambdaSun: s.lambdaSun,
      date: nextSolarLongitude(s.lambdaSun, from),
    });
  }
  for (const s of SEASONS) {
    out.push({
      id: s.id,
      name: s.name,
      icon: s.icon,
      kind: 'season',
      lambdaSun: s.lambdaSun,
      date: nextSolarLongitude(s.lambdaSun, from),
    });
  }
  for (const a of APSIDES) {
    const date = nextSolarLongitude(a.lambdaSun, from);
    out.push({
      id: a.id,
      name: a.name,
      icon: a.icon,
      kind: 'apsis',
      detail: `${orbit(date).rMkm.toFixed(1)} M km`,
      lambdaSun: a.lambdaSun,
      date,
    });
  }
  out.push({
    id: 'full-moon',
    name: 'Full moon',
    icon: '🌕',
    kind: 'moon',
    date: nextMoonElongation(180, from),
  });
  out.push({
    id: 'new-moon',
    name: 'New moon',
    icon: '🌑',
    kind: 'moon',
    date: nextMoonElongation(0, from),
  });
  return out.sort((a, b) => a.date - b.date);
}
