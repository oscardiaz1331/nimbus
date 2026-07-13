// Self-contained low-precision astronomy (Meeus-style truncated series).
// Good to ~0.1-0.5° — plenty for schematic views — and fully offline: the
// orrery widget must keep working with no providers and no network.
// The server's astronomy provider (astral) stays the precise source for
// rise/set times; this module only drives visualizations.

const DEG = Math.PI / 180;

const norm360 = (d) => ((d % 360) + 360) % 360;

export function julianDay(date) {
  return date.getTime() / 86400000 + 2440587.5;
}

// Sun: subsolar point + apparent ecliptic longitude.
export function solar(date) {
  const d = julianDay(date) - 2451545.0; // days since J2000
  const g = norm360(357.529 + 0.98560028 * d) * DEG; // mean anomaly
  const q = norm360(280.459 + 0.98564736 * d); // mean longitude (deg)
  const lambda = norm360(q + 1.915 * Math.sin(g) + 0.02 * Math.sin(2 * g)); // apparent
  const eps = (23.439 - 0.00000036 * d) * DEG; // obliquity
  const sinL = Math.sin(lambda * DEG);
  const declination = Math.asin(Math.sin(eps) * sinL) / DEG;
  const ra = norm360(Math.atan2(Math.cos(eps) * sinL, Math.cos(lambda * DEG)) / DEG);
  // Equation of time as (mean - apparent) hour offset, normalized to ±12 h.
  const eotHours = (((q - ra + 540) % 360) - 180) / 15;
  const utcHours =
    date.getUTCHours() + date.getUTCMinutes() / 60 + date.getUTCSeconds() / 3600;
  const subsolarLon = ((-15 * (utcHours - 12 + eotHours) + 540) % 360) - 180;
  return { subsolarLat: declination, subsolarLon, eclipticLon: lambda };
}

// Latitude of the day/night terminator at a given longitude.
export function terminatorLat(lonDeg, subsolarLat, subsolarLon) {
  const H = (lonDeg - subsolarLon) * DEG;
  const tanDecl = Math.tan((Math.abs(subsolarLat) < 1e-4 ? 1e-4 : subsolarLat) * DEG);
  return Math.atan(-Math.cos(H) / tanDecl) / DEG;
}

const PHASES = [
  ['New Moon', '\u{1F311}'],
  ['Waxing Crescent', '\u{1F312}'],
  ['First Quarter', '\u{1F313}'],
  ['Waxing Gibbous', '\u{1F314}'],
  ['Full Moon', '\u{1F315}'],
  ['Waning Gibbous', '\u{1F316}'],
  ['Last Quarter', '\u{1F317}'],
  ['Waning Crescent', '\u{1F318}'],
];

// Moon: mean elongation from the sun (drives the Earth–Moon diagram).
export function moon(date) {
  const d = julianDay(date) - 2451545.0;
  const elongation = norm360(297.8501921 + 12.19074912 * d);
  const illumination = (1 - Math.cos(elongation * DEG)) / 2;
  const [phaseName, icon] = PHASES[Math.floor(norm360(elongation + 22.5) / 45)];
  return { elongation, illumination, phaseName, icon };
}

// Mollweide projection (equal-area, elliptical outline — landmasses keep
// their true relative sizes). Returns [x, y] with x in ±2√2, y in ±√2.
export function mollweide(lonDeg, latDeg) {
  const phi = latDeg * DEG;
  let theta;
  if (Math.abs(latDeg) > 89.99) {
    theta = Math.sign(phi) * (Math.PI / 2);
  } else {
    theta = phi;
    for (let i = 0; i < 5; i++) {
      theta -=
        (2 * theta + Math.sin(2 * theta) - Math.PI * Math.sin(phi)) /
        (2 + 2 * Math.cos(2 * theta));
    }
  }
  return [
    ((2 * Math.SQRT2) / Math.PI) * lonDeg * DEG * Math.cos(theta),
    Math.SQRT2 * Math.sin(theta),
  ];
}

// Geographic (lon, lat) -> unit vector in the Earth-fixed (ECEF) frame.
// All the 3D globe math runs in this frame: the subsolar point pins the
// sun's direction, so seasons, time of day and axial tilt come for free.
export function vec(lonDeg, latDeg) {
  const lon = lonDeg * DEG;
  const lat = latDeg * DEG;
  return [Math.cos(lat) * Math.cos(lon), Math.cos(lat) * Math.sin(lon), Math.sin(lat)];
}

// Greenwich mean sidereal time, degrees (low precision, ~0.01°).
export function gmst(date) {
  return norm360(280.46062 + 360.98564737 * (julianDay(date) - 2451545.0));
}

// Where the ecliptic north pole pierces the Earth (the sub-viewer point
// when looking at Earth from "above" the solar system). Fixed latitude
// 90° − 23.44°; the longitude sweeps daily with Earth's rotation.
export function eclipticPoleGeo(date) {
  return { lat: 66.56, lon: ((270 - gmst(date) + 540) % 360) - 180 };
}

const E = 0.01671; // Earth orbit eccentricity
const PERIHELION_LON = 102.94; // heliocentric ecliptic longitude of perihelion
const AU_MKM = 149.5978707;

// Earth's position on its orbit (heliocentric).
export function orbit(date) {
  const lambdaEarth = norm360(solar(date).eclipticLon + 180);
  const nu = norm360(lambdaEarth - PERIHELION_LON); // true anomaly
  const rAU = (1.000001018 * (1 - E * E)) / (1 + E * Math.cos(nu * DEG));
  return { lambdaEarth, nu, rAU, rMkm: rAU * AU_MKM };
}

export { PERIHELION_LON };
