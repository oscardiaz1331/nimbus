// Decide the ambient-effect kind from the freshest signals available.
//
// Priority: precipitation (Open-Meteo current weather_code / mm) beats
// cloudiness; for cloudiness our own observation wins while fresh
// (< 30 min), falling back to Open-Meteo's current cloud cover.
// WMO weather codes: 51-67 drizzle/rain, 71-77+85/86 snow, 80-82 showers,
// 95-99 thunderstorms.

const SNOW_CODES = new Set([71, 73, 75, 77, 85, 86]);
const RAIN_CODES = new Set([51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]);

const FRESH_MS = 30 * 60 * 1000;

export function decideAmbient(current, obs) {
  const code = current?.weather_code;
  if (code != null) {
    if (SNOW_CODES.has(code) || (current?.snowfall ?? 0) > 0) return 'snow';
    if (RAIN_CODES.has(code) || (current?.precipitation ?? 0) > 0) return 'rain';
  }

  let cloudCover = null; // [0,1]
  if (obs && Date.now() - new Date(obs.timestamp).getTime() < FRESH_MS) {
    cloudCover = obs.cloud_cover;
  } else if (current?.cloud_cover != null) {
    cloudCover = current.cloud_cover / 100;
  }
  if (cloudCover == null) return null;

  const day = current ? current.is_day !== 0 : true;
  if (cloudCover > 0.7) return 'overcast';
  if (cloudCover >= 0.3) return day ? 'partly-day' : 'partly-night';
  return day ? 'clear-day' : 'clear-night';
}
