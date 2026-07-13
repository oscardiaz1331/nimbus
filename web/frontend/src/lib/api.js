// Thin fetch wrapper for the /api/v1 endpoints + a polling helper.

export async function get(path) {
  const res = await fetch(path);
  if (!res.ok) {
    let detail;
    try {
      detail = (await res.json()).detail;
    } catch {
      detail = res.statusText;
    }
    throw new Error(typeof detail === 'string' ? detail : `HTTP ${res.status}`);
  }
  return res.json();
}

// Runs fn immediately, then every `seconds`. Returns a stop function
// suitable for onDestroy.
export function poll(fn, seconds) {
  fn();
  const id = setInterval(fn, seconds * 1000);
  return () => clearInterval(id);
}
