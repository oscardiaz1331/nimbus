<!-- Hourly forecast table from the open-meteo provider (cache-first on the
     server, so polling here never hammers the external API). Open-Meteo
     units are native (% and °C), unlike nimbus's own [0,1] fractions. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { timeShort, dateTime } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30, maxHours = 12 } = $props();

  let rows = $state([]);
  let stale = $state(false);
  let fetchedAt = $state(null);
  let error = $state(null);
  let loading = $state(true);

  onMount(() =>
    poll(async () => {
      try {
        const res = await get('/api/v1/providers/open-meteo/forecast');
        const h = res.data.hourly;
        const now = Date.now();
        rows = h.time
          .map((t, i) => ({
            // timezone=UTC is requested server-side; times arrive without a suffix
            date: new Date(`${t}Z`),
            clouds: h.cloud_cover[i],
            low: h.cloud_cover_low[i],
            mid: h.cloud_cover_mid[i],
            high: h.cloud_cover_high[i],
            temp: h.temperature_2m[i],
            precip: h.precipitation_probability[i],
          }))
          .filter((r) => r.date.getTime() >= now - 3600 * 1000)
          .slice(0, maxHours);
        stale = res.stale;
        fetchedAt = res.fetched_at;
        error = null;
      } catch (e) {
        error = e.message;
      } finally {
        loading = false;
      }
    }, refreshSeconds)
  );
</script>

<Widget
  {title}
  {loading}
  {error}
  footer={fetchedAt ? `source: Open-Meteo · fetched ${dateTime(fetchedAt * 1000)}` : null}
>
  {#if stale}
    <span class="badge">stale — external API unreachable</span>
  {/if}
  <div style="overflow-x: auto">
    <table class="data">
      <thead>
        <tr>
          <th>Time</th>
          <th>Clouds</th>
          <th>Low / Mid / High</th>
          <th>Temp</th>
          <th>Precip</th>
        </tr>
      </thead>
      <tbody>
        {#each rows as r (r.date)}
          <tr>
            <td>{timeShort(r.date)}</td>
            <td>{r.clouds}%</td>
            <td>{r.low} / {r.mid} / {r.high}</td>
            <td>{r.temp}°C</td>
            <td>{r.precip}%</td>
          </tr>
        {/each}
      </tbody>
    </table>
  </div>
</Widget>
