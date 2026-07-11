<!-- Raw data view: the most recent observation rows, straight off the API. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { pct, dateTime } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30, stationId = null, limit = 12 } = $props();

  let rows = $state([]);
  let error = $state(null);
  let loading = $state(true);

  onMount(() => {
    const params = new URLSearchParams({ limit: String(limit) });
    if (stationId) params.set('station_id', stationId);
    return poll(async () => {
      try {
        rows = await get(`/api/v1/observations?${params}`);
        error = null;
      } catch (e) {
        error = e.message;
      } finally {
        loading = false;
      }
    }, refreshSeconds);
  });
</script>

<Widget {title} {loading} {error}>
  {#if rows.length === 0}
    <p class="muted">No observations yet.</p>
  {:else}
    <div style="overflow-x: auto">
      <table class="data">
        <thead>
          <tr>
            <th>Time</th>
            <th>Clouds</th>
            <th>Valid sky</th>
            <th>Inference</th>
            <th>Image</th>
          </tr>
        </thead>
        <tbody>
          {#each rows as r (r.id)}
            <tr>
              <td>{dateTime(r.timestamp)}</td>
              <td>{pct(r.cloud_cover)}</td>
              <td>{r.sky_fraction != null ? pct(r.sky_fraction) : '—'}</td>
              <td>{r.inference_ms != null ? `${r.inference_ms.toFixed(0)} ms` : '—'}</td>
              <td>
                {#if r.image_url}
                  <a href={r.image_url} target="_blank" rel="noreferrer">view</a>
                {:else}
                  —
                {/if}
              </td>
            </tr>
          {/each}
        </tbody>
      </table>
    </div>
  {/if}
</Widget>
