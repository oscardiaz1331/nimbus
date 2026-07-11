<!-- Stat tile: latest cloud cover as a hero number (no chart needed). -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { pct, dateTime } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30, stationId = null } = $props();

  let obs = $state(null);
  let error = $state(null);
  let loading = $state(true);

  onMount(() => {
    const qs = stationId ? `?station_id=${encodeURIComponent(stationId)}` : '';
    return poll(async () => {
      try {
        obs = await get(`/api/v1/observations/latest${qs}`);
        error = null;
      } catch (e) {
        error = e.message;
      } finally {
        loading = false;
      }
    }, refreshSeconds);
  });
</script>

<Widget {title} {loading} {error} footer={obs ? dateTime(obs.timestamp) : null}>
  {#if obs}
    <div class="big-number">{pct(obs.cloud_cover)}</div>
    <p class="muted">cloud cover</p>
    {#if obs.sky_fraction != null}
      <p class="muted">valid sky: {pct(obs.sky_fraction)}</p>
    {/if}
  {/if}
</Widget>
