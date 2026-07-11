<!-- Per-class cloud breakdown from the latest observation's `classes` map.
     Today's model is single-class ("cloud"), but the v1 schema already
     carries the breakdown, so multi-class models light this up for free. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { pct, dateTime } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30, stationId = null } = $props();

  let obs = $state(null);
  let error = $state(null);
  let loading = $state(true);

  const entries = $derived(
    Object.entries(obs?.classes ?? {}).sort((a, b) => b[1] - a[1])
  );

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

<Widget {title} {loading} {error} footer={obs && entries.length ? dateTime(obs.timestamp) : null}>
  {#if entries.length === 0}
    <p class="muted">
      The latest observation has no per-class breakdown (single-class model).
    </p>
  {:else}
    {#each entries as [name, value] (name)}
      <div class="bar-row">
        <span class="bar-label">{name}</span>
        <div class="bar-track">
          <div class="bar-fill" style:width="{Math.min(value * 100, 100)}%"></div>
        </div>
        <span class="bar-val">{pct(value)}</span>
      </div>
    {/each}
  {/if}
</Widget>
