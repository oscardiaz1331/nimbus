<!-- LoRa/IoT node health: nodes auto-register when they first POST
     telemetry; online/stale/offline is derived from last_seen age. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import Widget from './Widget.svelte';

  let {
    title = '',
    refreshSeconds = 30,
    staleSeconds = 900,
    offlineSeconds = 3600,
  } = $props();

  let nodes = $state([]);
  let error = $state(null);
  let loading = $state(true);

  function status(node) {
    const age = (Date.now() - new Date(node.last_seen).getTime()) / 1000;
    if (age > offlineSeconds) return 'offline';
    if (age > staleSeconds) return 'stale';
    return 'online';
  }

  function relative(iso) {
    const s = Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000);
    if (s < 90) return `${Math.round(s)} s ago`;
    if (s < 5400) return `${Math.round(s / 60)} min ago`;
    if (s < 129600) return `${Math.round(s / 3600)} h ago`;
    return `${Math.round(s / 86400)} d ago`;
  }

  onMount(() =>
    poll(async () => {
      try {
        nodes = await get('/api/v1/nodes');
        error = null;
      } catch (e) {
        error = e.message;
      } finally {
        loading = false;
      }
    }, refreshSeconds)
  );
</script>

<Widget {title} {loading} {error}>
  {#if nodes.length === 0}
    <p class="muted">No LoRa nodes have reported yet.</p>
  {:else}
    {#each nodes as n (n.node_id)}
      {@const st = status(n)}
      <div class="node-row">
        <span class="dot {st}"></span>
        <div class="node-main">
          <div class="node-id">{n.node_id}</div>
          <p class="muted">
            {st} · {relative(n.last_seen)}
            {#if n.rssi_dbm != null}&nbsp;· {n.rssi_dbm} dBm{/if}
            {#if n.snr_db != null}&nbsp;· SNR {n.snr_db} dB{/if}
          </p>
        </div>
        {#if n.battery_pct != null}
          <span class="node-battery">{Math.round(n.battery_pct)}%</span>
        {/if}
      </div>
    {/each}
  {/if}
</Widget>
