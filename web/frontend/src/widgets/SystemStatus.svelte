<!-- Pi health at a glance: /health + station last_seen. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { dateTime } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30 } = $props();

  let health = $state(null);
  let stations = $state([]);
  let error = $state(null);
  let loading = $state(true);

  const gb = (mb) => (mb / 1024).toFixed(1);
  const age = (s) => (s < 120 ? `${s}s` : `${Math.round(s / 60)}min`);

  onMount(() =>
    poll(async () => {
      try {
        [health, stations] = await Promise.all([
          get('/api/v1/health'),
          get('/api/v1/stations'),
        ]);
        error = null;
      } catch (e) {
        error = e.message;
      } finally {
        loading = false;
      }
    }, refreshSeconds)
  );
</script>

<Widget {title} {loading} {error} footer={health ? `nimbus web v${health.version}` : null}>
  {#if health}
    <dl class="kv">
      <dt>Status</dt>
      <dd>{health.status}{health.db_ok ? ' · db ok' : ' · DB ERROR'}</dd>
      <dt>Disk free</dt>
      <dd>{gb(health.disk_free_mb)} GB</dd>
      {#if health.system.load_1m != null}
        <dt>CPU load</dt>
        <dd>{health.system.load_1m} / {health.system.cpu_count} cores</dd>
      {/if}
      {#if health.system.mem_available_mb != null}
        <dt>RAM free</dt>
        <dd>{gb(health.system.mem_available_mb)} / {gb(health.system.mem_total_mb)} GB</dd>
      {/if}
      {#if health.system.cpu_temp_c != null}
        <dt>CPU temp</dt>
        <dd>{health.system.cpu_temp_c} °C</dd>
      {/if}
      {#if health.system.uptime_hours != null}
        <dt>Uptime</dt>
        <dd>
          {health.system.uptime_hours < 48
            ? `${health.system.uptime_hours} h`
            : `${(health.system.uptime_hours / 24).toFixed(1)} d`}
        </dd>
      {/if}
      {#each stations as s (s.id)}
        <dt>{s.id}</dt>
        <dd>{s.last_seen ? `seen ${dateTime(s.last_seen)}` : 'no data yet'}</dd>
      {/each}
      {#each health.provider_cache as c (`${c.provider}/${c.resource}`)}
        <dt>{c.provider}</dt>
        <dd>cache {age(c.age_seconds)} old</dd>
      {/each}
    </dl>
  {/if}
</Widget>
