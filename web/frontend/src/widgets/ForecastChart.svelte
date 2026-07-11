<!-- 48 h outlook: total cloud cover + precipitation probability from the
     open-meteo provider, both 0-100 % on a single axis (one scale, one
     chart — never dual-axis). Two series, so the live legend is the
     identity + hover readout. -->
<script>
  import { onMount } from 'svelte';
  import uPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';
  import { get, poll } from '../lib/api.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30 } = $props();

  let el;
  let chart = null;
  let hasChart = $state(false);
  let error = $state(null);

  const pctFmt = (v) => (v == null ? '—' : `${Math.round(v)}%`);

  function cssVar(name) {
    return getComputedStyle(el).getPropertyValue(name).trim();
  }

  function makeChart(data) {
    const axis = {
      stroke: cssVar('--text-muted'),
      grid: { stroke: cssVar('--gridline'), width: 1 },
      ticks: { stroke: cssVar('--baseline'), width: 1 },
    };
    chart = new uPlot(
      {
        width: el.clientWidth,
        height: 220,
        scales: { y: { range: [0, 100] } },
        axes: [axis, { ...axis, values: (u, vals) => vals.map(pctFmt) }],
        series: [
          {},
          {
            label: 'cloud cover',
            stroke: cssVar('--series-1'),
            width: 2,
            points: { show: false },
            value: (u, v) => pctFmt(v),
          },
          {
            label: 'precip. prob.',
            stroke: cssVar('--series-2'),
            width: 2,
            points: { show: false },
            value: (u, v) => pctFmt(v),
          },
        ],
      },
      data,
      el
    );
    hasChart = true;
  }

  async function refresh() {
    try {
      const h = (await get('/api/v1/providers/open-meteo/forecast')).data.hourly;
      const nowSec = Date.now() / 1000;
      const ts = [];
      const clouds = [];
      const precip = [];
      h.time.forEach((t, i) => {
        const sec = new Date(`${t}Z`).getTime() / 1000; // UTC, no suffix from the API
        if (sec >= nowSec - 3600) {
          ts.push(sec);
          clouds.push(h.cloud_cover[i]);
          precip.push(h.precipitation_probability[i]);
        }
      });
      if (ts.length) {
        const data = [ts, clouds, precip];
        if (chart) chart.setData(data);
        else makeChart(data);
      }
      error = null;
    } catch (e) {
      error = e.message;
    }
  }

  onMount(() => {
    const stopPoll = poll(refresh, refreshSeconds);
    const ro = new ResizeObserver(() => {
      if (chart && el.clientWidth > 0) chart.setSize({ width: el.clientWidth, height: 220 });
    });
    ro.observe(el);
    return () => {
      ro.disconnect();
      stopPoll();
      chart?.destroy();
    };
  });
</script>

<Widget {title}>
  {#if !hasChart}
    {#if error}
      <p class="error">{error}</p>
    {:else}
      <p class="muted">Loading…</p>
    {/if}
  {:else if error}
    <p class="muted">refresh failed: {error}</p>
  {/if}
  <div bind:this={el}></div>
</Widget>
