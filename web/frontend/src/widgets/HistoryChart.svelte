<!-- Cloud-cover time series: avg line + min/max band. The columnar
     /observations/series response feeds uPlot directly; all imperative
     uPlot handling is confined to this component. The live legend row
     doubles as the crosshair readout (time / max / min / avg). -->
<script>
  import { onMount } from 'svelte';
  import uPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';
  import { get, poll } from '../lib/api.js';
  import Widget from './Widget.svelte';

  let {
    title = '',
    refreshSeconds = 30,
    stationId = null,
    hours = 24,
    bucket = '10m',
  } = $props();

  const RANGES = [
    { label: '24 h', hours: 24, bucket: '10m' },
    { label: '7 d', hours: 168, bucket: '1h' },
    { label: '30 d', hours: 720, bucket: '6h' },
  ];

  let el;
  let chart = null;
  let hasChart = $state(false);
  let empty = $state(false);
  let error = $state(null);
  // svelte-ignore state_referenced_locally -- layout props are static per
  // widget instance; they only seed the initial range.
  let range = $state(RANGES.find((r) => r.hours === hours) ?? { label: null, hours, bucket });

  function csvHref() {
    const end = new Date();
    const start = new Date(end.getTime() - range.hours * 3600 * 1000);
    const params = new URLSearchParams({ start: start.toISOString(), end: end.toISOString() });
    if (stationId) params.set('station_id', stationId);
    return `/api/v1/observations/export.csv?${params}`;
  }

  function setRange(r) {
    range = r;
    refresh();
  }

  const pctFmt = (v) => (v == null ? '—' : `${Math.round(v * 100)}%`);

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
        scales: { y: { range: [0, 1] } },
        axes: [axis, { ...axis, values: (u, vals) => vals.map(pctFmt) }],
        series: [
          {},
          { label: 'max', stroke: 'transparent', points: { show: false }, value: (u, v) => pctFmt(v) },
          { label: 'min', stroke: 'transparent', points: { show: false }, value: (u, v) => pctFmt(v) },
          {
            label: 'cloud cover',
            stroke: cssVar('--series-1'),
            width: 2,
            points: { show: false },
            value: (u, v) => pctFmt(v),
          },
        ],
        bands: [{ series: [1, 2], fill: cssVar('--series-1-band') }],
      },
      data,
      el
    );
    hasChart = true;
  }

  async function refresh() {
    try {
      const end = new Date();
      const start = new Date(end.getTime() - range.hours * 3600 * 1000);
      const params = new URLSearchParams({
        start: start.toISOString(),
        end: end.toISOString(),
        bucket: range.bucket,
      });
      if (stationId) params.set('station_id', stationId);
      const s = await get(`/api/v1/observations/series?${params}`);
      empty = s.ts.length === 0;
      if (!empty) {
        const data = [s.ts, s.max, s.min, s.avg];
        if (chart) chart.setData(data);
        else makeChart(data);
      } else if (chart) {
        chart.setData([[], [], [], []]);
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
  <div class="toolbar">
    {#each RANGES as r (r.label)}
      <button class="range-btn" class:active={range.label === r.label} onclick={() => setRange(r)}>
        {r.label}
      </button>
    {/each}
    <span class="spacer"></span>
    <a class="range-btn" href={csvHref()} download>CSV ↓</a>
  </div>
  {#if !hasChart}
    {#if error}
      <p class="error">{error}</p>
    {:else if empty}
      <p class="muted">No observations in the selected range yet.</p>
    {:else}
      <p class="muted">Loading…</p>
    {/if}
  {:else if error}
    <p class="muted">refresh failed: {error}</p>
  {:else if empty}
    <p class="muted">No observations in the selected range.</p>
  {/if}
  <div bind:this={el}></div>
</Widget>
