<!-- LoRa link quality over time for one node: RSSI and SNR as two stacked
     small-multiple charts (different units never share one axis). The
     node selector appears when more than one node has reported. -->
<script>
  import { onMount } from 'svelte';
  import uPlot from 'uplot';
  import 'uplot/dist/uPlot.min.css';
  import { get, poll } from '../lib/api.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30, limit = 300 } = $props();

  let nodes = $state([]);
  let nodeId = $state(null);
  let error = $state(null);
  let empty = $state(false);
  let hasCharts = $state(false);

  let rssiEl, snrEl;
  let rssiChart = null;
  let snrChart = null;

  function cssVar(name) {
    return getComputedStyle(rssiEl).getPropertyValue(name).trim();
  }

  function makeChart(el, label, color, unit, data) {
    const axis = {
      stroke: cssVar('--text-muted'),
      grid: { stroke: cssVar('--gridline'), width: 1 },
      ticks: { stroke: cssVar('--baseline'), width: 1 },
    };
    return new uPlot(
      {
        width: el.clientWidth,
        height: 120,
        axes: [axis, { ...axis }],
        series: [
          {},
          {
            label,
            stroke: color,
            width: 2,
            points: { show: false },
            value: (u, v) => (v == null ? '—' : `${v} ${unit}`),
          },
        ],
      },
      data,
      el
    );
  }

  async function refresh() {
    try {
      nodes = await get('/api/v1/nodes');
      if (!nodes.length) {
        empty = true;
        return;
      }
      empty = false;
      if (!nodeId || !nodes.some((n) => n.node_id === nodeId)) {
        nodeId = nodes[0].node_id;
      }
      const history = await get(
        `/api/v1/nodes/${encodeURIComponent(nodeId)}/telemetry?limit=${limit}`
      );
      const asc = history.slice().reverse(); // API is newest-first
      const ts = asc.map((r) => new Date(r.last_seen).getTime() / 1000);
      const rssiData = [ts, asc.map((r) => r.rssi_dbm)];
      const snrData = [ts, asc.map((r) => r.snr_db)];
      if (rssiChart) {
        rssiChart.setData(rssiData);
        snrChart.setData(snrData);
      } else {
        rssiChart = makeChart(rssiEl, 'RSSI', cssVar('--series-1'), 'dBm', rssiData);
        snrChart = makeChart(snrEl, 'SNR', cssVar('--series-2'), 'dB', snrData);
        hasCharts = true;
      }
      error = null;
    } catch (e) {
      error = e.message;
    }
  }

  function selectNode(id) {
    nodeId = id;
    refresh();
  }

  onMount(() => {
    const stopPoll = poll(refresh, refreshSeconds);
    const ro = new ResizeObserver(() => {
      if (rssiChart && rssiEl.clientWidth > 0) {
        rssiChart.setSize({ width: rssiEl.clientWidth, height: 120 });
        snrChart.setSize({ width: snrEl.clientWidth, height: 120 });
      }
    });
    ro.observe(rssiEl);
    return () => {
      ro.disconnect();
      stopPoll();
      rssiChart?.destroy();
      snrChart?.destroy();
    };
  });
</script>

<Widget {title}>
  {#if nodes.length > 1}
    <div class="toolbar">
      {#each nodes as n (n.node_id)}
        <button
          class="range-btn"
          class:active={n.node_id === nodeId}
          onclick={() => selectNode(n.node_id)}
        >
          {n.node_id}
        </button>
      {/each}
    </div>
  {/if}
  {#if empty}
    <p class="muted">No LoRa nodes have reported yet.</p>
  {:else if !hasCharts}
    {#if error}
      <p class="error">{error}</p>
    {:else}
      <p class="muted">Loading…</p>
    {/if}
  {:else if error}
    <p class="muted">refresh failed: {error}</p>
  {/if}
  <p class="chart-label muted">RSSI (dBm)</p>
  <div bind:this={rssiEl}></div>
  <p class="chart-label muted">SNR (dB)</p>
  <div bind:this={snrEl}></div>
</Widget>
