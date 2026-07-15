<!-- Precipitation radar: Leaflet + OpenStreetMap base tiles + RainViewer
     radar frames. The server caches only the frame index; tiles load
     straight from the CDNs in the browser, so this widget needs internet
     on the client side. Frames animate by toggling per-frame layer
     opacity (no tile reload flicker). -->
<script>
  import { onMount } from 'svelte';
  import L from 'leaflet';
  import 'leaflet/dist/leaflet.css';
  import { get } from '../lib/api.js';
  import { timeShort } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', zoom = 7, height = 320 } = $props();

  let el;
  let error = $state(null);
  let loading = $state(true);
  let frames = $state([]);
  let pastCount = $state(0);
  let frameIdx = $state(0);
  let playing = $state(false);

  let map = null;
  let host = '';
  let activeLayer = null;
  let playTimer = null;
  const layerCache = new Map();

  function frameLayer(frame) {
    if (!layerCache.has(frame.path)) {
      layerCache.set(
        frame.path,
        L.tileLayer(`${host}${frame.path}/256/{z}/{x}/{y}/2/1_1.png`, {
          opacity: 0,
          attribution: '<a href="https://www.rainviewer.com/">RainViewer</a>',
        }).addTo(map)
      );
    }
    return layerCache.get(frame.path);
  }

  function show(i) {
    frameIdx = i;
    const layer = frameLayer(frames[i]);
    if (activeLayer && activeLayer !== layer) activeLayer.setOpacity(0);
    layer.setOpacity(0.7);
    activeLayer = layer;
  }

  function togglePlay() {
    playing = !playing;
    if (playing) {
      playTimer = setInterval(() => show((frameIdx + 1) % frames.length), 600);
    } else {
      clearInterval(playTimer);
    }
  }

  onMount(() => {
    (async () => {
      try {
        const [stations, res] = await Promise.all([
          get('/api/v1/stations'),
          get('/api/v1/providers/rain-viewer/frames'),
        ]);
        const st = stations[0] ?? {};
        host = res.data.host;
        frames = [...res.data.past, ...res.data.nowcast];
        pastCount = res.data.past.length;

        map = L.map(el, {
          center: [st.latitude ?? 40.4, st.longitude ?? -3.7],
          zoom,
          scrollWheelZoom: false, // don't hijack page scrolling
        });
        L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
          attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
        }).addTo(map);
        if (st.latitude != null && st.longitude != null) {
          L.circleMarker([st.latitude, st.longitude], {
            radius: 6,
            weight: 2,
            fillOpacity: 0.8,
          })
            .addTo(map)
            .bindTooltip(st.name ?? st.id);
        }
        if (frames.length) show(Math.max(pastCount - 1, 0)); // most recent real frame
        error = null;
      } catch (e) {
        error = e.message;
      } finally {
        loading = false;
      }
    })();

    const ro = new ResizeObserver(() => map?.invalidateSize());
    ro.observe(el);
    return () => {
      ro.disconnect();
      clearInterval(playTimer);
      map?.remove();
    };
  });
</script>

<Widget {title}>
  {#if error}
    <p class="error">{error}</p>
  {:else if loading}
    <p class="muted">Loading…</p>
  {/if}
  <div bind:this={el} class="rain-map" style:height="{height}px"></div>
  {#if frames.length}
    <div class="map-controls">
      <button class="map-btn" onclick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? '⏸' : '▶'}
      </button>
      <input
        type="range"
        min="0"
        max={frames.length - 1}
        value={frameIdx}
        oninput={(e) => show(+e.target.value)}
      />
      <span class="muted map-time">
        {frameIdx >= pastCount ? 'forecast ' : ''}{timeShort(frames[frameIdx].time * 1000)}
      </span>
    </div>
  {/if}
</Widget>
