<!-- Replays the most recent all-sky frames (observations that carry an
     image reference) as a timelapse. Frames are preloaded so playback
     doesn't stutter on slow clients. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { dateTime } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 60, stationId = null, frames = 48, fps = 4 } = $props();

  let imgs = $state([]);
  let idx = $state(0);
  let playing = $state(false);
  let error = $state(null);
  let loading = $state(true);

  let timer = null;

  function togglePlay() {
    playing = !playing;
    if (playing) {
      timer = setInterval(() => {
        if (imgs.length) idx = (idx + 1) % imgs.length;
      }, 1000 / fps);
    } else {
      clearInterval(timer);
    }
  }

  onMount(() => {
    const params = new URLSearchParams({ limit: String(frames) });
    if (stationId) params.set('station_id', stationId);
    const stop = poll(async () => {
      try {
        const rows = await get(`/api/v1/observations?${params}`);
        const withImages = rows
          .filter((r) => r.image_url)
          .reverse() // API is newest-first; play chronologically
          .map((r) => ({ url: r.image_url, timestamp: r.timestamp }));
        withImages.forEach((f) => {
          const pre = new Image();
          pre.src = f.url;
        });
        imgs = withImages;
        if (idx >= imgs.length) idx = Math.max(imgs.length - 1, 0);
        error = null;
      } catch (e) {
        error = e.message;
      } finally {
        loading = false;
      }
    }, Math.max(refreshSeconds, 60));
    return () => {
      stop();
      clearInterval(timer);
    };
  });
</script>

<Widget {title} {loading} {error}>
  {#if imgs.length === 0}
    <p class="muted">No frames with images in the recent observations yet.</p>
  {:else}
    <div class="allsky">
      <img src={imgs[idx].url} alt="All-sky timelapse frame" />
    </div>
    <div class="map-controls">
      <button class="map-btn" onclick={togglePlay} aria-label={playing ? 'Pause' : 'Play'}>
        {playing ? '⏸' : '▶'}
      </button>
      <input
        type="range"
        min="0"
        max={imgs.length - 1}
        value={idx}
        oninput={(e) => (idx = +e.target.value)}
      />
      <span class="muted map-time">{dateTime(imgs[idx].timestamp)}</span>
    </div>
  {/if}
</Widget>
