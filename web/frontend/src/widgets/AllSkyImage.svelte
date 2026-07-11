<!-- Latest all-sky frame with the AI segmentation mask as an overlay whose
     opacity is a slider: Original <-> AI. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { dateTime } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30, stationId = null } = $props();

  let obs = $state(null);
  let error = $state(null);
  let loading = $state(true);
  let maskOpacity = $state(0); // 0 = original, 100 = full AI mask

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

<Widget {title} {loading} {error} footer={obs?.image_url ? dateTime(obs.timestamp) : null}>
  {#if obs?.image_url}
    <div class="allsky">
      <img src={obs.image_url} alt="Latest all-sky frame" />
      {#if obs.mask_url && maskOpacity > 0}
        <img
          class="mask"
          style:opacity={maskOpacity / 100}
          src={obs.mask_url}
          alt="Cloud segmentation mask overlay"
        />
      {/if}
    </div>
    {#if obs.mask_url}
      <label class="allsky-toggle">
        <span>original</span>
        <input type="range" min="0" max="100" bind:value={maskOpacity} />
        <span>AI mask</span>
      </label>
    {/if}
  {:else if obs}
    <p class="muted">The latest observation carries no image reference.</p>
  {/if}
</Widget>
