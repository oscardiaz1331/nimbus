<!-- Sun & moon ephemeris from the local astronomy provider. The
     astronomical-darkness window is the all-sky camera's observing window. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { timeShort } from '../lib/format.js';
  import Widget from './Widget.svelte';

  let { title = '', refreshSeconds = 30 } = $props();

  let eph = $state(null);
  let error = $state(null);
  let loading = $state(true);

  const t = (iso) => (iso ? timeShort(iso) : '—');

  onMount(() =>
    poll(async () => {
      try {
        eph = (await get('/api/v1/providers/astronomy/ephemeris')).data;
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
  {#if eph}
    <div class="astro-moon">
      <span class="astro-icon">{eph.moon.icon}</span>
      <div>
        <div class="astro-phase">{eph.moon.phase_name}</div>
        <p class="muted">{Math.round(eph.moon.illumination * 100)}% illuminated</p>
      </div>
    </div>
    <dl class="kv">
      <dt>Sun</dt>
      <dd>{t(eph.sun.sunrise)} → {t(eph.sun.sunset)}</dd>
      <dt>Now</dt>
      <dd>
        {eph.sun.elevation_now}° elev · {eph.sun.azimuth_now}° az
        {#if eph.sun.is_up}
          · ≤{eph.sun.clear_sky_irradiance_wm2} W/m²
        {/if}
      </dd>
      <dt>Moon</dt>
      <dd>{t(eph.moon.moonrise)} → {t(eph.moon.moonset)}</dd>
      <dt>Darkness</dt>
      <dd>
        {#if eph.darkness.hours != null}
          {t(eph.darkness.astronomical_dusk)} → {t(eph.darkness.astronomical_dawn)}
          ({eph.darkness.hours} h)
        {:else}
          no astronomical darkness tonight
        {/if}
      </dd>
    </dl>
  {/if}
</Widget>
