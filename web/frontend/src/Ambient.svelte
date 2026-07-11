<!-- Full-page ambient weather layer (config: dashboard.ambient). Pure CSS
     animations behind the cards, pointer-events: none, hidden entirely
     under prefers-reduced-motion except for the static glows. -->
<script>
  import { onMount } from 'svelte';
  import { get, poll } from './lib/api.js';
  import { decideAmbient } from './lib/ambient.js';

  let { refreshSeconds = 60 } = $props();

  let kind = $state(null);

  const rand = (a, b) => a + Math.random() * (b - a);
  const drops = Array.from({ length: 36 }, () => ({
    left: rand(0, 100), delay: rand(0, 1.6), dur: rand(0.9, 1.6),
  }));
  const flakes = Array.from({ length: 28 }, () => ({
    left: rand(0, 100), delay: rand(0, 6), dur: rand(6, 12), size: rand(3, 6),
  }));
  const stars = Array.from({ length: 44 }, () => ({
    left: rand(0, 100), top: rand(0, 55), delay: rand(0, 4), dur: rand(2, 5), size: rand(1, 2.5),
  }));
  const clouds = [
    { top: 3, dur: 130, delay: 0, s: 1 },
    { top: 14, dur: 170, delay: -60, s: 1.5 },
    { top: 26, dur: 110, delay: -30, s: 0.8 },
  ];

  const cloudy = () => ['overcast', 'partly-day', 'partly-night', 'rain', 'snow'].includes(kind);

  onMount(() =>
    poll(async () => {
      let current = null;
      let obs = null;
      try {
        current = (await get('/api/v1/providers/open-meteo/forecast')).data.current;
      } catch { /* provider disabled or offline — obs may still decide */ }
      try {
        obs = await get('/api/v1/observations/latest');
      } catch { /* no observations yet */ }
      kind = decideAmbient(current, obs);
    }, Math.max(refreshSeconds, 60))
  );
</script>

{#if kind}
  <div class="ambient" aria-hidden="true">
    {#if kind === 'rain'}
      {#each drops as d}
        <i class="drop" style="left:{d.left}%;animation-delay:{d.delay}s;animation-duration:{d.dur}s"></i>
      {/each}
    {/if}
    {#if kind === 'snow'}
      {#each flakes as f}
        <i
          class="flake"
          style="left:{f.left}%;width:{f.size}px;height:{f.size}px;animation-delay:{f.delay}s;animation-duration:{f.dur}s"
        ></i>
      {/each}
    {/if}
    {#if kind === 'clear-day' || kind === 'partly-day'}
      <i class="sun-glow"></i>
    {/if}
    {#if kind === 'clear-night' || kind === 'partly-night'}
      {#each stars as s}
        <i
          class="star"
          style="left:{s.left}%;top:{s.top}vh;width:{s.size}px;height:{s.size}px;animation-delay:{s.delay}s;animation-duration:{s.dur}s"
        ></i>
      {/each}
      <i class="moon-glow"></i>
    {/if}
    {#if cloudy()}
      {#each clouds as c}
        <i
          class="cloud-blob"
          style="top:{c.top}vh;--s:{c.s};animation-duration:{c.dur}s;animation-delay:{c.delay}s"
        ></i>
      {/each}
    {/if}
  </div>
{/if}

<style>
  .ambient {
    position: fixed;
    inset: 0;
    z-index: 0;
    overflow: hidden;
    pointer-events: none;
  }

  .ambient i {
    position: absolute;
    display: block;
  }

  /* rain */
  .drop {
    top: -12vh;
    width: 1.5px;
    height: 14px;
    background: linear-gradient(to bottom, transparent, var(--series-1));
    opacity: 0.4;
    animation: fall linear infinite;
  }

  @keyframes fall {
    to {
      transform: translateY(124vh);
    }
  }

  /* snow */
  .flake {
    top: -6vh;
    border-radius: 50%;
    background: var(--text-muted);
    opacity: 0.5;
    animation: drift linear infinite;
  }

  @keyframes drift {
    25% { margin-left: 14px; }
    75% { margin-left: -14px; }
    to { transform: translateY(112vh); }
  }

  /* sun */
  .sun-glow {
    top: -18vmin;
    right: -18vmin;
    width: 56vmin;
    height: 56vmin;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(253, 184, 19, 0.3), transparent 68%);
    animation: pulse 9s ease-in-out infinite alternate;
  }

  /* moon + stars */
  .moon-glow {
    top: -12vmin;
    right: -12vmin;
    width: 42vmin;
    height: 42vmin;
    border-radius: 50%;
    background: radial-gradient(circle, rgba(195, 200, 220, 0.16), transparent 68%);
    animation: pulse 12s ease-in-out infinite alternate;
  }

  @keyframes pulse {
    to { transform: scale(1.12); }
  }

  .star {
    border-radius: 50%;
    background: var(--text-secondary);
    opacity: 0.25;
    animation: twinkle ease-in-out infinite alternate;
  }

  @keyframes twinkle {
    to { opacity: 0.9; }
  }

  /* drifting clouds */
  .cloud-blob {
    left: 0;
    width: 42vw;
    height: 14vh;
    border-radius: 50%;
    background: var(--text-muted);
    opacity: 0.1;
    filter: blur(34px);
    animation: cross linear infinite;
  }

  @keyframes cross {
    from { transform: translateX(-50vw) scale(var(--s, 1)); }
    to { transform: translateX(150vw) scale(var(--s, 1)); }
  }

  @media (prefers-reduced-motion: reduce) {
    .drop, .flake, .star, .cloud-blob { display: none; }
    .sun-glow, .moon-glow { animation: none; }
  }
</style>
