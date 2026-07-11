<script>
  import { get } from './lib/api.js';
  import { WIDGETS } from './widgets/registry.js';
  import Widget from './widgets/Widget.svelte';
  import Ambient from './Ambient.svelte';

  let dashboard = $state(null);
  let error = $state(null);
  let hash = $state(location.hash);

  get('/api/v1/dashboard')
    .then((d) => (dashboard = d))
    .catch((e) => (error = e.message));

  $effect(() => {
    const onHash = () => (hash = location.hash);
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  });

  const section = $derived.by(() => {
    if (!dashboard) return null;
    const id = hash.replace(/^#\/?/, '');
    return dashboard.sections.find((s) => s.id === id) ?? dashboard.sections[0] ?? null;
  });
</script>

{#if dashboard?.ambient}
  <Ambient refreshSeconds={dashboard.refresh_seconds} />
{/if}

<header>
  <h1>{dashboard?.title ?? 'Nimbus'}</h1>
  <span class="subtitle">all-sky cloud observation</span>
</header>

{#if dashboard && dashboard.sections.length > 1}
  <nav aria-label="Dashboard sections">
    {#each dashboard.sections as s (s.id)}
      <a href="#/{s.id}" class:active={section?.id === s.id}>
        {#if s.icon}<span aria-hidden="true">{s.icon}</span>{/if}
        {s.title}
      </a>
    {/each}
  </nav>
{/if}

<main>
  {#if error}
    <p class="error">Cannot reach the API: {error}</p>
  {:else if section}
    {#each section.layout as item (item)}
      {@const Component = WIDGETS[item.type]}
      <div class="cell" style:grid-column="span {Math.min(item.span ?? 1, 4)}">
        {#if Component}
          <Component
            title={item.title}
            refreshSeconds={dashboard.refresh_seconds}
            {...item.props}
          />
        {:else}
          <Widget title={item.title || item.type} error={`unknown widget type "${item.type}"`} />
        {/if}
      </div>
    {/each}
  {/if}
</main>
