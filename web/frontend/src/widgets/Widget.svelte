<!-- Shared card frame: title / loading / error handling, so each widget
     body is just data fetching + markup. -->
<script>
  let { title = '', loading = false, error = null, footer = null, children } = $props();
</script>

<section class="widget">
  {#if title}<h2>{title}</h2>{/if}
  <div class="body">
    {#if error}
      <p class="error">{error}</p>
    {:else if loading}
      <p class="muted">Loading…</p>
    {:else}
      {@render children?.()}
    {/if}
  </div>
  {#if footer && !error && !loading}
    <p class="muted footer">{footer}</p>
  {/if}
</section>
