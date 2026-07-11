<!-- Orrery: three schematic views driven entirely by client-side math
     (lib/astro.js) — no providers, no network, works offline:
       · Day/Night  — Mollweide (equal-area) map with the live terminator
       · Earth–Moon — per-pixel rendered 3D globe (default camera: ecliptic
                      north, so the 23.4° axial tilt is visible; drag to
                      rotate), Moon at its elongation, phase geometry
       · Orbit      — Earth's position on its orbit around the Sun
     Every view has a time slider (±24 h; ±6 months for the orbit) with a
     reset back to live "now".

     The globe camera is stored in the INERTIAL frame (lat + right
     ascension) and converted to Earth-fixed coordinates per frame via
     sidereal time — that is what makes the time slider spin the Earth
     around its own (tilted) N–S axis instead of dragging the scene. -->
<script module>
  // Land/sea bitmask, rasterized once from the Natural Earth rings and
  // shared by every widget instance: per-frame globe rendering is then
  // pure lookups (no polygon clipping, so no fill artifacts ever).
  const MASK_W = 1024;
  const MASK_H = 512;
  let landMask = null;

  function getLandMask(rings) {
    if (landMask) return landMask;
    const c = document.createElement('canvas');
    c.width = MASK_W;
    c.height = MASK_H;
    const ctx = c.getContext('2d', { willReadFrequently: true });
    ctx.fillStyle = '#fff';
    ctx.beginPath();
    for (const ring of rings) {
      ring.forEach(([lon, lat], i) => {
        const x = ((lon + 180) / 360) * MASK_W;
        const y = ((90 - lat) / 180) * MASK_H;
        i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      });
      ctx.closePath();
    }
    ctx.fill();
    const rgba = ctx.getImageData(0, 0, MASK_W, MASK_H).data;
    landMask = new Uint8Array(MASK_W * MASK_H);
    for (let i = 0; i < landMask.length; i++) landMask[i] = rgba[i * 4] > 127 ? 1 : 0;
    return landMask;
  }

  // Globe palette (fixed: the globe is "photographic", not theme-tinted)
  const OCEAN = [30, 95, 175];
  const LAND = [64, 142, 88];
  const MOON_GRAY = [150, 148, 146];

  const DEG2 = Math.PI / 180;

  // Scratch canvas reused across rasterSphere() calls (Earth, then Moon):
  // each sphere is painted into it fresh and composited onto the real
  // canvas with drawImage, which alpha-blends normally. putImageData
  // does NOT blend — it replaces destination pixels outright, so an
  // out-of-circle transparent corner would punch a hole through whatever
  // was already drawn underneath (visible as a "box" wherever two bodies'
  // bounding squares overlapped). drawImage avoids that entirely.
  let scratch = null;

  // Rasterize one sphere (Earth or Moon) onto `ctx` at (cx, cy) with
  // logical radius r. `rx, up, f` is the camera basis (screen-x, screen-y,
  // toward-viewer) — the *same* basis for both bodies, so both react to
  // camera drag identically. `sample(p0,p1,p2)` returns the surface color
  // for the 3D point on the unit sphere; `sunV` (same frame as rx/up/f)
  // drives per-pixel terminator shading. `twilight` widens/narrows the
  // soft day/night band (the Moon has no atmosphere, so its terminator is
  // sharper than Earth's).
  function rasterSphere(ctx, S, cx, cy, r, rx, up, f, sunV, sample, twilight = 0.16) {
    const prF = r * S;
    const pr = Math.ceil(prF);
    const size = 2 * pr + 2;
    if (!scratch) scratch = document.createElement('canvas');
    if (scratch.width !== size || scratch.height !== size) {
      scratch.width = size;
      scratch.height = size;
    }
    const sctx = scratch.getContext('2d');
    const img = sctx.createImageData(size, size);
    const px = img.data;
    for (let j = 0; j < size; j++) {
      const Y = (j - pr) / prF;
      for (let i = 0; i < size; i++) {
        const X = (i - pr) / prF;
        const rr = X * X + Y * Y;
        if (rr > 1.02) continue;
        const edge = (1 - Math.sqrt(rr)) * prF + 1; // soft antialiased rim
        const alpha = edge <= 0 ? 0 : edge >= 1 ? 1 : edge;
        if (alpha === 0) continue;
        const Z = Math.sqrt(Math.max(0, 1 - rr));
        // 3D point on the sphere: X·rx − Y·up + Z·f  (screen y is down)
        const p0 = X * rx[0] - Y * up[0] + Z * f[0];
        const p1 = X * rx[1] - Y * up[1] + Z * f[1];
        const p2 = X * rx[2] - Y * up[2] + Z * f[2];
        const c = sample(p0, p1, p2);
        const lit = p0 * sunV[0] + p1 * sunV[1] + p2 * sunV[2];
        const t = Math.max(0, Math.min(1, lit / twilight + 0.5));
        const b = (0.3 + 0.5 * t + 0.2 * Math.max(0, lit)) * (0.72 + 0.28 * Z);
        const o4 = (j * size + i) * 4;
        px[o4] = c[0] * b;
        px[o4 + 1] = c[1] * b;
        px[o4 + 2] = c[2] * b;
        px[o4 + 3] = alpha * 255;
      }
    }
    sctx.putImageData(img, 0, 0); // scratch was empty; a plain write is fine here
    // Compose onto the real canvas with normal alpha blending, bypassing
    // its logical-unit transform (the scratch is already device pixels).
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(scratch, Math.round(cx * S) - pr, Math.round(cy * S) - pr);
    ctx.restore();
  }
</script>

<script>
  import { onMount } from 'svelte';
  import { get, poll } from '../lib/api.js';
  import { dateTime } from '../lib/format.js';
  import {
    solar,
    moon,
    orbit,
    terminatorLat,
    mollweide,
    vec,
    gmst,
    eclipticPoleGeo,
    PERIHELION_LON,
  } from '../lib/astro.js';
  import LAND_RINGS from '../assets/land-rings.json';
  import Widget from './Widget.svelte';

  // svelte-ignore state_referenced_locally -- layout prop seeds the initial view
  let { title = '', refreshSeconds = 30, view: initialView = 'daynight' } = $props();

  const VIEWS = [
    { id: 'daynight', label: 'Day / Night' },
    { id: 'earthmoon', label: 'Earth – Moon' },
    { id: 'orbit', label: 'Orbit' },
  ];
  const DEG = Math.PI / 180;

  // svelte-ignore state_referenced_locally
  let view = $state(initialView);
  let now = $state(new Date());
  let station = $state(null);

  // Time travel: hours slider shared by Day/Night + Earth–Moon, days
  // slider for the orbit. 0 = live (keeps ticking with the poll).
  let offsetH = $state(0);
  let offsetD = $state(0);
  const t12 = $derived(new Date(now.getTime() + offsetH * 3600_000));
  const t3 = $derived(new Date(now.getTime() + offsetD * 86_400_000));

  const s = $derived(solar(t12));
  const m = $derived(moon(t12));
  const o = $derived(orbit(t3));

  const simTime = $derived(view === 'orbit' ? t3 : t12);
  const simulated = $derived(view === 'orbit' ? offsetD !== 0 : offsetH !== 0);

  // Camera for the globe, in the inertial frame: latitude + right
  // ascension. Default = ecliptic north pole (dec 66.56°, RA 270°).
  const DEFAULT_VIEW = { lat: 66.56, ra: 270 };
  let viewRA = $state({ ...DEFAULT_VIEW });
  const isDefaultView = $derived(
    viewRA.lat === DEFAULT_VIEW.lat && viewRA.ra === DEFAULT_VIEW.ra
  );

  function resetTime() {
    offsetH = 0;
    offsetD = 0;
    viewRA = { ...DEFAULT_VIEW };
    now = new Date();
  }

  let wrapEl;
  let canvasEl = $state(null);
  let emCanvas = $state(null);

  function cssVar(name) {
    return getComputedStyle(wrapEl).getPropertyValue(name).trim();
  }

  function dot(ctx, x, y, r, fill) {
    ctx.fillStyle = fill;
    ctx.beginPath();
    ctx.arc(x, y, r, 0, 2 * Math.PI);
    ctx.fill();
  }

  // ------------------------------------------------------- Day/Night map
  function drawMap() {
    const w = wrapEl?.clientWidth ?? 0;
    if (!w || !canvasEl) return;
    const h = Math.round(w / 2); // Mollweide outline is exactly 2:1
    const dpr = window.devicePixelRatio || 1;
    canvasEl.width = w * dpr;
    canvasEl.height = h * dpr;
    canvasEl.style.height = `${h}px`;
    const ctx = canvasEl.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const rx = w / 2 - 2;
    const ry = h / 2 - 1;
    // Equal-area projection: landmasses keep their true relative sizes.
    const P = (lon, lat) => {
      const [mx, my] = mollweide(lon, lat);
      return [w / 2 + (mx / (2 * Math.SQRT2)) * rx, h / 2 - (my / Math.SQRT2) * ry];
    };
    const trace = (ctx2, points) => {
      points.forEach(([lon, lat], i) => {
        const [x, y] = P(lon, lat);
        i === 0 ? ctx2.moveTo(x, y) : ctx2.lineTo(x, y);
      });
    };

    // globe outline
    ctx.strokeStyle = cssVar('--gridline');
    ctx.beginPath();
    ctx.ellipse(w / 2, h / 2, rx, ry, 0, 0, 2 * Math.PI);
    ctx.stroke();

    ctx.fillStyle = cssVar('--baseline');
    ctx.beginPath();
    for (const ring of LAND_RINGS) {
      trace(ctx, ring);
      ctx.closePath();
    }
    ctx.fill();

    // Night overlay: terminator curve, then hug the ±180° boundary meridians
    // down to the dark pole (they project onto the ellipse edge).
    const darkPole = s.subsolarLat > 0 ? -90 : 90;
    const night = [];
    for (let lon = -180; lon <= 180; lon += 2) {
      night.push([lon, terminatorLat(lon, s.subsolarLat, s.subsolarLon)]);
    }
    const eastEdgeLat = night[night.length - 1][1];
    for (let lat = eastEdgeLat; Math.abs(lat - darkPole) > 3; lat += Math.sign(darkPole - eastEdgeLat) * 3) {
      night.push([180, lat]);
    }
    night.push([180, darkPole]);
    for (let lat = darkPole; Math.abs(lat - night[0][1]) > 3; lat -= Math.sign(darkPole - night[0][1]) * 3) {
      night.push([-180, lat]);
    }
    ctx.fillStyle = 'rgba(8, 10, 30, 0.42)';
    ctx.beginPath();
    trace(ctx, night);
    ctx.closePath();
    ctx.fill();

    const [sx, sy] = P(s.subsolarLon, s.subsolarLat);
    dot(ctx, sx, sy, 5, cssVar('--status-warning'));
    if (station?.longitude != null && station?.latitude != null) {
      const [stx, sty] = P(station.longitude, station.latitude);
      dot(ctx, stx, sty, 5.5, cssVar('--surface'));
      dot(ctx, stx, sty, 4, cssVar('--series-1'));
    }
  }

  // ------------------------------------------- Earth–Moon 3D globe view
  // Logical scene coordinates (like an SVG viewBox of 400x250).
  const EM = { w: 400, h: 250, ex: 150, ey: 122, re: 36, orbitR: 88, rm: 10 };

  const V3 = {
    dot: (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2],
    cross: (a, b) => [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ],
    scale: (a, k) => [a[0] * k, a[1] * k, a[2] * k],
    sub: (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]],
    norm: (a) => {
      const l = Math.hypot(a[0], a[1], a[2]);
      return l ? [a[0] / l, a[1] / l, a[2] / l] : a;
    },
  };

  const sunV = $derived(vec(s.subsolarLon, s.subsolarLat));

  // Inertial camera -> Earth-fixed longitude for the current sim time.
  const viewCenter = $derived({
    lat: viewRA.lat,
    lon: ((viewRA.ra - gmst(t12) + 540) % 360) - 180,
  });

  // Screen basis: +x locked to the sun's direction (rays always from the
  // right), forward = out of the screen.
  const basis = $derived.by(() => {
    const f = vec(viewCenter.lon, viewCenter.lat);
    let rx = V3.sub(sunV, V3.scale(f, V3.dot(sunV, f)));
    rx = Math.hypot(rx[0], rx[1], rx[2]) < 1e-6 ? [1, 0, 0] : V3.norm(rx);
    return { f, rx, up: V3.cross(f, rx) };
  });

  // The Moon's orbital plane is within ~5° of the ecliptic — close enough
  // for a schematic. e1/e2 span that plane, expressed in the same
  // Earth-fixed frame as sunV/basis, so the Moon's screen position reacts
  // to both the time slider (elongation advances) and camera drag (the
  // orbit foreshortens into an ellipse when viewed off-axis), exactly like
  // the globe and the station marker already do.
  const moonPlane = $derived.by(() => {
    const pole = eclipticPoleGeo(t12);
    const seaEcl = vec(pole.lon, pole.lat);
    const e1 = sunV;
    const e2 = V3.norm(V3.cross(seaEcl, sunV));
    return { e1, e2 };
  });
  const moonDir = $derived.by(() => {
    const { e1, e2 } = moonPlane;
    const c = Math.cos(m.elongation * DEG);
    const sn = Math.sin(m.elongation * DEG);
    return [c * e1[0] + sn * e2[0], c * e1[1] + sn * e2[1], c * e1[2] + sn * e2[2]];
  });

  function drawEarthMoon() {
    const w = wrapEl?.clientWidth ?? 0;
    if (!w || !emCanvas) return;
    const h = Math.round((w * EM.h) / EM.w);
    const dpr = window.devicePixelRatio || 1;
    emCanvas.width = w * dpr;
    emCanvas.height = h * dpr;
    emCanvas.style.height = `${h}px`;
    const S = (w / EM.w) * dpr; // logical -> device pixels
    const ctx = emCanvas.getContext('2d');
    ctx.setTransform(S, 0, 0, S, 0, 0);
    ctx.clearRect(0, 0, EM.w, EM.h);
    const { f, rx, up } = basis;
    const warning = cssVar('--status-warning'); // used by the station marker below

    // Moon orbit: the real 3D circle (moon plane) projected through the
    // current camera — an ellipse when viewed off-axis, a circle only
    // when looking straight down the orbital pole (the default view).
    const { e1, e2 } = moonPlane;
    const orbitPt = (deg) => {
      const c = Math.cos(deg * DEG);
      const sn = Math.sin(deg * DEG);
      const d = [c * e1[0] + sn * e2[0], c * e1[1] + sn * e2[1], c * e1[2] + sn * e2[2]];
      return [EM.ex + EM.orbitR * V3.dot(d, rx), EM.ey - EM.orbitR * V3.dot(d, up)];
    };
    ctx.strokeStyle = cssVar('--gridline');
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    for (let deg = 0; deg <= 360; deg += 6) {
      const [x, y] = orbitPt(deg);
      deg === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.setLineDash([]);
    // orbital direction arrow, tangent to the real path near the Moon
    const [ax, ay] = orbitPt(m.elongation + 8);
    const [bx, by] = orbitPt(m.elongation + 14);
    const ang = Math.atan2(by - ay, bx - ax);
    ctx.strokeStyle = cssVar('--text-muted');
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(ax - 6 * Math.cos(ang - 0.5), ay - 6 * Math.sin(ang - 0.5));
    ctx.lineTo(ax, ay);
    ctx.lineTo(ax - 6 * Math.cos(ang + 0.5), ay - 6 * Math.sin(ang + 0.5));
    ctx.stroke();

    // The globe, per pixel: invert the orthographic projection, look up
    // land/sea in the precomputed mask, light with the real sun vector.
    const mask = getLandMask(LAND_RINGS);
    const sampleEarth = (p0, p1, p2) => {
      const lat = Math.asin(Math.max(-1, Math.min(1, p2)));
      const lon = Math.atan2(p1, p0);
      const mi =
        ((MASK_H - 1 - Math.floor(((lat / DEG2 + 90) / 180) * MASK_H)) * MASK_W +
          Math.floor(((lon / DEG2 + 180) / 360) * MASK_W)) |
        0;
      return mask[mi] ? LAND : OCEAN;
    };
    rasterSphere(ctx, S, EM.ex, EM.ey, EM.re, rx, up, f, sunV, sampleEarth);

    // rotation axis (23.4° tilt visible from the default view)
    const npX = V3.dot([0, 0, 1], rx);
    const npY = -V3.dot([0, 0, 1], up); // screen y
    const npFront = V3.dot([0, 0, 1], f) > 0;
    const npLen = Math.hypot(npX, npY);
    const ink = cssVar('--text-primary');
    ctx.fillStyle = ink;
    ctx.strokeStyle = ink;
    ctx.font = '10px system-ui, sans-serif';
    if (npLen < 0.05) {
      dot(ctx, EM.ex, EM.ey, 2.5, ink);
      ctx.fillText(npFront ? 'N' : 'S', EM.ex + 6, EM.ey - 6);
    } else {
      const u = [npX / npLen, npY / npLen];
      const baseR = npFront ? npLen * EM.re : EM.re;
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.moveTo(EM.ex + u[0] * baseR, EM.ey + u[1] * baseR);
      ctx.lineTo(EM.ex + u[0] * (EM.re + 10), EM.ey + u[1] * (EM.re + 10));
      ctx.stroke();
      if (npFront) dot(ctx, EM.ex + npX * EM.re, EM.ey + npY * EM.re, 2, ink);
      ctx.textAlign = 'center';
      ctx.fillText('N', EM.ex + u[0] * (EM.re + 18), EM.ey + u[1] * (EM.re + 18) + 3);
      ctx.textAlign = 'start';
    }

    // station marker
    if (station?.longitude != null && station?.latitude != null) {
      const p = vec(station.longitude, station.latitude);
      if (V3.dot(p, f) > 0) {
        const x = EM.ex + EM.re * V3.dot(p, rx);
        const y = EM.ey - EM.re * V3.dot(p, up);
        dot(ctx, x, y, 4.5, cssVar('--surface'));
        dot(ctx, x, y, 3.2, warning);
      }
    }

    // Moon: real 3D orbital position (moonDir, in the same Earth-fixed
    // frame as sunV/basis) projected through the current camera — so it
    // moves correctly with both the time slider and camera drag, landing
    // on the orbit ellipse traced above. Lit per-pixel with the same
    // sphere rasterizer as Earth (sharper terminator: no atmosphere).
    const d = moonDir;
    const mx = EM.ex + EM.orbitR * V3.dot(d, rx);
    const my = EM.ey - EM.orbitR * V3.dot(d, up);
    rasterSphere(ctx, S, mx, my, EM.rm, rx, up, f, sunV, () => MOON_GRAY, 0.05);
  }

  $effect(() => {
    if (view === 'daynight' && canvasEl) {
      void s.subsolarLon; // track sim time; station tracked below
      void station;
      drawMap();
    }
  });

  $effect(() => {
    if (view === 'earthmoon' && emCanvas) {
      void basis; // camera + sim time
      void m.elongation;
      void station;
      drawEarthMoon();
    }
  });

  onMount(() => {
    get('/api/v1/stations')
      .then((list) => (station = list[0] ?? null))
      .catch(() => {}); // no station dot, everything else still works
    const stopPoll = poll(() => (now = new Date()), Math.max(refreshSeconds, 30));
    const ro = new ResizeObserver(() => {
      drawMap();
      drawEarthMoon();
    });
    ro.observe(wrapEl);
    return () => {
      stopPoll();
      ro.disconnect();
    };
  });

  // Drag to rotate the globe: adjusts the inertial camera, so the Earth
  // still spins about its own axis when time moves afterwards.
  let drag = null;
  function dragStart(e) {
    e.currentTarget.setPointerCapture(e.pointerId);
    drag = { x: e.clientX, y: e.clientY, lat: viewRA.lat, ra: viewRA.ra };
  }
  function dragMove(e) {
    if (!drag) return;
    const k = 360 / (emCanvas?.clientWidth || EM.w); // full width = one turn
    viewRA = {
      lat: Math.max(-89.9, Math.min(89.9, drag.lat + (e.clientY - drag.y) * k)),
      ra: (((drag.ra - (e.clientX - drag.x) * k) % 360) + 360) % 360,
    };
  }
  function dragEnd() {
    drag = null;
  }

  // ------------------------------------------------------- Orbit view
  const OR = { cx: 200, cy: 126, a: 100, e: 0.18 }; // eccentricity exaggerated ~11x
  function orbitPos(lambdaDeg) {
    const nu = (lambdaDeg - PERIHELION_LON) * DEG;
    const r = (OR.a * (1 - OR.e * OR.e)) / (1 + OR.e * Math.cos(nu));
    return [OR.cx + r * Math.cos(lambdaDeg * DEG), OR.cy - r * Math.sin(lambdaDeg * DEG)];
  }
  const orbitPath = $derived.by(() => {
    const pts = [];
    for (let lam = 0; lam <= 360; lam += 3) pts.push(orbitPos(lam).map((v) => v.toFixed(1)));
    return `M${pts.map((p) => p.join(',')).join('L')}Z`;
  });
  const earthPos = $derived(orbitPos(o.lambdaEarth));
  // Screen angle of the sun->Earth direction: the night half points this way.
  const earthAwayDeg = $derived(
    Math.atan2(earthPos[1] - OR.cy, earthPos[0] - OR.cx) / DEG
  );
  const SEASONS = [
    { lam: 180, label: 'Mar' },
    { lam: 270, label: 'Jun' },
    { lam: 0, label: 'Sep' },
    { lam: 90, label: 'Dec' },
  ];
  const APSIDES = [
    { lam: PERIHELION_LON, label: 'perihelion · ~Jan 3' },
    { lam: PERIHELION_LON + 180, label: 'aphelion · ~Jul 4' },
  ];
</script>

<Widget {title} footer={`${dateTime(simTime)}${simulated ? ' · simulated' : ''}`}>
  <div class="toolbar">
    {#each VIEWS as v (v.id)}
      <button class="range-btn" class:active={view === v.id} onclick={() => (view = v.id)}>
        {v.label}
      </button>
    {/each}
  </div>
  <div class="timebar">
    {#if view === 'orbit'}
      <input type="range" min="-183" max="183" step="1" value={offsetD}
        oninput={(e) => (offsetD = +e.target.value)} aria-label="Time offset in days" />
      <span class="muted tlabel">{offsetD === 0 ? 'now' : `${offsetD > 0 ? '+' : ''}${offsetD} d`}</span>
    {:else}
      <input type="range" min="-24" max="24" step="0.5" value={offsetH}
        oninput={(e) => (offsetH = +e.target.value)} aria-label="Time offset in hours" />
      <span class="muted tlabel">{offsetH === 0 ? 'now' : `${offsetH > 0 ? '+' : ''}${offsetH} h`}</span>
    {/if}
    <button class="range-btn" onclick={resetTime} title="Back to live time and default view">
      ⟲ now
    </button>
  </div>
  <div bind:this={wrapEl}>
    {#if view === 'daynight'}
      <canvas bind:this={canvasEl} class="orrery-canvas"></canvas>
      <p class="muted">
        ● sun overhead at {s.subsolarLat.toFixed(1)}°, {s.subsolarLon.toFixed(1)}°
        {#if station}&nbsp;· ● {station.name ?? station.id}{/if}
        &nbsp;· Mollweide (equal-area)
      </p>
    {:else if view === 'earthmoon'}
      <!-- svelte-ignore a11y_no_static_element_interactions -->
      <canvas
        bind:this={emCanvas}
        class="orrery-canvas orrery-drag"
        aria-label="Earth and Moon diagram, drag to rotate the globe"
        onpointerdown={dragStart}
        onpointermove={dragMove}
        onpointerup={dragEnd}
        onpointercancel={dragEnd}
      ></canvas>
      <p class="muted">
        {m.icon} {m.phaseName} · {Math.round(m.illumination * 100)}% lit ·
        {isDefaultView ? 'seen from ecliptic north' : 'custom view'} · drag to rotate ·
        not to scale
      </p>
    {:else}
      <svg viewBox="0 0 400 250" class="orrery-svg" role="img" aria-label="Earth orbit diagram">
        <path d={orbitPath} fill="none" stroke="var(--gridline)" stroke-width="1.5" />
        <!-- Sun at the focus -->
        <circle cx={OR.cx} cy={OR.cy} r="22" fill="var(--status-warning)" opacity="0.2" />
        <circle cx={OR.cx} cy={OR.cy} r="10" fill="var(--status-warning)" />
        <!-- season markers -->
        {#each SEASONS as sm (sm.label)}
          {@const p = orbitPos(sm.lam)}
          <circle cx={p[0]} cy={p[1]} r="2.5" fill="var(--text-muted)" />
          <text x={p[0] + (p[0] >= OR.cx ? 6 : -6)} y={p[1] + 3}
            text-anchor={p[0] >= OR.cx ? 'start' : 'end'} class="svg-label">{sm.label}</text>
        {/each}
        <!-- perihelion / aphelion -->
        {#each APSIDES as ap (ap.label)}
          {@const p = orbitPos(ap.lam)}
          <circle cx={p[0]} cy={p[1]} r="2" fill="var(--series-2)" />
          <text x={p[0] + (p[0] >= OR.cx ? -8 : 8)} y={p[1] + 14}
            text-anchor={p[0] >= OR.cx ? 'end' : 'start'} class="svg-label svg-label-accent">{ap.label}</text>
        {/each}
        <!-- Earth: night half points radially away from the sun -->
        <g transform="translate({earthPos[0]} {earthPos[1]}) rotate({earthAwayDeg})">
          <circle r="7" fill="var(--series-1)" />
          <path d="M 0 -7 A 7 7 0 0 1 0 7 Z" fill="rgba(8, 10, 30, 0.5)" />
        </g>
        <text x="8" y="16" class="svg-label">
          Earth–Sun: {o.rAU.toFixed(4)} AU · {o.rMkm.toFixed(1)} M km
        </text>
        <text x="8" y="242" class="svg-label">eccentricity exaggerated for clarity</text>
      </svg>
    {/if}
  </div>
</Widget>
