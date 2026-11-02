// Coded graphics. Each <div class="figure" data-figure="name"> is filled with an
// SVG drawn by FIGURES[name](svg, w, h, div). Coordinates are canvas px: the
// default figure box is 1700 × 750 (see .figure in theme.css); data-w / data-h
// change it. Groups made with frag() step in with the arrow keys.

(function () {
  const NS = 'http://www.w3.org/2000/svg';
  const W = 1700, H = 750;

  function el(tag, attrs = {}, parent) {
    const n = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (parent) parent.append(n);
    return n;
  }
  function text(parent, x, y, str, attrs = {}) {
    const t = el('text', { x, y, ...attrs }, parent);
    t.textContent = str;
    return t;
  }
  function frag(parent, index) {
    const g = el('g', { class: 'fragment' }, parent);
    if (index !== undefined) g.setAttribute('data-fragment-index', index);
    return g;
  }

  // Run step(dt) every frame while the figure's slide is showing
  function animate(step) {
    let raf = 0, last = 0;
    const tick = now => { step(Math.min(0.05, (now - last) / 1000)); last = now; raf = requestAnimationFrame(tick); };
    return {
      start() { if (!raf) { last = performance.now(); raf = requestAnimationFrame(tick); } },
      stop() { cancelAnimationFrame(raf); raf = 0; },
    };
  }
  // n click-steps owned by this figure: onStep(k) gets the number of steps shown so far
  function steps(svg, n, onStep) {
    const marks = Array.from({ length: n }, () => frag(svg));
    const update = () => onStep(marks.filter(m => m.classList.contains('visible')).length);
    Reveal.on('fragmentshown', update); Reveal.on('fragmenthidden', update); Reveal.on('ready', update);
    onStep(0);
  }

  const FIGURES = {};

  // Wide → narrow → wide, four bands of half-width hw and height dy, labels to the right
  function funnel(svg, labels, hw, dy, size, lit) {
    const widths = [1, 0.42, 0.14, 0.42, 1];
    labels.forEach((label, i) => {
      const y0 = i * dy, y1 = y0 + dy - size / 6, a = widths[i] * hw, b = widths[i + 1] * hw, on = i === lit;
      el('path', { d: `M${hw - a},${y0} H${hw + a} L${hw + b},${y1} H${hw - b} Z`,
        fill: on ? 'var(--warm)' : 'none', stroke: on ? 'var(--warm)' : 'var(--ink-faint)', 'stroke-width': 3 }, svg);
      text(svg, 2 * hw + size, (y0 + y1) / 2 + size / 3, label, { 'font-size': size, fill: on ? 'var(--warm)' : 'var(--ink-dim)' });
    });
  }

  // "You are here" map on the meta slides, with data-at's section lit
  FIGURES['hourglass'] = (svg, w, h, div) => {
    const parts = ['Motivation', 'Theory', 'Results', 'Conclusion'];
    funnel(svg, parts, 55, h / 4, 26, parts.findIndex(p => p.toLowerCase() === div.dataset.at));
  };

  // The shape of a talk
  FIGURES['funnel'] = (svg, w, h) =>
    funnel(svg, ['a hook the room cares about', 'the problem', 'your contribution', 'what it changes'], 190, h / 4, 36);

  // Vague claims, then the concrete versions
  FIGURES['word-soup'] = (svg, w, h) => {
    const soup = [['important', 60, 120], ['widely used', 380, 200], ['significant', 120, 300], ['many applications', 330, 400],
                  ['state of the art', 40, 500], ['efficient', 480, 560], ['novel', 200, 640]];
    const facts = [['~3% of global CO₂', 60, 150], ['Kelvin, 1887', 60, 300], ['10⁶ evaluations', 60, 450], ['30 µs each', 60, 600]];
    const vague = el('g', {}, svg), sharp = frag(svg, 2);
    soup.forEach(([s, x, y]) => text(vague, x, y, s, { 'font-size': 44, class: 'faint', 'font-style': 'italic' }));
    el('rect', { x: 0, y: 0, width: w, height: h, fill: 'var(--bg)' }, sharp);
    facts.forEach(([s, x, y]) => text(sharp, x, y, s, { 'font-size': 64, class: 'warm' }));
  };

  // A room of mixed backgrounds: ask before you pick the hook
  FIGURES['audience'] = (svg, w, h) => {
    const kinds = [['engineers', 'warm'], ['physicists', 'cool'], ['mathematicians', 'green'], ['everyone else', 'ink-faint']];
    const seat = [0, 1, 0, 3, 2, 0, 1, 3, 0, 0, 1, 2, 3, 0, 1, 0, 3, 1, 0, 2, 0, 1, 3, 0, 1, 0, 0, 2, 3, 1, 0, 3, 1, 0, 0, 2];
    seat.forEach((k, n) => el('circle', { cx: 90 + (n % 9) * 80, cy: 80 + Math.floor(n / 9) * 80, r: 26, fill: `var(--${kinds[k][1]})` }, svg));
    kinds.forEach(([s, c], i) => {
      el('circle', { cx: 100 + (i % 2) * 360, cy: 470 + Math.floor(i / 2) * 60, r: 14, fill: `var(--${c})` }, svg);
      text(svg, 130 + (i % 2) * 360, 482 + Math.floor(i / 2) * 60, s, { 'font-size': 34, class: 'dim' });
    });
    text(svg, 90, 680, 'Consider the room', { 'font-size': 44, class: 'warm' });
  };

  // Four point idealisations, animated (Kelvin's wake at the surface follows as a photo layer).
  // Mass: particles fall in at speed ∝ 1/r². Charge: pushed out at speed ∝ 1/r². Vortex: orbit at
  // speed ∝ 1/r. Moving source: rings spreading at a third of its speed, whose envelope is the
  // Kelvin wedge (sin 19.47° = 1/3).
  FIGURES['points'] = svg => {
    const items = [['Point mass', 'Newton · 1687'], ['Point charge', 'Coulomb · 1785'],
                   ['Point vortex', 'Helmholtz · 1858'], ['Moving wave source', 'Kelvin · 1887']];
    const R = 170, cy = 300, N = 60, defs = el('defs', {}, svg), rand = (a, b) => a + Math.random() * (b - a);
    const panels = items.map(([name, who], i) => {
      const cx = W / 4 * (i + 0.5), kelvin = i === 3;
      const clip = el('clipPath', { id: `pt-clip-${i}` }, defs);
      el(kelvin ? 'rect' : 'circle', kelvin ? { x: cx - R, y: cy - R, width: 2 * R, height: 2 * R } : { cx, cy, r: R }, clip);
      const g = el('g', { 'clip-path': `url(#pt-clip-${i})` }, svg);
      text(svg, cx, cy + R + 70, name, { 'font-size': 40, 'text-anchor': 'middle', class: kelvin ? 'warm' : '' });
      text(svg, cx, cy + R + 120, who, { 'font-size': 32, 'text-anchor': 'middle', class: 'dim' });
      if (kelvin) return { cx, g, rings: [], src: el('circle', { r: 9, fill: 'var(--warm)' }, svg), x: -R, clock: 0 };
      el('circle', { cx, cy, r: 9, fill: 'var(--ink)' }, svg);
      const dots = Array.from({ length: N }, () => ({ r: rand(12, R), q: rand(0, 2 * Math.PI), n: el('circle', { r: 4, fill: 'var(--cool)' }, g) }));
      return { cx, g, dots };
    });
    const k = 6e5, vmax = 900, U = 90;
    const move = [
      (d, dt) => { d.r -= Math.min(vmax, k / d.r ** 2) * dt; if (d.r < 10) d.r = R; },
      (d, dt) => { d.r += Math.min(vmax, k / d.r ** 2) * dt; if (d.r > R) d.r = rand(10, 20); },
      (d, dt) => { d.q += 4000 / d.r ** 2 * dt; },
    ];
    function step(dt) {
      panels.slice(0, 3).forEach((p, i) => p.dots.forEach(d => {
        move[i](d, dt);
        d.n.setAttribute('cx', p.cx + d.r * Math.cos(d.q)); d.n.setAttribute('cy', cy + d.r * Math.sin(d.q));
        d.n.setAttribute('opacity', Math.min(1, (R - d.r) / 30));
      }));
      const p = panels[3];
      p.x += U * dt; p.clock += dt;
      if (p.x > R + 40) { p.x = -R - 40; p.rings.forEach(r => r.n.remove()); p.rings = []; }
      if (p.clock > 0.15) { p.clock = 0; p.rings.push({ x: p.cx + p.x, age: 0, n: el('circle', { cy, fill: 'none', stroke: 'var(--cool)', 'stroke-width': 2 }, p.g) }); }
      p.rings.forEach(r => { r.age += dt; r.n.setAttribute('cx', r.x); r.n.setAttribute('r', U / 3 * r.age); r.n.setAttribute('opacity', Math.max(0, 1 - r.age / 4)); });
      p.src.setAttribute('cx', p.cx + p.x); p.src.setAttribute('cy', cy);
    }
    return animate(step);
  };

  // Kelvin's wavelike integral at x = -8, y = 1, just below the surface: the phase g(t), the
  // running sum S of exp(i g) over t (a Cornu spiral, as in Fresnel diffraction), and which
  // samples sit near the two stationary points, where |g'| is small. The range [t0, t1] gives
  // the same phase, 60 rad (~10 loops), beyond each stationary point.
  function kelvinSum(t0 = -5.09, t1 = 11.7, n = 9000, z = -0.002) {
    const x = -8, y = 1, g = t => (x + y * t) * Math.hypot(1, t), dg = t => y * Math.hypot(1, t) + (x + y * t) * t / Math.hypot(1, t);
    const t = Array.from({ length: n + 1 }, (_, i) => t0 + (t1 - t0) * i / n), dt = (t1 - t0) / n;
    let re = 0, im = 0;
    const S = t.map(s => { const a = Math.exp(z * (1 + s * s)) * dt; re += a * Math.cos(g(s)); im += a * Math.sin(g(s)); return [re, im]; });
    S.unshift([0, 0]); S.pop();   // the sum before each step, so S[0] is the origin
    const sp = [1, -1].map(s => Math.round(((-(x + s * Math.sqrt(x * x - 8 * y * y)) / (4 * y)) - t0) / (t1 - t0) * n));   // g'(t±) = 0
    return { t, g, S, n, t0, t1, sp, flat: t.map(s => Math.abs(dg(s)) < 1.5) };
  }
  // S scaled to fit the box (x0, y0) → (x0 + bw, y0 + bh), keeping its aspect
  function fit(S, x0, y0, bw, bh) {
    const [xs, ys] = [0, 1].map(j => S.map(p => p[j])), lo = [Math.min(...xs), Math.min(...ys)], hi = [Math.max(...xs), Math.max(...ys)];
    const sc = Math.min(bw / (hi[0] - lo[0]), bh / (hi[1] - lo[1]));
    return S.map(([a, b]) => [x0 + (a - lo[0]) * sc, y0 + bh - (b - lo[1]) * sc]);
  }
  const polyline = (parent, pts, attrs) =>
    el('path', { d: pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join(''), fill: 'none', ...attrs }, parent);

  // The integral built up as t sweeps: on top the integrand cos g(t), below its running sum.
  // It only travels near the two stationary points (warm, marked on both); elsewhere it curls
  // up, ever tighter but never quite finished: at the surface nothing decays. When the sweep
  // ends, the integral itself appears as the arrow from the origin to the end of the sum.
  FIGURES['cornu'] = (svg, w, h) => {
    const { t, g, S, n, t0, t1, sp } = kelvinSum(), top = 170;
    const P = fit(S, 40, top + 30, w - 80, h - top - 60), Q = t.map(s => [40 + (s - t0) / (t1 - t0) * (w - 80), 95 - 45 * Math.cos(g(s))]);
    const traced = pts => {   // revealed along its length by stroke-dashoffset
      const len = [0];
      pts.slice(1).forEach((p, i) => len.push(len[i] + Math.hypot(p[0] - pts[i][0], p[1] - pts[i][1])));
      return { len, n: polyline(svg, pts, { stroke: 'var(--cool)', 'stroke-width': 2.5, 'stroke-dasharray': `${len[n]} ${len[n]}` }) };
    };
    const strip = traced(Q), spiral = traced(P);
    const marks = sp.flatMap(i => [P, Q].map(pts => ({ i, n: el('circle', { cx: pts[i][0], cy: pts[i][1], r: 10, fill: 'var(--warm)', opacity: 0 }, svg) })));
    text(svg, 40, 20, 'each wave adds a small step in its own direction (phase)', { 'font-size': 28, class: 'dim' });
    text(svg, 40, top + 10, 'the steps added head to tail', { 'font-size': 28, class: 'dim' });
    el('circle', { cx: P[0][0], cy: P[0][1], r: 8, fill: 'var(--ink)' }, svg);
    const defs = el('defs', {}, svg), head = el('marker', { id: 'cornu-head', viewBox: '0 0 10 10', refX: 9, refY: 5, markerWidth: 5, markerHeight: 5, orient: 'auto-start-reverse' }, defs);
    el('path', { d: 'M0,0 L10,5 L0,10 Z', fill: 'var(--ink)' }, head);
    const answer = el('g', { opacity: 0 }, svg), mid = [(P[0][0] + P[n][0]) / 2, (P[0][1] + P[n][1]) / 2];
    el('line', { x1: P[0][0], y1: P[0][1], x2: P[n][0], y2: P[n][1], stroke: 'var(--ink)', 'stroke-width': 6, 'marker-end': 'url(#cornu-head)' }, answer);
    text(answer, mid[0] + 30, mid[1] + 50, 'the answer: the wave at this point', { 'font-size': 32 });
    marks.push({ i: n, n: answer });
    const first = { i: Math.max(...sp) }, name = text(svg, P[first.i][0] + 24, P[first.i][1] + 44, 'stationary point: the phase stands still',
      { 'font-size': 30, class: 'warm', opacity: 0 });
    let s = 0;
    return animate(dt => {
      s = (s + dt / 5) % 1.4;   // 5 s sweep, then 2 s to read the answer
      const i = Math.min(n, Math.round(Math.min(1, s) * n));
      [strip, spiral].forEach(p => p.n.setAttribute('stroke-dashoffset', p.len[n] - p.len[i]));
      marks.forEach(m => m.n.setAttribute('opacity', m.i <= i ? 1 : 0));
      name.setAttribute('opacity', first.i <= i ? 1 : 0);
    });
  };

  // The same spiral, brute force (blue, every curl) against our method (orange): real-line
  // quadrature only through the stationary regions, and each curled-up tail in one jump.
  FIGURES['shortcut'] = (svg, w, h) => {
    const { S, n, flat } = kelvinSum(), P = fit(S, 40, 90, w - 80, h - 130);
    polyline(svg, P, { stroke: 'var(--cool)', 'stroke-width': 2, opacity: 0.7 });
    text(svg, 40, 40, 'brute force: follow every curl', { 'font-size': 34, class: 'cool' });
    const g = frag(svg, 0), runs = [];
    flat.forEach((f, i) => { if (f && !flat[i - 1]) runs.push([i]); if (!f && flat[i - 1]) runs[runs.length - 1].push(i); });
    let prev = 0;
    runs.forEach(([a, b]) => {
      polyline(g, [P[prev], P[a]], { stroke: 'var(--warm)', 'stroke-width': 4, 'stroke-dasharray': '12 10' });
      polyline(g, P.slice(a, b + 1), { stroke: 'var(--warm)', 'stroke-width': 7, 'stroke-linecap': 'round' });
      prev = b;
    });
    polyline(g, [P[prev], P[n]], { stroke: 'var(--warm)', 'stroke-width': 4, 'stroke-dasharray': '12 10' });
    [0, n].forEach(i => el('circle', { cx: P[i][0], cy: P[i][1], r: 9, fill: 'var(--warm)' }, g));
    text(g, 40, 84, 'only ~50 points: 10⁴–10⁵ times faster', { 'font-size': 34, class: 'warm' });
    const [a, b] = runs.reduce((r, s) => s[1] - s[0] > r[1] - r[0] ? s : r), m = P[(a + b) >> 1];
    text(g, m[0] + 24, m[1] + 8, 'stationary region', { 'font-size': 30, class: 'warm' });
  };

  // Log time axis: what a prediction costs vs what applications can afford.
  FIGURES['timescale'] = svg => {
    const L = e => 100 + (e + 6) / 10 * 1500, y = 380;   // 1 µs … 10⁴ s
    el('line', { x1: L(-6), x2: L(4), y1: y, y2: y, class: 'axis' }, svg);
    [[-6, '1 µs'], [-3, '1 ms'], [0, '1 s'], [3.56, '1 hour']].forEach(([e, s]) => {
      el('line', { x1: L(e), x2: L(e), y1: y - 12, y2: y + 12, class: 'axis' }, svg);
      text(svg, L(e), y + 60, s, { 'font-size': 32, 'text-anchor': 'middle', class: 'dim' });
    });
    [[3.9, 'CFD run', 'hours', ''], [0.3, 'brute force', '~2 s per point', ''],
     [-2, 'autopilot budget', 'milliseconds', 'cool'], [-4.5, 'our method', '30 µs', 'warm']].forEach(([e, name, v, cls], i) => {
      const g = i === 3 ? frag(svg) : svg, up = i % 2 ? -1 : 1;
      el('circle', { cx: L(e), cy: y, r: 14, fill: cls ? `var(--${cls})` : 'var(--ink)' }, g);
      text(g, L(e), y - up * 150, name, { 'font-size': 40, 'text-anchor': 'middle', class: cls });
      text(g, L(e), y - up * 150 + 46, v, { 'font-size': 32, 'text-anchor': 'middle', class: 'dim' });
    });
  };

  // From "lecture" to "magic trick": the needle starts at lecture, swings to magic on the first
  // click, and settles on "mechanism" on the second.
  FIGURES['dial'] = (svg, w, h) => {
    const cx = w / 2, cy = h - 40, r = h - 100;
    el('path', { d: `M${cx - r},${cy} A${r},${r} 0 0 1 ${cx + r},${cy}`, class: 'axis', 'stroke-width': 6 }, svg);
    text(svg, cx - r, cy + 50, 'lecture', { 'font-size': 40, 'text-anchor': 'middle', class: 'dim' });
    text(svg, cx + r, cy + 50, 'magic trick', { 'font-size': 40, 'text-anchor': 'middle', class: 'dim' });
    const needle = el('line', { x1: cx, y1: cy, x2: cx, y2: cy - r + 20, stroke: 'var(--warm)', 'stroke-width': 8, 'stroke-linecap': 'round',
      style: `transform-origin: ${cx}px ${cy}px; transition: transform 1.2s cubic-bezier(.4,1.6,.5,1)` }, svg);
    const label = text(svg, cx, cy - r - 10, 'mechanism', { 'font-size': 44, 'text-anchor': 'middle', class: 'warm', style: 'transition: opacity 0.6s' });
    steps(svg, 2, k => {
      needle.style.transform = `rotate(${[-80, 80, 0][k]}deg)`;
      label.style.opacity = k === 2 ? 1 : 0;
    });
  };

  // Slide 12's wave cut (point source just below the surface, x = -8) copied across a width
  // b = 1 with elliptic weights, as for a finite wing. Their weighted average (orange) tends
  // to the smooth width-integrated cut, but the spikes only cancel slowly as points are added.
  // Stops once the leftover is under 1%, which is visually smooth.
  FIGURES['cancel'] = (svg, w, h) => {
    const x = -8, z = -0.01, b = 1, Y = 4.8, T = 40, nt = 40000, ny = 600, nd = 300;
    const ys = Array.from({ length: nd + 1 }, (_, i) => -Y + 2 * Y * i / nd);
    const eta = N => Array.from({ length: N }, (_, j) => -1 + (2 * j + 1) / N);
    const X = y => 40 + (y + Y) / (2 * Y) * (w - 80), V = v => 170 - 2.4 * Math.max(-50, Math.min(50, v));
    let at, average, seq = [];   // built on first showing: the cut takes a moment to integrate
    function build() {
      const dt = 2 * T / nt, grid = Array.from({ length: ny + 1 }, (_, i) => -Y - b + 2 * (Y + b) * i / ny);
      const cut = grid.map(y => {   // ∂ₓW = 4∫exp(z(1+t²)) cos(g) √(1+t²) dt
        let s = 0;
        for (let k = 0; k <= nt; k++) { const t = -T + k * dt, r = Math.hypot(1, t); s += Math.exp(z * (1 + t * t)) * Math.cos((x + y * t) * r) * r; }
        return 4 * s * dt;
      });
      at = y => { const f = (y + Y + b) / (2 * (Y + b)) * ny, i = Math.min(ny - 1, Math.floor(f)), u = f - i; return cut[i] * (1 - u) + cut[i + 1] * u; };
      average = N => {
        const E = eta(N), wt = E.map(e => Math.sqrt(1 - e * e)), W = wt.reduce((a, c) => a + c);
        return ys.map(y => E.reduce((s, e, j) => s + wt[j] * at(y - b * e), 0) / W);
      };
      const exact = average(2000), scale = Math.max(...average(1).map(Math.abs));
      for (let m = 0, N = 1; ; m++, N = Math.max(N + 1, Math.round(1.25 ** m))) {
        const A = average(N), left = Math.max(...A.map((v, i) => Math.abs(v - exact[i]))) / scale;
        seq.push({ N, A, left });
        if (left < 0.01) break;
      }
    }
    const Nmax = () => seq[seq.length - 1].N;
    const LX = N => 100 + Math.log(N) / Math.log(Nmax()) * (w - 160), LY = A => 380 + Math.log10(1 / Math.max(A, 0.01)) / 2 * 200;
    const axes = el('g', {}, svg), waves = el('g', {}, svg), dots = el('g', {}, svg), label = text(svg, 40, 30, '', { 'font-size': 34 });
    function drawAxes() {
      el('line', { x1: LX(1), x2: LX(Nmax()), y1: LY(0.01), y2: LY(0.01), class: 'axis' }, axes);
      el('line', { x1: LX(1), x2: LX(1), y1: LY(1), y2: LY(0.01), class: 'axis' }, axes);
      [1, Nmax()].forEach(N => text(axes, LX(N), LY(0.01) + 36, N, { 'font-size': 26, 'text-anchor': 'middle', class: 'dim' }));
      text(axes, (LX(1) + LX(Nmax())) / 2, LY(0.01) + 72, 'points across the width', { 'font-size': 28, 'text-anchor': 'middle', class: 'dim' });
      [[1, '100%'], [0.01, '1%']].forEach(([A, s]) => text(axes, LX(1) - 14, LY(A) + 8, s, { 'font-size': 26, 'text-anchor': 'end', class: 'dim' }));
      text(axes, LX(1) + 16, LY(1) - 14, 'leftover spikes', { 'font-size': 28, class: 'dim' });
    }
    let k = 0, clock = 0;
    function show() {
      const { N, A, left } = seq[k];
      waves.replaceChildren();
      eta(N).filter((_, j) => j % Math.ceil(N / 10) === 0).forEach(e =>
        polyline(waves, ys.map(y => [X(y), V(at(y - b * e))]), { stroke: 'var(--cool)', 'stroke-width': 1.5, opacity: 0.35 }));
      polyline(waves, ys.map((y, i) => [X(y), V(A[i])]), { stroke: 'var(--warm)', 'stroke-width': 4 });
      el('circle', { cx: LX(N), cy: LY(left), r: 7, fill: 'var(--warm)' }, dots);
      label.textContent = `${N} point${N > 1 ? 's' : ''} across the width: ${Math.round(100 * left)}% left`;
    }
    const run = animate(dt => {
      clock += dt;
      if (clock < (k === seq.length - 1 ? 4 : 0.7)) return;
      clock = 0; k = (k + 1) % seq.length;
      if (!k) dots.replaceChildren();
      show();
    });
    return {
      start() { if (!at) { build(); drawAxes(); show(); } run.start(); },
      stop: run.stop,
    };
  };

  // <ol class="titles">: the slide titles marked data-story, so it stays in step with edits
  document.querySelectorAll('.titles').forEach(ol => document.querySelectorAll('[data-story]').forEach(h => {
    const c = h.cloneNode(true);
    c.querySelectorAll('.source, .opt').forEach(n => n.remove());
    ol.append(Object.assign(document.createElement('li'), { innerHTML: c.innerHTML.trim() }));
  }));

  const controls = [];
  document.querySelectorAll('.figure[data-figure]').forEach(div => {
    const draw = FIGURES[div.dataset.figure];
    if (!draw) { div.textContent = `unknown figure: ${div.dataset.figure}`; return; }
    const w = +div.dataset.w || W, h = +div.dataset.h || H;
    const svg = el('svg', { viewBox: `0 0 ${w} ${h}`, 'aria-hidden': 'true' }, div);
    const ctl = draw(svg, w, h, div);
    if (div.closest('.steps')) svg.querySelectorAll('.fragment').forEach(f => f.classList.remove('fragment'));   // thumbnails show every step
    if (ctl) controls.push([div, ctl]);
  });

  // Figures with script-driven animation run only while their slide is showing.
  function sync() {
    const cur = Reveal.getCurrentSlide();
    controls.forEach(([div, ctl]) => (cur && cur.contains(div) ? ctl.start() : ctl.stop()));
  }
  Reveal.on('ready', sync);
  Reveal.on('slidechanged', sync);
})();
