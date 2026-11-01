// Live wave-field demo (slide 17). Fields come from talk/server.jl on
// localhost:8081; if it isn't running, baked frames in assets/demo/ are used.
//   fast:  /field?F=&beta=&model=&z=   → whole frame, with its compute time
//   brute: /brute?F=&beta=&k0=&n=      → the next n pixels by direct quadrature

(function () {
  const root = document.getElementById('demo');
  if (!root) return;
  const SERVER = 'http://localhost:8081', ZMAX = 10;
  const [fastCv, bruteCv] = ['fast', 'brute'].map(m => root.querySelector(`canvas[data-method=${m}]`));
  const clock = m => root.querySelector(`[data-clock=${m}]`);
  const status = root.querySelector('[data-status]');
  const param = p => root.querySelector(`[data-param=${p}]`).value;
  const query = () => {
    const s = +param('depth');
    return `F=${(+param('speed')).toFixed(2)}&beta=${(+param('width')).toFixed(2)}` +
           `&model=${param('model')}&z=${s ? (-0.5 * 10 ** (-2.7 * s)).toPrecision(3) : -0}`;
  };

  // Diverging blue–white–red, like the paper's figures
  const STOPS = [[22, 52, 120], [96, 150, 200], [241, 237, 234], [205, 110, 90], [120, 20, 30]];
  function colour(v) {
    const s = Math.max(0, Math.min(1, (v / ZMAX + 1) / 2)) * (STOPS.length - 1), i = Math.min(STOPS.length - 2, Math.floor(s)), f = s - i;
    return STOPS[i].map((c, j) => c + f * (STOPS[i + 1][j] - c));
  }
  function paint(cv, nx, ny, z, k0 = 0) {   // z is column-major from the bow, mirrored about y = 0
    const ctx = cv.getContext('2d');
    if (cv.width !== nx || cv.height !== ny) { cv.width = nx; cv.height = ny; }
    const img = z.length < nx * ny ? ctx.getImageData(0, 0, nx, ny) : ctx.createImageData(nx, ny);
    z.forEach((v, n) => {
      const k = k0 + n, i = nx - 1 - Math.floor(k / ny), j = k % ny, [r, g, b] = colour(v);
      const p = 4 * (j * nx + i);
      img.data.set([r, g, b, 255], p);
    });
    ctx.putImageData(img, 0, 0);
  }

  let live = false, version = 0;
  async function fast() {
    const v = ++version;
    try {
      const r = await (await fetch(`${SERVER}/field?${query()}`)).json();
      if (v !== version) return;
      paint(fastCv, r.nx, r.ny, r.z);
      clock('fast').textContent = `computed in ${r.time.toFixed(2)} s`;
      status.textContent = '';
      brute(v, r.nx, r.ny, r.kbow);
    } catch {
      baked(v);
    }
  }
  async function baked(v) {
    status.textContent = 'server offline: saved frames';
    clock('brute').textContent = '';
    const q = query().replace(/&model.*/, '').replace(/[=&]/g, '_');
    try {
      const r = await (await fetch(`assets/demo/${q}.json`)).json();
      if (v === version) { paint(fastCv, r.nx, r.ny, r.z); clock('fast').textContent = `computed in ${r.time.toFixed(2)} s`; }
    } catch { clock('fast').textContent = 'no saved frame'; }
  }
  // Brute force paints from the bow while the slide is showing; any slider move restarts it.
  async function brute(v, nx, ny, kbow) {
    const t0 = performance.now(), total = nx * ny - kbow;
    const ctx = bruteCv.getContext('2d');
    bruteCv.width = nx; bruteCv.height = ny; ctx.fillStyle = '#888'; ctx.fillRect(0, 0, nx, ny);
    paint(bruteCv, nx, ny, new Array(kbow).fill(0), 0);
    let done = 0, eta = '';
    const tick = setInterval(() => {
      if (v !== version) return clearInterval(tick);
      clock('brute').textContent = `${done} of ${total} pixels in ${((performance.now() - t0) / 1000).toFixed(0)} s${eta}`;
    }, 250);
    // one pixel at a time on one thread, so the fast frames keep the rest of the machine
    for (let k0 = kbow; k0 < nx * ny && live && v === version; k0++) {
      const r = await (await fetch(`${SERVER}/brute?${query()}&k0=${k0}&n=1`)).json();
      if (v !== version) return;
      paint(bruteCv, nx, ny, r.z, k0);
      done++;
      eta = ` · done in ~${((performance.now() - t0) / 1000 / done * total / 3600).toFixed(0)} hours`;
    }
    clearInterval(tick);
  }

  root.querySelectorAll('[data-param]').forEach(i => i.addEventListener('input', fast));
  const show = e => {
    live = e.currentSlide.contains(root);
    if (live) fast(); else version++;
  };
  Reveal.on('ready', show);
  Reveal.on('slidechanged', show);
})();
