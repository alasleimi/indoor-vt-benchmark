// Loaded only when the 3D tab is opened. Coordinates retain their saved float64 precision.
export function mountVTMap(
  container,
  { job, meta, filters, request, roomLabel },
) {
  container.innerHTML = `<div class="map-intro">One saved survey-time source map. Rotate to inspect height; VTs can lie outside the room, below the floor, or above the ceiling.</div>
    <div class="map-selectors"><label>Room<select data-map="room"></select></label><label>Noise level<select data-map="noise"></select></label><label>APs<select data-map="ap"><option>A</option><option>B</option><option>A+B</option></select></label></div>
    <div class="map-metrics" aria-live="polite"></div>
    <div class="map-toolbar"><button data-camera="all">Fit all</button><button data-camera="room">Room view</button><button data-camera="3d">3D</button><button data-camera="top">Top</button><button data-camera="side">Side</button><button data-camera="in" aria-label="Zoom in">+</button><button data-camera="out" aria-label="Zoom out">−</button></div>
    <div class="map-layers"><label><input type="checkbox" data-layer="predicted" checked> <span class="map-dot predicted"></span>Predicted</label><label><input type="checkbox" data-layer="truth" checked> <span class="map-dot truth"></span>Ground truth</label><label><input type="checkbox" data-layer="survey" checked>Survey</label><label><input type="checkbox" data-layer="links" checked>Match links</label></div>
    <div class="map-plot"><canvas tabindex="0" aria-label="Interactive three-dimensional VT map. Drag or use arrow keys to rotate; use plus and minus to zoom."></canvas><div class="map-tooltip" hidden></div><div class="map-loading" role="status">Loading saved coordinates…</div></div>
    <div class="map-status" aria-live="polite"></div><p class="stat-caption">Drag to rotate · Shift-drag to pan · Scroll or +/− to zoom · Hover or tap a source for its coordinates. Dark diamonds mark physical APs. Links use the recorded one-to-one F1 matches within 0.5 m.</p>
    <div class="map-point" aria-live="polite">Select a source to inspect its coordinates and reflection labels.</div>`;
  const $ = (selector) => container.querySelector(selector);
  const roomSelect = $('[data-map="room"]'),
    noiseSelect = $('[data-map="noise"]'),
    apSelect = $('[data-map="ap"]');
  const canvas = $("canvas"),
    ctx = canvas.getContext("2d"),
    tooltip = $(".map-tooltip");
  const rooms = meta.rooms.filter(
    (r) =>
      (!filters.family || r.family === filters.family) &&
      (!filters.room || r.id === filters.room),
  );
  for (const room of rooms)
    roomSelect.add(new Option(roomLabel(room.id), room.id));
  for (const noise of meta.levels[job.track])
    noiseSelect.add(
      new Option(
        job.track === "receiver" ? `${noise} dB` : `σ ${noise} m`,
        String(noise),
      ),
    );
  noiseSelect.value =
    filters.noise ||
    String(
      job.track === "receiver"
        ? Math.max(...meta.levels[job.track])
        : Math.min(...meta.levels[job.track]),
    );
  apSelect.value = filters.ap || "A";
  let data,
    controller,
    disposed = false,
    frame = 0,
    width = 0,
    height = 0;
  let yaw = -0.6,
    pitch = 0.65,
    scale = 1,
    center = [0, 0, 0],
    pan = [0, 0],
    fitMode = "all";
  let predictedScreen = new Float32Array(),
    truthScreen = [],
    matchedPredicted = new Set(),
    matchedTruth = new Set();
  const layers = { predicted: true, truth: true, survey: true, links: true },
    pointers = new Map();
  let drag = null,
    pinch = null;
  const fmt = (value) =>
    Number(value).toLocaleString("en-US", { maximumFractionDigits: 4 });
  const count = (value) => value.toLocaleString("en-US");
  function project(x, y, z) {
    x -= center[0];
    y -= center[1];
    z -= center[2];
    const xx = Math.cos(yaw) * x - Math.sin(yaw) * y;
    const yy = Math.sin(yaw) * x + Math.cos(yaw) * y;
    return [
      width / 2 + pan[0] + scale * xx,
      height / 2 +
        pan[1] -
        scale * (Math.cos(pitch) * z - Math.sin(pitch) * yy),
    ];
  }
  function fit(mode = fitMode) {
    if (!data || !width) return;
    fitMode = mode;
    const min = [Infinity, Infinity, Infinity],
      max = [-Infinity, -Infinity, -Infinity];
    const add = (x, y, z) => {
      for (const [i, v] of [x, y, z].entries()) {
        min[i] = Math.min(min[i], v);
        max[i] = Math.max(max[i], v);
      }
    };
    for (const [x, y] of data.scene.polygon_xy) {
      add(x, y, 0);
      add(x, y, data.scene.ceiling_m);
    }
    if (mode === "all") {
      for (const source of data.truth) add(...source.xyz);
      for (let i = 0; i < data.predicted.length; i += 3)
        add(data.predicted[i], data.predicted[i + 1], data.predicted[i + 2]);
    }
    center = min.map((v, i) => (v + max[i]) / 2);
    scale = 1;
    pan = [0, 0];
    const bounds = [Infinity, Infinity, -Infinity, -Infinity];
    const screenBounds = (x, y, z) => {
      const p = project(x, y, z);
      bounds[0] = Math.min(bounds[0], p[0] - width / 2);
      bounds[1] = Math.min(bounds[1], p[1] - height / 2);
      bounds[2] = Math.max(bounds[2], p[0] - width / 2);
      bounds[3] = Math.max(bounds[3], p[1] - height / 2);
    };
    for (const [x, y] of data.scene.polygon_xy) {
      screenBounds(x, y, 0);
      screenBounds(x, y, data.scene.ceiling_m);
    }
    if (mode === "all") {
      for (const source of data.truth) screenBounds(...source.xyz);
      for (let i = 0; i < data.predicted.length; i += 3)
        screenBounds(
          data.predicted[i],
          data.predicted[i + 1],
          data.predicted[i + 2],
        );
    }
    scale = Math.min(
      (width - 70) / Math.max(bounds[2] - bounds[0], 1),
      (height - 70) / Math.max(bounds[3] - bounds[1], 1),
    );
    pan = [
      (-scale * (bounds[0] + bounds[2])) / 2,
      (-scale * (bounds[1] + bounds[3])) / 2,
    ];
    schedule();
  }
  function schedule() {
    if (!disposed && !frame) frame = requestAnimationFrame(draw);
  }
  function line(a, b, color, lineWidth = 1) {
    ctx.beginPath();
    ctx.moveTo(...a);
    ctx.lineTo(...b);
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
  }
  function draw() {
    frame = 0;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#f8fafb";
    ctx.fillRect(0, 0, width, height);
    if (!data) return;
    const polygon = data.scene.polygon_xy;
    const floor = polygon.map(([x, y]) => project(x, y, 0)),
      ceiling = polygon.map(([x, y]) => project(x, y, data.scene.ceiling_m));
    ctx.beginPath();
    floor.forEach((p, i) => (i ? ctx.lineTo(...p) : ctx.moveTo(...p)));
    ctx.closePath();
    ctx.fillStyle = "#d9e8e655";
    ctx.fill();
    polygon.forEach((_, i) => {
      line(floor[i], floor[(i + 1) % floor.length], "#6c858b");
      line(ceiling[i], ceiling[(i + 1) % ceiling.length], "#b2c3c7");
      line(floor[i], ceiling[i], "#b2c3c7");
    });
    if (layers.survey) {
      ctx.fillStyle = "#94a7b077";
      for (const xyz of data.scene.survey_xyz) {
        const p = project(...xyz);
        ctx.fillRect(p[0] - 1, p[1] - 1, 2, 2);
      }
    }
    let visible = 0;
    ctx.fillStyle = "#d27a26";
    for (let i = 0; i < data.predicted.length / 3; i++) {
      const p = project(
        data.predicted[3 * i],
        data.predicted[3 * i + 1],
        data.predicted[3 * i + 2],
      );
      predictedScreen[2 * i] = p[0];
      predictedScreen[2 * i + 1] = p[1];
      if (p[0] >= 0 && p[0] <= width && p[1] >= 0 && p[1] <= height) {
        visible++;
        if (layers.predicted) ctx.fillRect(p[0] - 2, p[1] - 2, 4, 4);
      }
    }
    truthScreen = data.truth.map((source) => project(...source.xyz));
    if (layers.links && layers.truth && layers.predicted)
      for (const [a, b] of data.matches)
        line(
          [predictedScreen[2 * a], predictedScreen[2 * a + 1]],
          truthScreen[b],
          "#765ca988",
          1.2,
        );
    if (layers.truth)
      for (const [x, y] of truthScreen) {
        ctx.beginPath();
        ctx.arc(x, y, 4.5, 0, 2 * Math.PI);
        ctx.strokeStyle = "#007f79";
        ctx.lineWidth = 1.7;
        ctx.stroke();
      }
    for (const ap of data.ap.split("+")) {
      const p = project(...data.scene.access_points_xyz[ap]);
      ctx.fillStyle = "#243a43";
      ctx.beginPath();
      ctx.moveTo(p[0], p[1] - 6);
      ctx.lineTo(p[0] + 6, p[1]);
      ctx.lineTo(p[0], p[1] + 6);
      ctx.lineTo(p[0] - 6, p[1]);
      ctx.closePath();
      ctx.fill();
      ctx.font = "bold 11px system-ui";
      ctx.fillText(`AP ${ap}`, p[0] + 9, p[1] - 7);
    }
    // Orientation compass follows the same 3D camera, independently of the data extent.
    const origin = project(...center),
      base = [36, height - 37];
    for (const [i, label, color] of [
      [0, "x", "#b45151"],
      [1, "y", "#39835e"],
      [2, "z", "#477bb2"],
    ]) {
      const v = [...center];
      v[i] += 26 / scale;
      const p = project(...v);
      const end = [base[0] + p[0] - origin[0], base[1] + p[1] - origin[1]];
      line(base, end, color, 1.6);
      ctx.fillStyle = color;
      ctx.font = "11px system-ui";
      ctx.fillText(label, end[0] + 3, end[1] - 3);
    }
    const hidden = data.predicted.length / 3 - visible;
    $(".map-status").textContent =
      `${count(visible)} of ${count(data.predicted.length / 3)} predictions in view${hidden ? " · Use Fit all to include off-screen predictions" : ""} · Coordinates in meters`;
  }
  function sourceInfo(kind, index) {
    if (kind === "truth") {
      const s = data.truth[index],
        paths = s.surface_sequences
          .map((seq) => (seq.length ? seq.join(" → ") : "direct / physical AP"))
          .join("; ");
      return `Ground truth · ${s.id} · ${matchedTruth.has(index) ? "matched" : "missed"}\nx ${fmt(s.xyz[0])} · y ${fmt(s.xyz[1])} · z ${fmt(s.xyz[2])} m\n${paths}`;
    }
    const channel = data.channels.find(
      (c) => index >= c.start && index < c.start + c.count,
    );
    return `Prediction ${index + 1} · AP ${channel.ap} · ${matchedPredicted.has(index) ? "matched" : "unmatched"}\nx ${fmt(data.predicted[3 * index])} · y ${fmt(data.predicted[3 * index + 1])} · z ${fmt(data.predicted[3 * index + 2])} m`;
  }
  function pick(x, y) {
    let best = null,
      distance = 81;
    if (layers.predicted)
      for (let i = 0; i < predictedScreen.length / 2; i++) {
        const d =
          (x - predictedScreen[2 * i]) ** 2 +
          (y - predictedScreen[2 * i + 1]) ** 2;
        if (d < distance) {
          distance = d;
          best = ["predicted", i];
        }
      }
    if (layers.truth)
      truthScreen.forEach((p, i) => {
        const d = (x - p[0]) ** 2 + (y - p[1]) ** 2;
        if (d < distance) {
          distance = d;
          best = ["truth", i];
        }
      });
    return best;
  }
  function hover(event, pin = false) {
    if (!data) return;
    const rect = canvas.getBoundingClientRect(),
      x = event.clientX - rect.left,
      y = event.clientY - rect.top,
      hit = pick(x, y);
    tooltip.hidden = !hit;
    if (hit) {
      tooltip.textContent = sourceInfo(...hit);
      tooltip.style.left = `${Math.max(8, Math.min(x + 14, width - 260))}px`;
      tooltip.style.top = `${Math.max(8, Math.min(y + 12, height - 90))}px`;
      if (pin) $(".map-point").textContent = tooltip.textContent;
    }
  }
  canvas.addEventListener("pointerdown", (e) => {
    canvas.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    drag = {
      x: e.clientX,
      y: e.clientY,
      moved: false,
      pan: e.shiftKey || e.button === 2,
    };
    tooltip.hidden = true;
  });
  canvas.addEventListener("pointermove", (e) => {
    if (!pointers.has(e.pointerId)) {
      hover(e);
      return;
    }
    pointers.set(e.pointerId, [e.clientX, e.clientY]);
    if (pointers.size === 2) {
      const [a, b] = [...pointers.values()],
        d = Math.hypot(a[0] - b[0], a[1] - b[1]);
      if (pinch && d > 0) scale *= d / pinch;
      pinch = d;
      drag.moved = true;
      schedule();
      return;
    }
    const dx = e.clientX - drag.x,
      dy = e.clientY - drag.y;
    if (Math.abs(dx) + Math.abs(dy) > 1) drag.moved = true;
    if (drag.pan) {
      pan[0] += dx;
      pan[1] += dy;
    } else {
      yaw += dx * 0.009;
      pitch = Math.max(-Math.PI / 2, Math.min(Math.PI / 2, pitch + dy * 0.008));
    }
    drag.x = e.clientX;
    drag.y = e.clientY;
    schedule();
  });
  const end = (e) => {
    if (drag && !drag.moved) hover(e, true);
    pointers.delete(e.pointerId);
    pinch = null;
    if (!pointers.size) drag = null;
  };
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);
  canvas.addEventListener("pointerleave", () => {
    tooltip.hidden = true;
  });
  canvas.addEventListener("contextmenu", (e) => e.preventDefault());
  canvas.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      scale *= Math.exp(Math.max(-1, Math.min(1, -e.deltaY * 0.001)));
      schedule();
    },
    { passive: false },
  );
  canvas.addEventListener("keydown", (e) => {
    if (
      ![
        "ArrowLeft",
        "ArrowRight",
        "ArrowUp",
        "ArrowDown",
        "+",
        "=",
        "-",
        "Home",
      ].includes(e.key)
    )
      return;
    e.preventDefault();
    if (e.key === "Home") fit("all");
    else if (e.key === "+" || e.key === "=") scale *= 1.25;
    else if (e.key === "-") scale /= 1.25;
    else if (e.key === "ArrowLeft") yaw -= 0.12;
    else if (e.key === "ArrowRight") yaw += 0.12;
    else
      pitch = Math.max(
        -Math.PI / 2,
        Math.min(Math.PI / 2, pitch + (e.key === "ArrowUp" ? 0.12 : -0.12)),
      );
    schedule();
  });
  container.querySelectorAll("[data-camera]").forEach(
    (button) =>
      (button.onclick = () => {
        const value = button.dataset.camera;
        if (value === "all" || value === "room") fit(value);
        else if (value === "in") scale *= 1.4;
        else if (value === "out") scale /= 1.4;
        else if (value === "top") {
          yaw = 0;
          pitch = Math.PI / 2;
        } else if (value === "side") {
          yaw = 0;
          pitch = 0;
        } else {
          yaw = -0.6;
          pitch = 0.65;
        }
        if (["top", "side", "3d"].includes(value)) fit();
        schedule();
      }),
  );
  container.querySelectorAll("[data-layer]").forEach(
    (input) =>
      (input.onchange = () => {
        layers[input.dataset.layer] = input.checked;
        schedule();
      }),
  );
  const resize = new ResizeObserver(() => {
    const rect = canvas.parentElement.getBoundingClientRect(),
      oldWidth = width;
    width = rect.width;
    height = rect.height;
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    if (!oldWidth) fit();
    schedule();
  });
  resize.observe(canvas.parentElement);
  async function load() {
    controller?.abort();
    controller = new AbortController();
    const signal = controller.signal;
    data = null;
    predictedScreen = new Float32Array();
    truthScreen = [];
    tooltip.hidden = true;
    $(".map-loading").hidden = false;
    $(".map-loading").textContent = "Loading saved coordinates…";
    $(".map-metrics").textContent = "";
    $(".map-status").textContent = "";
    schedule();
    try {
      const result = await request(
        "map",
        {
          id: job.id,
          room: roomSelect.value,
          noise: noiseSelect.value,
          ap: apSelect.value,
        },
        signal,
      );
      if (signal.aborted || disposed) return;
      data = result;
      predictedScreen = new Float32Array((data.predicted.length / 3) * 2);
      matchedPredicted = new Set(data.matches.map((m) => m[0]));
      matchedTruth = new Set(data.matches.map((m) => m[1]));
      $(".map-metrics").textContent =
        `${count(data.predicted.length / 3)} predicted · ${count(data.truth.length)} true · OSPA ${fmt(data.metrics.ospa)} m · F1 ${fmt(data.metrics.f1 * 100)}% · ${count(data.matches.length)} matches`;
      $(".map-point").textContent =
        "Select a source to inspect its coordinates and reflection labels.";
      $(".map-loading").hidden = true;
      fit("all");
    } catch (error) {
      if (!signal.aborted && !disposed)
        $(".map-loading").textContent = error.message;
    }
  }
  for (const select of [roomSelect, noiseSelect, apSelect])
    select.onchange = load;
  load();
  return () => {
    disposed = true;
    controller?.abort();
    resize.disconnect();
    if (frame) cancelAnimationFrame(frame);
    data = null;
    predictedScreen = new Float32Array();
    truthScreen = [];
    pointers.clear();
  };
}
