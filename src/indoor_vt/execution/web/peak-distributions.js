const colors = ['#2563a6', '#c94e34'];
const names = ['Correct pairs · same VT', 'Incorrect pairs · different VTs'];
const fmt = (v, digits = 3) => Number(v).toLocaleString('en-US', { maximumFractionDigits: digits });
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);

export function histogramRows(hist) {
  return hist.bins.map(([low, correct, incorrect]) => ({ low, high: low + hist.bin_width,
    counts: [correct, incorrect],
    density: [correct, incorrect].map((n, i) => hist.totals[i] ? n / (hist.totals[i] * hist.bin_width) : null),
  }));
}

export function mountDistributions(container, { request, meta }) {
  container.innerHTML = `<div class="match-section-heading"><div><h2>MPUrge dissimilarity distributions</h2>
    <p data-distribution="condition" aria-live="polite">Loading candidate peak pairs…</p></div>
    <button class="button subtle" data-distribution="export" disabled>Export distributions</button></div>
    <p>Every finite <strong>first-round candidate dissimilarity</strong> is included, whether or not MPUrge accepts the pair. Correct means the two peaks share the exact same true VT, including the direct-path AP. The selected AP, noise and frozen parameters above apply to all three plots.</p>
    <p class="spatial-caption">These are empirical histograms, not fitted bell curves. Each class has area 1, so the much larger number of incorrect pairs does not hide the correct-pair distribution. Blue and red can overlap; their sample counts are shown below each plot. The two raw plots share both axis scales.</p>
    <div class="distribution-grid">${[
      ['neighborhood', 'Within MPUrge neighborhoods', 'Each eligible unordered fingerprint pair is counted once.'],
      ['all', 'All fingerprint pairs', 'Every unordered pair in the empty-room survey, including pairs outside MPUrge neighborhoods.'],
      ['normalized', 'Normalized within each fingerprint’s neighborhood', 'Each local candidate pair contributes once from each endpoint’s perspective.'],
    ].map(([key, title, note]) => `<article class="distribution-panel" data-population="${key}"><h3>${title}</h3><p>${note}</p><div data-chart="${key}" class="distribution-chart">Loading…</div><div data-counts="${key}" class="distribution-counts"></div><p data-hover="${key}" class="spatial-caption" aria-live="polite">Hover or focus a histogram interval for both classes.</p></article>`).join('')}</div>
    <details class="matching-explanation"><summary>Exactly how is the third plot normalized?</summary>
      <p>Let <var>N(i)</var> be the other fingerprints that share at least one MPUrge local group with fingerprint <var>i</var>. Let <var>δ(i,a;j,b)</var> be the first-round dissimilarity between peak <var>a</var> of fingerprint <var>i</var> and peak <var>b</var> of fingerprint <var>j</var>.</p>
      <p class="distribution-formula"><var>mᵢ</var> = median { δ(i,a;j,b) : j ∈ N(i), all retained peak pairs (a,b) with finite δ }<br><var>z(i,a;j,b)</var> = δ(i,a;j,b) / mᵢ</p>
      <p>One median is computed per whole fingerprint across its neighborhood comparisons, using both correct and incorrect candidates together. Truth labels do not enter the normalization. A local pair therefore contributes δ/mᵢ and δ/mⱼ. This is a diagnostic transformation; the matcher still uses its original dissimilarity. Anchors with a zero or undefined median cannot be normalized and are reported as excluded.</p>
      <p data-distribution="exclusions"></p>
    </details>`;
  const $ = selector => container.querySelector(selector), cache = new Map();
  let data = null, settings = null, revision = 0;

  function binText(row, hist) {
    return `${fmt(row.low)} ≤ dissimilarity < ${fmt(row.high)} · ` + names.map((name, i) =>
      `${name}: ${fmt(row.counts[i], 0)} pairs (${hist.totals[i] ? fmt(100 * row.counts[i] / hist.totals[i], 2) + '% of this class' : 'class absent; density undefined'})`).join(' · ');
  }
  function render() {
    const rows = Object.fromEntries(['all', 'neighborhood', 'normalized'].map(k => [k, histogramRows(data[k])]));
    const rawMax = Math.max(.25, ...rows.all.map(r => r.high), ...rows.neighborhood.map(r => r.high));
    const rawY = Math.max(.01, ...[...rows.all, ...rows.neighborhood].flatMap(r => r.density.filter(v => v != null)));
    for (const key of ['neighborhood', 'all', 'normalized']) {
      const hist = data[key], normalized = key === 'normalized', records = rows[key];
      const xmax = normalized ? Math.max(.05, ...records.map(r => r.high)) : rawMax;
      const ymax = (normalized ? Math.max(.01, ...records.flatMap(r => r.density.filter(v => v != null))) : rawY) * 1.08;
      const x = v => 62 + 640 * v / xmax, y = v => 263 - 222 * v / ymax;
      const byId = new Map(records.map(r => [Math.round(r.low / hist.bin_width), r]));
      const count = Math.round(xmax / hist.bin_width), paths = [];
      for (let c = 0; c < 2; c++) {
        if (!hist.totals[c]) continue;
        let d = `M${x(0)},${y(0)}`;
        for (let i = 0; i < count; i++) {
          const value = byId.get(i)?.density[c] || 0;
          d += `L${x(i * hist.bin_width)},${y(value)}H${x((i + 1) * hist.bin_width)}`;
        }
        paths.push(`<path d="${d}L${x(xmax)},${y(0)}Z" fill="${colors[c]}" fill-opacity=".17" stroke="${colors[c]}" stroke-width="1.8"/>`);
      }
      $(`[data-chart="${key}"]`).innerHTML = `<svg viewBox="0 0 740 330" role="img" aria-label="${esc(key)} dissimilarity distributions: correct versus incorrect candidate peak pairs">
        ${[0, .25, .5, .75, 1].map(f => `<path d="M62 ${y(f * ymax)}H702" stroke="#e2e8eb"/><text x="53" y="${y(f * ymax) + 4}" text-anchor="end">${fmt(f * ymax)}</text>`).join('')}
        <text x="62" y="20">${normalized ? 'Density (unitless dissimilarity)' : 'Density (1/m)'}</text>
        ${paths.join('')}
        ${records.map(r => `<rect x="${x(r.low)}" y="41" width="${Math.max(.2, x(r.high) - x(r.low))}" height="222" fill="transparent" tabindex="0" data-pop="${key}" data-low="${r.low}" aria-label="${esc(binText(r, hist))}"><title>${esc(binText(r, hist))}</title></rect>`).join('')}
        ${[0, .2, .4, .6, .8, 1].map(f => `<text x="${x(f * xmax)}" y="285" text-anchor="middle">${fmt(f * xmax, 2)}</text>`).join('')}
        <text x="382" y="313" text-anchor="middle">${normalized ? 'δ / fingerprint neighborhood median (unitless)' : 'First-round dissimilarity δ (m)'}</text></svg>`;
      $(`[data-counts="${key}"]`).innerHTML = names.map((name, i) => `<span><i style="background:${colors[i]}"></i>${name}: <strong>${fmt(hist.totals[i], 0)}</strong>${hist.totals[i] ? '' : ' (no curve)'}</span>`).join('');
      const total = hist.totals[0] + hist.totals[1];
      $(`[data-hover="${key}"]`).textContent = `${fmt(hist.pairs, 0)} ${normalized ? 'directed fingerprint comparisons' : 'fingerprint pairs'} · ${total ? fmt(100 * hist.totals[0] / total, 2) + '% correct candidates before matching' : 'No finite candidates'} · Bin width ${hist.bin_width}${normalized ? '' : ' m'}. Hover an interval for counts.`;
    }
    $('[data-distribution="exclusions"]').textContent = `Unscored or nonfinite cells excluded: ${fmt(data.all.excluded_cells, 0)} from all pairs; ${fmt(data.neighborhood.excluded_cells, 0)} from neighborhood pairs. These include optional size-unification removals and triangle-bound rejections. Normalization excludes ${fmt(data.normalization.skipped_zero_median_cells, 0)} directed finite cells because the anchor median is zero.`;
    $('[data-distribution="export"]').disabled = false;
  }
  for (const event of ['pointerover', 'focusin']) container.addEventListener(event, e => {
    const element = e.target.closest('[data-low]');
    if (!element || !data) return;
    const key = element.dataset.pop, hist = data[key];
    const row = histogramRows(hist).find(r => r.low === Number(element.dataset.low));
    $(`[data-hover="${key}"]`).textContent = binText(row, hist);
  });
  $('[data-distribution="export"]').onclick = () => {
    const records = [['trial', 'ap', 'noise_sigma_m', 'population', 'low', 'high_exclusive', 'correct', 'incorrect', 'correct_density', 'incorrect_density'],
      ...['neighborhood', 'all', 'normalized'].flatMap(k => histogramRows(data[k]).map(r => [meta.trial, settings.ap, settings.noise, k, r.low, r.high, ...r.counts, ...r.density]))];
    const url = URL.createObjectURL(new Blob([records.map(r => r.join(',')).join('\n')], { type: 'text/csv' }));
    const a = document.createElement('a'); a.href = url; a.download = `mpurge-dissimilarities-${settings.ap}-${settings.noise}.csv`; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  };
  return { async update(next) {
    settings = { ...next }; const current = ++revision, key = `${next.ap}:${next.noise}`;
    data = null; $('[data-distribution="export"]').disabled = true;
    container.querySelectorAll('[data-chart]').forEach(el => { el.textContent = 'Calculating first-round dissimilarities…'; });
    container.querySelectorAll('[data-counts], [data-hover]').forEach(el => { el.textContent = ''; });
    $('[data-distribution="condition"]').textContent = `AP ${next.ap} · σ ${next.noise} m · Empty room · Loading…`;
    try {
      if (!cache.has(key)) cache.set(key, request('distributions', { ap: next.ap, noise: next.noise }).catch(e => { cache.delete(key); throw e; }));
      const result = await cache.get(key);
      if (revision !== current) return;
      data = result; while (cache.size > 3) cache.delete(cache.keys().next().value);
      $('[data-distribution="condition"]').textContent = `AP ${next.ap} · σ ${next.noise} m · Empty room · First-round dissimilarities (unchanged by the matching-stage selector)`;
      render();
    } catch (e) {
      if (revision !== current) return;
      $('[data-distribution="condition"]').textContent = `Dissimilarity distributions unavailable: ${e.message}`;
      container.querySelectorAll('[data-chart]').forEach(el => { el.textContent = 'Unavailable'; });
    }
  } };
}
