const $ = s => document.querySelector(s);
const api = async (u, o) => { const r = await fetch(u, o); const j = await r.json(); if (!r.ok) throw new Error(j.error || 'Request failed'); return j; };
const COL = {ink:'#8b5cf6', red:'#f472b6', blue:'#22d3ee', teal:'#34d399', grey:'#4b5675', amber:'#fbbf24'};
const MCOL = {'Linear Regression':COL.blue, 'Decision Tree Regressor':COL.teal, 'Random Forest Regressor':COL.ink};
const BAND = {'Outstanding':['#34d399','#22d3ee'], 'Stable':['#818cf8','#22d3ee'], 'Needs support':['#fb7185','#f472b6']};
// vertical/horizontal gradient fill for bars
const vg = (c1, c2, horiz) => ctx => { const a = ctx.chart.chartArea; if (!a) return c1; const g = horiz ? ctx.chart.ctx.createLinearGradient(a.left, 0, a.right, 0) : ctx.chart.ctx.createLinearGradient(0, a.bottom, 0, a.top); g.addColorStop(0, c1); g.addColorStop(1, c2); return g; };
const mix = c => ({[COL.ink]:[COL.ink,'#c4b5fd'], [COL.blue]:[COL.blue,'#a5f3fc'], [COL.teal]:[COL.teal,'#a7f3d0']}[c] || [c, c]);
const pretty = s => s.replace(/_/g, ' ');
const f2 = v => Number(v).toFixed(2), f3 = v => Number(v).toFixed(3);
Chart.defaults.font.family = "'Instrument Sans', sans-serif"; Chart.defaults.color = '#98a0c3'; Chart.defaults.borderColor = 'rgba(160,170,255,.09)';
Chart.defaults.animation.duration = 900; Chart.defaults.animation.easing = 'easeOutQuart';
Object.assign(Chart.defaults.plugins.tooltip, {backgroundColor:'rgba(12,14,30,.95)', borderColor:'rgba(139,92,246,.5)', borderWidth:1, padding:10, cornerRadius:10, titleColor:'#fff', boxPadding:4});
Chart.defaults.plugins.legend.labels.usePointStyle = true; Chart.defaults.datasets.bar.borderSkipped = false;
const charts = {};
function mk(id, cfg) { charts[id]?.destroy(); cfg.options = Object.assign({responsive:true, maintainAspectRatio:false}, cfg.options); charts[id] = new Chart($('#'+id), cfg); }
const fill = (sel, items, fn = x => x) => { $(sel).innerHTML = items.map(x => `<option value="${x}">${fn(x)}</option>`).join(''); };
let EDA, MODELS, IMP, AVP;

async function init() {
  const [sum, eda, mod, imp, avp, schema] = await Promise.all([api('/api/summary'), api('/api/eda'), api('/api/models'), api('/api/importance'), api('/api/actual-vs-predicted'), api('/api/schema')]);
  EDA = eda; MODELS = mod; IMP = imp; AVP = avp;
  renderOverview(sum); renderEDA(); renderModels(); renderForm(schema); loadHistory();
}

function renderOverview(s) {
  $('#k-rows').textContent = s.dataset.rows.toLocaleString();
  const b = s.best, k = [
    ['Students', s.dataset.rows.toLocaleString(), `${s.dataset.n_train} train · ${s.dataset.n_test} test`, ''],
    ['Mean exam score', s.eda_stats.mean, `range ${s.eda_stats.min}–${s.eda_stats.max}, SD ${s.eda_stats.std}`, ''],
    ['Best model', s.best_model.replace(' Regressor',''), 'chosen by 5-fold CV RMSE', 'best'],
    ['Test R²', f3(b.test.r2), `MAE ${f2(b.test.mae)} · RMSE ${f2(b.test.rmse)}`, 'best']];
  $('#kpis').innerHTML = k.map(([l, v, sub, c]) => `<div class="col-6 col-lg-3"><div class="kpi ${c}"><div class="l">${l}</div><div class="v" data-count="${v}">${v}</div><div class="l">${sub}</div></div></div>`).join('');
  document.querySelectorAll('[data-count]').forEach(el => countUp(el, el.dataset.count));
  mk('c-hist', {type:'bar', data:{labels:EDA.hist.labels, datasets:[{data:EDA.hist.counts, backgroundColor:vg('#6366f1','#22d3ee'), borderRadius:5}]},
    options:{plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Exam score'}}, y:{title:{display:true,text:'Students'}}}}});
  const ct = EDA.corr_target;
  mk('c-corr', {type:'bar', data:{labels:ct.map(e => pretty(e[0])), datasets:[{data:ct.map(e => e[1]), backgroundColor:ct.map(e => e[1] >= 0 ? vg('#6366f1','#22d3ee',true) : vg('#f472b6','#fb7185',true)), borderRadius:5}]},
    options:{indexAxis:'y', plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Pearson r'}}}}});
}

function renderEDA() {
  const nums = Object.keys(EDA.scatter), cats = Object.keys(EDA.categories);
  fill('#sel-num', nums, pretty); fill('#sel-cat', cats, pretty);
  $('#sel-num').value = 'Hours_Studied'; $('#sel-num').onchange = drawScatter; $('#sel-cat').onchange = drawCat;
  drawScatter(); drawCat();
  const {labels, values} = EDA.corr_matrix, short = l => pretty(l).split(' ').map(w => w.slice(0, 5)).join(' ');
  const cell = v => { const a = Math.abs(v), c = v >= 0 ? '99,102,241' : '244,114,182'; return `<td title="r = ${v.toFixed(3)}" style="background:rgba(${c},${Math.max(.07, a)});color:${a > .55 ? '#fff' : '#cfd3f2'}">${v.toFixed(2)}</td>`; };
  $('#heat').innerHTML = `<table class="heat w-100"><tr><th></th>${labels.map(l => `<th>${short(l)}</th>`).join('')}</tr>` +
    values.map((r, i) => `<tr><th class="text-end">${short(labels[i])}</th>${r.map(cell).join('')}</tr>`).join('') + '</table>';
  const miss = Object.entries(EDA.missing);
  $('#quality').innerHTML = `<p class="note">Missing values found in the dataset and how the pipeline handles them:</p><ul class="small ps-3">` +
    miss.map(([c, n]) => `<li><b>${pretty(c)}</b>: ${n} missing</li>`).join('') + `</ul><p class="note mb-0">Categorical gaps are filled with the most frequent value inside the pipeline (fitted on training data only), so there is no leakage.</p>`;
}
function drawScatter() {
  const c = $('#sel-num').value;
  mk('c-scatter', {type:'scatter', data:{datasets:[{label:'Student', data:EDA.scatter[c], backgroundColor:'rgba(34,211,238,.45)', pointRadius:3.5, pointHoverRadius:7, hoverBackgroundColor:'#fff'},
    {label:'Linear trend', data:EDA.trend[c], type:'line', borderColor:COL.red, borderWidth:3, pointRadius:0}]},
    options:{scales:{x:{title:{display:true,text:pretty(c)}}, y:{title:{display:true,text:'Exam score'}, min:50}}}});
}
function drawCat() {
  const c = $('#sel-cat').value, d = EDA.categories[c];
  mk('c-cat', {type:'bar', data:{labels:d.map(x => x.label), datasets:[{data:d.map(x => x.mean), backgroundColor:vg('#6366f1','#c084fc'), borderRadius:8}]},
    options:{plugins:{legend:{display:false}, tooltip:{callbacks:{label: t => `mean ${t.raw} (n=${d[t.dataIndex].count})`}}}, scales:{y:{min:60, title:{display:true,text:'Mean exam score'}}}}});
}

function renderModels() {
  const names = Object.keys(MODELS.models), best = MODELS.best_model;
  $('#metric-table').innerHTML = `<thead><tr><th>Model</th><th>MAE</th><th>RMSE</th><th>R² (test)</th><th>R² (train)</th><th>CV RMSE (5-fold)</th><th>CV R²</th></tr></thead><tbody>` +
    names.map(n => { const m = MODELS.models[n]; return `<tr class="${n === best ? 'metric-best' : ''}"><td><b>${n}</b> ${n === best ? '<span class="badge badge-best">best</span>' : ''}</td>
      <td>${f3(m.test.mae)}</td><td>${f3(m.test.rmse)}</td><td>${f3(m.test.r2)}</td><td>${f3(m.train_r2)}</td>
      <td>${f3(m.cv.rmse_mean)} ± ${f3(m.cv.rmse_std)}</td><td>${f3(m.cv.r2_mean)} ± ${f3(m.cv.r2_std)}</td></tr>`; }).join('') + '</tbody>';
  $('#criterion').textContent = `Best model = ${MODELS.criterion}. Test metrics come from the 20% hold-out set that was never used for fitting or selection.`;
  const col = names.map(n => MCOL[n]), short = names.map(n => n.replace(' Regressor', ''));
  mk('c-err', {type:'bar', data:{labels:['MAE','RMSE'], datasets:names.map((n, i) => ({label:short[i], data:[MODELS.models[n].test.mae, MODELS.models[n].test.rmse], backgroundColor:vg(...mix(col[i])), borderRadius:6, maxBarThickness:60}))},
    options:{scales:{y:{title:{display:true,text:'Exam-score points'}}}}});
  mk('c-r2', {type:'bar', data:{labels:short, datasets:[{data:names.map(n => MODELS.models[n].test.r2), backgroundColor:col.map(c => vg(...mix(c))), borderRadius:8, maxBarThickness:90}]},
    options:{plugins:{legend:{display:false}}, scales:{y:{min:0, max:1}}}});
  fill('#sel-avp', names); $('#sel-avp').value = best; $('#sel-avp').onchange = drawAVP; drawAVP();
  fill('#sel-imp', [...names, 'RF impurity']); $('#sel-imp').value = best; $('#sel-imp').onchange = drawImp; drawImp();
}
function drawImp() {
  const v = $('#sel-imp').value, gini = v === 'RF impurity';
  let e = gini ? Object.entries(IMP.rf_gini).map(([k, x]) => [k, x, 0]) : Object.entries(IMP.permutation[v]).map(([k, x]) => [k, x.mean, x.std]);
  e = e.sort((a, b) => b[1] - a[1]).slice(0, 12);
  $('#imp-note').textContent = gini ? 'Random Forest impurity importance (share of total split gain), one-hot columns summed per feature.'
    : `Permutation importance on the test set: drop in R² when a feature is shuffled (mean of 10 repeats) for ${v}. Works for every model.`;
  mk('c-imp', {type:'bar', data:{labels:e.map(x => pretty(x[0])), datasets:[{data:e.map(x => x[1]), backgroundColor:vg(...mix(MCOL[v] || COL.ink), true), borderRadius:5}]},
    options:{indexAxis:'y', plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:gini ? 'importance' : 'R² drop'}}}}});
}
function drawAVP() {
  const n = $('#sel-avp').value, a = AVP.actual, p = AVP[n];
  mk('c-avp', {type:'scatter', data:{datasets:[{label:'Test student', data:a.map((x, i) => ({x, y:p[i]})), backgroundColor:'rgba(165,180,252,.4)', pointRadius:3, pointHoverRadius:7},
    {label:'Perfect prediction', data:[{x:55, y:55}, {x:100, y:100}], type:'line', borderColor:COL.red, borderDash:[6, 4], pointRadius:0, borderWidth:2}]},
    options:{scales:{x:{title:{display:true,text:'Actual exam score'}, min:55}, y:{title:{display:true,text:'Predicted exam score'}, min:55}}}});
  const res = a.map((x, i) => x - p[i]), lo = -10, w = 1, bins = Array(20).fill(0);
  res.forEach(r => { const k = Math.min(19, Math.max(0, Math.floor((r - lo) / w))); bins[k]++; });
  mk('c-res', {type:'bar', data:{labels:bins.map((_, i) => lo + i * w), datasets:[{data:bins, backgroundColor:vg(...mix(MCOL[n])), borderRadius:4}]},
    options:{plugins:{legend:{display:false}}, scales:{x:{title:{display:true,text:'Residual (points; edge bins include outliers)'}}, y:{title:{display:true,text:'Students'}}}}});
}

let SCHEMA;
function renderForm(s) {
  SCHEMA = s;
  $('#f-num').innerHTML = s.numeric.map(f => `<div class="col-sm-6"><div class="field-num"><label class="form-label" for="${f.name}">${f.label}<output id="o-${f.name}">${f.default}</output></label>
    <input class="form-range" type="range" id="${f.name}" name="${f.name}" min="${f.min}" max="${f.max}" value="${f.default}" step="1"></div></div>`).join('');
  $('#f-cat').innerHTML = s.categorical.map(f => `<div class="col-sm-6 ${f.options.length <= 3 ? 'col-md-6' : 'col-md-4'}"><label class="form-label">${f.label}</label>` +
    (f.options.length <= 3
      ? `<div class="seg" role="radiogroup">${f.options.map((o, i) => `<input type="radio" name="${f.name}" id="${f.name}_${i}" value="${o}" ${o === f.default ? 'checked' : ''}><label for="${f.name}_${i}">${o}</label>`).join('')}</div>`
      : `<select class="form-select form-select-sm" id="${f.name}" name="${f.name}">${f.options.map(o => `<option ${o === f.default ? 'selected' : ''}>${o}</option>`).join('')}</select>`) + '</div>').join('');
  document.querySelectorAll('input[type=range]').forEach(r => { r.oninput = () => syncRange(r, true); syncRange(r); });
  $('#form').onsubmit = predict;
  $('#btn-reset').onclick = () => { setDefaults(); resultEmpty(); };
  $('#btn-random').onclick = randomize;
  resultEmpty();
}
function syncRange(r, bump) {
  const o = $('#o-' + r.name); r.style.setProperty('--p', ((r.value - r.min) / (r.max - r.min) * 100) + '%'); o.textContent = r.value;
  if (bump) { o.classList.add('bump'); setTimeout(() => o.classList.remove('bump'), 140); }
}
function setDefaults() {
  SCHEMA.numeric.forEach(f => { const r = $('#' + f.name); r.value = f.default; syncRange(r); });
  SCHEMA.categorical.forEach(f => setCat(f, f.default));
}
function setCat(f, v) { if (f.options.length <= 3) { const i = f.options.indexOf(v); $(`#${f.name}_${i}`).checked = true; } else $('#' + f.name).value = v; }
function randomize() {
  SCHEMA.numeric.forEach(f => { const r = $('#' + f.name); r.value = Math.round(+f.min + Math.random() * (f.max - f.min)); syncRange(r, true); });
  SCHEMA.categorical.forEach(f => setCat(f, f.options[Math.floor(Math.random() * f.options.length)]));
  $('#form').requestSubmit();
}

const C = 2 * Math.PI * 90;
const ringSVG = (id, band) => { const [c1, c2] = BAND[band] || ['#8b5cf6', '#22d3ee'];
  return `<svg viewBox="0 0 210 210"><defs><linearGradient id="${id}" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="${c1}"/><stop offset="1" stop-color="${c2}"/></linearGradient></defs>
  <circle class="ring-bg" cx="105" cy="105" r="90"/><circle class="ring-fg" id="ring-fg" cx="105" cy="105" r="90" stroke="url(#${id})" style="--ring-glow:${c1}aa"/></svg>`; };
function resultEmpty() {
  $('#result').innerHTML = `<div class="empty"><div class="ring-wrap ring">${ringSVG('rg0')}<div class="ring-center"><div class="big" style="color:var(--mute)">--</div><small class="note">/ 100</small></div></div>
    <h2 class="justify-content-center">Ready when you are</h2><p class="note mb-0">Adjust the inputs and press <b>Predict</b> (or hit 🎲 Randomize). The result comes from the trained pipeline on the server, and every prediction is saved to History.</p></div>`;
}
function countUp(el, target, dec, dur = 1100) {
  const t = parseFloat(String(target).replace(/,/g, '')); if (isNaN(t)) return;
  const d = dec ?? (String(target).includes('.') ? String(target).split('.')[1].length : 0), comma = String(target).includes(',');
  const t0 = performance.now(), fmt = v => comma ? Math.round(v).toLocaleString() : v.toFixed(d);
  (function step(now) { const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 4); el.textContent = fmt(t * e); if (k < 1) requestAnimationFrame(step); else el.textContent = String(target); })(t0);
}
async function predict(e) {
  e.preventDefault(); $('#form-err').textContent = '';
  const body = Object.fromEntries(new FormData($('#form')));
  const btn = $('#btn-predict'), lbl = btn.querySelector('span'); btn.disabled = true; lbl.textContent = 'Predicting…';
  try {
    const r = await api('/api/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(body)});
    const maxD = Math.max(...r.whatif.map(w => +w.delta), 0.01), [c1] = BAND[r.band] || [COL.ink];
    $('#result').classList.remove('fresh'); void $('#result').offsetWidth; $('#result').classList.add('fresh');
    $('#result').innerHTML = `<p class="note mb-0 text-center">Predicted exam score · ${r.model}</p>
      <div class="ring-wrap">${ringSVG('rg1', r.band)}<div class="ring-center"><div class="big" id="big-n" style="color:${c1}">0.0</div><small class="note">out of 100</small></div></div>
      <div class="text-center mt-2"><span class="pill ${r.band}">${r.band}</span></div>
      <div class="range-bar" title="likely range"><i style="left:${r.low}%;width:${Math.max(1, r.high - r.low)}%"></i><b style="left:${r.score}%"></b></div>
      <p class="note text-center mb-2">likely range ${r.low}–${r.high} (± test RMSE ${r.margin})</p><p class="text-center">${r.message}</p>
      ${r.raw_score !== r.score ? `<p class="note text-center">Raw model output ${r.raw_score}, capped to the valid 0–100 range.</p>` : ''}
      ${r.whatif.length ? `<div class="divider"></div><h2 style="font-size:.95rem">✨ What would raise this score?</h2><ul class="whatif">${r.whatif.map(w => `<li><div><span>${w.change}</span><b>+${w.delta}</b></div><div class="bar"><i data-w="${(+w.delta / maxD * 100).toFixed(0)}"></i></div></li>`).join('')}</ul><p class="note">Each line re-runs the model with only that factor changed. These are associations learned from data, not guaranteed effects.</p>` : ''}
      <div class="divider"></div><h2 style="font-size:.95rem">Same student, all three models</h2><div class="chart-box short"><canvas id="c-pred"></canvas></div>
      <p class="note mb-0 mt-2">Saved to history as #${r.id}.</p>`;
    requestAnimationFrame(() => { $('#ring-fg').style.strokeDasharray = `${r.score / 100 * C} 999`; document.querySelectorAll('.whatif .bar i').forEach(i => i.style.width = i.dataset.w + '%'); });
    countUp($('#big-n'), r.score.toFixed(1), 1, 1300);
    const n = Object.keys(r.all_models);
    mk('c-pred', {type:'bar', data:{labels:n.map(x => x.replace(' Regressor', '')), datasets:[{data:n.map(x => r.all_models[x]), backgroundColor:n.map(x => vg(...mix(MCOL[x]), true)), borderRadius:6}]},
      options:{indexAxis:'y', plugins:{legend:{display:false}}, scales:{x:{min:Math.max(0, Math.min(...Object.values(r.all_models)) - 10), max:100}}}});
    loadHistory();
    if (window.innerWidth < 992) $('#result').scrollIntoView({behavior:'smooth', block:'start'});
  } catch (err) { $('#form-err').textContent = err.message; }
  btn.disabled = false; lbl.textContent = 'Predict exam score';
}

async function loadHistory() {
  const rows = await api('/api/history'), keys = ['Hours_Studied', 'Attendance', 'Previous_Scores', 'Sleep_Hours', 'Tutoring_Sessions', 'Motivation_Level'];
  $('#hist-table').innerHTML = !rows.length ? '<tbody><tr><td class="note p-3">No predictions yet. Make one above and it will appear here.</td></tr></tbody>' :
    `<thead><tr><th>#</th><th>Time</th><th>Model</th><th>Score</th>${keys.map(k => `<th>${pretty(k)}</th>`).join('')}</tr></thead><tbody>` +
    rows.map(r => `<tr><td>${r.id}</td><td>${r.created_at}</td><td>${r.model}</td><td><b style="color:var(--cyan)">${f2(r.prediction)}</b></td>${keys.map(k => `<td>${r.inputs[k]}</td>`).join('')}</tr>`).join('') + '</tbody>';
}
$('#btn-clear').onclick = async () => { if (confirm('Delete all saved predictions?')) { await api('/api/history', {method:'DELETE'}); loadHistory(); } };
init().catch(e => { document.body.insertAdjacentHTML('afterbegin', `<div class="alert alert-danger m-3">Failed to load dashboard: ${e.message}</div>`); });

// scroll progress bar + reveal-on-scroll
addEventListener('scroll', () => { const h = document.documentElement; $('#scroll-progress').style.width = (h.scrollTop / (h.scrollHeight - h.clientHeight) * 100) + '%'; }, {passive:true});
const io = new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) { e.target.classList.add('in'); io.unobserve(e.target); } }), {threshold:.05});
document.querySelectorAll('.reveal').forEach(el => io.observe(el));
