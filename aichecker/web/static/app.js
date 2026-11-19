'use strict';

const state = { csrf: '', session: null, jobTimer: null, jobId: null };
const POLL_MS = 1500;
const STATUS_MS = 20000;
const BAND_TEXT = {
  not_enough_text: 'Too short to judge',
  little_sign: 'Little sign of AI writing',
  some_sign: 'Some signs of AI writing',
  strong_sign: 'Strong signs of AI writing',
};
const CAVEATS = [
  'The score measures how predictable the wording is to a pair of small language models. It cannot know who wrote the text.',
  'Formal, technical or simple writing, writing by non-native English speakers and text that has been edited can all look machine-like. Check the per-domain false positive numbers.',
  'It was measured on RAID, a public set of English texts from older generators (2023 to early 2024 models). Newer models, other languages, very short texts and heavily edited text may do worse.',
  'It is clearly weaker for some kinds of writing and some AI models, and it can miss AI text that was sampled with a repetition penalty. The tables below show where.',
  'Use it as one clue among others, never as proof about a person.',
];

const $ = (id) => document.getElementById(id);

function el(tag, props, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(props || {})) {
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else if (value !== false && value !== null && value !== undefined) node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}

function clear(node) { node.replaceChildren(); return node; }

async function api(method, path, body) {
  const headers = {};
  let payload;
  if (method !== 'GET') headers['X-CSRF-Token'] = state.csrf;
  if (body instanceof FormData) payload = body;
  else if (body !== undefined) { headers['Content-Type'] = 'application/json'; payload = JSON.stringify(body); }
  let response;
  try {
    response = await fetch(path, { method, headers, body: payload, credentials: 'same-origin' });
  } catch (error) {
    throw new Error('Could not reach the app. Check the network and try again.');
  }
  if (response.redirected || response.status === 401) { window.location.assign('/'); throw new Error('Please sign in again.'); }
  let data = null;
  try { data = await response.json(); } catch (error) { data = null; }
  if (!response.ok) {
    const failure = new Error((data && data.error) || `The request failed (${response.status}).`);
    failure.busy = Boolean(data && data.busy);
    throw failure;
  }
  return data;
}

function show(node, kind, text) {
  node.hidden = !text;
  node.className = `message ${kind || ''}`.trim();
  node.textContent = text || '';
}

function likelihoodText(value) {
  const rounded = Math.round(value * 100);
  if (rounded >= 99) return '99+';
  if (rounded <= 1) return '<1';
  return String(rounded);
}

function formatCount(n) { return `${n.toLocaleString()} characters`; }
function percent(value) { return value === null || value === undefined ? 'n/a' : `${(value * 100).toFixed(1)}%`; }
function number(value) { return value === null || value === undefined ? 'n/a' : value.toFixed(3); }

function selectTab(name) {
  for (const tab of document.querySelectorAll('.tab')) tab.setAttribute('aria-selected', String(tab.dataset.panel === name));
  for (const panel of document.querySelectorAll('[role="tabpanel"]')) panel.hidden = panel.id !== `panel-${name}`;
  if (name === 'voice') loadVoice();
  if (name === 'accuracy') loadAccuracy();
  if (name === 'humanise') loadVoiceLine();
  if (location.hash !== `#${name}`) history.replaceState(null, '', `#${name}`);
}

function renderMarked(text, sentences) {
  const box = el('div', { class: 'marked' });
  let cursor = 0;
  for (const sentence of [...sentences].sort((a, b) => a.start - b.start)) {
    if (sentence.start < cursor) continue;
    if (sentence.start > cursor) box.append(text.slice(cursor, sentence.start));
    const piece = text.slice(sentence.start, sentence.end);
    if (sentence.level === 'strong' || sentence.level === 'some') {
      box.append(el('span', { class: `s-${sentence.level}`, title: `Sentence predictability ${sentence.score.toFixed(2)} (lower is more machine-like)` }, piece));
    } else box.append(piece);
    cursor = sentence.end;
  }
  if (cursor < text.length) box.append(text.slice(cursor));
  return box;
}

function renderAssessment(result, heading) {
  const assessment = result.assessment;
  const likelihood = assessment.likelihood;
  const verdict = el('div', { class: 'verdict', 'data-band': assessment.band },
    el('div', {}, el('div', { class: 'big' }, likelihood === null ? 'n/a' : likelihoodText(likelihood), likelihood === null ? null : el('small', { text: '% likely AI-written' })),
      el('div', { class: 'muted', text: assessment.headline })));
  if (likelihood !== null) {
    const mark = el('div', { class: 'mark' });
    mark.style.left = `${Math.min(Math.max(likelihood, 0), 1) * 100}%`;
    verdict.append(el('div', { class: 'meter', role: 'img', 'aria-label': `Likelihood ${likelihoodText(likelihood)} percent` }, mark));
  }
  const parts = [heading ? el('h2', { text: heading }) : null, verdict];
  if (assessment.reasons.length) parts.push(el('ul', { class: 'reasons' }, assessment.reasons.map((reason) => el('li', { 'data-leans': reason.leans, text: reason.text }))));
  for (const note of assessment.notes) parts.push(el('p', { class: 'faint', text: note }));
  if (result.truncated) parts.push(el('div', { class: 'message warn', text: `Only the first ${result.characters_scored.toLocaleString()} characters were scored (${result.scored_tokens.toLocaleString()} of ${result.tokens.toLocaleString()} tokens).` }));
  return parts;
}

async function runCheck() {
  const text = $('check-text').value;
  const message = $('check-message');
  if (!text.trim()) { show(message, 'error', 'Paste some text or open a file first.'); return; }
  const button = $('check-run');
  button.disabled = true;
  show(message, '', 'Checking. This can take up to a minute while the models load.');
  $('check-result').hidden = true;
  try {
    const result = await api('POST', '/api/check', { text });
    show(message, '', '');
    const card = el('div', { class: 'card' }, renderAssessment(result));
    if (result.sentences.length) {
      card.append(el('h3', { text: 'Sentences that look most machine-like' }),
        el('div', { class: 'legend' }, el('span', {}, el('i', { class: 's-strong' }), 'Strong'), el('span', {}, el('i', { class: 's-some' }), 'Some'),
          el('span', { class: 'faint', text: 'Marks on single sentences are rough: short sentences are not marked.' })),
        renderMarked(text, result.sentences));
    }
    card.append(el('button', { class: 'btn ghost small', type: 'button', onclick: () => selectTab('accuracy') }, 'How accurate this is'));
    clear($('check-result')).append(card);
    $('check-result').hidden = false;
    $('check-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
  } catch (error) {
    show(message, error.busy ? 'warn' : 'error', error.busy ? `${error.message} Try again in a few seconds.` : error.message);
  } finally {
    button.disabled = false;
  }
}

async function openFileInto(input, textarea, counter, message) {
  const file = input.files[0];
  input.value = '';
  if (!file) return;
  show(message, '', `Reading ${file.name}`);
  const form = new FormData();
  form.append('file', file);
  try {
    const result = await api('POST', '/api/extract', form);
    textarea.value = result.text;
    counter.textContent = formatCount(result.text.length);
    show(message, result.truncated ? 'warn' : '', result.truncated ? `${file.name} has ${result.characters.toLocaleString()} characters; only the first ${result.text.length.toLocaleString()} were loaded.` : `Loaded ${file.name}.`);
  } catch (error) {
    show(message, 'error', error.message);
  }
}

function wireCounter(textarea, counter) {
  textarea.addEventListener('input', () => { counter.textContent = formatCount(textarea.value.length); });
}

async function loadVoiceLine() {
  try {
    const voice = await api('GET', '/api/voice');
    const totals = voice.totals;
    const line = $('humanise-voice');
    if (totals.ready) {
      show(line, '', `Your voice: ${totals.passages} passages from ${totals.sources} file${totals.sources === 1 ? '' : 's'}.`);
    } else {
      clear(line).hidden = false;
      line.className = 'message warn';
      line.append('The Humaniser needs some of your own writing first. ', el('a', { href: '#voice', onclick: () => selectTab('voice') }, 'Add it on the Your voice tab'), '.');
    }
    $('humanise-run').disabled = !totals.ready;
  } catch (error) {
    show($('humanise-voice'), 'error', error.message);
  }
}

function renderHumaniseResult(result) {
  const parts = [el('h2', { text: 'Before and after' })];
  const columns = el('div', { class: 'two' },
    el('div', {}, el('h3', { text: 'Before' }), result.before ? el('div', { class: 'scorebox' }, scoreSummary(result.before)) : el('p', { class: 'faint', text: 'No score for this text.' }), el('div', { class: 'compare', text: result.original })),
    el('div', {}, el('h3', { text: 'After' }), result.after ? el('div', { class: 'scorebox' }, scoreSummary(result.after)) : el('p', { class: 'faint', text: 'No score for this text.' }), el('div', { class: 'compare', id: 'humanise-after', text: result.rewritten })));
  parts.push(columns);
  parts.push(el('div', { class: 'row' }, el('button', { class: 'btn ghost small', type: 'button', onclick: () => copyText(result.rewritten) }, 'Copy the rewrite')));
  if (result.score_error) parts.push(el('div', { class: 'message warn', text: `The scores could not be worked out: ${result.score_error}` }));
  if (result.kept_original_parts) parts.push(el('div', { class: 'message warn', text: `${result.kept_original_parts} part(s) came back unusable twice, so the original wording was kept there.` }));
  if (result.warnings.length) parts.push(el('div', {}, el('h3', { text: 'Check these against the original' }), el('ul', { class: 'plain' }, result.warnings.map((warning) => el('li', { text: warning })))));
  parts.push(el('p', { class: 'faint', text: 'Always read the rewrite against the original before you use it. The scores are this checker\'s own reading and say nothing about any other tool.' }));
  return parts;
}

function scoreSummary(result) {
  const assessment = result.assessment;
  return [el('span', { class: 'big' }, assessment.likelihood === null ? 'n/a' : likelihoodText(assessment.likelihood), assessment.likelihood === null ? null : el('small', { text: '%' })), el('span', { class: 'muted', text: assessment.headline })];
}

async function copyText(text) {
  try { await navigator.clipboard.writeText(text); } catch (error) { show($('humanise-message'), 'warn', 'Copying was blocked by the browser; select the text and copy it by hand.'); }
}

function stopPolling() {
  if (state.jobTimer) clearTimeout(state.jobTimer);
  state.jobTimer = null;
  state.jobId = null;
  $('humanise-progress').hidden = true;
  $('humanise-run').disabled = false;
}

async function pollJob(jobId) {
  if (state.jobId !== jobId) return;
  try {
    const job = await api('GET', `/api/jobs/${jobId}`);
    $('humanise-bar').style.width = `${Math.round((job.step / job.steps) * 100)}%`;
    $('humanise-step').textContent = job.message;
    if (job.status === 'running') { state.jobTimer = setTimeout(() => pollJob(jobId), POLL_MS); return; }
    stopPolling();
    if (job.status === 'done') {
      show($('humanise-message'), '', '');
      clear($('humanise-result')).append(el('div', { class: 'card' }, renderHumaniseResult(job.result)));
      $('humanise-result').hidden = false;
      $('humanise-result').scrollIntoView({ behavior: 'smooth', block: 'start' });
    } else if (job.status === 'failed') show($('humanise-message'), job.busy ? 'warn' : 'error', job.busy ? `${job.error} Try again in a few seconds.` : job.error);
    else show($('humanise-message'), '', 'Stopped.');
  } catch (error) {
    stopPolling();
    show($('humanise-message'), 'error', error.message);
  }
}

async function runHumanise() {
  const text = $('humanise-text').value;
  const message = $('humanise-message');
  if (!text.trim()) { show(message, 'error', 'Paste some text or open a file first.'); return; }
  show(message, '', '');
  $('humanise-result').hidden = true;
  $('humanise-run').disabled = true;
  try {
    const started = await api('POST', '/api/humanise', { text });
    state.jobId = started.job;
    $('humanise-progress').hidden = false;
    $('humanise-bar').style.width = '0';
    $('humanise-step').textContent = 'Starting';
    pollJob(started.job);
  } catch (error) {
    $('humanise-run').disabled = false;
    show(message, 'error', error.message);
  }
}

async function stopHumanise() {
  if (!state.jobId) return;
  try { await api('DELETE', `/api/jobs/${state.jobId}`); } catch (error) { show($('humanise-message'), 'error', error.message); }
}

function renderVoice(voice) {
  const totals = voice.totals;
  const summary = $('voice-summary');
  summary.hidden = false;
  summary.className = `message${totals.ready ? '' : ' warn'}`;
  summary.textContent = totals.ready
    ? `Ready: ${totals.passages} passages from ${totals.sources} file${totals.sources === 1 ? '' : 's'} (${totals.characters.toLocaleString()} characters).`
    : `Not enough yet: ${totals.passages} passages, ${totals.characters.toLocaleString()} characters. Add at least a few pages of your own writing.`;
  $('voice-rescan').hidden = !voice.folderConfigured;
  const list = clear($('voice-list'));
  for (const source of voice.sources) {
    list.append(el('li', {}, el('span', { class: 'name', text: source.name }), el('span', { class: 'faint', text: `${source.passages} passages${source.origin === 'folder' ? ', from the folder' : ''}` }),
      el('button', { class: 'btn ghost small', type: 'button', onclick: () => removeVoice(source.id) }, 'Remove')));
  }
}

async function loadVoice() {
  try { renderVoice(await api('GET', '/api/voice')); } catch (error) { show($('voice-message'), 'error', error.message); }
}

function describeVoiceChange(result) {
  const lines = [];
  if (result.added !== undefined && Array.isArray(result.added)) lines.push(`Added ${result.added.length} file(s).`);
  else if (result.added !== undefined) lines.push(`Added ${result.added}, kept ${result.unchanged}, removed ${result.removed}.`);
  for (const skipped of result.skipped || []) lines.push(`Skipped ${skipped.name}: ${skipped.reason}`);
  return lines.join(' ');
}

async function addVoiceFiles() {
  const input = $('voice-files');
  const files = [...input.files];
  input.value = '';
  if (!files.length) return;
  const form = new FormData();
  for (const file of files) form.append('files', file);
  $('voice-busy').textContent = 'Reading and indexing. This takes a little while.';
  try {
    const result = await api('POST', '/api/voice/files', form);
    show($('voice-message'), result.skipped.length ? 'warn' : '', describeVoiceChange(result));
    await loadVoice();
  } catch (error) {
    show($('voice-message'), 'error', error.message);
  } finally {
    $('voice-busy').textContent = '';
  }
}

async function rescanVoice() {
  $('voice-busy').textContent = 'Scanning the folder.';
  try {
    const result = await api('POST', '/api/voice/rescan');
    show($('voice-message'), result.skipped.length ? 'warn' : '', describeVoiceChange(result));
    await loadVoice();
  } catch (error) {
    show($('voice-message'), 'error', error.message);
  } finally {
    $('voice-busy').textContent = '';
  }
}

async function removeVoice(id) {
  try { await api('DELETE', `/api/voice/files/${id}`); await loadVoice(); } catch (error) { show($('voice-message'), 'error', error.message); }
}

function table(columns, rows) {
  return el('div', { class: 'table-wrap' }, el('table', { class: 'data' },
    el('thead', {}, el('tr', {}, columns.map((column) => el('th', { text: column })))),
    el('tbody', {}, rows.map((row) => el('tr', {}, row.map((cell) => el('td', { text: cell })))))));
}

function groupRows(groups) {
  return Object.entries(groups).map(([name, entry]) => [name, entry.n_human ?? '', entry.n_ai ?? '', number(entry.auroc), percent(entry.tpr_at_1pct_fpr), percent(entry.tpr_at_5pct_fpr), percent(entry.fpr_at_fpr1_threshold), percent(entry.tpr_at_fpr1_threshold), percent(entry.fpr_at_fpr5_threshold), percent(entry.tpr_at_fpr5_threshold)]);
}

const GROUP_COLUMNS = ['Group', 'Human texts', 'AI texts', 'AUROC', 'AI caught at 1% false alarms', 'at 5%', 'Human flagged strong', 'AI flagged strong', 'Human flagged some', 'AI flagged some'];

async function loadAccuracy() {
  const body = clear($('accuracy-body'));
  body.append(el('h1', { text: 'How accurate this is' }));
  let report;
  try { report = await api('GET', '/api/accuracy'); } catch (error) { body.append(el('div', { class: 'message error', text: error.message })); return; }
  const accuracy = report.accuracy;
  const overall = accuracy.overall;
  body.append(
    el('p', { class: 'intro', text: `Measured on ${overall.n_human.toLocaleString()} human and ${overall.n_ai.toLocaleString()} AI texts that were never used to set the thresholds (calibrated ${report.created}). Reader pair: ${report.pair.observer} with ${report.pair.performer}.` }),
    el('div', { class: 'stats' },
      el('div', { class: 'stat' }, el('b', { text: number(overall.auroc) }), el('span', { text: 'AUROC (1.0 is perfect, 0.5 is a coin toss)' })),
      el('div', { class: 'stat' }, el('b', { text: percent(overall.tpr_at_1pct_fpr) }), el('span', { text: 'of AI texts caught when only 1 in 100 human texts is wrongly flagged' })),
      el('div', { class: 'stat' }, el('b', { text: percent(overall.tpr_at_5pct_fpr) }), el('span', { text: 'of AI texts caught at 5 in 100 wrongly flagged' })),
      el('div', { class: 'stat' }, el('b', { text: percent(overall.fpr_at_fpr1_threshold) }), el('span', { text: 'of human texts actually marked "strong" on the held-out set' }))),
    el('h2', { text: 'Things to keep in mind' }), el('ul', { class: 'plain' }, CAVEATS.map((item) => el('li', { text: item }))),
    el('h2', { text: 'By kind of writing' }), table(GROUP_COLUMNS, groupRows(accuracy.per_domain)),
    el('h2', { text: 'By length (tokens, roughly three quarters of a word each)' }), table(GROUP_COLUMNS, groupRows(accuracy.per_length)),
    el('h2', { text: 'By AI model that wrote the text' }),
    table(['Model', 'AI texts', 'AUROC', 'Caught as strong', 'Caught as some'], Object.entries(accuracy.per_generator).map(([name, entry]) => [name, entry.n_ai, number(entry.auroc), percent(entry.tpr_at_fpr1_threshold), percent(entry.tpr_at_fpr5_threshold)])),
    el('h2', { text: 'By how the AI text was generated' }),
    table(['Setting', 'AI texts', 'AUROC', 'Caught as strong', 'Caught as some'], Object.entries(accuracy.per_decoding).map(([name, entry]) => [name, entry.n_ai, number(entry.auroc), percent(entry.tpr_at_fpr1_threshold), percent(entry.tpr_at_fpr5_threshold)])),
    el('h2', { text: 'Single sentences' }),
    el('p', { class: 'muted', text: `Marks on single sentences are much rougher than the whole-text reading: AUROC ${number(accuracy.sentence_level.auroc)}. A strong mark lands on ${percent(accuracy.sentence_level.ai_sentences_marked_strong)} of AI sentences and ${percent(accuracy.sentence_level.human_sentences_marked_strong)} of human ones.` }),
    el('p', { class: 'faint', text: `"Strong" and "some" are the cut-offs set so that about 1 and 5 in 100 human texts in the training half were flagged. Texts shorter than ${report.minTokens} tokens are not judged at all. Long texts are read in windows of ${report.windowTokens} tokens, up to ${report.maxTokens}.` }));
}

async function refreshStatus() {
  try {
    const status = await api('GET', '/api/status');
    $('worker-dot').dataset.state = status.worker;
    $('worker-text').textContent = status.worker === 'up' ? (status.pairLoaded ? 'Checker ready' : 'Checker idle') : 'Checker offline';
  } catch (error) {
    $('worker-dot').dataset.state = 'down';
    $('worker-text').textContent = 'No answer';
  }
}

async function start() {
  state.session = await api('GET', '/api/session');
  state.csrf = state.session.csrfToken;
  if (state.session.signOutPath) {
    $('signout').href = state.session.signOutPath;
    $('signout').hidden = false;
  }
  for (const tab of document.querySelectorAll('.tab')) tab.addEventListener('click', () => selectTab(tab.dataset.panel));
  wireCounter($('check-text'), $('check-count'));
  wireCounter($('humanise-text'), $('humanise-count'));
  $('check-run').addEventListener('click', runCheck);
  $('check-clear').addEventListener('click', () => { $('check-text').value = ''; $('check-count').textContent = formatCount(0); $('check-result').hidden = true; show($('check-message'), '', ''); });
  $('check-file').addEventListener('change', () => openFileInto($('check-file'), $('check-text'), $('check-count'), $('check-message')));
  $('humanise-file').addEventListener('change', () => openFileInto($('humanise-file'), $('humanise-text'), $('humanise-count'), $('humanise-message')));
  $('humanise-run').addEventListener('click', runHumanise);
  $('humanise-stop').addEventListener('click', stopHumanise);
  $('voice-files').addEventListener('change', addVoiceFiles);
  $('voice-rescan').addEventListener('click', rescanVoice);
  const wanted = location.hash.slice(1);
  selectTab(['check', 'humanise', 'voice', 'accuracy'].includes(wanted) ? wanted : 'check');
  refreshStatus();
  setInterval(refreshStatus, STATUS_MS);
}

start().catch((error) => show($('check-message'), 'error', error.message));
