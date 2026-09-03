// ========== Initialization ==========

window.addEventListener('DOMContentLoaded', async () => {
  await loadGames();
  await loadAIDefaults();
  document.getElementById('game-select').addEventListener('change', () => {
    // 选中拉黑游戏时提示取消拉黑
    const sel = document.getElementById('game-select');
    const bl = getBlacklist();
    if (bl.includes(sel.value)) {
      if (confirm(`${sel.value} 已被拉黑，取消拉黑吗？`)) {
        saveBlacklist(bl.filter(id => id !== sel.value));
        renderGameSelect(sel.value);
      }
    }
    onGameChange();
  });
  onGameChange();
  // 实时更新 Frame Skip 叠加提示
  ['cfg-repeat', 'cfg-frameskip', 'cfg-cb-frameskip'].forEach(id => {
    document.getElementById(id).addEventListener('input', updateFrameSkipEffect);
  });
  document.addEventListener('keydown', handleKeyboard);
});

// ========== Blacklist ==========
function getBlacklist() {
  try { return JSON.parse(localStorage.getItem('blacklistedGames') || '[]'); }
  catch { return []; }
}
function saveBlacklist(list) {
  localStorage.setItem('blacklistedGames', JSON.stringify(list));
}

async function loadGames() {
  const resp = await fetch('/api/games');
  games = await resp.json();
  renderGameSelect();
}

function renderGameSelect(preserveValue) {
  const sel = document.getElementById('game-select');
  const oldVal = preserveValue || sel.value;
  const bl = getBlacklist();
  // 正常游戏在前，拉黑游戏沉底
  const normal = games.filter(g => !bl.includes(g.game_id));
  const blacked = games.filter(g => bl.includes(g.game_id));
  sel.innerHTML = '';
  for (const g of normal) {
    const opt = document.createElement('option');
    opt.value = g.game_id;
    opt.textContent = `${g.key}. ${g.game_id}`;
    opt.dataset.tag = g.tag;
    sel.appendChild(opt);
  }
  for (const g of blacked) {
    const opt = document.createElement('option');
    opt.value = g.game_id;
    opt.textContent = `\u2716 ${g.key}. ${g.game_id}`;
    opt.dataset.tag = g.tag;
    opt.style.color = '#888';
    opt.classList.add('blacklisted-option');
    sel.appendChild(opt);
  }
  if (oldVal && [...sel.options].some(o => o.value === oldVal)) sel.value = oldVal;
}

function toggleBlacklist() {
  console.log('[Blacklist] toggleBlacklist called');
  const sel = document.getElementById('game-select');
  const gameId = sel.value;
  console.log('[Blacklist] gameId:', gameId);
  if (!gameId) { console.log('[Blacklist] no gameId, returning'); return; }
  const bl = getBlacklist();
  console.log('[Blacklist] current list:', bl);
  if (bl.includes(gameId)) {
    if (!confirm(`取消拉黑 ${gameId} 吗？`)) return;
    saveBlacklist(bl.filter(id => id !== gameId));
    console.log('[Blacklist] removed', gameId);
  } else {
    if (!confirm(`拉黑 ${gameId} 吗？`)) return;
    saveBlacklist([...bl, gameId]);
    console.log('[Blacklist] added', gameId);
  }
  console.log('[Blacklist] after save:', getBlacklist());
  renderGameSelect(gameId);
  onGameChange();
}

async function loadAIDefaults() {
  try {
    const resp = await fetch('/api/config/ai-defaults');
    const data = await resp.json();
    if (data.base_url) document.getElementById('ai-base-url').value = data.base_url;
    if (data.model) document.getElementById('ai-model').value = data.model;
    if (data.temperature != null) document.getElementById('ai-temperature').value = data.temperature;
    if (data.max_tokens != null) document.getElementById('ai-max-tokens').value = data.max_tokens;
    if (data.has_api_key) {
      document.getElementById('ai-api-key').placeholder = '(config.json default)';
    }
  } catch (e) { /* config.json not found, use hardcoded defaults */ }
}
