function getBuiltInGameRules(gameId) {
  const game = games.find(g => g.game_id === gameId);
  return game && game.rules ? game.rules : 'No rules available for this game.';
}

function getConfigGameRules(configData, gameId) {
  const configuredRules = typeof configData?.game_rules === 'string'
    ? configData.game_rules.trim()
    : '';
  return configuredRules || getBuiltInGameRules(gameId);
}

function showGameRulesFromConfig(configData, gameId) {
  const rules = getConfigGameRules(configData, gameId);
  const rulesEl = document.getElementById('game-rules-text');
  if (rulesEl) rulesEl.textContent = rules;
  return rules;
}

function onGameChange() {
  const sel = document.getElementById('game-select');
  const opt = sel.selectedOptions[0];
  if (!opt) return;
  const tag = opt.dataset.tag;

  // Show/hide config sections
  document.getElementById('cfg-ale').classList.toggle('hidden', tag !== 'ale');
  document.getElementById('cfg-custom-breakout').classList.toggle('hidden', tag !== 'custom_breakout');
  document.getElementById('cfg-minigrid').classList.toggle('hidden', tag !== 'minigrid');
  document.getElementById('cfg-frozenlake').classList.toggle('hidden', !sel.value.includes('FrozenLake'));
  document.getElementById('cfg-highway').classList.toggle('hidden', !sel.value.includes('highway'));
  document.getElementById('cfg-procgen').classList.toggle('hidden', !sel.value.includes('procgen'));

  // Update game rules display
  const gameId = sel.value;

  // Per-game custom MiniGrid panels
  document.getElementById('cfg-custom-lava-crossing')?.classList.toggle('hidden', gameId !== 'CustomLavaCrossing-v0');
  document.getElementById('cfg-custom-multiroom')?.classList.toggle('hidden', gameId !== 'CustomMultiRoom-v0');
  document.getElementById('cfg-custom-unlock-pickup')?.classList.toggle('hidden', gameId !== 'CustomUnlockPickup-v0');
  const rulesEl = document.getElementById('game-rules-text');
  rulesEl.textContent = getBuiltInGameRules(gameId);

  // Update Frame Skip effect hint
  updateFrameSkipEffect();

  // Apply game-specific defaults from GAME_DEFAULTS table
  // crossval 模式下不应用：配置应来自 AI config，避免覆盖
  if (currentMode !== 'crossval') {
    const globalBase = GAME_DEFAULTS['_default'] || {};
    const categoryBase = tag === 'ale' ? (GAME_DEFAULTS['_ale_default'] || {})
                       : tag === 'custom_breakout' ? (GAME_DEFAULTS['_custom_breakout_default'] || {})
                       : tag === 'minigrid' ? (GAME_DEFAULTS['_minigrid_default'] || {})
                       : {};
    const overrides = GAME_DEFAULTS[gameId] || {};
    const merged = { ...globalBase, ...categoryBase, ...overrides };
    for (const [key, val] of Object.entries(merged)) {
      const el = document.getElementById('cfg-' + key);
      if (!el) continue;
      if (el.type === 'checkbox') {
        el.checked = !!val;
      } else {
        el.value = val;
      }
    }

    // Apply seed default based on current mode
    const seedDefaults = GAME_DEFAULTS['_seed'] || {};
    const seedEl = document.getElementById('cfg-seed');
    if (seedEl && seedEl.value === '') {
      seedEl.value = seedDefaults[currentMode] ?? '';
    }
  }

  // 在 AI 模式下，切换游戏时刷新 context 文件夹列表（过滤匹配当前游戏）
  if (currentMode === 'ai') loadContextFolderList();

  // 加载游戏标签
  loadGameTags();

  // 拉黑功能（放在末尾，避免阻断核心逻辑）
  updateBlacklistButton();
}

function updateBlacklistButton() {
  try {
    const btnBl = document.getElementById('btn-blacklist');
    console.log('[Blacklist] updateBlacklistButton, btn:', btnBl);
    if (!btnBl) return;
    const gameId = document.getElementById('game-select').value;
    const bl = getBlacklist();
    console.log('[Blacklist] gameId:', gameId, 'blacklist:', bl, 'isBlacked:', bl.includes(gameId));
    if (bl.includes(gameId)) {
      btnBl.textContent = 'Unblock';
      btnBl.style.color = '#888';
      btnBl.style.borderColor = '#888';
    } else {
      btnBl.textContent = 'Blacklist';
      btnBl.style.color = '#ef5350';
      btnBl.style.borderColor = '#ef5350';
    }
  } catch (e) { console.error('[Blacklist] updateBlacklistButton error:', e); }
}

function updateFrameSkipEffect() {
  const sel = document.getElementById('game-select');
  const opt = sel ? sel.selectedOptions[0] : null;
  const tag = opt ? opt.dataset.tag : null;

  const effectEl = document.getElementById('cfg-repeat-effect');
  if (!effectEl) return;

  const outerSkip = parseInt(document.getElementById('cfg-repeat').value) || 1;

  if (tag === 'ale') {
    const innerSkip = parseInt(document.getElementById('cfg-frameskip').value) || 4;
    effectEl.textContent = `实际帧数 = 游戏自带 ${innerSkip} × 外层叠加 ${outerSkip} = ${innerSkip * outerSkip} 帧/动作`;
  } else if (tag === 'custom_breakout') {
    const innerSkip = parseInt(document.getElementById('cfg-cb-frameskip').value) || 4;
    effectEl.textContent = `实际帧数 = 游戏自带 ${innerSkip} × 外层叠加 ${outerSkip} = ${innerSkip * outerSkip} 帧/动作`;
  } else {
    effectEl.textContent = outerSkip > 1 ? `每次动作推进 ${outerSkip} 帧` : '';
  }
}

// ========== Mode Switch ==========
// Context frames cache for player (演示视频/文件夹加载的帧)
let contextFrames = [];
// AI session frames cache (AI游戏运行时产生的帧)
let aiSessionFrames = [];

function switchMode(mode) {
  if (mode === currentMode) return;

  // 有未保存进度时提示确认
  const hasProgress = sessionId && totalFrames > 0;
  if (hasProgress && !confirm('确定切换吗？未保存的记录将会清空。')) return;

  // 清理当前模式的状态
  cleanupCurrentMode();

  // 离开 crossval 模式：解锁左侧 panel + 清状态
  if (currentMode === 'crossval') {
    crossvalUnlockPanel();
    crossvalClearState();
  }

  currentMode = mode;
  // AI 表单跨模式保留，因此返回 AI 时按实际会提交的表单规则恢复面板；
  // Human/Cross 空状态则回到当前游戏的项目规则。
  const currentGameId = document.getElementById('game-select')?.value;
  if (currentGameId) {
    const rulesEl = document.getElementById('game-rules-text');
    const aiRules = mode === 'ai'
      ? document.getElementById('ai-game-rules')?.value.trim()
      : '';
    if (rulesEl) rulesEl.textContent = aiRules || getBuiltInGameRules(currentGameId);
  }
  document.querySelectorAll('.tab-btn').forEach(b => {
    b.classList.toggle('active', b.dataset.mode === mode);
  });
  // Toggle AI panels
  document.getElementById('config-actions').style.display = mode === 'ai' ? 'flex' : 'none';
  document.getElementById('ai-config-panel').classList.toggle('hidden', mode !== 'ai');
  document.getElementById('ai-log-container').classList.toggle('hidden', mode !== 'ai');
  // Human controls 在 human 和 crossval 都需要（人在玩）
  document.getElementById('human-controls').classList.toggle('hidden', mode === 'ai');
  document.getElementById('ai-controls').classList.toggle('hidden', mode !== 'ai');
  // Human-extra（save/undo/optimal）只在 human 模式显示——crossval 用自己的控制条
  document.getElementById('human-extra-btns').classList.toggle('hidden', mode !== 'human');
  // Cross-val 专属
  const cvBrief = document.getElementById('crossval-brief');
  const cvEmpty = document.getElementById('crossval-empty');
  const cvBriefing = document.getElementById('crossval-briefing');
  const cvPlayCtl = document.getElementById('crossval-play-controls');
  const cvLevelSelector = document.getElementById('crossval-level-selector');
  const standardLeftContent = document.getElementById('standard-left-content');
  const frameContainer = document.getElementById('frame-container');
  const progressBar = document.getElementById('progress-bar-container');
  if (cvBrief) cvBrief.classList.toggle('hidden', mode !== 'crossval');
  if (cvPlayCtl) cvPlayCtl.classList.add('hidden');  // 仅 playing 时显示
  if (cvEmpty) cvEmpty.classList.toggle('hidden', mode !== 'crossval');
  if (cvBriefing) cvBriefing.classList.add('hidden');  // 仅 loaded 时显示
  if (cvLevelSelector) cvLevelSelector.classList.toggle('hidden', mode !== 'crossval');
  if (standardLeftContent) standardLeftContent.classList.toggle('hidden', mode === 'crossval');
  // crossval 模式下中央默认隐藏游戏帧 + 进度条（empty/briefing 自占位）
  if (mode === 'crossval') {
    frameContainer.classList.add('hidden');
    progressBar.classList.add('hidden');
  } else {
    frameContainer.classList.remove('hidden');
    progressBar.classList.remove('hidden');
  }

  // Set default seed based on mode (from GAME_DEFAULTS)
  const seedDefaults = GAME_DEFAULTS['_seed'] || {};
  // crossval 不主动覆盖 seed（seed 由 AI config 提供）
  if (mode !== 'crossval') {
    document.getElementById('cfg-seed').value = seedDefaults[mode] ?? '';
  }

  if (mode === 'ai') {
    loadContextFolderList();
    // Show context frames if loaded, otherwise clear player
    if (contextFrames.length > 0) {
      showContextInPlayer();
    }
  } else if (mode === 'crossval' && typeof crossvalInitializeQueue === 'function') {
    crossvalInitializeQueue().catch(err => {
      console.error('[CrossVal] queue initialization failed:', err);
      showEvaluationSaveToast('加载评测队列失败', true);
    });
  }
}

function cleanupCurrentMode() {
  // 关闭 WebSocket 连接
  if (ws) {
    ws.close();
    ws = null;
  }

  // 清理会话
  if (sessionId) {
    fetch(`/api/session/${sessionId}`, { method: 'DELETE' }).catch(e => console.error('Failed to delete session:', e));
    sessionId = null;
  }

  // 重置UI状态
  isLive = true;
  totalFrames = 0;
  lastGameOver = false;

  // 清空进度条
  const progressBar = document.getElementById('progress-bar');
  progressBar.value = 0;
  progressBar.max = 0;
  document.getElementById('progress-label').textContent = '0 / 0';

  // 清空游戏画面（显示灰色占位符）
  const gameFrame = document.getElementById('game-frame');
  // 创建一个灰色占位图片
  const canvas = document.createElement('canvas');
  canvas.width = 160;
  canvas.height = 210;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#16213e';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#555';
  ctx.font = '14px monospace';
  ctx.textAlign = 'center';
  ctx.fillText('No Game', canvas.width / 2, canvas.height / 2);
  gameFrame.src = canvas.toDataURL();

  // 清空状态信息
  document.getElementById('info-step').textContent = '0';
  document.getElementById('info-reward').textContent = '0.00';
  document.getElementById('info-cumulative').textContent = '0.00';
  document.getElementById('info-status').textContent = 'Not Started';
  document.getElementById('info-status').style.color = '#999';
  const endingEl = document.getElementById('info-ending');
  const passEl = document.getElementById('info-pass');
  const scoreEl = document.getElementById('info-score');
  const progressEl = document.getElementById('info-progress');
  if (endingEl)   { endingEl.textContent   = '-'; endingEl.style.color   = '#999'; }
  if (passEl)     { passEl.textContent     = '-'; passEl.style.color     = '#999'; }
  if (scoreEl)    { scoreEl.textContent    = '-'; scoreEl.style.color    = '#999'; }
  if (progressEl) { progressEl.innerHTML   = '-'; progressEl.style.color = '#999'; }
  document.getElementById('info-ram').innerHTML = '';

  // 清空AI日志和对话
  document.getElementById('ai-log').innerHTML = '';
  clearChatView();

  // 重置AI按钮状态
  document.getElementById('ai-btn-start').disabled = false;

  // 清空AI会话帧缓存（但保留演示视频缓存contextFrames）
  aiSessionFrames = [];
}

/**
 * 动态填充 ALE Mode/Difficulty 下拉列表
 * @param {string} selectId - select 元素 id
 * @param {number[]} values - 可用值列表
 * @param {string} hintId - hint 元素 id
 * @param {string} type - 'mode' 或 'difficulty'
 */
// ========== Game Tags ==========

function loadGameTags() {
  const sel = document.getElementById('game-select');
  const gameId = sel.value;
  const game = games.find(g => g.game_id === gameId);
  const tags = game && game.game_tags;

  document.getElementById('tag-info-horizon').value = (tags && tags.info_horizon) || '';
  document.getElementById('tag-env-dynamics').value = (tags && tags.env_dynamics) || '';

  updateAITagDisplay(tags);
}

function updateAITagDisplay(tags) {
  const el = document.getElementById('ai-tag-display');
  if (!el) return;
  if (!tags || (!tags.info_horizon && !tags.env_dynamics)) {
    el.textContent = '';
    const warn = document.createElement('span');
    warn.style.color = '#ef6c00';
    warn.textContent = '\u26a0 Not tagged';
    el.appendChild(warn);
    return;
  }
  const parts = [];
  if (tags.info_horizon) {
    const labels = { single_frame: 'Single Frame', few_frames: 'Few Frames', full_history: 'Full History' };
    parts.push(labels[tags.info_horizon] || tags.info_horizon);
  }
  if (tags.env_dynamics) {
    const labels = { static: 'Static', dynamic: 'Dynamic' };
    parts.push(labels[tags.env_dynamics] || tags.env_dynamics);
  }
  el.textContent = parts.join(' | ');
  el.style.color = '#81c784';
}

async function onTagChange() {
  const sel = document.getElementById('game-select');
  const gameId = sel.value;
  if (!gameId) return;

  const infoHorizon = document.getElementById('tag-info-horizon').value || null;
  const envDynamics = document.getElementById('tag-env-dynamics').value || null;

  const tags = {};
  if (infoHorizon) tags.info_horizon = infoHorizon;
  if (envDynamics) tags.env_dynamics = envDynamics;

  try {
    await fetch('/api/game_tags', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ game_id: gameId, tags: Object.keys(tags).length > 0 ? tags : null }),
    });
    // Update cached game data
    const game = games.find(g => g.game_id === gameId);
    if (game) game.game_tags = Object.keys(tags).length > 0 ? tags : null;
    updateAITagDisplay(game ? game.game_tags : null);
  } catch (e) {
    console.error('[GameTags] Failed to save:', e);
  }
}

