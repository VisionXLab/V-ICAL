// ========== Game Control ==========
async function startGame() {
  // Close existing session
  if (sessionId) {
    await fetch(`/api/session/${sessionId}`, { method: 'DELETE' });
    if (ws) { ws.close(); ws = null; }
  }

  const gameName = document.getElementById('game-select').value;
  const config = buildConfig(gameName);

  document.getElementById('btn-start').disabled = true;
  document.getElementById('btn-start').textContent = 'Starting...';

  try {
    const resp = await fetch('/api/session/create', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        game_name: gameName,
        config,
        mode: currentMode,
        evaluation_config_name: currentMode === 'crossval'
          ? window.crossvalState?.loadedConfigName
          : null,
      }),
    });
    if (!resp.ok) {
      const err = await resp.json();
      alert('Error: ' + (typeof err.detail === 'string' ? err.detail : JSON.stringify(err.detail)));
      return;
    }
    await applySessionData(await resp.json(), gameName);
  } catch (e) {
    alert('Failed to start: ' + e.message);
  } finally {
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-start').textContent = 'Start Game';
  }
}

/** 创建成功后的通用初始化 */
async function applySessionData(data, gameName) {
  sessionId = data.session_id;
  actionInfo = data.action_info;
  loadedSequence = null;
  updateSeqBadge();

  // 保留 contextList（演示上下文跨 game session 保持），仅更新 contextFrames
  updateContextFrames();
  aiSessionFrames = [];
  showFrame(data.frame);
  updateState(data.state);
  setupActionButtons();
  setupKeyMap();
  totalFrames = 1;
  updateProgress(0, 1);
  isLive = true;

  try {
    await connectWS();
  } catch (e) {
    console.error('[startGame] WS failed to open:', e);
    throw e;
  }

  // AI 模式下，context 由 aiStart() 统一加载，这里不再提前推送

  document.getElementById('btn-reset').disabled = false;
  document.getElementById('btn-undo').disabled = true;
  lastGameOver = false;
  document.getElementById('btn-save').disabled = false;
  document.getElementById('ai-btn-save').disabled = false;
  document.getElementById('no-game-msg').classList.add('hidden');
  document.getElementById('game-frame').classList.remove('hidden');

  // Show/hide Export Optimal button for custom MiniGrid games
  const OPTIMAL_GAMES = ['CustomLavaCrossing-v0', 'CustomMultiRoom-v0', 'CustomUnlockPickup-v0'];
  const btnOptimal = document.getElementById('btn-export-optimal');
  if (btnOptimal) {
    const show = OPTIMAL_GAMES.includes(gameName) && currentMode !== 'ai';
    btnOptimal.classList.toggle('hidden', !show);
  }
}

function buildConfig(gameName) {
  const cfg = {};
  const seed = document.getElementById('cfg-seed').value;
  if (seed !== '') cfg.seed = parseInt(seed);
  cfg.repeat = parseInt(document.getElementById('cfg-repeat').value) || 1;
  cfg.noop_fill = document.getElementById('cfg-noop-fill').checked;
  cfg.pre_score=parseFloat(document.getElementById('cfg-pre-score').value)
  if (cfg.noop_fill) {
    cfg.action_repeat = parseInt(document.getElementById('cfg-action-repeat').value) || 1;
  }

  // Max Steps — 通用参数，所有游戏类型共用
  const maxSteps = document.getElementById('cfg-max-steps').value;
  if (maxSteps !== '') cfg.max_steps = parseInt(maxSteps);
  if (currentMode === 'crossval') {
    const maxScore = window.crossvalState?.loadedConfigData?.game_settings?.['cfg-max-score'];
    if (maxScore !== undefined && maxScore !== null && maxScore !== '') {
      cfg.max_score = parseFloat(maxScore);
    }
  }

  const sel = document.getElementById('game-select');
  const tag = sel.selectedOptions[0]?.dataset.tag;

  if (tag === 'ale') {
    cfg.frameskip = parseInt(document.getElementById('cfg-frameskip').value) || 4;
    cfg.auto_respawn = document.getElementById('cfg-auto-respawn').checked;
    const lives = document.getElementById('cfg-initial-lives').value;
    if (lives !== '') cfg.initial_lives = parseInt(lives);
    cfg.end_on_life_loss = document.getElementById('cfg-end-on-life-loss').checked;
    const ramText = document.getElementById('cfg-initial-ram').value.trim();
    if (ramText) {
      const ramObj = {};
      for (const line of ramText.split('\n')) {
        const m = line.trim().match(/^(\d+)\s*=\s*(\d+)$/);
        if (m) ramObj[parseInt(m[1])] = parseInt(m[2]);
      }
      if (Object.keys(ramObj).length > 0) cfg.initial_ram = ramObj;
    }
    const skipSteps = document.getElementById('cfg-skip-initial-steps').value;
    if (skipSteps !== '' && parseInt(skipSteps) > 0) cfg.skip_initial_steps = parseInt(skipSteps);
    const skipAction = (GAME_DEFAULTS[gameName] || {})['skip-initial-action'];
    if (skipAction != null) cfg.skip_initial_action = skipAction;
    cfg.action_set = document.getElementById('cfg-action-set').value;
    const modeVal = document.getElementById('cfg-mode').value;
    if (modeVal !== '') cfg.mode = parseInt(modeVal);
    const diffVal = document.getElementById('cfg-difficulty').value;
    if (diffVal !== '') cfg.difficulty = parseInt(diffVal);
  } else if (tag === 'custom_breakout') {
    cfg.paddle_width = parseInt(document.getElementById('cfg-paddle-width').value) || 20;
    cfg.ball_speed = parseFloat(document.getElementById('cfg-ball-speed').value) || 2.0;
    cfg.brick_rows = parseInt(document.getElementById('cfg-brick-rows').value) || 6;
    cfg.brick_cols = parseInt(document.getElementById('cfg-brick-cols').value) || 10;
    cfg.brick_area_top_offset = parseInt(document.getElementById('cfg-brick-offset').value) || 10;
    cfg.launch_angle_mode = document.getElementById('cfg-launch-angle-mode').value;
    cfg.frameskip = parseInt(document.getElementById('cfg-cb-frameskip').value) || 4;
    cfg.initial_lives = parseInt(document.getElementById('cfg-cb-lives').value) || 5;
    cfg.end_on_life_loss = document.getElementById('cfg-cb-end-on-life-loss').checked;
    const cbSkipSteps = document.getElementById('cfg-cb-skip-initial-steps').value;
    if (cbSkipSteps !== '' && parseInt(cbSkipSteps) > 0) cfg.skip_initial_steps = parseInt(cbSkipSteps);
  }

  if (tag === 'minigrid') {
    cfg.partial_obs = document.getElementById('cfg-partial-obs').checked;
  }

  // Custom MiniGrid environment parameters
  if (gameName === 'CustomLavaCrossing-v0') {
    cfg.size = parseInt(document.getElementById('cfg-lc-size').value) || 11;
    cfg.num_crossings = parseInt(document.getElementById('cfg-lc-crossings').value) || 5;
  }
  if (gameName === 'CustomMultiRoom-v0') {
    cfg.grid_size = parseInt(document.getElementById('cfg-mr-grid-size').value) || 25;
    cfg.num_rooms = parseInt(document.getElementById('cfg-mr-rooms').value) || 6;
    cfg.max_room_size = parseInt(document.getElementById('cfg-mr-room-size').value) || 10;
  }
  if (gameName === 'CustomUnlockPickup-v0') {
    cfg.room_size = parseInt(document.getElementById('cfg-up-room-size').value) || 6;
    cfg.num_rows = parseInt(document.getElementById('cfg-up-rows').value) || 1;
    cfg.num_cols = parseInt(document.getElementById('cfg-up-cols').value) || 2;
  }

  if (gameName.includes('FrozenLake')) {
    cfg.is_slippery = document.getElementById('cfg-slippery').checked;
  }

  if(gameName.includes("highway"))
  {
    cfg.lanes_count=parseInt(document.getElementById("cfg-lanes-count").value);
    cfg.vehicles_density=parseFloat(document.getElementById("cfg-vehicles-density").value);
  }

  if(gameName.includes("procgen"))
  {
    const procgenSkipSteps = document.getElementById('cfg-procgen-skip-initial-steps').value;
    if (procgenSkipSteps !== '' && parseInt(procgenSkipSteps) > 0) cfg.skip_initial_steps = parseInt(procgenSkipSteps);

  }

  return cfg;
}

async function exportOptimalSolution() {
  if (!sessionId) return;
  const btn = document.getElementById('btn-export-optimal');
  btn.disabled = true;
  btn.textContent = 'Computing...';
  try {
    const resp = await fetch(`/api/session/${sessionId}/export-optimal`, { method: 'POST' });
    if (!resp.ok) {
      const err = await resp.json();
      alert('Export failed: ' + (err.detail || JSON.stringify(err)));
      return;
    }
    const data = await resp.json();
    alert(`Optimal solution exported!\n${data.total_steps} steps, reward=${data.episode_reward.toFixed(4)}\nSaved to: ${data.folder_name}`);
  } catch (e) {
    alert('Export error: ' + e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = 'Export Optimal';
  }
}

async function resetGame() {
  if (!sessionId) return;
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'reset' }));
  }
  // 清空对话和日志
  clearChatView();
  document.getElementById('ai-log').innerHTML = '';
  aiSessionFrames = [];
  document.getElementById('ai-btn-start').disabled = false;
}
