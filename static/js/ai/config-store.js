// ========== AI Config Save/Load ==========

async function saveAIConfig() {
  const gameId = document.getElementById('game-select').value;
  if (!gameId) { alert('Select a game first'); return; }

  const name = prompt('Config name:');
  if (!name) return;

  // Collect AI hyperparams (api_key/base_url/model only come from config.json, never saved here)
  const config = {
    temperature: parseFloat(document.getElementById('ai-temperature').value),
    max_tokens: parseInt(document.getElementById('ai-max-tokens').value),
    video_description: document.getElementById('ai-video-desc').value.trim(),
    game_rules: document.getElementById('ai-game-rules').value.trim(),
    action_mode: document.getElementById('ai-action-mode').value,
    frame_source: document.getElementById('ai-frame-source').value,
    frame_window: parseInt(document.getElementById('ai-frame-window').value) || 0,
    hide_reward: document.getElementById('ai-hide-reward').checked,
    video_mode: document.getElementById('ai-video-mode').checked,
  };

  // Collect all game settings (left panel)
  const _gsIds = [
    'cfg-seed', 'cfg-repeat', 'cfg-noop-fill', 'cfg-action-repeat', 'cfg-max-steps',
    'cfg-frameskip', 'cfg-auto-respawn', 'cfg-initial-lives', 'cfg-end-on-life-loss',
    'cfg-skip-initial-steps', 'cfg-action-set', 'cfg-initial-ram', 'cfg-mode', 'cfg-difficulty',
    'cfg-paddle-width', 'cfg-ball-speed', 'cfg-brick-rows', 'cfg-brick-cols',
    'cfg-brick-offset', 'cfg-launch-angle-mode', 'cfg-cb-frameskip', 'cfg-cb-lives',
    'cfg-cb-end-on-life-loss', 'cfg-cb-skip-initial-steps',
    'cfg-partial-obs', 'cfg-slippery',
    'cfg-lc-size', 'cfg-lc-crossings',
    'cfg-mr-grid-size', 'cfg-mr-rooms', 'cfg-mr-room-size',
    'cfg-up-room-size', 'cfg-up-rows', 'cfg-up-cols',
    "cfg-game-mode","cfg-game-difficulty",
    "cfg-lanes-count","cfg-vehicles-density",
    "cfg-pre-score","cfg-procgen-skip-initial-steps",
  ];
  const game_settings = {};
  for (const id of _gsIds) {
    const el = document.getElementById(id);
    if (!el) continue;
    game_settings[id] = el.type === 'checkbox' ? el.checked : el.value;
  }
  config.game_settings = game_settings;

  // Collect all context frames + save item boundaries
  const allFrames = contextList.flatMap(item => item.frames);
  config.context_items = contextList.map(item => ({
    label: item.label,
    type: item.type,
    folder: item.folder || null,
    frame_count: item.frameCount,
  }));

  // Include loaded action sequence reference
  if (loadedSequence) {
    config.action_sequence = loadedSequence;  // { game_safe, name }
  }

  const body = { game_id: gameId, name, config, frames: allFrames };

  try {
    const resp = await fetch('/api/ai-configs/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const data = await resp.json();
    alert(`Config saved: ${data.path}`);
  } catch (err) {
    alert(`Failed to save config: ${err.message}`);
  }
}

async function showLoadConfigDialog() {
  const gameId = document.getElementById('game-select').value;
  if (!gameId) { alert('Select a game first'); return; }

  let allConfigs;
  try {
    const resp = await fetch('/api/ai-configs');
    allConfigs = await resp.json();
  } catch (err) {
    alert(`Failed to load configs: ${err.message}`);
    return;
  }

  const configs = allConfigs[gameId] || [];
  if (configs.length === 0) {
    alert('No saved configs for this game');
    return;
  }

  const overlay = document.getElementById('config-load-overlay');
  const list = document.getElementById('config-load-list');
  // Clear old entries
  while (list.firstChild) list.removeChild(list.firstChild);

  for (const cfg of configs) {
    // Frames entry
    if (cfg.frame_count > 0) {
      list.appendChild(_buildConfigRow(cfg.name, 'frames', `${cfg.frame_count} frames`, cfg.created_at));
    }

    // Video entry
    if (cfg.has_video) {
      list.appendChild(_buildConfigRow(cfg.name, 'video', 'video.mp4', cfg.created_at));
    }

    // Config-only (no frames, no video)
    if (cfg.frame_count === 0 && !cfg.has_video) {
      list.appendChild(_buildConfigRow(cfg.name, 'none', 'no context', cfg.created_at));
    }
  }

  overlay.classList.remove('hidden');
}

function _buildConfigRow(name, mode, detail, createdAt) {
  const row = document.createElement('div');
  row.className = 'config-load-item';

  const nameSpan = document.createElement('span');
  nameSpan.className = 'cfg-name';
  nameSpan.textContent = name;
  row.appendChild(nameSpan);

  const typeSpan = document.createElement('span');
  typeSpan.className = mode === 'video' ? 'cfg-type cfg-type-video' : 'cfg-type';
  typeSpan.textContent = mode === 'none' ? 'config only' : mode;
  row.appendChild(typeSpan);

  const detailSpan = document.createElement('span');
  detailSpan.className = 'cfg-detail';
  detailSpan.textContent = detail;
  row.appendChild(detailSpan);

  const dateSpan = document.createElement('span');
  dateSpan.className = 'cfg-date';
  dateSpan.textContent = createdAt ? createdAt.slice(0, 10) : '';
  row.appendChild(dateSpan);

  row.onclick = () => loadAIConfig(name, mode);
  return row;
}

function closeLoadConfigDialog() {
  document.getElementById('config-load-overlay').classList.add('hidden');
}

async function loadAIConfig(configName, mode) {
  // mode: 'frames' | 'video' | 'none'
  const gameId = document.getElementById('game-select').value;
  const gameSafe = gameId.replace(/\//g, '_').replace(/:/g,"_");

  closeLoadConfigDialog();

  try {
    // Fetch config.json
    const cfgResp = await fetch(`/api/ai-configs/${gameSafe}/${configName}/config`);
    if (!cfgResp.ok) throw new Error(`Config fetch failed: ${cfgResp.status}`);
    const configData = await cfgResp.json();

    // Fill hyperparams (api_key/base_url/model not touched — they come from config.json only)
    if (configData.temperature != null) document.getElementById('ai-temperature').value = configData.temperature;
    if (configData.max_tokens != null) document.getElementById('ai-max-tokens').value = configData.max_tokens;
    if (configData.video_description != null) document.getElementById('ai-video-desc').value = configData.video_description;
    const resolvedGameRules = showGameRulesFromConfig(configData, gameId);
    document.getElementById('ai-game-rules').value = resolvedGameRules;
    if (configData.frame_window != null) document.getElementById('ai-frame-window').value = configData.frame_window;
    if (configData.hide_reward != null) document.getElementById('ai-hide-reward').checked = configData.hide_reward;

    // Restore AI mode from action_mode + frame_source
    const am = configData.action_mode || 'natural_language';
    const fs = configData.frame_source || 'subbed_nl';
    let aiMode = 'subbed-nl';
    if (fs === 'original' && am === 'numeric') aiMode = 'original-numeric';
    else if (fs === 'original' && am === 'natural_language') aiMode = 'original-nl';
    else if (fs === 'subbed' && am === 'numeric') aiMode = 'subbed-numeric';
    else if (fs === 'subbed_nl' && am === 'natural_language') aiMode = 'subbed-nl';
    document.getElementById('ai-mode').value = aiMode;
    updateAIModeFields();

    // Clear existing context
    contextList = [];
    updateContextFrames();
    renderContextList();

    // Fetch frames (needed for both frames and video modes)
    const framesResp = await fetch(`/api/ai-configs/${gameSafe}/${configName}/frames`);
    const framesData = await framesResp.json();
    const frames = framesData.frames || [];

    // Restore context items: use saved boundaries or fallback to single blob
    const savedItems = configData.context_items;
    if (frames.length > 0) {
      if (savedItems && savedItems.length > 0) {
        // 按保存时的边界拆分为独立 context 项
        let offset = 0;
        for (const item of savedItems) {
          const itemFrames = frames.slice(offset, offset + item.frame_count);
          contextList.push({
            id: 'config_' + Date.now() + '_' + offset,
            label: item.label,
            type: item.type || 'upload',
            folder: item.folder || null,
            frames: itemFrames,
            frameCount: itemFrames.length,
          });
          offset += item.frame_count;
        }
      } else {
        // 旧格式兼容：无 context_items，作为单个 blob
        contextList.push({
          id: 'config_' + Date.now(),
          label: configName + ' (' + frames.length + (mode === 'video' ? 'f, video)' : ' frames)'),
          type: 'upload',
          frames: frames,
          frameCount: frames.length,
        });
      }
      updateContextFrames();
      renderContextList();
      showContextInPlayer(frames);
    }

    // Restore video mode
    if (configData.video_mode != null) {
      document.getElementById('ai-video-mode').checked = configData.video_mode;
    } else if (mode === 'video') {
      document.getElementById('ai-video-mode').checked = true;
    } else {
      document.getElementById('ai-video-mode').checked = false;
    }

    if (document.getElementById('ai-video-mode').checked) {
      const videoUrl = '/api/ai-configs/' + gameSafe + '/' + configName + '/video';
      const videoPreview = document.getElementById('context-video-preview');
      videoPreview.src = videoUrl;
      document.getElementById('context-video-preview-section').classList.remove('hidden');
    } else {
      document.getElementById('context-video-preview-section').classList.add('hidden');
    }

    // Restore game settings BEFORE startGame() so buildConfig() picks them up
    const _idCompat = { 'cfg-game-mode': 'cfg-mode', 'cfg-game-difficulty': 'cfg-difficulty' };
    if (configData.game_settings) {
      for (const [id, val] of Object.entries(configData.game_settings)) {
        const el = document.getElementById(_idCompat[id] || id);
        if (!el) continue;
        if (el.type === 'checkbox') {
          el.checked = val;
        } else {
          el.value = val;
        }
      }
    }

    // Auto-start game, then replay action sequence if present
    await startGame();  // applySessionData() clears loadedSequence
    if (configData.action_sequence && configData.action_sequence.name) {
      loadedSequence = configData.action_sequence;
      updateSeqBadge();
      // Actually replay the sequence so game reaches the saved state
      try {
        const seqResp = await fetch(`/api/action-sequences/${encodeURIComponent(loadedSequence.game_safe)}/${encodeURIComponent(loadedSequence.name)}/sequence`);
        if (seqResp.ok) {
          const seqData = await seqResp.json();
          await new Promise(resolve => {
            window._onReplayDone = resolve;
            ws.send(JSON.stringify({ type: 'replay_sequence', actions: seqData.actions, delay_ms: 80 }));
          });
        }
      } catch (e) {
        console.warn('Failed to replay sequence:', e);
      }
    }

  } catch (err) {
    alert('Failed to load config: ' + err.message);
  }
}
