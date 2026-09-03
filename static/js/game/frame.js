// ========== Frame Display ==========
function showFrame(b64) {
  const img = document.getElementById('game-frame');
  img.src = 'data:image/jpeg;base64,' + b64;
}

function showEvaluationSaveToast(message = '已经保存了', isError = false) {
  const toast = document.getElementById('evaluation-save-toast');
  if (!toast) return;
  toast.textContent = message;
  toast.classList.toggle('error', isError);
  toast.classList.remove('hidden');
  toast.style.animation = 'none';
  void toast.offsetWidth;
  toast.style.animation = '';
  clearTimeout(window._evaluationSaveToastTimer);
  window._evaluationSaveToastTimer = setTimeout(() => toast.classList.add('hidden'), 2500);
}

// ========== State Display ==========
function updateState(state) {
  lastState = state;
  document.getElementById('info-step').textContent = state.step;

  const cvHide = currentMode === 'crossval' && window.crossvalHideReward
                 && window.crossvalState && window.crossvalState.isPlaying;
  const rewardEl = document.getElementById('info-reward');
  const cumEl = document.getElementById('info-cumulative');
  const scoreEl = document.getElementById('info-score');
  if (cvHide) {
    rewardEl.textContent = '■■■';
    cumEl.textContent = '■■■';
    rewardEl.classList.add('cv-hidden-reward');
    cumEl.classList.add('cv-hidden-reward');
    if (scoreEl) {
      scoreEl.textContent = '■■■';
      scoreEl.classList.add('cv-hidden-reward');
    }
  } else {
    rewardEl.textContent = (state.reward ?? 0).toFixed(2);
    cumEl.textContent = (state.cumulative_reward ?? 0).toFixed(2);
    rewardEl.classList.remove('cv-hidden-reward');
    cumEl.classList.remove('cv-hidden-reward');
    // Score：直接读后端 compute_score 注入的 state.info.score。
    //   - REWARD_AS_SCORE 游戏（18 个）→ 等价 cumulative reward
    //   - 9 个自定义规则游戏 → 归一到 [0,100]
    //   - 其它游戏 → 后端不写该字段，前端显示 "-"
    if (scoreEl) {
      const scoreVal = state.info && state.info.score;
      if (typeof scoreVal === 'number' && isFinite(scoreVal)) {
        scoreEl.textContent = scoreVal.toFixed(2);
        scoreEl.style.color = '';
      } else {
        scoreEl.textContent = '-';
        scoreEl.style.color = '#999';
      }
      scoreEl.classList.remove('cv-hidden-reward');
    }
  }
  document.getElementById('info-status').textContent = state.is_game_over ? 'Game Over' : 'Playing';
  document.getElementById('info-status').style.color = state.is_game_over ? '#ef5350' : '#81c784';

  // Ending / Pass — 由后端 GameSession.step() 在 game_over 时写入 state.info
  const endingEl = document.getElementById('info-ending');
  const passEl = document.getElementById('info-pass');
  const ending = state.info && state.info.ending;
  const passed = state.info && state.info.pass;
  if (state.is_game_over === true && ending) {
    endingEl.textContent = ending;
    endingEl.style.color = (
      ending === 'victory'  ? '#81c784' :
      ending === 'timeout'  ? '#64b5f6' :
                              '#ef5350'   // gameover
    );
  } else {
    endingEl.textContent = '-';
    endingEl.style.color = '#999';
  }
  if (state.is_game_over === true && (passed === true || passed === false)) {
    passEl.textContent = passed ? 'True' : 'False';
    passEl.style.color = passed ? '#81c784' : '#ef5350';
  } else {
    passEl.textContent = '-';
    passEl.style.color = '#999';
  }

  // Progress：展示 event_counts 原始计数（每个事件占一行 "name: N"）
  const progressEl = document.getElementById('info-progress');
  if (progressEl) {
    const ec = state.info && state.info.event_counts;
    if (ec && Object.keys(ec).length > 0) {
      progressEl.innerHTML = Object.entries(ec)
        .map(([k, v]) => `${k}: ${v}`)
        .join('<br>');
      progressEl.style.color = '#81c784';
    } else {
      progressEl.textContent = '-';
      progressEl.style.color = '#999';
    }
  }

  // RAM info
  const ramDiv = document.getElementById('info-ram');
  ramDiv.innerHTML = '';
  if (state.ram_info) {
    for (const [k, v] of Object.entries(state.ram_info)) {
      const row = document.createElement('div');
      row.className = 'info-row';
      row.innerHTML = `<span class="info-label">${k}</span><span class="info-value">${v}</span>`;
      ramDiv.appendChild(row);
    }
  }

  // Update totalFrames
  if (state.total_frames) {
    totalFrames = state.total_frames;
    if (isLive) updateProgress(totalFrames - 1, totalFrames);
  }

  // Human evaluation save result (non-blocking)
  if (currentMode === 'human' && state.is_game_over && !lastGameOver && !isReplaying) {
    showEvaluationSaveToast(
      state.human_eval_save_error ? '保存失败' : '已经保存了',
      !!state.human_eval_save_error,
    );
  }
  // Cross-validation: only show a non-blocking save notification
  if (currentMode === 'crossval' && state.is_game_over && !lastGameOver && !isReplaying) {
    const reason = ending || ((state.info && state.info.truncated) ? 'timeout' : 'gameover');
    if (typeof crossvalShowReveal === 'function') {
      crossvalShowReveal(reason, passed, state.human_eval_save_error);
    }
  }
  lastGameOver = !!state.is_game_over;
}

// ========== Progress Bar ==========
function updateProgress(current, total) {
  const bar = document.getElementById('progress-bar');
  bar.max = Math.max(0, total - 1);
  bar.value = current;
  document.getElementById('progress-label').textContent = `${current} / ${Math.max(0, total - 1)}`;
}

function onProgressChange(val) {
  isLive = false;
  const idx = parseInt(val);
  fetchFrame(idx);
}

function goFirst() {
  isLive = false;
  fetchFrame(0);
}

function goPrev() {
  isLive = false;
  const bar = document.getElementById('progress-bar');
  const idx = Math.max(0, parseInt(bar.value) - 1);
  fetchFrame(idx);
}

function goNext() {
  const bar = document.getElementById('progress-bar');
  const idx = parseInt(bar.value) + 1;
  const max = parseInt(bar.max);
  if (idx >= max) {
    goLive();
  } else {
    isLive = false;
    fetchFrame(idx);
  }
}

function goLive() {
  isLive = true;
  fetchFrame(totalFrames - 1);
}

function fetchFrame(idx) {
  if (currentMode === 'human') {
    // Human mode: always fetch from session via WebSocket
    fetchHumanFrame(idx);
  } else if (currentMode === 'ai') {
    // AI mode: context preview or AI session
    fetchAIFrame(idx);
  }
}

function fetchHumanFrame(idx) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'get_frame', index: idx }));
  }
}

function fetchAIFrame(idx) {
  // If AI is running (WebSocket active), fetch session frames
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'get_frame', index: idx }));
    return;
  }

  // 优先使用AI游戏运行时的帧（回放AI游戏）
  if (aiSessionFrames.length > 0) {
    idx = Math.max(0, Math.min(idx, aiSessionFrames.length - 1));
    showFrame(aiSessionFrames[idx]);
    updateProgress(idx, aiSessionFrames.length);
    return;
  }

  // 否则使用预览帧（最近一次上传/加载的 context）
  if (previewFrames.length > 0) {
    idx = Math.max(0, Math.min(idx, previewFrames.length - 1));
    showFrame(previewFrames[idx]);
    updateProgress(idx, previewFrames.length);
  }
}

