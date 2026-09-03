// ========== Cross-Validation Loader ==========
// 流程：左栏显式选关（或继续下一个未完成）→ briefing → Start Playing
//      → 终局自动保存 → 用户点“下一个测试”进入下一项。
//
// 全局 crossval state（命名空间挂在 window 上以避免冲突）
window.crossvalState = {
  loadedConfigName: null,        // string
  loadedConfigData: null,        // 完整 config.json 对象
  loadedGameId: null,            // 'ALE/Seaquest-v5'
  loadedGameSafe: null,          // 'ALE_Seaquest-v5'
  segments: [],                  // [{index, label, frame_count, url}]
  currentSeg: 0,                 // 当前播放段下标
  watchedSegs: null,             // Set<int>
  isPlaying: false,              // 是否在 playing 阶段
  queueIndex: null,
  evaluationDone: false,
};

const CROSSVAL_PROGRESS_KEY = 'video_cl_crossval_completed_tasks_v3';
window.crossvalQueue = [];
window.crossvalQueueIndex = 0;
window.crossvalQueueReady = null;
window.crossvalCompletedTasks = new Set();

function _crossvalTaskKey(task) {
  return `${task.gameId}\u0000${task.name}`;
}

async function crossvalInitializeQueue(forceReload = false) {
  if (forceReload) window.crossvalQueueReady = null;
  if (window.crossvalQueueReady) return window.crossvalQueueReady;
  window.crossvalQueueReady = (async () => {
    const resp = await fetch('/api/ai-configs');
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const allConfigs = await resp.json();
    const queue = [];
    const gameSelect = document.getElementById('game-select');
    const gameOrder = (typeof games !== 'undefined' && games.length)
      ? games.slice()
          .sort((a, b) => Number(a.key) - Number(b.key))
          .map(game => game.game_id)
      : gameSelect
        ? Array.from(gameSelect.options)
            .sort((a, b) => Number(a.textContent.split('.')[0]) - Number(b.textContent.split('.')[0]))
            .map(option => option.value)
        : Object.keys(allConfigs || {});
    for (const gameId of gameOrder) {
      for (const cfg of (allConfigs[gameId] || [])) {
        const gameMeta = (typeof games !== 'undefined' ? games : [])
          .find(game => game.game_id === gameId);
        queue.push({
          gameId,
          pageId: gameMeta?.key ?? '',
          name: cfg.name,
          mode: cfg.frame_count > 0 ? 'frames' : cfg.has_video ? 'video' : 'none',
        });
      }
    }
    window.crossvalQueue = queue;
    try {
      const saved = JSON.parse(localStorage.getItem(CROSSVAL_PROGRESS_KEY) || '[]');
      window.crossvalCompletedTasks = new Set(Array.isArray(saved) ? saved : []);
    } catch (_) {
      window.crossvalCompletedTasks = new Set();
    }
    const firstIncomplete = queue.findIndex(task =>
      !window.crossvalCompletedTasks.has(_crossvalTaskKey(task)));
    window.crossvalQueueIndex = firstIncomplete >= 0 ? firstIncomplete : queue.length;
    crossvalRenderLevelSelector();
    return queue;
  })().catch(err => {
    window.crossvalQueueReady = null;
    throw err;
  });
  return window.crossvalQueueReady;
}

function crossvalRenderLevelSelector() {
  const list = document.getElementById('cv-level-list');
  const progressText = document.getElementById('cv-level-progress-text');
  const progressFill = document.getElementById('cv-level-progress-fill');
  const continueBtn = document.getElementById('cv-level-continue');
  if (!list) return;

  const queue = window.crossvalQueue || [];
  const completed = queue.filter(task =>
    window.crossvalCompletedTasks.has(_crossvalTaskKey(task))).length;
  if (progressText) progressText.textContent = `${completed} / ${queue.length} 已完成`;
  if (progressFill) {
    progressFill.style.width = queue.length ? `${completed / queue.length * 100}%` : '0%';
  }
  if (continueBtn) {
    continueBtn.disabled = !!window.crossvalState?.isPlaying || completed >= queue.length;
    continueBtn.textContent = completed >= queue.length ? '全部测试已完成' : '继续下一个未完成';
  }

  list.innerHTML = '';
  if (!queue.length) {
    const empty = document.createElement('p');
    empty.className = 'cv-level-empty';
    empty.textContent = '没有找到可评测的 AI 配置。';
    list.appendChild(empty);
    return;
  }

  let previousGame = null;
  queue.forEach((task, index) => {
    if (task.gameId !== previousGame) {
      const label = document.createElement('div');
      label.className = 'cv-level-game-label';
      label.textContent = `${task.pageId ? `${task.pageId}. ` : ''}${task.gameId}`;
      list.appendChild(label);
      previousGame = task.gameId;
    }

    const key = _crossvalTaskKey(task);
    const done = window.crossvalCompletedTasks.has(key);
    const active = window.crossvalState?.queueIndex === index
      && !!window.crossvalState?.loadedConfigName;
    const button = document.createElement('button');
    button.type = 'button';
    button.className = `cv-level-item${done ? ' is-done' : ''}${active ? ' is-active' : ''}`;
    button.disabled = !!window.crossvalState?.isPlaying;
    button.title = done ? '已完成；点击可重新评测并覆盖 run_0' : '加载这项测试';
    button.onclick = () => crossvalSelectTask(index);

    const id = document.createElement('span');
    id.className = 'cv-level-id';
    id.textContent = task.pageId ? `#${task.pageId}` : '—';
    const name = document.createElement('span');
    name.className = 'cv-level-name';
    name.textContent = task.name;
    const status = document.createElement('span');
    status.className = 'cv-level-status';
    status.textContent = active ? '当前' : done ? '✓' : '○';
    button.append(id, name, status);
    list.appendChild(button);
  });
}

function _crossvalSetSelectedGame(gameId) {
  const sel = document.getElementById('game-select');
  if (!sel) return;
  sel.value = gameId;
  if (typeof onGameChange === 'function') onGameChange();
}

async function crossvalNextTest() {
  const nextBtn = document.getElementById('cv-btn-next-test');
  const afterBtn = document.getElementById('cv-btn-next-after-run');
  const continueBtn = document.getElementById('cv-level-continue');
  if (nextBtn) nextBtn.disabled = true;
  if (afterBtn) afterBtn.disabled = true;
  if (continueBtn) continueBtn.disabled = true;
  try {
    const queue = await crossvalInitializeQueue(true);
    const idx = window.crossvalQueueIndex;
    if (idx >= queue.length) {
      showEvaluationSaveToast('所有测试已经完成了');
      return;
    }
    await crossvalSelectTask(idx);
  } catch (err) {
    showEvaluationSaveToast('加载下一个测试失败', true);
    console.error('[CrossVal] next test failed:', err);
  } finally {
    if (nextBtn) nextBtn.disabled = false;
    if (afterBtn) afterBtn.disabled = false;
    crossvalRenderLevelSelector();
  }
}

async function crossvalSelectTask(index) {
  const queue = await crossvalInitializeQueue();
  const task = queue[index];
  if (!task || window.crossvalState?.isPlaying) return;

  if (sessionId || ws) cleanupCurrentMode();
  crossvalClearState();
  document.getElementById('frame-container')?.classList.add('hidden');
  document.getElementById('progress-bar-container')?.classList.add('hidden');
  _crossvalSetSelectedGame(task.gameId);
  await crossvalLoadConfig(task.name, task.mode, index);
}

function crossvalMarkEvaluationComplete(saveError = null) {
  const cv = window.crossvalState;
  if (cv.evaluationDone) return;
  cv.evaluationDone = true;
  // 终局已经冻结，允许“下一个测试”清理当前 session 并加载队列下一项。
  // 若保留 true，crossvalSelectTask() 会把它误判为仍在游玩而直接 return。
  cv.isPlaying = false;
  if (!saveError && Number.isInteger(cv.queueIndex)) {
    const currentTask = window.crossvalQueue[cv.queueIndex];
    if (currentTask) {
      window.crossvalCompletedTasks.add(_crossvalTaskKey(currentTask));
      localStorage.setItem(
        CROSSVAL_PROGRESS_KEY,
        JSON.stringify(Array.from(window.crossvalCompletedTasks)),
      );
    }
    const nextIndex = window.crossvalQueue.findIndex((task, index) =>
      index > cv.queueIndex && !window.crossvalCompletedTasks.has(_crossvalTaskKey(task)));
    window.crossvalQueueIndex = nextIndex >= 0 ? nextIndex : window.crossvalQueue.length;
  }
  const nextBtn = document.getElementById('cv-btn-next-after-run');
  if (nextBtn) nextBtn.classList.remove('hidden');
  document.getElementById('cv-btn-end-run')?.classList.add('hidden');
  document.getElementById('cv-btn-abort-run')?.classList.add('hidden');
  document.getElementById('cv-btn-rewatch-run')?.classList.add('hidden');
  crossvalRenderLevelSelector();
}

// LEFT-PANEL 锁定
const _CV_LOCK_PANELS = ['standard-left-content'];

function crossvalLockPanel() {
  for (const id of _CV_LOCK_PANELS) {
    const el = document.getElementById(id);
    if (el) el.classList.add('locked-panel');
  }
}

function crossvalUnlockPanel() {
  for (const id of _CV_LOCK_PANELS) {
    const el = document.getElementById(id);
    if (el) el.classList.remove('locked-panel');
  }
}

function crossvalClearState() {
  window.crossvalState = {
    loadedConfigName: null,
    loadedConfigData: null,
    loadedGameId: null,
    loadedGameSafe: null,
    segments: [],
    currentSeg: 0,
    watchedSegs: null,
    isPlaying: false,
    queueIndex: null,
    evaluationDone: false,
  };
  if (typeof clearCrossvalHudMask === 'function') clearCrossvalHudMask();
  // 重置 UI 状态
  const briefing = document.getElementById('crossval-briefing');
  const empty = document.getElementById('crossval-empty');
  const playCtl = document.getElementById('crossval-play-controls');
  if (briefing) briefing.classList.add('hidden');
  if (playCtl) playCtl.classList.add('hidden');
  if (empty && currentMode === 'crossval') empty.classList.remove('hidden');
  // brief 文本清空
  const ids = ['cv-brief-config','cv-brief-game','cv-brief-seed','cv-brief-action-mode',
               'cv-brief-frame-source','cv-brief-hide-reward','cv-brief-action-set','cv-brief-rules',
               'cv-briefing-config-name'];
  for (const id of ids) {
    const el = document.getElementById(id);
    if (el) el.textContent = '—';
  }
  // 重置 video element
  const v = document.getElementById('cv-demo-video');
  if (v) { v.pause(); v.onended = null; v.removeAttribute('src'); }
  // 清段切换 UI
  const tabs = document.getElementById('cv-segment-tabs');
  if (tabs) tabs.innerHTML = '';
  const lbl = document.getElementById('cv-segment-current-label');
  if (lbl) lbl.textContent = '—';
  const prog = document.getElementById('cv-segment-progress');
  if (prog) prog.textContent = '0 / 0';
  const toast = document.getElementById('cv-segment-toast');
  if (toast) { toast.classList.add('hidden'); toast.style.animation = ''; }
  document.getElementById('cv-btn-next-after-run')?.classList.add('hidden');
  document.getElementById('cv-btn-end-run')?.classList.remove('hidden');
  document.getElementById('cv-btn-abort-run')?.classList.remove('hidden');
  document.getElementById('cv-btn-rewatch-run')?.classList.remove('hidden');
  const currentGameId = document.getElementById('game-select')?.value;
  if (currentGameId) showGameRulesFromConfig(null, currentGameId);
  crossvalRenderLevelSelector();
}

async function crossvalLoadConfig(configName, configMode, queueIndex = null) {
  const gameId = document.getElementById('game-select').value;
  const gameSafe = gameId.replace(/\//g, '_').replace(/:/g, '_');

  let configData;
  try {
    const cfgResp = await fetch(`/api/ai-configs/${gameSafe}/${configName}/config`);
    if (!cfgResp.ok) throw new Error(`Config fetch failed: ${cfgResp.status}`);
    configData = await cfgResp.json();
  } catch (err) {
    throw new Error('Failed to load config: ' + err.message);
  }

  // 1) 把 game_settings 应用到左侧 panel (供后续 startGame() 的 buildConfig 读取)
  // 先恢复表单的初始值，避免当前配置未提供的可选字段沿用上一配置的值。
  document.querySelectorAll(
    '#standard-left-content input[id^="cfg-"], ' +
    '#standard-left-content select[id^="cfg-"], ' +
    '#standard-left-content textarea[id^="cfg-"]',
  ).forEach(el => {
    if (el.type === 'checkbox' || el.type === 'radio') {
      el.checked = el.defaultChecked;
    } else if (el.tagName === 'SELECT') {
      for (const option of el.options) option.selected = option.defaultSelected;
      if (el.selectedIndex < 0 && el.options.length) el.selectedIndex = 0;
    } else {
      el.value = el.defaultValue;
    }
  });

  const _idCompat = { 'cfg-game-mode': 'cfg-mode', 'cfg-game-difficulty': 'cfg-difficulty' };
  if (configData.game_settings) {
    for (const [id, val] of Object.entries(configData.game_settings)) {
      const el = document.getElementById(_idCompat[id] || id);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = !!val;
      else el.value = val;
    }
  }

  // 2) 设置 crossval 全局状态
  window.crossvalState.loadedConfigName = configName;
  window.crossvalState.loadedConfigData = configData;
  window.crossvalState.loadedGameId = gameId;
  window.crossvalState.loadedGameSafe = gameSafe;
  window.crossvalState.queueIndex = queueIndex === null
    ? window.crossvalQueueIndex
    : queueIndex;
  window.crossvalState.evaluationDone = false;
  window.crossvalState.videoWatched = false;
  window.crossvalState.isPlaying = false;
  window.crossvalHideReward = !!configData.hide_reward;
  crossvalRenderLevelSelector();

  // 3) 填充 brief panel
  document.getElementById('cv-brief-config').textContent = configName;
  document.getElementById('cv-brief-game').textContent = gameId;
  const seedVal = (configData.game_settings && configData.game_settings['cfg-seed']) || '(random)';
  document.getElementById('cv-brief-seed').textContent = seedVal === '' ? '(random)' : seedVal;
  document.getElementById('cv-brief-action-mode').textContent = configData.action_mode || 'natural_language';
  document.getElementById('cv-brief-frame-source').textContent = configData.frame_source || 'subbed_nl';
  document.getElementById('cv-brief-hide-reward').textContent = configData.hide_reward ? 'true' : 'false';
  const actionSet = (configData.game_settings && configData.game_settings['cfg-action-set']) || '—';
  document.getElementById('cv-brief-action-set').textContent = actionSet;
  // AI 配置中的规则优先；没有配置时才回退项目 game_rules.json。
  const rulesText = showGameRulesFromConfig(configData, gameId);
  document.getElementById('cv-brief-rules').textContent = rulesText;
  document.getElementById('cv-briefing-config-name').textContent = `${gameId} · ${configName}`;

  // 4) 切到 briefing 状态(在加载 segments 前先显示出来)
  document.getElementById('crossval-empty').classList.add('hidden');
  document.getElementById('crossval-briefing').classList.remove('hidden');

  // 5) 锁定左侧 panel
  crossvalLockPanel();

  // 6) 加载 segments 元信息(与 AI 实际看到的多段视频边界对齐)
  let segments = [];
  try {
    const segResp = await fetch(`/api/ai-configs/${gameSafe}/${configName}/segments`);
    if (segResp.ok) {
      const segData = await segResp.json();
      segments = (segData.segments || []).map(s => ({
        index: s.index,
        label: s.label,
        frame_count: s.frame_count,
        url: `/api/ai-configs/${gameSafe}/${configName}/segment/${s.index}`,
      }));
    }
  } catch (e) {
    console.warn('[Crossval] failed to fetch segments:', e);
  }

  window.crossvalState.segments = segments;
  window.crossvalState.currentSeg = 0;
  window.crossvalState.watchedSegs = new Set();

  // 7) 渲染段按钮 + 绑定视频事件
  _crossvalRenderSegmentTabs();
  _crossvalBindVideoEvents();

  if (segments.length > 0) {
    _crossvalLoadSegment(0, /*autoplay=*/true);
  } else {
    // 没有 demo 段:直接放行
    document.getElementById('cv-demo-video').removeAttribute('src');
    document.getElementById('cv-segment-current-label').textContent = '(无演示)';
    document.getElementById('cv-segment-progress').textContent = '0 / 0';
    _crossvalUpdateStartGate(true);
  }
  _crossvalUpdateNavButtons();
}

// ---------- 多段视频核心 ----------
function _crossvalRenderSegmentTabs() {
  const tabs = document.getElementById('cv-segment-tabs');
  tabs.innerHTML = '';
  const segs = window.crossvalState.segments;
  for (const s of segs) {
    const btn = document.createElement('button');
    btn.className = 'cv-seg-btn';
    btn.textContent = String(s.index + 1);
    btn.title = s.label;
    btn.onclick = () => _crossvalLoadSegment(s.index, /*autoplay=*/true);
    tabs.appendChild(btn);
  }
  _crossvalUpdateSegmentTabs();
}

function _crossvalUpdateSegmentTabs() {
  const tabs = document.getElementById('cv-segment-tabs').children;
  const cur = window.crossvalState.currentSeg;
  const watched = window.crossvalState.watchedSegs || new Set();
  for (let i = 0; i < tabs.length; i++) {
    tabs[i].classList.toggle('active', i === cur);
    tabs[i].classList.toggle('watched', watched.has(i));
  }
}

function _crossvalBindVideoEvents() {
  const v = document.getElementById('cv-demo-video');
  // 防止重复绑定
  v.onended = () => {
    const cv = window.crossvalState;
    cv.watchedSegs.add(cv.currentSeg);
    _crossvalUpdateSegmentTabs();

    const next = cv.currentSeg + 1;
    if (next < cv.segments.length) {
      _crossvalShowSegmentToast(`已自动播放第 ${next + 1} 段：${cv.segments[next].label}`);
      _crossvalLoadSegment(next, /*autoplay=*/true);
    } else {
      // 全部看完
      _crossvalRefreshGate();
    }
    _crossvalUpdateNavButtons();
  };
}

function _crossvalLoadSegment(idx, autoplay) {
  const cv = window.crossvalState;
  if (idx < 0 || idx >= cv.segments.length) return;
  cv.currentSeg = idx;
  const seg = cv.segments[idx];
  const v = document.getElementById('cv-demo-video');
  v.muted = true;
  v.src = seg.url;
  if (autoplay) {
    v.play().catch(() => {});
  }
  document.getElementById('cv-segment-current-label').textContent =
    `第 ${idx + 1} / ${cv.segments.length} 段：${seg.label} (${seg.frame_count} 帧)`;
  document.getElementById('cv-segment-progress').textContent = `${idx + 1} / ${cv.segments.length}`;
  _crossvalUpdateSegmentTabs();
  _crossvalUpdateNavButtons();
}

function _crossvalShowSegmentToast(text) {
  const t = document.getElementById('cv-segment-toast');
  if (!t) return;
  t.textContent = text;
  t.classList.remove('hidden');
  // 重启动画
  t.style.animation = 'none';
  void t.offsetWidth;
  t.style.animation = '';
}

function _crossvalUpdateNavButtons() {
  const cv = window.crossvalState;
  const prev = document.getElementById('cv-btn-prev-seg');
  const next = document.getElementById('cv-btn-next-seg');
  if (!prev || !next) return;
  prev.disabled = cv.currentSeg <= 0;
  next.disabled = cv.currentSeg >= cv.segments.length - 1;
}

function _crossvalUpdateStartGate(_unlocked) {
  const startBtn = document.getElementById('cv-btn-start-play');
  const watchStatus = document.getElementById('cv-briefing-watch-status');
  startBtn.disabled = false;
  const cv = window.crossvalState;
  const total = (cv.segments || []).length;
  const seen = cv.watchedSegs ? cv.watchedSegs.size : 0;
  if (total === 0) {
    watchStatus.textContent = '(无演示片段)';
    watchStatus.classList.remove('watched');
  } else if (seen >= total) {
    watchStatus.textContent = '✓ 所有片段已观看完毕';
    watchStatus.classList.add('watched');
  } else {
    watchStatus.textContent = `已观看 ${seen} / ${total} 段(可随时开始)`;
    watchStatus.classList.remove('watched');
  }
}

function _crossvalRefreshGate() {
  _crossvalUpdateStartGate(true);
}

function crossvalPrevSegment() {
  const cv = window.crossvalState;
  if (cv.currentSeg > 0) _crossvalLoadSegment(cv.currentSeg - 1, /*autoplay=*/true);
}

function crossvalNextSegment() {
  const cv = window.crossvalState;
  if (cv.currentSeg < cv.segments.length - 1) _crossvalLoadSegment(cv.currentSeg + 1, /*autoplay=*/true);
}

function crossvalRewatchDemo() {
  const cv = window.crossvalState;
  if (!cv.segments || cv.segments.length === 0) {
    alert('该配置没有演示视频');
    return;
  }

  if (cv.isPlaying) {
    // playing 阶段:用 overlay,从第一段开始顺播
    const overlay = document.getElementById('crossval-rewatch-overlay');
    const v = document.getElementById('cv-rewatch-video');
    overlay.classList.remove('hidden');
    let idx = 0;
    const playNext = () => {
      if (idx >= cv.segments.length) {
        v.onended = null;
        return;
      }
      v.src = cv.segments[idx].url;
      v.currentTime = 0;
      idx += 1;
      v.play().catch(() => {});
    };
    v.onended = playNext;
    playNext();
  } else {
    // briefing 阶段:从当前段开始重看
    const v = document.getElementById('cv-demo-video');
    v.currentTime = 0;
    v.play().catch(()=>{});
  }
}

function crossvalCloseRewatch() {
  const overlay = document.getElementById('crossval-rewatch-overlay');
  const v = document.getElementById('cv-rewatch-video');
  v.pause(); v.removeAttribute('src');
  v.onended = null;
  overlay.classList.add('hidden');
}

async function crossvalStartPlaying() {
  // 切换中央 panel: briefing → frame_container
  document.getElementById('crossval-briefing').classList.add('hidden');
  document.getElementById('frame-container').classList.remove('hidden');
  document.getElementById('progress-bar-container').classList.remove('hidden');
  document.getElementById('crossval-play-controls').classList.remove('hidden');
  window.crossvalState.isPlaying = true;
  crossvalRenderLevelSelector();

  // startGame() 内部已 await connectWS() onopen，回调返回时 ws 必已 OPEN
  try {
    await startGame();
  } catch (e) {
    console.error('[Crossval] startGame failed:', e);
    return;
  }
  if (!sessionId) return;  // startGame 失败时不会更新 sessionId

  // 与 AI mode 对齐:若 config 带 action_sequence,先回放到 AI 起点再交给人类
  const configData = window.crossvalState.loadedConfigData;
  if (configData && configData.action_sequence && configData.action_sequence.name) {
    loadedSequence = configData.action_sequence;
    if (typeof updateSeqBadge === 'function') updateSeqBadge();
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
      console.warn('[Crossval] Failed to replay action_sequence:', e);
    }
  }

  // 应用 HUD 遮蔽
  if (typeof applyCrossvalHudMask === 'function') applyCrossvalHudMask();
}

function crossvalAbort() {
  if (!confirm('确定中止本次代入测试? 当前进度将丢失。')) return;
  // 清理 session
  if (sessionId) {
    fetch(`/api/session/${sessionId}`, { method: 'DELETE' }).catch(()=>{});
    sessionId = null;
  }
  if (ws) { ws.close(); ws = null; }
  // 回到 empty
  crossvalClearState();
  crossvalUnlockPanel();
  document.getElementById('crossval-play-controls').classList.add('hidden');
  document.getElementById('frame-container').classList.add('hidden');
  document.getElementById('progress-bar-container').classList.add('hidden');
  // 重置 game frame
  document.getElementById('game-frame').classList.add('hidden');
}
