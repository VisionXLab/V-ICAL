function undoAction() {
  if (!ws || ws.readyState !== WebSocket.OPEN) return;
  isLive = true;
  ws.send(JSON.stringify({ type: 'undo' }));
}

let _endingTypeResolve = null;

function showEndingTypeModal() {
  return new Promise((resolve) => {
    _endingTypeResolve = resolve;

    // 自动推荐逻辑
    let recommended = 'demo_end';
    if (lastState && lastState.is_game_over) {
      if (lastState.info && lastState.info.truncated) {
        recommended = 'time_out';
      } else {
        recommended = 'game_over';
      }
    }

    // 清除旧推荐，设置新推荐
    document.querySelectorAll('.ending-card').forEach(card => {
      card.classList.toggle('recommended', card.dataset.type === recommended);
    });

    document.getElementById('ending-type-overlay').classList.remove('hidden');
  });
}

function resolveEndingType(type) {
  document.getElementById('ending-type-overlay').classList.add('hidden');
  if (_endingTypeResolve) {
    _endingTypeResolve(type);
    _endingTypeResolve = null;
  }
}

async function saveSession() {
  if (!sessionId) return;

  const endingType = await showEndingTypeModal();
  if (endingType === null) return;  // cancelled

  // 默认名称：游戏名_时间戳
  const gameName = document.getElementById('game-select').value.replace('ALE/', '').replace(/\//g, '_');
  const now = new Date();
  const ts = now.getFullYear()
    + String(now.getMonth() + 1).padStart(2, '0')
    + String(now.getDate()).padStart(2, '0') + '_'
    + String(now.getHours()).padStart(2, '0')
    + String(now.getMinutes()).padStart(2, '0')
    + String(now.getSeconds()).padStart(2, '0');
  const defaultName = `${gameName}_${ts}`;
  const customName = prompt('Session name:', defaultName);
  if (customName === null) return;  // cancelled

  const chatHtml = document.getElementById('chat-messages').innerHTML;
  const resp = await fetch(`/api/session/${sessionId}/save`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ mode: currentMode, chat_html: chatHtml, ending_type: endingType, custom_name: customName }),
  });
  const data = await resp.json();
  let msg = 'Saved to: ' + data.path + '\nFrames: ' + data.frames_saved + ', Actions: ' + data.actions_saved + '\nVideo: ' + (data.video || 'N/A');
  if (data.full_path) {
    msg += '\n\nFull version (with sequence): ' + data.full_path;
  }
  alert(msg);
  await loadContextFolderList();
}

async function loadContextFolderList() {
  const resp = await fetch('/api/human-operations');
  const ops = await resp.json();
  const sel = document.getElementById('ai-context-folder');
  const currentGame = document.getElementById('game-select').value;

  // 只显示匹配当前游戏的操作（game_name 为空的旧数据也保留显示）
  const filtered = ops.filter(op => !op.game_name || op.game_name === currentGame);

  // Clear existing options
  while (sel.options.length > 0) sel.remove(0);
  const defaultOpt = document.createElement('option');
  defaultOpt.value = '';
  defaultOpt.textContent = filtered.length ? '-- select --' : '-- no saved sessions for this game --';
  sel.appendChild(defaultOpt);
  for (const op of filtered) {
    const opt = document.createElement('option');
    opt.value = (op.source || 'human') + '/' + op.name;
    const tag = op.source === 'ai' ? '[AI]' : '[Human]';
    opt.textContent = `${tag} ${op.name} (${op.frame_count} frames)`;
    sel.appendChild(opt);
  }
}

async function loadContextFolder() {
  const folder = document.getElementById('ai-context-folder').value;
  if (!folder) { alert('Select a folder first'); return; }

  const videoMode = document.getElementById('ai-video-mode').checked;
  const frameSource = document.getElementById('ai-frame-source').value;

  console.log('[loadContextFolder] videoMode:', videoMode, 'frameSource:', frameSource, 'folder:', folder);

  // 加载帧数据
  const framesResp = await fetch(`/api/operations/${folder}/frames?source=${frameSource}`);
  const framesData = await framesResp.json();
  const frames = framesData.frames || [];

  if (frames.length === 0) { alert('No frames found in this folder'); return; }

  // 生成列表项标签
  const sel = document.getElementById('ai-context-folder');
  const label = sel.selectedOptions[0]?.textContent || folder;

  // 添加到列表
  contextList.push({
    id: 'folder_' + Date.now(),
    label: label,
    type: 'folder',
    folder: folder,
    frames: frames,
    frameCount: frames.length,
  });
  updateContextFrames();
  renderContextList();

  // 记录最近一次 folder（用于 regenerateContextVideo）
  pendingContextFolder = folder;

  if (videoMode) {
    // 视频模式：显示视频预览（仅最近一项）
    const actionMode = document.getElementById('ai-action-mode').value;
    const videoFilename = actionMode === 'natural_language' ? 'video_nl.mp4' : 'video.mp4';
    const [source, name] = folder.includes('/') ? folder.split('/') : ['human', folder];
    const videoUrl = `/api/video/${source}/${name}/${videoFilename}`;
    const videoPreview = document.getElementById('context-video-preview');
    videoPreview.src = videoUrl;
    document.getElementById('context-video-preview-section').classList.remove('hidden');
  } else {
    document.getElementById('context-video-preview-section').classList.add('hidden');
  }

  showContextInPlayer(frames);
}

function removeContextItem(id) {
  contextList = contextList.filter(item => item.id !== id);
  updateContextFrames();
  renderContextList();
  if (contextFrames.length > 0) {
    showContextInPlayer();
  } else {
    // 没有 context 了，清空预览
    document.getElementById('context-video-preview-section').classList.add('hidden');
  }
}

function updateContextFrames() {
  contextFrames = contextList.flatMap(item => item.frames);
}

function renderContextList() {
  const container = document.getElementById('context-list');
  if (contextList.length === 0) {
    container.innerHTML = '';
    document.getElementById('context-count').textContent = '';
    return;
  }
  const totalFrames_ = contextList.reduce((s, it) => s + it.frameCount, 0);
  document.getElementById('context-count').textContent = `${contextList.length} context(s), ${totalFrames_} frames total`;
  container.innerHTML = contextList.map(item => `
    <div class="context-item">
      <span class="ctx-label" title="${item.label}">${item.label}</span>
      <span class="ctx-count">${item.frameCount}f</span>
      <button class="ctx-del" onclick="removeContextItem('${item.id}')" title="Remove">&times;</button>
    </div>
  `).join('');
}

function updateAIModeFields() {
  const mode = document.getElementById('ai-mode').value;

  // Parse mode: "original-numeric", "original-nl", "subbed-numeric", "subbed-nl"
  let frameSource, actionMode;

  if (mode === 'original-numeric') {
    frameSource = 'original';
    actionMode = 'numeric';
  } else if (mode === 'original-nl') {
    frameSource = 'original';
    actionMode = 'natural_language';
  } else if (mode === 'subbed-numeric') {
    frameSource = 'subbed';
    actionMode = 'numeric';
  } else if (mode === 'subbed-nl') {
    frameSource = 'subbed_nl';
    actionMode = 'natural_language';
  }

  // Update hidden fields
  document.getElementById('ai-frame-source').value = frameSource;
  document.getElementById('ai-action-mode').value = actionMode;
}

// Initialize on page load
window.addEventListener('DOMContentLoaded', () => {
  updateAIModeFields();

  // 监听 video mode 切换，自动显示/隐藏预览区域
  document.getElementById('ai-video-mode').addEventListener('change', (e) => {
    const section = document.getElementById('context-video-preview-section');
    if (e.target.checked) {
      // 如果已经加载了 context，显示视频预览
      const video = document.getElementById('context-video-preview');
      if (video.src && pendingContextFolder) {
        section.classList.remove('hidden');
      }
    } else {
      // 切换到图片模式，隐藏视频预览
      section.classList.add('hidden');
    }
  });
});

async function regenerateContextVideo() {
  if (!pendingContextFolder) { alert('Load a folder first'); return; }

  const frameSource = document.getElementById('ai-frame-source').value;
  const actionMode = document.getElementById('ai-action-mode').value;
  const fps = parseFloat(document.getElementById('context-video-fps').value) || 1.0;

  document.getElementById('context-video-status').textContent = 'Regenerating from frames...';

  try {
    const sid = sessionId || 'none';
    const resp = await fetch(`/api/session/${sid}/operations/${pendingContextFolder}/load-video`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        frame_source: frameSource,
        action_mode: actionMode,
        fps: fps,
        force_regenerate: true
      })
    });

    const data = await resp.json();

    // Update video preview
    const videoPreview = document.getElementById('context-video-preview');
    videoPreview.src = data.video_url + '?t=' + Date.now(); // Add timestamp to force reload

    // Update status
    document.getElementById('context-video-status').textContent = `Regenerated at ${fps} FPS`;
    document.getElementById('context-count').textContent = `${data.count} frames (${fps} FPS)`;

    // 更新 contextList 中对应 folder 的帧
    const newFrames = data.frames_b64 || [];
    const item = contextList.find(it => it.folder === pendingContextFolder);
    if (item) {
      item.frames = newFrames;
      item.frameCount = newFrames.length;
      renderContextList();
    }
    updateContextFrames();
  } catch (err) {
    alert(`Failed to regenerate video: ${err.message}`);
    document.getElementById('context-video-status').textContent = 'Error';
  }
}

function showContextInPlayer(frames) {
  previewFrames = frames || contextFrames;
  if (previewFrames.length === 0) return;
  document.getElementById('no-game-msg').classList.add('hidden');
  document.getElementById('game-frame').classList.remove('hidden');
  showFrame(previewFrames[0]);
  totalFrames = previewFrames.length;
  updateProgress(0, totalFrames);
  isLive = false;
}

