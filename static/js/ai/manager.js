// ========== AI Controls ==========
async function uploadContextFiles() {
  const files = document.getElementById('ai-context-files').files;
  if (!files.length) return;

  const videoMode = document.getElementById('ai-video-mode').checked;
  const firstFile = files[0];
  const isVideo = firstFile.type.startsWith('video/');

  if (isVideo && videoMode) {
    // 视频模式：需要后端提取帧（需要 session）
    if (!sessionId) { alert('Start a game first (video extraction needs a session)'); return; }
    const fps = parseFloat(document.getElementById('context-video-fps').value) || 2;
    const fd = new FormData();
    fd.append('video', firstFile);
    fd.append('fps', fps);

    document.getElementById('context-count').textContent = 'Extracting frames from video...';

    try {
      const resp = await fetch(`/api/session/${sessionId}/ai/upload-context-video`, {
        method: 'POST',
        body: fd
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}: ${resp.statusText}`);
      const data = await resp.json();
      const frames = data.frames || [];

      contextList.push({
        id: 'upload_' + Date.now(),
        label: firstFile.name + ` (${frames.length}f @${fps}fps)`,
        type: 'upload',
        frames: frames,
        frameCount: frames.length,
      });
      updateContextFrames();
      renderContextList();
      showContextInPlayer(frames);
    } catch (err) {
      alert(`Failed to extract frames: ${err.message}`);
    }
  } else {
    // 图片模式：在前端读取为 base64
    const readAsB64 = (file) => new Promise((res) => {
      const reader = new FileReader();
      reader.onload = (e) => res(e.target.result.split(',')[1]);
      reader.readAsDataURL(file);
    });
    const frames = await Promise.all(Array.from(files).map(readAsB64));

    const fileNames = Array.from(files).map(f => f.name).join(', ');
    contextList.push({
      id: 'upload_' + Date.now(),
      label: fileNames + ` (${frames.length} images)`,
      type: 'upload',
      frames: frames,
      frameCount: frames.length,
    });
    updateContextFrames();
    renderContextList();
    showContextInPlayer(frames);

    // 如果有 session，同时上传到后端
    if (sessionId) {
      const fd = new FormData();
      for (const f of files) fd.append('files', f);
      await fetch(`/api/session/${sessionId}/ai/upload-context`, { method: 'POST', body: fd });
    }
  }
  // 清空文件选择器
  document.getElementById('ai-context-files').value = '';
}

async function aiStart() {
  if (!sessionId) { alert('Start a game first'); return; }
  if (!ws || ws.readyState !== WebSocket.OPEN) return;

  clearChatView();
  document.getElementById('ai-log').innerHTML = '';
  // 清空之前AI游戏的帧缓存
  aiSessionFrames = [];

  // 先发 AI 配置
  const body = {
    api_key: document.getElementById('ai-api-key').value,
    base_url: document.getElementById('ai-base-url').value,
    model: document.getElementById('ai-model').value,
    temperature: parseFloat(document.getElementById('ai-temperature').value),
    max_tokens: parseInt(document.getElementById('ai-max-tokens').value),
  };
  const videoDesc = document.getElementById('ai-video-desc').value.trim();
  if (videoDesc) body.video_description = videoDesc;
  const gameRules = document.getElementById('ai-game-rules').value.trim();
  body.game_rules = gameRules;  // always send (empty string resets to built-in rules)
  body.action_mode = document.getElementById('ai-action-mode').value;
  body.frame_source = document.getElementById('ai-frame-source').value;
  body.video_mode = document.getElementById('ai-video-mode').checked;
  body.frame_window = parseInt(document.getElementById('ai-frame-window').value) || 0;
  body.hide_reward = document.getElementById('ai-hide-reward').checked;

  await fetch(`/api/session/${sessionId}/ai/configure`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });

  // 将所有 context 项加载到 AIPlayer
  if (contextList.length > 0) {
    // 先清空后端 context
    await fetch(`/api/session/${sessionId}/ai/clear-context`, { method: 'POST' });
    // 逐项加载
    for (const item of contextList) {
      if (item.type === 'folder' && item.folder) {
        await fetch(`/api/session/${sessionId}/ai/load-context-folder`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ folder: item.folder, append: true })
        });
      } else if (item.type === 'upload' && item.frames.length > 0) {
        await fetch(`/api/session/${sessionId}/ai/upload-context-frames`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frames: item.frames })
        });
      }
    }
  }

  // 如果有已加载序列，先通过 replay_sequence 回放，等完成后再启动 AI
  if (loadedSequence) {
    try {
      const seqResp = await fetch(`/api/action-sequences/${encodeURIComponent(loadedSequence.game_safe)}/${encodeURIComponent(loadedSequence.name)}/sequence`);
      if (seqResp.ok) {
        const seqData = await seqResp.json();
        // 发送 replay_sequence 并等待 replay_done
        await new Promise(resolve => {
          window._onReplayDone = resolve;
          ws.send(JSON.stringify({ type: 'replay_sequence', actions: seqData.actions, delay_ms: 80 }));
        });
        // 回放完成，发 ai_start 跳过 reset
        ws.send(JSON.stringify({ type: 'ai_start', skip_reset: true }));
        document.getElementById('ai-btn-start').disabled = true;
        return;
      }
    } catch (e) {
      console.warn('[AI] Failed to load sequence for ai_start:', e);
    }
  }
  // 无序列，正常启动（会 reset）
  ws.send(JSON.stringify({ type: 'ai_start' }));
  document.getElementById('ai-btn-start').disabled = true;
}

function formatUsage(usage) {
  if (!usage || (usage.input == null && usage.output == null)) return '';
  const parts = [];
  if (usage.input != null) parts.push(`In: ${usage.input.toLocaleString()}`);
  if (usage.output != null) parts.push(`Out: ${usage.output.toLocaleString()}`);
  if (usage.thoughts != null) parts.push(`Thoughts: ${usage.thoughts.toLocaleString()}`);
  if (usage.total != null) parts.push(`Total: ${usage.total.toLocaleString()}`);
  let line = `<span class="ai-log-usage">${parts.join(' | ')} tokens`;
  const media = [];
  if (usage.input_image_tokens != null) media.push(`img: ${usage.input_image_tokens.toLocaleString()} tok`);
  else if (usage.input_image_count != null) media.push(`${usage.input_image_count} img`);
  if (usage.input_video_tokens != null) media.push(`video: ${usage.input_video_tokens.toLocaleString()} tok`);
  else if (usage.input_video_count != null) media.push(`${usage.input_video_count} video`);
  if (media.length) line += ` · ${media.join(' · ')}`;
  return line + '</span>';
}

function appendAILog(data) {
  const log = document.getElementById('ai-log');
  const entry = document.createElement('div');
  const isValid = data.is_valid;
  entry.className = 'ai-log-entry' + (isValid ? '' : ' ai-log-entry-invalid');
  const usageHTML = formatUsage(data.usage);
  const actionText = isValid
    ? `Action: ${data.parsed_action}${data.action_meaning ? ' - ' + data.action_meaning : ''}`
    : '<span style="color:#ef5350">Action: (invalid)</span>';
  const thinking = data.usage && data.usage.thinking;
  const thinkingHTML = thinking ? `
    <details class="ai-log-thinking">
      <summary>💭 Thinking</summary>
      <div class="ai-log-thinking-body">${escapeHTML(thinking)}</div>
    </details>` : '';
  entry.innerHTML = `
    <span class="ai-log-step">[Step ${data.step+1}]</span>
    ${actionText}${usageHTML ? '<br>' + usageHTML : ''}<br>
    ${thinkingHTML}<span class="ai-log-raw">${escapeHTML(data.raw_response || '')}</span>
  `;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function appendAIRetry(data) {
  const log = document.getElementById('ai-log');
  const entry = document.createElement('div');
  entry.className = 'ai-log-entry ai-log-entry-retry';
  const rawPart = data.raw_response
    ? `<br><span class="ai-log-raw">${escapeHTML(data.raw_response)}</span>`
    : '';
  entry.innerHTML = `
    <span class="ai-log-step">[Retry]</span> <span style="color:#ef6c00">${escapeHTML(data.message)}</span>${rawPart}
  `;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
}

function appendAIError(message) {
  const log = document.getElementById('ai-log');
  const entry = document.createElement('div');
  entry.className = 'ai-log-entry ai-log-entry-error';
  entry.innerHTML = `
    <span class="ai-log-step">[ERROR]</span><br>
    <span class="ai-log-raw">${escapeHTML(message)}</span>
  `;
  log.appendChild(entry);
  log.scrollTop = log.scrollHeight;
  document.getElementById('ai-btn-start').disabled = false;
}

