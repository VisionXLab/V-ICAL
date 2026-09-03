// ========== Cross-Validation HUD ==========
// 控制 HUD 中的 reward 字段是否被遮蔽（hide_reward=true 时）
// 注：现存的 frame.js updateState() 会读取 window.crossvalHideReward
// 在 crossval-loader.js 中设置；此文件提供一个统一的应用函数。

window.crossvalHideReward = false;

function applyCrossvalHudMask() {
  const hide = !!window.crossvalHideReward && currentMode === 'crossval';
  const rewardEl = document.getElementById('info-reward');
  const cumEl = document.getElementById('info-cumulative');
  const scoreEl = document.getElementById('info-score');
  if (!rewardEl || !cumEl) return;
  rewardEl.classList.toggle('cv-hidden-reward', hide);
  cumEl.classList.toggle('cv-hidden-reward', hide);
  if (scoreEl) scoreEl.classList.toggle('cv-hidden-reward', hide);
  if (hide) {
    rewardEl.textContent = '■■■';
    cumEl.textContent = '■■■';
    if (scoreEl) scoreEl.textContent = '■■■';
  }
}

function clearCrossvalHudMask() {
  window.crossvalHideReward = false;
  const rewardEl = document.getElementById('info-reward');
  const cumEl = document.getElementById('info-cumulative');
  const scoreEl = document.getElementById('info-score');
  if (rewardEl) rewardEl.classList.remove('cv-hidden-reward');
  if (cumEl) cumEl.classList.remove('cv-hidden-reward');
  if (scoreEl) scoreEl.classList.remove('cv-hidden-reward');
}
