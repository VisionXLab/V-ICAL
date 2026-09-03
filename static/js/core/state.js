// ========== State ==========
let currentMode = 'human';  // 'human' | 'ai'
let sessionId = null;
let ws = null;
let isLive = true;
let actionInfo = {};
let pendingContextFolder = '';  // 最近一次 load 的 folder（用于 regenerateContextVideo）
let contextList = [];  // [{id, label, type:'folder'|'upload', folder?, frames, frameCount}]
let games = [];
let totalFrames = 0;
let lastGameOver = false;
let lastState = null;

// Keyboard mapping: key -> action index (set per game)
let keyMap = {};

// Preview frames: the subset currently shown in the player (last upload/load)
let previewFrames = [];

// Sequence replay state
let isReplaying = false;

// Currently loaded action sequence (null = none)
let loadedSequence = null;  // { game_safe, name }

// ========== Game Defaults ==========
// 集中管理所有游戏的默认配置。切换游戏时自动填入对应值。
// 字段名对应 HTML input 的 id（去掉 cfg- 前缀），值为默认值。
// '_ale_default' 是所有 ALE 游戏的兜底默认值。
const GAME_DEFAULTS = {
  '_default': {                 // 所有游戏的兜底默认值
    'repeat': 1,
    'max-steps': 30,
  },
  '_ale_default': {
    'initial-lives': '',        // 空 = 使用游戏默认
    'max-steps': 50,
    'skip-initial-steps': 0,
    'end-on-life-loss': false,
    'action-set': 'full',       // 默认 full（对角线等组合动作在大多数游戏中有效）
    'auto-respawn': false,      // 默认关闭，仅需要的游戏单独开启（死后自动复活）
  },
  '_seed': { human: 41, ai: 43 },

  '_minigrid_default': {
    'partial-obs': true,          // 默认局部观察（fog of war）
  },

  // MiniGrid 每个游戏的默认步数上限 = max(seed42,seed43) BFS + 20，向上取整 10
  'MiniGrid-LavaCrossingS11N5-v0':        { 'max-steps': 40 },   // BFS≤20
  'MiniGrid-UnlockPickup-v0':             { 'max-steps': 40 },   // BFS≤18
  'MiniGrid-MultiRoom-N6-v0':             { 'max-steps': 70 },   // BFS≤43
  // Custom MiniGrid (with per-game param defaults)
  'CustomLavaCrossing-v0':  { 'max-steps': 40, 'lc-size': 11, 'lc-crossings': 5 },
  'CustomMultiRoom-v0':     { 'max-steps': 70, 'mr-grid-size': 25, 'mr-rooms': 6, 'mr-room-size': 10 },
  'CustomUnlockPickup-v0':  { 'max-steps': 40, 'up-room-size': 6, 'up-rows': 1, 'up-cols': 2 },
  '_custom_breakout_default': {
    'cb-lives': 5,
    'max-steps': 50,
    'cb-skip-initial-steps': 0,
    'cb-end-on-life-loss': false,
  },
  'CustomBreakout':       { 'cb-lives': 1, 'ball-speed': 5, 'brick-rows': 3, 'brick-cols': 3 },
  'CartPole-v1':          { 'repeat': 2 },
  'ALE/Enduro-v5':        { 'repeat': 5 },
  'ALE/Freeway-v5':       { 'repeat': 5 },
  'ALE/Seaquest-v5':      { 'skip-initial-steps': 60, 'auto-respawn': true, 'repeat': 5, 'end-on-life-loss': true, 'max-steps': 50, 'difficulty': 0 },
  'ALE/Pong-v5':          { 'skip-initial-steps': 50, 'repeat': 3, 'noop-fill': true, 'action-set': 'simple' },  // Fire 无效，保持 simple
  'ALE/Frostbite-v5':     { 'skip-initial-steps': 30, 'end-on-life-loss': true, 'action-set': 'simple', 'repeat': 4, 'noop-fill': true, 'mode': 0 },
  'ALE/Asterix-v5':       { 'skip-initial-steps': 140, 'end-on-life-loss': true, 'max-steps': 50, 'repeat': 3, 'noop-fill': true },
  // New ALE games
  'ALE/Tennis-v5':            { 'max-steps': 30, 'repeat': 4, 'end-on-life-loss': true, 'skip-initial-steps': 1, 'skip-initial-action': 1 },
  // Classic Control / Box2D

  // Batch 3 — S/A/B tier ALE games
  'ALE/ChopperCommand-v5':    { 'max-steps': 30, 'repeat': 4, 'end-on-life-loss': true },
  'ALE/Berzerk-v5':           { 'max-steps': 25, 'repeat': 4, 'end-on-life-loss': true },
};

