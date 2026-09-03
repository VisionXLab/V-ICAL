"""
交互式游戏环境脚本
支持选择游戏、设置参数、输入动作、保存状态图片
"""

from datetime import datetime
from pathlib import Path
import sys
import gymnasium as gym
import imageio
import numpy as np
from env_wrapper import create_env, do_action

# 导入自定义 Breakout
CUSTOM_BREAKOUT_AVAILABLE = False
try:
    # 获取当前脚本所在目录
    project_root = "/Users/chris/Desktop/Video-CL"
    
    # 添加到 sys.path
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    print(f"sys.path: {sys.path}")
    
    # 尝试导入
    from new_gym.breakout_wrapper import create_breakout_env, BreakoutGameEnv
    print(f"Successfully imported breakout_wrapper")
    CUSTOM_BREAKOUT_AVAILABLE = True
except Exception as e:
    # 静默失败，用户选择 CustomBreakout 时会显示错误
    print(f"Failed to import breakout_wrapper: {e}")
    pass

# breakpoint()
# 常用游戏列表
# 分类：Toy Text（最机械）、MiniGrid（有画面+机械）、Classic Control、ALE（实时游戏）
GAMES = {
    # === Toy Text 系列（最机械，离散步进）===
    "1": ("Taxi-v3", "出租车（网格接送，超机械）"),
    "2": ("CliffWalking-v1", "悬崖漫步（上下左右一步）"),
    "3": ("FrozenLake-v1", "冰湖（建议 is_slippery=False）"),
    "4": ("Blackjack-v1", "21点（hit/stand，回合制）"),
    
    # === MiniGrid 系列（有画面+机械步进）===
    "5": ("MiniGrid-Empty-16x16-v0", "MiniGrid 空房间 16×16（机械步进+画面）"),
    "6": ("MiniGrid-DoorKey-16x16-v0", "MiniGrid 钥匙门 16×16（机械步进+画面）"),
    "7": ("MiniGrid-FourRooms-v0", "MiniGrid 四房间（机械步进+画面）"),
    
    # === Classic Control 系列 ===
    "8": ("CartPole-v1", "倒立摆（连续控制但动作明确）"),
    "9": ("Acrobot-v1", "Acrobot（连续控制）"),
    
    # === ALE/Atari 系列（实时游戏，相对较慢的）===
    "10": ("ALE/Enduro-v5", "耐力赛（Atari，较慢）"),
    "11": ("ALE/Freeway-v5", "高速公路（Atari，上下移动）"),
    "12": ("ALE/Seaquest-v5", "海底任务（Atari，节奏中等）"),
    "13": ("ALE/Boxing-v5", "拳击（Atari，动作离散但实时）"),
    
    # === ALE/Atari 系列（经典但较实时）===
    "14": ("ALE/Pong-v5", "乒乓球（Atari，实时）"),
    "15": ("CustomBreakout", "打砖块（可调参数，无惯性，正确掉球处理）"),
    "16": ("ALE/SpaceInvaders-v5", "太空侵略者（Atari，实时）"),
    "17": ("ALE/Qbert-v5", "Q*bert（Atari，实时）"),
}


def save_frame(image: np.ndarray, save_dir: Path, step: int) -> Path:
    """保存当前帧图片"""
    save_dir.mkdir(parents=True, exist_ok=True)
    frame_path = save_dir / f"step_{step:06d}.png"
    imageio.imwrite(frame_path, image)
    return frame_path


def print_menu():
    """打印主菜单"""
    print("\n" + "=" * 60)
    print("可用游戏:")
    print("-" * 60)
    for key, (game_id, desc) in GAMES.items():
        print(f"  {key}. {game_id} - {desc}")
    print("-" * 60)
    print("=" * 60)


def get_game_choice() -> str:
    """获取用户选择的游戏"""
    print_menu()
    
    while True:
        max_key = max([int(k) for k in GAMES.keys() if k.isdigit()])
        choice = input(f"\n请选择游戏 (1-{max_key} 或输入完整游戏名称，默认 Taxi-v3): ").strip()
        
        if not choice:
            return GAMES["1"][0]  # 默认 Taxi-v3
        
        if choice in GAMES:
            return GAMES[choice][0]
        
        # 尝试作为完整游戏名称（支持 ALE 和非 ALE）
        # 如果已经是完整名称，直接返回
        if "/" in choice or "-v" in choice:
            return choice
        
        # 否则尝试添加 ALE/ 前缀（向后兼容）
        return f"ALE/{choice}"


def get_config(game_name: str = "") -> dict:
    """获取用户配置
    
    Args:
        game_name: 游戏名称，用于提供特定游戏的配置选项
    """
    print("\n" + "=" * 60)
    print("环境配置")
    print("=" * 60)
    
    # 随机种子配置（最先配置，确保公平性）
    print("\n🎲 随机种子 (Seed):")
    print("  - 设置种子可保证每次游戏的初始状态完全一致")
    print("  - 适用于模型对比测试，确保公平性")
    seed_input = input("请输入随机种子 (整数，默认=None/随机): ").strip()
    seed = None
    if seed_input:
        try:
            seed = int(seed_input)
            print(f"✅ 已设置种子: {seed}")
        except:
            print("⚠️  无效的种子值，将使用随机种子")
            seed = None
    else:
        print("✅ 将使用随机种子（每次不同）")
    
    is_ale_game = game_name.startswith('ALE/')
    is_custom_breakout = game_name == 'CustomBreakout'
    
    # 自定义 Breakout 的特殊配置
    if is_custom_breakout:
        print("\n🎮 自定义 Breakout 配置:")
        
        paddle_width_input = input("木板宽度 (默认20, 范围10-40): ").strip()
        paddle_width = int(paddle_width_input) if paddle_width_input else 20
        paddle_width = max(10, min(40, paddle_width))
        
        ball_speed_input = input("小球速度 (默认2.0, 范围1.0-5.0): ").strip()
        ball_speed = float(ball_speed_input) if ball_speed_input else 2.0
        ball_speed = max(1.0, min(5.0, ball_speed))
        
        lives_input = input("初始生命数 (默认5, 范围1-99): ").strip()
        initial_lives = int(lives_input) if lives_input else 5
        initial_lives = max(1, min(99, initial_lives))
        
        frameskip_input = input("frameskip (默认4, 范围1-8): ").strip()
        frameskip = int(frameskip_input) if frameskip_input else 4
        frameskip = max(1, min(8, frameskip))
        
        brick_rows_input = input("砖块行数 (默认6, 范围3-12): ").strip()
        brick_rows = int(brick_rows_input) if brick_rows_input else 6
        brick_rows = max(3, min(12, brick_rows))
        
        brick_cols_input = input("砖块列数 (默认10, 范围5-15): ").strip()
        brick_cols = int(brick_cols_input) if brick_cols_input else 10
        brick_cols = max(5, min(15, brick_cols))
        
        brick_offset_input = input("砖块距顶部偏移 (默认10, 范围5-50, 越大砖块越靠下): ").strip()
        brick_area_top_offset = int(brick_offset_input) if brick_offset_input else 10
        brick_area_top_offset = max(5, min(50, brick_area_top_offset))
        
        print("\n发球角度模式:")
        print("  1. random - 随机角度（游戏性好，每次不同）")
        print("  2. fixed - 固定角度 0.3 弧度/约17度（便于比较和调试）")
        print("  3. fixed_angle - 自定义固定角度")
        launch_mode_input = input("请选择 (1/2/3, 默认1): ").strip()
        
        if launch_mode_input == "2":
            launch_angle_mode = "fixed"
            launch_angle = 0.3
        elif launch_mode_input == "3":
            launch_angle_mode = "fixed_angle"
            angle_input = input("请输入发球角度 (弧度, 范围-0.5到0.5, 默认0.3): ").strip()
            launch_angle = float(angle_input) if angle_input else 0.3
            launch_angle = max(-0.5, min(0.5, launch_angle))
        else:
            launch_angle_mode = "random"
            launch_angle = 0.3
        
        print(f"\n✅ 自定义 Breakout 配置:")
        print(f"   木板宽度: {paddle_width}")
        print(f"   小球速度: {ball_speed}")
        print(f"   初始生命: {initial_lives}")
        print(f"   frameskip: {frameskip}")
        print(f"   砖块: {brick_rows}行 x {brick_cols}列")
        print(f"   砖块偏移: {brick_area_top_offset}像素")
        if launch_angle_mode == "random":
            print(f"   发球模式: 随机角度")
        elif launch_angle_mode == "fixed":
            print(f"   发球模式: 固定角度 0.3 弧度 (约17度)")
        else:
            print(f"   发球模式: 固定角度 {launch_angle} 弧度 (约{launch_angle*57.3:.1f}度)")
    
    # frameskip（仅 ALE 游戏有效，CustomBreakout 已在上面配置）
    if is_ale_game and not is_custom_breakout:
        frameskip_input = input("frameskip (每步执行帧数, 4=推荐, 1=精细, 仅ALE游戏有效, 默认4): ").strip()
        try:
            frameskip = int(frameskip_input) if frameskip_input else 4
            frameskip = max(1, min(1000, frameskip))
        except:
            frameskip = 4
    elif not is_custom_breakout:
        frameskip = 1  # 非 ALE 游戏不需要 frameskip，但保留参数以兼容
        print("frameskip: 跳过（非 ALE 游戏不支持此参数）")
    # CustomBreakout 的 frameskip 已在上面配置，不需要重置
    
    # mode 和 difficulty（仅 ALE 游戏有效）
    mode = None
    difficulty = None
    if is_ale_game:
        mode_input = input("mode (游戏模式, 不同模式=不同玩法, 默认0, 输入?查看可用值): ").strip()
        if mode_input == '?':
            # 临时创建环境查看可用 mode
            try:
                import ale_py
                _tmp = gym.make(game_name, render_mode='rgb_array', frameskip=1)
                _ale = _tmp.unwrapped.ale
                print(f"  可用 modes: {list(_ale.getAvailableModes())}")
                print(f"  可用 difficulties: {list(_ale.getAvailableDifficulties())}")
                _tmp.close()
            except:
                print("  无法获取可用值，请参考文档")
            mode_input = input("mode (默认0): ").strip()
        try:
            mode = int(mode_input) if mode_input else 0
        except:
            mode = 0
        
        diff_input = input("difficulty (0=Easy/标准paddle, 1=Hard/小paddle, 默认0): ").strip()
        try:
            difficulty = int(diff_input) if diff_input else 0
        except:
            difficulty = 0
    
    # auto_respawn 和 initial_lives（仅 ALE 游戏有效）
    auto_respawn = True
    # 注意：CustomBreakout 已经在上面设置了 initial_lives，不要覆盖
    if not is_custom_breakout:
        initial_lives = None
    if is_ale_game:
        af_input = input("auto_respawn (死后自动复活, y=开启/n=关闭, 默认y): ").strip().lower()
        auto_respawn = af_input not in ['n', 'no', 'false', '0']
        
        lives_input = input("initial_lives (初始生命数, 默认=游戏原始值, 如99=99条命): ").strip()
        if lives_input:
            try:
                initial_lives = int(lives_input)
                initial_lives = max(1, min(255, initial_lives))
            except:
                initial_lives = None
    
    # 特殊游戏参数
    is_slippery = None
    if 'FrozenLake' in game_name:
        slippery_input = input("is_slippery (是否滑, False=机械可控, True=随机, 默认False): ").strip().lower()
        if slippery_input in ['false', 'f', '0', '']:
            is_slippery = False
        elif slippery_input in ['true', 't', '1']:
            is_slippery = True
        # 否则保持 None，使用环境默认值
    
    # repeat
    repeat_input = input("repeat (总执行次数, 1=精确, 4=快速, 建议1-4): ").strip()
    try:
        repeat = int(repeat_input) if repeat_input else 1
        repeat = max(1, min(2000, repeat))
    except:
        repeat = 1
    
    # repeat 模式选择
    print("\nrepeat 模式:")
    print("  1. 传统模式: 所有次数都执行指定 action")
    print("  2. 新模式: action 执行 k 次，剩余用 NOOP 填充")
    mode_input = input("请选择模式 (1/2, 默认1): ").strip()
    use_noop_fill = (mode_input == "2")
    
    action_repeat = None
    if use_noop_fill:
        action_repeat_input = input(f"action 执行次数 (1-{repeat}, 默认1): ").strip()
        try:
            action_repeat = int(action_repeat_input) if action_repeat_input else 1
            action_repeat = max(1, min(repeat, action_repeat))
        except:
            action_repeat = 1
    
    print(f"\n✅ 配置完成:")
    if seed is not None:
        print(f"   seed: {seed} (保证初始状态一致)")
    else:
        print(f"   seed: None (随机)")
    if is_ale_game:
        print(f"   frameskip: {frameskip}")
        if mode is not None:
            print(f"   mode: {mode}")
        if difficulty is not None:
            print(f"   difficulty: {difficulty}")
        print(f"   auto_respawn: {'开启' if auto_respawn else '关闭'}")
        if initial_lives is not None:
            print(f"   initial_lives: {initial_lives}")
    if is_slippery is not None:
        print(f"   is_slippery: {is_slippery}")
    print(f"   repeat: {repeat}")
    if use_noop_fill:
        print(f"   模式: action 执行 {action_repeat} 次，NOOP 填充 {repeat - action_repeat} 次")
    else:
        print(f"   模式: 传统模式（action 执行 {repeat} 次）")
    print("=" * 60)
    
    config = {
        'frameskip': frameskip,
        'render_mode': 'rgb_array',
        'repeat': repeat,
        'action_repeat': action_repeat,
        'noop_fill': use_noop_fill,
        'seed': seed,
    }
    
    # 自定义 Breakout 配置
    if is_custom_breakout:
        config['paddle_width'] = paddle_width
        config['ball_speed'] = ball_speed
        config['initial_lives'] = initial_lives
        config['brick_rows'] = brick_rows
        config['brick_cols'] = brick_cols
        config['brick_area_top_offset'] = brick_area_top_offset
        config['launch_angle_mode'] = launch_angle_mode
        config['launch_angle'] = launch_angle
    
    if mode is not None:
        config['mode'] = mode
    if difficulty is not None:
        config['difficulty'] = difficulty
    config['auto_respawn'] = auto_respawn
    if initial_lives is not None and not is_custom_breakout:
        config['initial_lives'] = initial_lives
    if is_slippery is not None:
        config['is_slippery'] = is_slippery
    
    return config


def interactive_play(env, game_name: str, save_dir: Path, repeat: int, 
                     action_repeat: int = None, noop_fill: bool = False, seed: int = None):
    """交互式游戏循环"""
    # 重置环境（使用种子）
    if seed is not None:
        state, info = env.reset(seed=seed)
    else:
        state, info = env.reset()
    step_count = 0
    
    # 保存初始状态
    frame_path = save_frame(state.image, save_dir, step_count)
    print(f"\n✅ 环境已创建并重置")
    print(f"   保存目录: {save_dir}")
    print(f"   初始状态已保存: {frame_path.name}")
    print(f"   图片形状: {state.image.shape}")
    
    # 显示动作信息
    action_info = env.get_action_info()
    print(f"\n可用动作:")
    for action_id, meaning in action_info.items():
        print(f"  {action_id}: {meaning}")
    
    print("\n" + "=" * 60)
    print("交互式游戏开始")
    print("=" * 60)
    print("输入说明:")
    print(f"  - 输入动作编号 (0 到 {env.action_space_size - 1}) 执行动作")
    print(f"  - 输入 'r' 或 'reset' 重置环境")
    print(f"  - 输入 'q' 或 'quit' 退出")
    print(f"  - 输入 'info' 查看当前信息")
    print(f"  - 环境 frameskip: {env.frameskip}")
    if noop_fill and action_repeat is not None:
        print(f"  - 执行模式: action 执行 {action_repeat} 次，NOOP 填充 {repeat - action_repeat} 次（总共 {repeat} 次）")
    else:
        print(f"  - 执行模式: action 执行 {repeat} 次（传统模式）")
    print("\n💡 提示:")
    is_ale_game = game_name.startswith('ALE/')
    if is_ale_game:
        print(f"  - 这是 ALE/Atari 游戏（实时模拟）")
        print(f"  - 即使 action=0 (NOOP)，游戏状态仍会变化（球和对手会继续移动）")
        if env.auto_respawn and env._needs_fire:
            print(f"  - 🔄 auto_respawn 已开启: 死后自动复活，无需手动操作")
        if env.initial_lives is not None:
            print(f"  - ❤️  初始生命已设为 {env.initial_lives}")
    else:
        print(f"  - 这是离散步进游戏（每次动作=一步）")
        print(f"  - 动作含义明确，不需要 frameskip")
    print("=" * 60 + "\n")
    
    episode_reward = 0.0
    episode_steps = 0
    
    while True:
        try:
            # 获取用户输入
            user_input = input(
                f"[步数 {step_count}] 请输入动作 "
                f"(0-{env.action_space_size-1}, r=重置, q=退出, info=信息): "
            ).strip().lower()
            
            # 退出
            if user_input in ['q', 'quit', 'exit']:
                print("\n退出游戏")
                break
            
            # 重置环境
            if user_input in ['r', 'reset']:
                if seed is not None:
                    state, info = env.reset(seed=seed)
                else:
                    state, info = env.reset()
                step_count += 1
                episode_reward = 0.0
                episode_steps = 0
                frame_path = save_frame(state.image, save_dir, step_count)
                print(f"\n✅ 环境已重置")
                if seed is not None:
                    print(f"   使用种子: {seed} (保证初始状态一致)")
                print(f"   状态已保存: {frame_path.name}")
                continue
            
            # 查看信息
            if user_input == 'info':
                print(f"\n当前信息:")
                print(f"  总步数: {step_count}")
                print(f"  当前回合步数: {episode_steps}")
                print(f"  当前回合奖励: {episode_reward:.2f}")
                print(f"  游戏是否结束: {state.terminated or state.truncated}")
                print(f"  图片形状: {state.image.shape}")
                continue
            
            # 解析动作
            try:
                action = int(user_input)
                if action < 0 or action >= env.action_space_size:
                    print(f"❌ 无效动作！请输入 0 到 {env.action_space_size - 1} 之间的数字\n")
                    continue
            except ValueError:
                print(f"❌ 无效输入！请输入数字、'r'、'q' 或 'info'\n")
                continue
            
            # 执行动作
            state = do_action(env, action, repeat=repeat, 
                            action_repeat=action_repeat, noop_fill=noop_fill)
            step_count += 1
            episode_steps += 1
            episode_reward += state.reward
            
            # 保存当前状态（只保存最终状态，不保存跳过的中间帧）
            frame_path = save_frame(state.image, save_dir, step_count)
            
            # 显示结果
            action_meaning = action_info[action]
            reward_str = f"奖励: {state.reward:.2f}" if state.reward != 0 else "奖励: 0.00"
            
            # 添加说明：对于 ALE 游戏，NOOP 时状态仍会变化
            note = ""
            is_ale_game = game_name.startswith('ALE/')
            if action == 0 and "NOOP" in action_meaning.upper() and is_ale_game:
                note = " (注意: NOOP 时球和对手仍会移动，这是正常的游戏行为)"
            
            print(f"\n🎯 动作 {action} ({action_meaning}){note} | {reward_str} | "
                  f"累计: {episode_reward:.2f} | 已保存: {frame_path.name}")
            
            # 显示丢球/生命损失提醒
            if state.info.get('life_lost'):
                lives_before = state.info.get('lives_before', '?')
                lives_after = state.info.get('lives_after', '?')
                print(f"   💀 丢球！生命: {lives_before} → {lives_after}")
                if state.info.get('auto_respawned'):
                    print(f"   🔄 已自动复活（auto_respawn）")
            
            # 显示 RAM 状态信息（如果可用）
            ram_info = env.get_ram_info()
            if ram_info:
                ram_str = " | ".join(f"{k}={v}" for k, v in ram_info.items())
                print(f"   📊 {ram_str}")
            
            if state.terminated or state.truncated:
                print(f"   ⚠️  游戏结束 ({'terminated' if state.terminated else 'truncated'})")
                print(f"\n回合结束！总步数: {episode_steps}, 总奖励: {episode_reward:.2f}")
                
                reset_choice = input("\n是否重置环境继续？(y/n，默认y): ").strip().lower()
                if reset_choice in ['', 'y', 'yes']:
                    if seed is not None:
                        state, info = env.reset(seed=seed)
                    else:
                        state, info = env.reset()
                    step_count += 1
                    episode_reward = 0.0
                    episode_steps = 0
                    frame_path = save_frame(state.image, save_dir, step_count)
                    print(f"\n✅ 环境已重置，状态已保存: {frame_path.name}")
                    if seed is not None:
                        print(f"   使用种子: {seed} (保证初始状态一致)")
        
        except KeyboardInterrupt:
            print("\n\n收到中断信号，退出游戏")
            break
        except Exception as e:
            print(f"\n❌ 发生错误: {e}\n")
            continue
    
    env.close()
    print(f"\n游戏结束！总共执行了 {step_count} 步")
    print(f"所有帧已保存到: {save_dir}")


def main():
    """主函数"""
    print("=" * 60)
    print("交互式游戏环境")
    print("=" * 60)
    
    # 选择游戏
    game_name = get_game_choice()
    print(f"\n✅ 已选择游戏: {game_name}")
    
    # 获取配置（传入游戏名称以提供特定选项）
    cfg = get_config(game_name)
    repeat = cfg.pop('repeat')  # 从cfg中取出repeat，因为create_env不需要它
    action_repeat = cfg.pop('action_repeat', None)  # 从cfg中取出action_repeat
    noop_fill = cfg.pop('noop_fill', False)  # 从cfg中取出noop_fill
    seed = cfg.pop('seed', None)  # 从cfg中取出seed，因为它用于reset而非create_env
    
    # 创建保存目录（以时间戳和游戏名命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 清理游戏名称，移除版本号和前缀
    game_short_name = game_name.replace("ALE/", "").replace("-v5", "").replace("-v4", "").replace("-v3", "").replace("-v1", "").replace("-v0", "")
    save_dir = Path("saved_frames") / f"{timestamp}_{game_short_name}"
    
    print(f"\n保存目录: {save_dir}")
    
    # 创建环境
    try:
        print("\n正在创建环境...")
        if game_name == 'CustomBreakout':
            if not CUSTOM_BREAKOUT_AVAILABLE:
                print("❌ 自定义 Breakout 不可用，请检查 new_gym 目录")
                return
            env = create_breakout_env(cfg)
        else:
            env = create_env(game_name, cfg)
    except Exception as e:
        print(f"\n❌ 创建环境失败: {e}")
        print("\n💡 提示: 请确保已安装以下依赖:")
        print("   基础: pip install gymnasium")
        print("   Atari: pip install 'gymnasium[atari]' 'gymnasium[accept-rom-license]' ale-py")
        print("   MiniGrid: pip install minigrid")
        print("   自定义 Breakout: 确保 new_gym 目录存在")
        return
    
    # 开始交互式游戏
    try:
        interactive_play(env, game_name, save_dir, repeat, action_repeat, noop_fill, seed)
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        env.close()


if __name__ == "__main__":
    main()
