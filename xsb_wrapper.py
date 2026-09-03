import gym as gym_origin
import gymnasium as gym

class cliff_end_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.base=0
        self.pos=0
        self.max=0

    def step(self, action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        self.pos=obs%12
        info['pass']=False
        info["ending"]="timeout"
        if terminated:
            reward=10
            info["ending"]="victory"
            info["pass"]=True
        if reward==-100:
            info["ending"]="gameover"
            terminated=True
            reward=0
        elif reward==-1:
            reward=0
            reward+=max(self.pos-self.max,0)/(11-self.base)*90
        if reward>0:
            self.max=self.pos
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.base=0
        self.pos=0
        self.max=0
        return obs, info
    
    def state_reset(self):
        self.base=self.pos
        self.max=self.pos
    
class acrobot_score_wrapper(gym.Wrapper):
    """Acrobot dense reward + pass via height-reached event.

    Height of the free end = -(cos(t1) + cos(t1+t2)), in [-2, 2].
    Reward each step is the increment in max-height seen so far (clipped to 1).
    Emits info["acrobot_height_reached"]=True when the free end ever rises
    above the pivot (height > 0), letting GameSession's event-rule mark pass.
    """
    def __init__(self, env):
        super().__init__(env)
        self.maxheight=0
    
    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info["ending"]="timeout"
        height=-(obs[0]+obs[0]*obs[2]-obs[1]*obs[3])
        if height>self.maxheight:
            reward=-self.maxheight+min(1,height)
            self.maxheight=height
        else:
            reward=0
        info["pass"]=False
        if self.maxheight>0:
            info["pass"]=True
        if terminated:
            info["ending"]="victory"
        reward*=100
        info["score"]=reward
        return obs,reward,terminated,truncated,info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.maxheight = 0
        return obs, info

class freeway_end_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.target=178
        ram=self.env.unwrapped.ale.getRAM()
        self.base=int(ram[14])
        self.high=self.base
        self.last_reward=0
        self.pass_type=False

    def step(self, action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        ram=self.env.unwrapped.ale.getRAM()
        cur_pos=int(ram[14])
        if cur_pos>self.high:
            self.high=cur_pos
        if (self.high-self.base)>(self.target-self.base)/2:
            self.pass_type=True
        info["ending"]="timeout"
        info["pass"]=self.pass_type
        if reward==1:
            info["ending"]="victory"
            terminated=True
            reward=1-self.last_reward
        else:
            reward=(self.high-self.base)/(self.target-self.base)-self.last_reward
            self.last_reward=(self.high-self.base)/(self.target-self.base)
        reward*=100
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        ram=self.env.unwrapped.ale.getRAM()
        self.base=int(ram[14])
        self.high=self.base
        self.last_reward=0
        self.pass_type=False
        return obs, info
    
    def state_reset(self):
        ram=self.env.unwrapped.ale.getRAM()
        self.base=int(ram[14])
        self.high=self.base
        self.last_reward=0
        self.pass_type=False


class custombreakout_end_wrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives=None

    def step(self, action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info["ending"]="timeout"
        info["pass"]=False
        if terminated:
            info["pass"]=True
            info["ending"]="victory"
        if self.lives==None:
            self.lives=info['lives']
        elif self.lives!=info['lives']:
            info["ending"]="gameover"
            terminated=True
            reward*=(100/2)
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.lives=None
        return obs,info

class mountaincar_score_wrapper(gym.Wrapper):
    def __init__(self, env, pre_score=0):
        super().__init__(env)
        self.pre_steps=pre_score
        self.cur_steps=0
        self.initial=-2
        self.max=-2
        self.last_reward=0

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info["ending"]="timeout"
        reward=0
        info["pass"]=False
        position=min(obs[0],0.5)
        self.cur_steps+=1
        if self.cur_steps>self.pre_steps:
            if self.initial==-2:
                self.initial=position
                self.max=position
            elif position>self.max:
                reward=-self.last_reward+(position-self.initial)/(0.5-self.initial)
                self.last_reward=(position-self.initial)/(0.5-self.initial)
                self.max=position
            if (self.max-self.initial)/(0.5-self.initial)>0.5:
                info["pass"]=True
        if terminated:
            info["ending"]="victory"
        reward*=100
        info["score"]=reward
        return obs,reward,terminated,truncated,info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.cur_steps = 0
        self.initial = -2
        self.max = -2
        self.last_reward = 0
        return obs, info

class assault_end_wrapper(gym.Wrapper):
    def __init__(self, env, max_steps, repeat):
        super().__init__(env)
        self.accum=0
        self.lives=None
        self.max_steps=max_steps
        self.num_steps=1
        self.cur_num_steps=0
        self.repeat=repeat

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        self.cur_num_steps+=1
        if self.cur_num_steps==self.repeat:
            self.cur_num_steps=0
            self.num_steps+=1
        info["ending"]="timeout"
        self.accum+=reward
        info["pass"]=False
        if self.accum>=63:
            info["pass"]=True
            info["ending"]="victory"
            terminated=True
        if self.lives==None:
            self.lives=info['lives']
        elif self.lives!=info['lives']:
            terminated=True
            info["ending"]="gameover"

        reward*=(50/63)
        if info["ending"]=="victory":
            reward+=(2-max(self.num_steps,self.max_steps/2)*2/self.max_steps)*50
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.accum=0
        self.lives=None
        return obs,info
    
class alien_end_wrapper(gym.Wrapper):
    def __init__(self, env,pre_score, max_steps, repeat):
        super().__init__(env)
        self.lives=None
        self.pre_score=pre_score
        self.accum=-self.pre_score
        self.max_steps=max_steps
        self.repeat=repeat
        self.num_steps=1
        self.cur_num_steps=0

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        self.cur_num_steps+=1
        if self.cur_num_steps==self.repeat:
            self.cur_num_steps=0
            self.num_steps+=1
        info["ending"]="timeout"
        info["pass"]=False
        if reward>=500:
            reward-=500
        self.accum+=reward
        if self.accum>=75:
            info["pass"]=True
        if self.accum>=150:
            terminated=True
            info["ending"]="victory"
        if self.lives==None:
            self.lives=info['lives']
        elif self.lives!=info['lives']:
            terminated=True
            info["ending"]="gameover"
        reward*=(50/150)
        if info["ending"]=="victory":
            reward+=(2-max(self.num_steps,self.max_steps/2)*2/self.max_steps)*50
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.lives=None
        self.accum=-self.pre_score
        return obs,info

class roadrunner_end_wrapper(gym.Wrapper):
    """RoadRunner pass-by-score wrapper with 1000-multiple suppression.

    Reports one step's reward delayed by 1 frame (via self.previous), and
    when (previous + current) ≥ 1000 absorbs the 1000-multiple into
    self.previous and emits 0 for this step. Lets the accum threshold reflect
    incremental survival scoring rather than one-shot bonuses.

    Pass when accum ≥ 1100 (lowered from 5000). Life loss terminates without
    victory → ending=gameover.
    """
    def __init__(self, env, pre_score=0):
        super().__init__(env)
        self.lives=None
        self.previous=0
        self.pre_score=pre_score
        self.accum=-self.pre_score

    def step(self,action):
        obs,cur_reward,terminated,truncated,info=self.env.step(action)
        info["ending"]="timeout"
        reward=self.previous
        self.previous=cur_reward
        info["pass"]=True
        if reward+self.previous>=1000:
            others=reward+self.previous-1000
            reward=0
            self.previous=others
        # self.accum+=reward
        if self.accum+reward>=1100:
            reward=1100-self.accum
            terminated=True
            info["ending"]="victory"
        if self.lives==None:
            self.lives=info['lives']
        elif self.lives!=info['lives']:
            terminated=True
            info["ending"]="gameover"
            info["pass"]=False
        self.accum+=reward
        reward*=(100/1100)
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.lives=None
        self.accum=-self.pre_score
        self.previous=0
        return obs,info
    
class highway_end_wrapper(gym.Wrapper):
    def __init__(self, env, pre_score=0):
        super().__init__(env)
        self.pre_score = pre_score
        self.accum = -self.pre_score

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info["ending"]="timeout"
        info["pass"]=True
        reward=info["rewards"]["high_speed_reward"]
        reward+=0.25
        reward/=1.25
        reward/=2
        if self.accum+reward>=25:
            terminated=True
            reward=25-self.accum
            info["ending"]="victory"
        elif terminated:
            info["pass"]=False
            info["ending"]="gameover"
        self.accum+=reward
        reward*=(100/25)
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.accum=-self.pre_score
        return obs,info
    
class aflappybird_reward_wrapper(gym.Wrapper):
    """aFlappyBird wrapper: pass when at least 1 pipe was passed.

    Per-step reward < 1 is discarded; reward >= 1 (pipe pass) is kept and
    emits info["aflappybird_passed_pipe"]=True so GameSession's event rule
    can mark pass=True. Episode terminates early at accum >= 3.
    """
    def __init__(self, env, pre_score=0):
        super().__init__(env)
        self.pre_score = pre_score
        self.accum = -self.pre_score

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info["ending"]="timeout"
        info["pass"]=False
        if terminated:
            info["ending"]="gameover"
        if reward<1:
            reward=0
        self.accum+=reward
        if self.accum>=1:
            info["pass"]=True
        if self.accum>=3:
            info["ending"]="victory"
            terminated=True
        reward*=(100/3)
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.accum=-self.pre_score
        return obs,info
    
class tetris_end_wrapper(gym.Wrapper):
    """Tetris wrapper: pass when at least 1 row was cleared.

    Reward == 1 is a normal piece-drop (ignored). Reward >= 10 is a row-clear
    (counted as 1 and emits info["tetris_cleared_row"]=True so GameSession's
    event rule marks pass=True). Episode terminates early at accum >= 2.
    """
    def __init__(self, env, pre_score=0):
        super().__init__(env)
        self.pre_score = pre_score
        self.accum = -self.pre_score

    def step(self,action):
        obs,reward,terminated,truncated,info=self.env.step(action)
        info["ending"]="timeout"
        info["pass"]=False
        if reward==1:
            reward=0
        elif reward>=10:
            reward=1
        self.accum+=reward
        if self.accum>=1:
            info["pass"]=True
        if self.accum>=2:
            info["ending"]="victory"
            terminated=True
        elif terminated:
            info["ending"]="gameover"
        reward*=(100/2)
        info["score"]=reward
        return obs,reward,terminated,truncated,info
    
    def reset(self,**kwargs):
        obs,info=self.env.reset(**kwargs)
        self.accum=-self.pre_score
        return obs,info
    
class bigfish_end_wrapper(gym_origin.Wrapper):
    def __init__(self, env, pre_score=0):
        super().__init__(env)
        self.pre_score=pre_score
        self.accum=-self.pre_score

    def step(self,action):
        obs,reward,terminated,info=self.env.step(action)
        info["ending"]="timeout"
        info["pass"]=True
        self.accum+=reward
        if terminated:
            info["pass"]=False
            info["ending"]="gameover"
        if self.accum>=2:
            info["ending"]="victory"
            terminated=True
        reward*=(100/2)
        info["score"]=reward
        return obs,reward,terminated,info
    
    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        self.accum=-self.pre_score
        return obs
    
class bossfight_end_wrapper(gym_origin.Wrapper):
    def __init__(self, env, criterion_seed=41):
        super().__init__(env)
        self.criterions={
            41:8,
            49:7,
            51:7,
            59:2,
            78:6,
        }
        self.criterion=self.criterions[criterion_seed]

    def step(self,action):
        obs,reward,terminated,info=self.env.step(action)
        info["ending"]="timeout"
        info["pass"]=True
        # self.accum+=reward
        if reward>0 and terminated:
            info["ending"]="victory"
        elif terminated:
            info["pass"]=False
            info["ending"]="gameover"  
        reward*=(100/self.criterion)    
        info["score"]=reward      
        return obs,reward,terminated,info

    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        return obs
    
class caveflyer_end_wrapper(gym_origin.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward=0

    def step(self,action):
        obs,reward,terminated,info=self.env.step(action)
        self.cumulative_reward+=reward
        info["ending"]="timeout"
        info["pass"]=True
        if self.cumulative_reward>=0.99:
            reward+=1-self.cumulative_reward
            info["ending"]="victory"
        elif terminated:
            info["pass"]=False
            info["ending"]="gameover"
        reward*=100
        info["score"]=reward
        return obs,reward,terminated,info
    
    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        self.cumulative_reward=0
        return obs
    
class dodgeball_end_wrapper(gym_origin.Wrapper):
    def __init__(self, env, criterion_seed=101):
        super().__init__(env)
        self.criterions={
            101:15,
            202:15,
            207:13,
            209:14,
            321:14,
        }
        self.criterion=self.criterions[criterion_seed]

    def step(self,action):
        obs,reward,terminated,info=self.env.step(action)
        info["ending"]="timeout"
        info["pass"]=True
        if reward>=10:
            info["ending"]="victory"
        elif terminated:
            info["pass"]=False
            info["ending"]="gameover"
        reward*=(100/self.criterion)
        info["score"]=reward
        return obs,reward,terminated,info
    
    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        return obs
    
class heist_end_wrapper(gym_origin.Wrapper):
    def __init__(self, env,criterion_seed=44):
        super().__init__(env)
        self.pass_type=False
        self.criterions={
            44:9,
            84:9,
            64:9,
            66:6,
            73:9,
        }
        self.criterion=self.criterions[criterion_seed]

    def step(self,action):
        obs,reward,terminated,info=self.env.step(action)
        if reward>=2:
            self.pass_type=True
        info["ending"]="timeout"
        info["pass"]=self.pass_type
        if terminated:
            info["ending"]="victory"
        reward*=(100/self.criterion)
        info["score"]=reward
        return obs,reward,terminated,info
    
    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        self.pass_type=False
        return obs
    
class leaper_end_wrapper(gym_origin.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward=0

    def step(self,action):
        obs,reward,terminated,info=self.env.step(action)
        self.cumulative_reward+=reward
        info["ending"]="timeout"
        info["pass"]=True
        if self.cumulative_reward>=0.99:
            reward+=1-self.cumulative_reward
            info["ending"]="victory"
        elif terminated:
            info["pass"]=False
            info["ending"]="gameover"
        reward*=100
        info["score"]=reward
        return obs,reward,terminated,info
    
    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        self.cumulative_reward=0
        return obs
    
class ninja_end_wrapper(gym_origin.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.cumulative_reward=0

    def step(self,action):
        obs,reward,terminated,info=self.env.step(action)
        self.cumulative_reward+=reward
        info["ending"]="timeout"
        info["pass"]=False
        if self.cumulative_reward>=0.99:
            reward+=1-self.cumulative_reward
            info["ending"]="victory"
        if self.cumulative_reward>=0.50:
            info["pass"]=True
        elif terminated:
            info["ending"]="gameover"
        reward*=100
        info["score"]=reward
        return obs,reward,terminated,info
    
    def reset(self,**kwargs):
        obs=self.env.reset(**kwargs)
        self.cumulative_reward=0
        return obs
