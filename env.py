from gymnasium import Env, spaces
from gymnasium.utils import seeding
import numpy as np


HMAX_NORMALIZE = 100 # shares normalization factor, maximum 100 shares per trade
INITIAL_CASH = {'US': 1e6, 'JP': 10e6, 'VN':1e6, 'VNn':1e6}
STOCK_DIM = 30
TRANSACTION_FEE_PERCENT = 0.001
REWARD_SCALING = {'US': 1e-4, 'JP': 1e-6, 'VN':1e-5, 'VNn':1e-5}


class StockEnvTrain(Env):
    def __init__(self, market, df):
        self.action_space = spaces.Box(low=-1, high=1, shape=(STOCK_DIM,)) 
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(181,))

        self.df = df
        self.market = market
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.n_trades = 0
        self.reward = 0
        self.terminal = False
              
        self.state = \
            [float(INITIAL_CASH[self.market])] + \
            self.data.adjcp.values.tolist() + \
            [0]*STOCK_DIM + \
            self.data.macd.values.tolist() + \
            self.data.rsi.values.tolist() + \
            self.data.cci.values.tolist() + \
            self.data.adx.values.tolist()

        self.asset_memory = [INITIAL_CASH[self.market]]
        self.reward_memory = []
        self.action_memory = []
        self.n_shares_memory = [np.array([0]*STOCK_DIM)]

        self._seed()


    def sell_stock(self, index, action):
        if self.state[index+STOCK_DIM+1] > 0:
            n_shares = min(-np.ceil(action), self.state[index+STOCK_DIM+1])
            self.state[0] += self.state[index+1] * n_shares * (1-TRANSACTION_FEE_PERCENT)
            self.state[index+STOCK_DIM+1] -= n_shares
            self.n_trades += 1

            return n_shares
        else:
            return 0

    
    def buy_stock(self, index, action):
        available_amount = self.state[0] // self.state[index+1]
        n_shares = min(available_amount, np.floor(action))
        self.state[0] -= self.state[index+1] * n_shares * (1+TRANSACTION_FEE_PERCENT)
        while self.state[0] < 0:
            n_shares -= 1
            self.state[0] += self.state[index+1] * (1+TRANSACTION_FEE_PERCENT)
        self.state[index+STOCK_DIM+1] += n_shares
        self.n_trades += 1

        return n_shares

        
    def step(self, actions):
        self.terminal = self.day >= len(self.df.index.unique())-1

        if self.terminal:
            self.action_memory.append(np.array([0]*STOCK_DIM))
            self.reward_memory.append(0)
            info_dict = {'asset_memory': self.asset_memory,
                         'reward_memory': self.reward_memory,
                         'action_memory': self.action_memory,
                         'n_shares_memory': self.n_shares_memory}

            return self.state, self.reward, self.terminal, False, info_dict
    
        else:
            actions = actions * HMAX_NORMALIZE
            begin_total_asset = self.state[0] + \
                np.inner(self.state[1:STOCK_DIM+1], self.state[STOCK_DIM+1:STOCK_DIM*2+1])
            argsort_actions = np.argsort(actions)
            
            sell_index = argsort_actions[:np.where(actions<0)[0].shape[0]]
            buy_index = argsort_actions[::-1][:np.where(actions>0)[0].shape[0]]
            hold_index = argsort_actions[np.where(actions==0)[0]]
            action_dict = {}

            for index in sell_index:
                n_shares_sell = self.sell_stock(index, float(actions[index]))
                action_dict[index] = -n_shares_sell

            for index in buy_index:
                n_shares_buy = self.buy_stock(index, float(actions[index]))
                action_dict[index] = n_shares_buy

            for index in hold_index:
                action_dict[index] = 0

            sorted_keys = np.sort(np.array(list(action_dict.keys())))
            self.action_memory.append(np.array([action_dict[key] for key in sorted_keys]).astype(int))

            self.day += 1
            self.data = self.df.loc[self.day,:]         
            
            self.state =  \
                [self.state[0]] + \
                self.data.adjcp.values.tolist() + \
                self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)] + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist() + \
                self.data.cci.values.tolist() + \
                self.data.adx.values.tolist()

            self.n_shares_memory.append(np.array(self.state[(STOCK_DIM+1):(STOCK_DIM*2+1)]).astype(int))

            end_total_asset = self.state[0] + \
                np.inner(self.state[1:STOCK_DIM+1], self.state[STOCK_DIM+1:STOCK_DIM*2+1])
            self.asset_memory.append(end_total_asset)
            reward = end_total_asset - begin_total_asset            
            self.reward_memory.append(reward)
            self.reward = reward*REWARD_SCALING[self.market]

            info_dict = {'state': self.state, 'n_trades': self.n_trades}

        return self.state, self.reward, self.terminal, False, info_dict
    

    def reset(self, seed=None):
        self.day = 0
        self.data = self.df.loc[self.day, :]
        self.n_trades = 0
        self.reward = 0
        self.terminal = False

        self.state = \
            [float(INITIAL_CASH[self.market])] + \
            self.data.adjcp.values.tolist() + \
            [0]*STOCK_DIM + \
            self.data.macd.values.tolist() + \
            self.data.rsi.values.tolist() + \
            self.data.cci.values.tolist() + \
            self.data.adx.values.tolist()
        
        self.asset_memory = [INITIAL_CASH[self.market]]
        self.rewards_memory = []
        self.action_memory = []
        self.n_shares_memory = [np.array([0]*STOCK_DIM)]

        return self.state, None


    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

class StockEnvTrade(StockEnvTrain):
    def __init__(self, market, df, initial, prev_state, prev_n_trades):
        super().__init__(market=market, df=df)
        self.initial = initial
        self.prev_state = prev_state
        self.prev_n_trades = prev_n_trades

    
    def reset(self, seed=None):  
        if self.initial:
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.n_trades = 0
            self.reward = 0
            self.terminal = False 

            self.state = \
                [float(INITIAL_CASH[self.market])] + \
                self.data.adjcp.values.tolist() + \
                [0]*STOCK_DIM + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist() + \
                self.data.cci.values.tolist() + \
                self.data.adx.values.tolist()
            
            self.asset_memory = [INITIAL_CASH[self.market]]
            self.rewards_memory = []
            self.action_memory = []
            self.n_shares_memory = [np.array([0]*STOCK_DIM)]
        else:
            self.day = 0
            self.data = self.df.loc[self.day, :]
            self.n_trades = self.prev_n_trades
            self.reward = 0
            self.terminal = False 

            self.state = \
                [self.prev_state[0]] + \
                self.data.adjcp.values.tolist() + \
                self.prev_state[(STOCK_DIM+1):(STOCK_DIM*2+1)] + \
                self.data.macd.values.tolist() + \
                self.data.rsi.values.tolist()  + \
                self.data.cci.values.tolist()  + \
                self.data.adx.values.tolist()
            
            previous_total_asset = self.prev_state[0] + \
                np.inner(self.prev_state[1:STOCK_DIM+1], self.prev_state[STOCK_DIM+1:STOCK_DIM*2+1])
            self.asset_memory = [previous_total_asset]
            self.rewards_memory = []
            self.action_memory = []
            self.n_shares_memory = [np.array(self.prev_state[(STOCK_DIM+1):(STOCK_DIM*2+1)]).astype(int)]
        
        return self.state, None