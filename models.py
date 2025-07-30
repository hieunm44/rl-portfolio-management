from env import StockEnvTrain, StockEnvTrade
from stable_baselines3 import A2C, DDPG, PPO, TD3
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import OrnsteinUhlenbeckActionNoise
import numpy as np
import pandas as pd
import time


def data_split(df, start, end):
    """
    split the dataset into training or testing using date
    :param data: (df) pandas dataframe, start, end
    :return: (df) pandas dataframe
    """
    data = df[(df['date'] >= start) & (df['date'] <= end)]
    data=data.sort_values(['date','ticker'], ignore_index=True)
    data.index = data['date'].factorize()[0]
    
    return data


def train_A2C(market, train_env, iter_num, timesteps=30000):
    start = time.time()
    model = A2C(policy='MlpPolicy', env=train_env)
    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f'trained_models/{market}/{market}_A2C_{iter_num}')
    print(f'Training time: {(end-start)/60} minutes')
    
    return model


def train_PPO(market, train_env, iter_num, timesteps=150000):
    start = time.time()
    model = PPO(policy='MlpPolicy', env=train_env, ent_coef=0.005, batch_size=8)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f'trained_models/{market}/{market}_PPO_{iter_num}')
    print(f'Training time: {(end-start)/60} minutes')

    return model



def train_DDPG(market, train_env, iter_num, timesteps=10000):
    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5*np.ones(n_actions))

    start = time.time()
    model = DDPG(policy='MlpPolicy', env=train_env, action_noise=action_noise)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f'trained_models/{market}/{market}_DDPG_{iter_num}')
    print(f'Training time: {(end-start)/60} minutes')

    return model


def train_TD3(market, train_env, iter_num, timesteps=10000):
    n_actions = train_env.action_space.shape[-1]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.5*np.ones(n_actions))

    start = time.time()
    model = TD3(policy='MlpPolicy', env=train_env, action_noise=action_noise)

    model.learn(total_timesteps=timesteps)
    end = time.time()

    model.save(f'trained_models/{market}/{market}_TD3_{iter_num}')
    print(f'Training time: {(end-start)/60} minutes')

    return model


def validate_model(market, model_name, valid_data, valid_env, valid_obs, iter_num):
    model = eval(model_name).load(f'trained_models/{market}/{market}_{model_name}_{iter_num}')
    for i in range(len(valid_data.index.unique())):
        action, _ = model.predict(valid_obs)
        valid_obs, reward, is_done, info = valid_env.step(action)

        if is_done:
            asset_memory = info[0]['asset_memory']
            reward_memory = info[0]['reward_memory']
            action_memory = info[0]['action_memory']
            n_shares_memory = info[0]['n_shares_memory']

            df_asset = pd.DataFrame({'account_value': asset_memory,
                                    'reward': reward_memory,
                                    'actions': action_memory,
                                    'n_shares': n_shares_memory})
            daily_return = df_asset['account_value'].pct_change(1)
            sharpe = (4**0.5) * daily_return.mean() / daily_return.std()

            df_asset.to_csv(f'results/{market}/{market}_{model_name}_memory_valid_{iter_num}.csv')
    
    return sharpe


def train_validate(market, model_name, data_df, trade_dates, rebalance_window):
    first_date = data_df['date'].values[0]
    print(f'Training {model_name} for {market} market...\n')
    n_mod_dates = (len(trade_dates)-1)%rebalance_window
    ceil_len = int(np.ceil((len(trade_dates)-1)/rebalance_window)*rebalance_window)
    end_len = len(trade_dates) if n_mod_dates==0 else ceil_len

    for i in range(rebalance_window, end_len, rebalance_window):
        end_train_id = 0 if i==rebalance_window else i-rebalance_window
        print(f'Training from {first_date} to {trade_dates[end_train_id]}')
        train_data = data_split(data_df, start=first_date, end=trade_dates[end_train_id])
        train_env = DummyVecEnv([lambda: StockEnvTrain(market=market, df=train_data)])
        model = eval(f'train_{model_name}(market=market, train_env=train_env, iter_num=i)')

        end_valid_id = i
        print(f'Validation from {trade_dates[end_train_id+1]} to {trade_dates[end_valid_id]}')
        valid_data = data_split(df=data_df, start=trade_dates[end_train_id+1], end=trade_dates[i+1])
        valid_env = DummyVecEnv([lambda: StockEnvTrain(market=market, df=valid_data)])
        valid_obs = valid_env.reset()
        sharpe = validate_model(market=market, model_name=model_name, valid_data=valid_data, valid_env=valid_env, valid_obs=valid_obs, iter_num=i)
        print(f'Sharpe: {sharpe}')
        
        end_trade_id = i+rebalance_window if i+rebalance_window+1 < len(trade_dates) else -1
        print(f'This model will be used for trading from {trade_dates[end_valid_id+1]} to {trade_dates[end_trade_id]}')
        print()



def trading(market, model_name, trade_data, initial, prev_state, prev_n_trades, iter_num):
    trade_env = DummyVecEnv([lambda: StockEnvTrade(market=market,
                                                   df=trade_data,
                                                   initial=initial,
                                                   prev_state=prev_state,
                                                   prev_n_trades=prev_n_trades)])
    trade_obs = trade_env.reset()
    model = eval(model_name).load(f'trained_models/{market}/{market}_{model_name}_{iter_num}')
    
    for i in range(len(trade_data.index.unique())):
        action, _ = model.predict(trade_obs)
        trade_obs, reward, is_done, info = trade_env.step(action)

        if i == len(trade_data.index.unique())-2:
            new_prev_state = info[0]['state']
            new_prev_n_trades = info[0]['n_trades']
            
        if is_done:
            asset_memory = info[0]['asset_memory']
            reward_memory = info[0]['reward_memory']
            action_memory = info[0]['action_memory']
            n_shares_memory = info[0]['n_shares_memory']

            df_asset = pd.DataFrame({'account_value': asset_memory,
                                    'reward': reward_memory,
                                    'actions': action_memory,
                                    'n_shares': n_shares_memory})
            daily_return = df_asset['account_value'].pct_change(1)
            sharpe = (4**0.5) * daily_return.mean() / daily_return.std()

            df_asset.to_csv(f'results/{market}/{market}_{model_name}_memory_trading_{iter_num}.csv')

    return new_prev_state, new_prev_n_trades, sharpe


def run_trading(market, model_name, data_df, trade_dates, rebalance_window):
    last_state = []
    last_n_trades = 0
    sharpe_list = []
    n_mod_dates = (len(trade_dates)-1)%rebalance_window
    ceil_len = int(np.ceil((len(trade_dates)-1)/rebalance_window)*rebalance_window)
    end_len = len(trade_dates) if n_mod_dates==0 else ceil_len

    for i in range(rebalance_window, end_len, rebalance_window):
        initial = True if i==rebalance_window else False
        end_trade_id = i+rebalance_window+1 if i+rebalance_window+1 < len(trade_dates) else -1
        print(f'Trading from {trade_dates[i+1]} to {trade_dates[end_trade_id]}')
        trade_data = data_split(data_df, start=trade_dates[i+1], end=trade_dates[end_trade_id])
        last_state, last_n_trades, sharpe = trading(market=market,
                                                    model_name=model_name,
                                                    trade_data=trade_data,
                                                    initial=initial,
                                                    prev_state=last_state,
                                                    prev_n_trades=last_n_trades,
                                                    iter_num=i)
        print(f'Sharpe: {sharpe}')
        sharpe_list.append(sharpe)

    print(f'\nAverage sharpe: {np.mean(sharpe_list)}')
    print(f'Total nummber of trades: {last_n_trades}')