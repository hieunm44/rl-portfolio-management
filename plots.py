import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import pyfolio


CURRENCIES = {'US': 'USD', 'JP': 'JPY', 'VN': 'VND (000s)', 'VNn': 'VND (000s)'}
MODELS = ['A2C', 'PPO', 'DDPG', 'TD3']
REBALANCE_WINDOW = 63


def list_to_matrix(value_list):
    final_list = []
    for sublist in value_list:
        sublist = sublist[1:-1]
        final_list.append([int(n) for n in sublist.split()])
    
    return np.array(final_list)


def retrieve_memory(model, markets, tickers, trade_dates, trade_data):
    asset_mem = {'US': [], 'JP': [], 'VN': [], 'VNn': []}
    action_mem = {'US': [], 'JP': [], 'VN': [], 'VNn': []}
    n_shares_mem = {'US': [], 'JP': [], 'VN': [], 'VNn': []}
    prices_ticker = {'US': np.zeros((len(trade_dates['US']), 30)), 
                    'JP': np.zeros((len(trade_dates['JP']), 30)),
                    'VN': np.zeros((len(trade_dates['VN']), 30)),
                    'VNn': np.zeros((len(trade_dates['VNn']), 30))}
    portfolio_weights = {'US': np.zeros((len(trade_dates['US']), 30)), 
                        'JP': np.zeros((len(trade_dates['JP']), 30)),
                        'VN': np.zeros((len(trade_dates['VN']), 30)),
                        'VNn': np.zeros((len(trade_dates['VNn']), 30))}

    for mk in markets:
        for i, tic in enumerate(tickers[mk]):
            df = trade_data[mk]
            prices_ticker[mk][:, i] = df[df['ticker']==tic]['adjcp'].values

        n_mod_dates = len(trade_dates[mk])%REBALANCE_WINDOW
        ceil_len = int(np.ceil(len(trade_dates[mk])/REBALANCE_WINDOW)*REBALANCE_WINDOW)
        end_len = len(trade_dates[mk]) if n_mod_dates==0 else ceil_len    

        for i in range(REBALANCE_WINDOW, end_len+1, REBALANCE_WINDOW):
            is_final = True if i==end_len else False

            asset = pd.read_csv(f'results/{mk}/{mk}_{model}_memory_trading_{i}.csv')
            if not is_final:
                asset_mem[mk] += asset['account_value'][:-1].to_list()
                action_mem[mk] += asset['actions'][:-1].to_list()
                n_shares_mem[mk] += asset['n_shares'][:-1].to_list()

            else:
                asset_mem[mk] += asset['account_value'].to_list()
                action_mem[mk] += asset['actions'].to_list()
                n_shares_mem[mk] += asset['n_shares'].to_list()
        
        action_matrix = list_to_matrix(action_mem[mk])
        n_shares_matrix = list_to_matrix(n_shares_mem[mk])
        
        portfolio_value_history = np.sum(prices_ticker[mk]*n_shares_matrix, axis=1)

        portfolio_weights[mk] = prices_ticker[mk] * n_shares_matrix / portfolio_value_history.reshape(-1, 1)
        portfolio_weights[mk][0] = np.zeros(len(tickers[mk]))

    return asset_mem, action_mem, n_shares_mem, portfolio_weights


def plot_series_returns(market, ticker, model, dates, Y):
    plt.figure(figsize=(15, 5.5))

    plt.subplot(2, 1, 1)
    plt.plot(dates, Y)
    plt.grid(True)
    plt.title(f'{ticker} Value')
    plt.ylabel(f'{CURRENCIES[market]}')
    plt.subplots_adjust(hspace=0.25)
    
    plt.subplot(2, 2, 3)
    daily_return = (Y[1:]-Y[:-1]) / Y[:-1]
    plt.plot(dates[1:], daily_return)
    plt.title(f'{ticker} Daily Return')

    plt.subplot(2, 2, 4)
    cum_return = Y / Y[0]
    plt.plot(dates, cum_return)
    plt.grid(True)
    plt.title(f'{ticker} Cumulative Return')

    if ticker=='portfolio':
        plt.savefig(f'figures/{market}/{market}_{ticker}_{model}_series_returns.pdf', bbox_inches='tight')
    else:
        plt.savefig(f'figures/{market}/{market}_{ticker}_series_returns.pdf', bbox_inches='tight')

    plt.show()


def plot_4series(market, dates, model_mem):
    plt.figure(figsize=(10, 3.5))

    print(f'Ending asset:')
    for model in model_mem.keys():
        print(f'{model}: {model_mem[model]['asset_mem'][market][-1]}')
        plt.plot(dates, model_mem[model]['asset_mem'][market], label=model)
        plt.grid(True)
        plt.legend()
        plt.ylabel(f'{CURRENCIES[market]}')

    plt.savefig(f'figures/{market}/{market}_portfolio_series.pdf', bbox_inches='tight')
    plt.show()


def plot_rmean_std(market, tics, Y_data, model_mem, com_names):
    rmeans = []
    rstds = []

    for tic in tics:
        Y = Y_data.loc[Y_data['ticker']==tic, 'adjcp'].to_numpy()
        cum_return = Y / Y[0]
        rmean = np.mean(cum_return)
        rstd = np.std(cum_return)
        rmeans.append(rmean)
        rstds.append(rstd)

    for model in model_mem.keys():
        Y = np.array(model_mem[model]['asset_mem'][market])
        cum_return = Y / Y[0]
        rmean = np.mean(cum_return)
        rstd = np.std(cum_return)
        rmeans.append(rmean)
        rstds.append(rstd)

    df = pd.DataFrame({'ticker': com_names+MODELS, 'rmean': rmeans, 'rstd': rstds})
    df_sort_rmean = df.sort_values(by='rmean')
    df_sort_rmean.reset_index(inplace=True)
    df_sort_rstd = df.sort_values(by='rstd')
    df_sort_rstd.reset_index(inplace=True)

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(9, 9))
    ticker_patch = mpatches.Patch(color='tab:blue', label='Stock')
    portfolio_patch = mpatches.Patch(color='tab:red', label='Portfolio')

    rl_ids1 = df_sort_rmean[df_sort_rmean['ticker'].isin(MODELS)].index.values
    colors1 = ['tab:blue']*34
    for i in rl_ids1:
        colors1[i] = 'tab:red'
    axs[0].barh(df_sort_rmean['ticker'], df_sort_rmean['rmean'], color=colors1)
    axs[0].set_title('Mean of cumulative return')
    axs[0].legend(handles=[ticker_patch, portfolio_patch], loc='lower right')

    rl_ids2 = df_sort_rmean[df_sort_rstd['ticker'].isin(MODELS)].index.values
    colors2 = ['tab:blue']*34
    for i in rl_ids2:
        colors2[i] = 'tab:red'
    axs[1].barh(df_sort_rstd['ticker'], df_sort_rstd['rstd'], color=colors2)
    axs[1].set_title('Std of cumulative return')
    axs[1].legend(handles=[ticker_patch, portfolio_patch], loc='lower right')
    fig.tight_layout()

    plt.savefig(f'figures/{market}/{market}_rmean_std.pdf', bbox_inches='tight')
    plt.show()


def plot_series_actions_shares(market, ticker, model, dates, Y, actions, n_shares_ticker):
    buy_steps = np.where(np.array(actions)>0)[0]
    sell_steps = np.where(np.array(actions)<0)[0]
    hold_steps = np.where(np.array(actions)==0)[0]

    plt.figure(figsize=(16, 12))
    plt.subplot(3, 1, 1)
    plt.plot(dates, Y)
    plt.ylabel(f'{CURRENCIES[market]}')
    plt.xticks([])
    plt.title(f'{ticker} Value')

    plt.subplot(3, 1, 2)
    plt.scatter(dates[buy_steps], np.ones_like(buy_steps), c='g', marker='^', label='buy')
    plt.scatter(dates[sell_steps], -np.ones_like(sell_steps), c='r', marker='v', label='sell')
    plt.scatter(dates[hold_steps], np.zeros_like(hold_steps), c='b', marker='>', label='hold')
    plt.legend()
    plt.yticks([])
    plt.xticks([])
    plt.title('Actions Made')

    plt.subplot(3, 1, 3)
    plt.plot(dates, n_shares_ticker)
    plt.title('Number of Shares Owned')
    plt.savefig(f'figures/{market}_{ticker}_{model}_series_actions_shares.pdf', bbox_inches='tight')
    plt.show()


def plot_top_drawdown_underwater(market, model, dates, Y, top=5):
    daily_return = (Y[1:]-Y[:-1]) / Y[:-1]
    daily_return_series = pd.Series(daily_return, index=dates[1:])
    df_drawdown = pyfolio.timeseries.gen_drawdown_table(returns=daily_return_series, top=top)
    print(df_drawdown)
    
    plt.figure(figsize=(10, 2.5))
    # top drawdown
    ax = plt.subplot(2, 1, 1)
    cum_return = Y / Y[0]
    ax.plot(dates, cum_return)
    lim = ax.get_ylim()
    colors = sns.cubehelix_palette(len(df_drawdown))[::-1]
    for i, (peak, recovery) in df_drawdown[
            ['Peak date', 'Recovery date']].iterrows():
        if pd.isnull(recovery):
            recovery = daily_return_series.index[-1]
        ax.fill_between((peak, recovery),
                        lim[0],
                        lim[1],
                        alpha=.4,
                        color=colors[i])
    plt.xticks([])
    plt.ylabel('Cum return')
    plt.subplots_adjust(hspace=0.25)

    # underwater
    ax = plt.subplot(2, 1, 2)
    running_max = np.maximum.accumulate(cum_return)
    underwater = -100 * ((running_max - cum_return) / running_max)
    underwater = pd.Series(underwater, index=dates)

    underwater.plot(kind='area', color='coral', alpha=0.7, rot=0)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')
    plt.ylabel('Drawdown (%)')


    plt.savefig(f'figures/{market}/{market}_{model}_drawdown.pdf', bbox_inches='tight')
    plt.show()


def plot_rolling_volatility(market, dates, model_mem, rolling_days=63):
    plt.figure(figsize=(10, 3.5))
    
    for model in model_mem.keys():
        Y = np.array(model_mem[model]['asset_mem'][market])
        daily_return = (Y[1:]-Y[:-1]) / Y[:-1]
        daily_return_series = pd.Series(daily_return, index=dates[1:])
        rolling_std = daily_return_series.rolling(rolling_days).std()
        mean_std = np.mean(rolling_std)

        plt.plot(dates[1:], rolling_std, label=model)
        plt.axhline(y=mean_std, linestyle='--')
        plt.grid(True)
        plt.legend()

    plt.savefig(f'figures/{market}/{market}_rolling_std.pdf', bbox_inches='tight')
    plt.show()


def plot_rolling_sharpe(market, dates, model_mem, rolling_days=63):
    plt.figure(figsize=(10, 3.5))
    
    for model in model_mem.keys():
        Y = np.array(model_mem[model]['asset_mem'][market])
        daily_return = (Y[1:]-Y[:-1]) / Y[:-1]
        daily_return_series = pd.Series(daily_return, index=dates[1:])
        rolling_std = daily_return_series.rolling(rolling_days).std()
        rolling_mean = daily_return_series.rolling(rolling_days).mean()
        rolling_sharpe = round(252/rolling_days)**0.5 * rolling_mean / rolling_std
        mean_sharpe = np.mean(rolling_sharpe)

        plt.plot(dates[1:], rolling_sharpe, label=model)
        plt.axhline(y=mean_sharpe, linestyle='--')
        plt.grid(True)
        plt.legend()


    plt.savefig(f'figures/{market}/{market}_rolling_sharp.pdf', bbox_inches='tight')
    plt.show()


def plot_portfolio_weights(market, model, dates, weights, tics, com_names):
    portfolio_weights_df = pd.DataFrame(weights, index=dates)
    portfolio_weights_df.columns = tics

    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', \
              'black', 'silver', 'lightcoral', 'chocolate', 'bisque', 'tan', 'gold', 'yellow', 'lawngreen', 'darkseagreen', \
              'lime', 'steelblue', 'navy', 'mediumblue', 'slateblue', 'darkviolet', 'thistle', 'purple', 'deeppink', 'lightpink']
    portfolio_weights_df.plot.area(figsize=(10, 8.5), color=colors, rot=0)
    
    ax = plt.gca()
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')

    L = ax.legend(bbox_to_anchor=(1.0, 1.0), labelspacing=0.4)
    for i, name in enumerate(com_names):
        L.get_texts()[i].set_text(name)
    plt.margins(x=0)
    plt.ylim([0, 1])

    plt.savefig(f'figures/{market}/{market}_{model}_weights.pdf', bbox_inches='tight')
    plt.show()