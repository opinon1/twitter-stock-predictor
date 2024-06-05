import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as tkr
import seaborn as sns

def price_plot_ma(df,ax=None, **plt_kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr

    n = df.shape[0] # number of dates
    if ax is None:
        ax = plt.gca()
        
    # format data for seaborn
    df=df.melt(id_vars='date',var_name='var', value_name='vals')
    df=df[df['var'].isin(['close_value','MA10','MA20','MA50','MA100'])]
    df['vals']=df['vals'].astype(float)
    df.index=df.date.dt.date
    df.date=df.date.dt.date
    # set axis formats / Set the locator
    if ax is None:
        ax = plt.gca()
        
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
        
    

        
    ax.set_ylabel('Close Price')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    sns.lineplot(data=df, x='date', y='vals',hue='var',palette='cool_r',ax=ax)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
    return ax

def price_plot_vol(df,ax=None, **plt_kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr
    
    n=df.shape[0]
    
    df.index=df.date.dt.date
    if ax is None:
        ax = plt.gca()
    
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
        
    
    ax.set_ylabel('Traded Volume (million)')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x/1000000), ',')))
    sns.lineplot(data=df, x='date', y='volume',palette='cool_r',ax=ax)

    return ax

def sentiment_barplot(df,ax=None, **plt_kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr
    
    df=df.groupby(['date','label'])['tweet_id'].agg('count').reset_index(name="count")
    
    n=len(df.date.unique())
    
    # format the data and make proportion
    df=df.pivot(index='date',columns='label',values='count')
    df=pd.DataFrame(df.to_records()).reset_index()
    df.loc[:,"total"]=df.loc[:,['bad','neutral','good']].sum(axis=1)
    df.loc[:,['bad','neutral','good']]=df.loc[:,['bad','neutral','good']].div(df.total,axis=0)
    df.loc[:,"total"]=df.loc[:,['bad','neutral','good']].sum(axis=1)
    df=df.drop(['total'], axis=1)
   
    df.index=df.date.dt.date
    if ax is None:
        ax = plt.gca()
    colors=['crimson','lightgrey','mediumseagreen']
    df.loc[:,['bad','neutral', 'good']].plot.bar(stacked=True, color=colors, width=1.0,alpha=0.5,ax=ax)
    
   
    # set axis formats / Set the locato
    
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
         
    
    ax.set_ylabel('Sentiment')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    
    ax.grid(True, which='major',axis='both')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5),fancybox=True, shadow=True)
    return ax
    
def sentiment_tweet_vol(df,ax=None,**kwargs):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import matplotlib.ticker as tkr
    df=df.groupby(['date'])['label'].agg('count').reset_index(name="count")
    df.index=df.date.dt.date
    n=len(df.date.unique())
    
    if ax is None:
        ax = plt.gca()
    # set axis formats / Set the locator
    
    major_locator = mdates.MonthLocator()  
    major_fmt = mdates.DateFormatter('%b')
    minor_locator = mdates.DayLocator(interval=1) 
    minor_fmt = mdates.DateFormatter('%d')
    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_fmt)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.grid(True, which='major',axis='both')
    
    if n > 750:
        major_locator = mdates.YearLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%Y')
        minor_locator =  mdates.MonthLocator()
        minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if((n > 250 ) & (n< 750 )):
        major_locator = mdates.MonthLocator()   # every year and quarter
        major_fmt = mdates.DateFormatter('%b-%Y')
        #minor_locator =  mdates.MonthLocator()
        #minor_fmt = mdates.DateFormatter('%b')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        #ax.xaxis.set_minor_locator(minor_locator)
        ax.grid(True, which='major',axis='both')
        
    if ((n > 90 ) & (n< 250 )):
        major_locator = mdates.MonthLocator()   # every  month
        major_fmt = mdates.DateFormatter('%b-%y')
        minor_locator = tkr.AutoMinorLocator(4)
        minor_fmt = mdates.DateFormatter('%d-%m')
        ax.xaxis.set_major_locator(major_locator)
        ax.xaxis.set_major_formatter(major_fmt)
        ax.xaxis.set_minor_locator(minor_locator)
        #ax.xaxis.set_minor_formatter(minor_fmt)
        ax.grid(True, which='major',axis='both')
        
    ax.set_ylabel('Tweet Volume')
    ax.set_xlabel('Date')
    ax.tick_params(axis='x', labelrotation = 45)
    ax.yaxis.set_major_formatter(tkr.FuncFormatter(lambda x, p: format(int(x), ',')))
    sns.lineplot(data=df, x='date', y='count',palette='cool_r',ax=ax)
    
    return ax

def corr_plot(sp,tw):
    
    x=tw.groupby(['date','label']).agg({"score":['count','mean']}).unstack('label') 
    sp=sp.reset_index(drop=True)
    # format the data and make proportion
    x=pd.DataFrame(x.to_records())
    # format columns names
    x.columns=['date','count_bad','count_neutral','count_good','score_mean_bad','score_mean_neutral','score_mean_good']
    x.loc[:,'tweet_volume']=x.loc[:,['count_bad','count_neutral','count_good']].sum(axis=1)
    x.loc[:,'count_ratio_gb']=x.count_good/x.count_bad # create a ratio good:bad
    # join price
    x=x.merge(sp.loc[:,['date','MA10', 'MA20', 'MA50','MA100', 'macd', 'rsi','volume']],how='left',left_on='date',right_on='date')

       #get tendencies as well
    x.loc[:, 'MA20_minus_MA50'] = x.MA20 - x.MA50
    x.loc[:, 'MA20_div_MA50'] = x.MA20 / x.MA50
    

    corr = x.corr()
    # Getting the Upper Triangle of the co-relation matrix
    matrix = np.triu(corr)
    ax = sns.heatmap(
        round(corr,3),
        mask = matrix,
        vmin=-1, vmax=1, center=0,
        cmap="YlGnBu",annot=True,annot_kws={"fontsize":8}, fmt=".2",
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    return ax
