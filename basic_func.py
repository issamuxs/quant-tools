def annualized_returns(rets, periods_per_year):
    n_periods = rets.shape[0]
    ann_rets = (1 + rets).prod()**(periods_per_year/n_periods) - 1
    return(ann_rets)

def annualized_vols(rets, periods_per_year):
    n_periods = rets.shape[0]
    ann_vols = rets.std(ddof=0)*(periods_per_year**0.5)
    return(ann_vols)

def drawdown(rets):
    wealth_index = 1000*(1+rets).cumprod()
    current_max = 0
    diff = []
    for i in wealth_index:
        if i > current_max:
            current_max=i
        diff.append((current_max - i)/current_max)
    pdd = pd.concat([wealth_index, pd.Series(diff, index=wealth_index.index)], axis=1)
    pdd.columns = [wealth_index.name, 'drawdown']
    return(pdd)

def semidev(r):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(semidev)
    elif isinstance(r, pd.Series):
        return r[r<0].std()
    else:
        raise TypeError('Expected pd.Series or pd.DataFrame')
    
def skew(r):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(skew)
    elif isinstance(r, pd.Series):
        return ((r - r.mean())**3).mean()/r.std()**3
    else:
        raise TypeError('Expected pd.Series or pd.DataFrame')
    
def kurt(r):
    if isinstance(r, pd.DataFrame):
        return r.aggregate(kurt)
    elif isinstance(r, pd.Series):
        return ((r - r.mean())**4).mean()/r.std()**4
    else:
        raise TypeError('Expected pd.Series or pd.DataFrame')

def pf_er(w, er):
    return w.T @ er

def pf_vol(w, cov): 
    return (w.T @ cov @ w)**(1/2)

def compute_ef_2(n_points, er, cov):
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [pf_er(w, er) for w in weights]
    vols = [pf_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Weights": weights,
        "Returns": rets,
        "Vols": vols
    })
    return ef

def minimize_vol(target_return, er, cov):
    n = er.shape[0]
    bounds = ((0,1),)*n
    w_sum_1 = {
        'type': 'eq',
        'fun': lambda w: np.sum(w)-1
    }
    target_return_constraint = {
        'type': 'eq',
        'fun': lambda w: w.T @ er - target_return
    }
    init_guess = np.ones(n)/n         
    opt = minimize(fun=pf_vol,
                   x0=init_guess,
                   args=(cov,),
                   method='SLSQP',
                   bounds=bounds,
                   constraints=[w_sum_1, target_return_constraint]
                  )
    return opt.x

def compute_ef_n(n_points, er, cov):
    r_max = er.max()
    r_min = er.min()
    target_rets = np.linspace(r_min, r_max, n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rets]
    rets = [pf_er(w, er) for w in weights]
    vols = [pf_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Vols": vols
    })
    return ef