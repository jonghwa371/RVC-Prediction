"""Forecast and origin-risk calculation for a demonstrator, not legal origin certification."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

RETURN_COL = 'krw_log_return_fx_100pct'


def read_history(path):
    df = pd.read_csv(path, dtype={'material_id': str, 'product_id': str})
    required = {'material_id','material_name','year_month',RETURN_COL}
    if not required.issubset(df.columns):
        raise ValueError('필수 열 누락: '+', '.join(sorted(required-set(df.columns))))
    df['year_month'] = pd.to_datetime(df['year_month'].astype(str)+'-01', errors='coerce')
    df[RETURN_COL] = pd.to_numeric(df[RETURN_COL], errors='coerce')
    return df


def catalog(df):
    c = df.sort_values('year_month').drop_duplicates('material_id').copy()
    c['bom_base_cost_krw'] = pd.to_numeric(c['bom_base_cost_krw'], errors='coerce').fillna(0)
    return c[['product_id','material_id','material_name','rvc_treatment','origin_reported','bom_base_cost_krw']].reset_index(drop=True)


def _contiguous_recent(series):
    s = series.sort_index().dropna()
    if s.empty: return s
    idx = s.index.to_period('M').astype(int).to_numpy()
    breaks = np.where(np.diff(idx)!=1)[0]
    return s.iloc[int(breaks[-1]+1):] if len(breaks) else s


def forecast_materials(df, material_ids, months, min_obs=12):
    """Fit ARMA candidates on contiguous, dated monthly returns; choose lowest AICc."""
    out=[]
    for mid in material_ids:
        part = df.loc[df.material_id.eq(mid), ['year_month',RETURN_COL]].dropna(subset=['year_month'])
        s = part.groupby('year_month')[RETURN_COL].mean().sort_index()
        s = _contiguous_recent(s)
        if len(s)<min_obs:
            out.append(dict(material_id=mid, available=False, n_obs=len(s), reason='연속 관측치 부족', model='—'))
            continue
        best=None
        for order in [(0,0,0),(1,0,0),(0,0,1),(1,0,1)]:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    fit=ARIMA(s.to_numpy(),order=order,trend='c',enforce_stationarity=True,enforce_invertibility=True).fit()
                k=len(fit.params); n=len(s)
                if n<=k+1: continue
                aicc=float(fit.aic+2*k*(k+1)/(n-k-1))
                if not np.isfinite(aicc): continue
                if best is None or aicc<best[0]: best=(aicc,order,fit)
            except Exception:
                continue
        if best is None:
            out.append(dict(material_id=mid,available=False,n_obs=len(s),reason='모형 적합 실패',model='—'))
            continue
        _,order,fit=best
        means=np.asarray(fit.forecast(steps=months),dtype=float)
        resid=np.asarray(fit.resid,dtype=float)
        resid=resid[np.isfinite(resid)]
        resid=resid-np.mean(resid) if len(resid) else np.array([0.])
        out.append(dict(material_id=mid,available=True,n_obs=len(s),model=f'ARMA({order[0]},{order[2]})',means=means,residuals=resid, aicc=best[0]))
    return out


def simulate(bom, forecasts, fob, months, n_sim=10000, seed=42, threshold=40):
    """Independent residual bootstrap across materials; no invented forecasts for missing data."""
    if fob<=0: raise ValueError('FOB는 0보다 커야 합니다.')
    if not len(bom): raise ValueError('BOM이 비어 있습니다.')
    if (bom['cost_krw']<0).any(): raise ValueError('원재료 비용은 음수일 수 없습니다.')
    current_vnm=float(bom.loc[bom.status.ne('원산지 확인'),'cost_krw'].sum())
    current_rvc=100*(1-current_vnm/fob)
    used=bom.loc[(bom.status.ne('원산지 확인')) & (bom.cost_krw>0)]
    lookup={f['material_id']:f for f in forecasts}
    missing=[f"{r.material_name} ({r.material_id})" for r in used.itertuples() if not lookup.get(r.material_id,{}).get('available',False)]
    if missing:
        return dict(current_vnm=current_vnm,current_rvc=current_rvc,available=False,missing=missing)
    rng=np.random.default_rng(seed)
    simulated_vnm=np.zeros((n_sim,months),dtype=float)
    material_details=[]
    for r in used.itertuples():
        f=lookup[r.material_id]
        shocks=rng.choice(f['residuals'],size=(n_sim,months),replace=True)
        returns=f['means'][None,:]+shocks
        cumulative=np.cumsum(returns,axis=1)
        prices=r.cost_krw*np.exp(np.clip(cumulative,-12,12))
        simulated_vnm+=prices
        material_details.append(dict(material=r.material_name,base_cost=float(r.cost_krw),forecast_cost=float(r.cost_krw*np.exp(np.sum(f['means']))),model=f['model']))
    rvcs=100*(1-simulated_vnm/fob)
    results=[]
    for h in range(months):
        vals=rvcs[:,h]
        results.append(dict(month=h+1,mean_rvc=float(vals.mean()),median_rvc=float(np.median(vals)),lo=float(np.quantile(vals,.05)),hi=float(np.quantile(vals,.95)),risk=float(np.mean(vals<threshold)*100),mean_vnm=float(simulated_vnm[:,h].mean())))
    return dict(current_vnm=current_vnm,current_rvc=current_rvc,available=True,monthly=pd.DataFrame(results),materials=pd.DataFrame(material_details),missing=[])
