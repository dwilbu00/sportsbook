"""Offline, research-only measurements for the separate improvement study.

Run from the repository root:
    python notes/improvement_study_20260908.py inventory|decisions|performance|nfl|quotes|mlb

Reads committed code and local mirror files. Network is blocked, including free
fallbacks, to prevent implicit provider calls. Writes only the selected aggregate
result to notes/improvement_study_20260908_<command>.json. No model promotion.
"""
import ast
from collections import Counter, defaultdict
import itertools
import hashlib
import json
import math
from pathlib import Path
import platform
import socket
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def blocked(*args, **kwargs):
    raise RuntimeError("Improvement study: provider requests disabled")


socket.socket.connect = blocked
socket.create_connection = blocked
import requests
requests.sessions.Session.request = blocked


def inventory():
    import pyarrow.parquet as pq
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):
                h.update(chunk)
        return h.hexdigest()
    code = []
    for p in ROOT.glob('*.py'):
        source = p.read_text(encoding='utf-8-sig')
        tree = ast.parse(source)
        functions = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
        code.append({'file': p.name, 'lines': len(source.splitlines()),
                     'sha256': digest(p),
                     'test': p.name.startswith('test_'),
                     'functions': len(functions),
                     'largest_function': max((n.end_lineno - n.lineno + 1 for n in functions), default=0)})
    data = []
    for p in sorted((ROOT / 'warehouse_mirror_data').glob('*.parquet')):
        try:
            f = pq.ParquetFile(p)
            data.append({'file': p.name, 'rows': f.metadata.num_rows,
                         'columns': f.schema_arrow.names, 'bytes': p.stat().st_size,
                         'sha256': digest(p)})
        except Exception:
            data.append({'file': p.name, 'status': 'unreadable_or_pointer'})
    return {'python': platform.python_version(), 'modules': len(code),
            'test_modules': sum(r['test'] for r in code),
            'production_lines': sum(r['lines'] for r in code if not r['test']),
            'largest_modules': sorted((r for r in code if not r['test']), key=lambda r: -r['lines'])[:12],
            'source_files': code,
            'mirrors': data}


def quotes():
    import pandas as pd
    import warehouse_mirror as wm
    from odds_client import american_to_decimal
    results=[]
    for sport in ('baseball_mlb','americanfootball_nfl','basketball_nba'):
        books={}
        for book in ('draftkings','fanduel'):
            frames=[wm._read(wm._team_file(sport,book,str(y))) for y in (2023,2024,2025,2026)]
            df=pd.concat([f for f in frames if f is not None],ignore_index=True)
            # Shared point sentinel applies only to moneylines, which have no line.
            df['point_key']=df['point'].fillna('moneyline')
            df['captured']=pd.to_datetime(df.captured_at,utc=True,errors='coerce')
            df['start']=pd.to_datetime(df.commence_time,utc=True,errors='coerce')
            df=df[(df.captured<=df.start)&df.price.notna()].copy()
            keys=['event_id','source','bet_type','selection','point_key']
            # One quote per side/window required; never select best retrospectively.
            df=df[~df.duplicated(keys,keep=False)]
            books[book]=df
        pairs=books['draftkings'].merge(books['fanduel'],on=keys,suffixes=('_dk','_fd'),validate='one_to_one')
        pairs=pairs[((pairs.captured_dk-pairs.captured_fd).dt.total_seconds().abs()<=60)&(pairs.start_dk==pairs.start_fd)].copy()
        pairs['decimal_dk']=pairs.price_dk.map(american_to_decimal)
        pairs['decimal_fd']=pairs.price_fd.map(american_to_decimal)
        for snapshot,sub in pairs.groupby('source'):
            gains=(sub.decimal_fd-sub.decimal_dk).clip(lower=0)
            results.append({'sport':sport,'snapshot':snapshot,'exact_line_side_pairs':len(sub),
                            'distinct_events':int(sub.event_id.nunique()),
                            'fd_better':int((gains>1e-9).sum()),
                            'same_price':int(((sub.decimal_fd-sub.decimal_dk).abs()<1e-9).sum()),
                            'mean_decimal_gain_if_choose_best':float(gains.mean()),
                            'mean_decimal_gain_when_fd_better':float(gains[gains>1e-9].mean()) if (gains>1e-9).any() else 0.0})
    return {'scope':'Historical team quotes only; complete same event/side/line/window; capture timestamps within 60s and both before start; not a selectable bet cohort or realized ROI',
            'results':results}


def mlb():
    import pandas as pd
    import warehouse_mirror as wm
    from stats import hits_at_least
    from props import _dist_p_over
    mechanics=[]
    for ab in (3.49,3.51,4.49,4.51):
        p,meta=_dist_p_over(.25,ab,None,None,None,1.,1.,.5,0.)
        mechanics.append({'expected_ab':ab,'rounded_ab':meta['n_ab_expected'],'p_hit':p})
    metrics=[]
    for season in (2024,2025):
        facts=wm._read(wm._batter_file('baseball_mlb',str(season)))
        facts=facts[(facts.game_type=='R')&facts.AB.notna()&facts.H.notna()].copy()
        facts['pid']=facts.athlete_id.astype(str); facts['gid']=facts.game_pk.astype(int)
        # The target is a recorded outcome, not a guarantee of bookmaker eligibility.
        # Prior AB-positive games replicate only the current D opportunity basis.
        forecasts={}
        for pid,player in facts.groupby('pid'):
            total_ab=total_h=0.; n=0; frequencies=Counter()
            for date,day in player.groupby('official_date',sort=True):
                if n>=20 and total_ab>0:
                    rate=total_h/total_ab; mean=total_ab/n
                    lo=math.floor(mean); frac=mean-lo
                    for row in day.itertuples():
                        if row.AB>=0 and row.H>=0:
                            for line in (.5,1.5):
                                k=int(line)+1
                                forecasts[(pid,row.gid,line)]={'outcome':int(row.H>line),
                                  'rounded':hits_at_least(k,max(1,round(mean)),rate),
                                  'adjacent_mixture':(1-frac)*hits_at_least(k,lo,rate)+frac*hits_at_least(k,lo+1,rate),
                                  'empirical_ab_mixture':sum(c*hits_at_least(k,ab,rate) for ab,c in frequencies.items())/n}
                # Same-day games never enter one another's feature history.
                for row in day.itertuples():
                    if row.AB>0:
                        total_ab+=row.AB; total_h+=row.H; n+=1; frequencies[int(row.AB)]+=1
        odds=wm._read(wm._prop_file('baseball_mlb','draftkings',str(season)))
        odds=odds[(odds.source=='closing')&(odds.prop_key=='batter_hits')&
                  (odds.direction=='OVER')&odds.game_pk.notna()&odds.player_mlb_id.notna()].copy()
        lead=(pd.to_datetime(odds.commence_time,utc=True,errors='coerce')-pd.to_datetime(odds.captured_at,utc=True,errors='coerce')).dt.total_seconds()/60
        odds=odds[(lead>=0)&(lead<=60)]
        odds=odds.drop_duplicates(['game_pk','player_mlb_id','point'],keep=False)
        selected=defaultdict(list)
        for row in odds.itertuples():
            try: pid=str(int(float(row.player_mlb_id)))
            except (ValueError,TypeError): continue
            key=(pid,int(row.game_pk),float(row.point))
            if key in forecasts: selected[key[2]].append(forecasts[key])
        for line,rr in sorted(selected.items()):
            for method in ('rounded','adjacent_mixture','empirical_ab_mixture'):
                metrics.append({'season':season,'line':line,'method':method,'n':len(rr),
                  'brier':statistics.mean((r[method]-r['outcome'])**2 for r in rr),
                  'mean_probability':statistics.mean(r[method] for r in rr),
                  'observed_rate':statistics.mean(r['outcome'] for r in rr)})
    return {'scope':'Exploratory opportunity-mechanism screen at exact-ID DK closing thresholds; 20 prior AB-positive games, same-season history, strict-before-date, quote lead 0..60 min. Empirical H/AB basis only: excludes shipped xBA, contextual adjustments, calibrators and full eligibility/participation rules. No ROI or live-model superiority claim.',
            'rounding_discontinuity':mechanics,'metrics':metrics}


def decisions():
    import bet_selector as bs
    import parlay as p
    from odds_client import parse_player_props, american_to_decimal
    # A conflicts with B and C. B and C are mutually compatible under real rules.
    common = {'event_id': 'fixture-game', 'team': 'A', 'edge_pct': 8.0}
    pool = [('A', 'total', 'UNDER', dict(common, over_expected_roi_pct=0,
             under_expected_roi_pct=10.0, over_hit_rate=40.0)),
            ('B', 'player_prop', None, dict(common, player='B', prop='player_points',
             direction='OVER', expected_roi_pct=9.0, over_rate=60.0)),
            ('C', 'player_prop', None, dict(common, team='B', player='C', prop='player_points',
             direction='OVER', expected_roi_pct=9.0, over_rate=60.0))]
    conflicts = {frozenset(('A', 'B')), frozenset(('A', 'C'))}
    recs=[{'sel_key':key,'bet_type':bt,'side':side,'cand':c,'leg':bs._leg(bt,side,c)} for key,bt,side,c in pool]
    actual_conflicts={frozenset((a['sel_key'],b['sel_key'])) for a,b in itertools.combinations(recs,2)
                      if bs._pair_conflict('basketball_nba',a,b)}
    assert actual_conflicts==conflicts
    selected = bs.select_top_bets(pool, 'basketball_nba', 2)
    ev = {'A': 10, 'B': 9, 'C': 9}
    optimal = max((c for k in range(3) for c in itertools.combinations(ev, k)
                   if not any(frozenset(pair) in conflicts for pair in itertools.combinations(c, 2))),
                  key=lambda c: sum(ev[x] for x in c))
    a = {'leg': {'game_key': 'game', 'bet_type': 'total_under', 'team': None}}
    b = {'leg': {'game_key': 'game', 'bet_type': 'player_prop_over', 'team': 'A',
                  'prop_key': 'player_points', 'player': 'player'}}
    pair_blocked = bs._pair_conflict('basketball_nba', a, b)
    # Joint Bernoulli distribution with p1=p2=.55, rho=-.4 is feasible.
    pr = .55; rho = -.4; joint = pr * pr + rho * pr * (1-pr)
    variance_independent = 2 * pr * (1-pr)
    variance_correlated = variance_independent + 2 * (joint - pr*pr)
    # Synthetic exact-line inventory: FD .5 is a valid offer lost by single-line output.
    def book(key, title, lines):
        return {'key': key, 'title': title, 'markets': [{'key': 'batter_hits',
                'outcomes': [{'description': 'Fixture Player', 'name': side, 'point': line,
                              'price': price} for line, price in lines for side in ('Over','Under')]}]}
    raw = {'id': 'fixture', 'home_team': 'A', 'away_team': 'B',
           'sport_key': 'baseball_mlb', 'bookmakers': [book('draftkings','DraftKings',[(1.5,-110)]),
                                                     book('fanduel','FanDuel',[(.5,-110),(1.5,100)])]}
    parsed = parse_player_props(raw)['props']['batter_hits']['Fixture Player']
    # SGP value is unidentified without ticket price, even if p_joint is known.
    legs = [{'game_key':'same','hist_prob':.55,'odds_price':-110} for _ in range(2)]
    sgp_model = .35
    return {'greedy_counterexample': {'scope': 'synthetic candidates, unmodified production NBA conflict rules and selector',
             'greedy': selected, 'greedy_sum_ev_pct': sum(ev[x] for x in selected),
             'feasible_optimum': optimal, 'optimum_sum_ev_pct': sum(ev[x] for x in optimal)},
            'negative_correlation': {'production_rejects_pair': pair_blocked,
             'p_each': pr, 'rho': rho, 'p_both': joint,
             'sum_bernoulli_variance_independent': variance_independent,
             'sum_bernoulli_variance_correlated': variance_correlated,
             'variance_reduction_fraction': 1 - variance_correlated / variance_independent},
            'single_line_parser': {'raw_fd_lines': [.5,1.5], 'analyzed_line': parsed['line'],
                                   'fd_lines_retained_in_output_offers': [parsed['line'] for o in parsed['offers'] if o['book']=='FanDuel']},
            'sgp_price_sensitivity': {'joint_probability': sgp_model,
              'production_surrogate_roi': p._parlay_expected_roi(legs, sgp_model),
              'hypothetical_ticket_decimal_2_5_roi': sgp_model*2.5-1,
              'hypothetical_ticket_decimal_3_roi': sgp_model*3-1},
            'line_shopping_example': {'probability': .54, 'dk_minus110_roi': .54*american_to_decimal(-110)-1,
                                     'fd_plus100_roi': .54*american_to_decimal(100)-1}}


def performance():
    import parlay as p
    import nfl_epa as e
    import pyarrow.parquet as pq
    # No installed-model or outcome changes: compare identical aggregate semantics.
    plays = e.load_plays(2024)
    dates = sorted({r['game_date'] for r in plays})
    t = time.perf_counter()
    reference = {d: e._aggregate(plays, d) for d in dates}
    scan_time = time.perf_counter() - t
    t = time.perf_counter()
    by_date = defaultdict(list)
    for row in plays:
        by_date[row['game_date']].append(row)
    off, deff = {}, {}
    snapshots = {}
    for d in dates:
        snapshots[d] = {team: {'off_epa':off.get(team,[0,0])[0]/off[team][1] if team in off else e.LEAGUE_AVG_EPA,
                              'def_epa':deff.get(team,[0,0])[0]/deff[team][1] if team in deff else e.LEAGUE_AVG_EPA,
                              'off_plays':off.get(team,[0,0])[1], 'def_plays':deff.get(team,[0,0])[1]}
                        for team in set(off)|set(deff)}
        for row in by_date[d]:
            for table, team in ((off,row['posteam']), (deff,row['defteam'])):
                acc = table.setdefault(team, [0.,0]); acc[0]+=row['epa']; acc[1]+=1
    prefix_time = time.perf_counter() - t
    assert all(set(reference[d]) == set(snapshots[d]) for d in dates)
    max_diff = max(abs(reference[d][team][key] - snapshots[d][team][key])
                   for d in dates for team in reference[d] for key in reference[d][team])
    assert max_diff < 1e-10
    mc=[]
    for n in (3,4,5):
        probs=[.55]*n; matrix=[[float(i==j) for j in range(n)] for i in range(n)]
        t=time.perf_counter()
        samples=[p._gaussian_copula_joint_prob(probs,matrix,seed=seed) for seed in range(30)]
        elapsed=time.perf_counter()-t
        exact=math.prod(probs)
        mc.append({'legs':n,'calls':30,'seconds':elapsed,'exact_independent':exact,
                   'sample_min':min(samples),'sample_max':max(samples),
                   'rmse_vs_exact':math.sqrt(statistics.mean((x-exact)**2 for x in samples)),
                   'theoretical_mc_se':math.sqrt(exact*(1-exact)/5000)})
    path=ROOT/'warehouse_mirror_data'/'nfl_pbp__americanfootball_nfl__2025.parquet'
    reads=[]
    for label,cols in [('all',None),('epa_only',['game_id','game_date','posteam','defteam','epa'])]:
        times=[]
        for _ in range(3):
            t=time.perf_counter(); table=pq.read_table(path, columns=cols); times.append(time.perf_counter()-t)
        reads.append({'columns':label,'rows':table.num_rows,'arrow_bytes':table.nbytes,
                      'median_seconds':statistics.median(times),'timings':times})
    return {'scope':'Local component benchmark; excludes network, Azure, and Streamlit UI',
            'epa_prefix': {'plays':len(plays),'dates':len(dates),'repeated_scan_seconds':scan_time,
                           'prefix_seconds':prefix_time,'ratio':scan_time/prefix_time,'max_abs_difference':max_diff},
            'copula_independence':mc,'parquet_projection':reads}


def nfl():
    import numpy as np
    import nfl_model as m
    import nfl_qb_asof as q
    import nfl_schedule as schedule
    import warehouse_mirror as wm
    import pandas as pd
    from nfl_market_scan import _load_closing_odds
    from r2_sharp import fair_two_way
    print('Building existing feature basis, with projected QB identity...', flush=True)
    rows=m.build_dataset([2023,2024,2025])
    games=schedule.game_index(['2023','2024','2025'])
    corrected=[]
    for r in rows:
        g=games[r[8]]; date=str(g['gameday'])[:10]
        qr,_=q.qb_ratings(r[7],date)
        qb=q.projected_qb_delta(r[7],date,g['home_team'],qr)-q.projected_qb_delta(r[7],date,g['away_team'],qr)
        corrected.append((*r[:2],qb,*r[3:]))
    def normal(z):
        return .5*(1+math.erf(float(z)/math.sqrt(2)))
    predictions=[]; fold_reports=[]
    for season in (2024,2025):
        tr=[r for r in corrected if r[7]<season]
        te=[r for r in corrected if r[7]==season]
        X=np.array([r[:6] for r in tr]); T=np.array([r[:6] for r in te]); y=np.array([r[6] for r in tr])
        scale=X.std(axis=0); scale[scale==0]=1
        center=X.mean(axis=0); center[0]=0
        Z=(X-center)/scale; V=(T-center)/scale
        penalty=np.diag([0,10,10,10,10,10])
        coef=np.linalg.lstsq(X,y,rcond=None)[0]
        margin=T@coef; residual=y-X@coef; sigma=float(np.sqrt(np.mean(residual**2)))
        ridge=np.linalg.solve(Z.T@Z+penalty,Z.T@y)
        ridge_res=y-Z@ridge; ridge_sigma=float(np.sqrt(np.mean(ridge_res**2)))
        # Direct probability model: training ties excluded, same test non-ties.
        keep=y!=0; L=Z[keep]; labels=(y[keep]>0).astype(float); beta=np.zeros(6)
        for _ in range(40):
            pp=1/(1+np.exp(-np.clip(L@beta,-30,30)))
            grad=L.T@(pp-labels)+penalty@beta
            hessian=L.T@((pp*(1-pp))[:,None]*L)+penalty+np.eye(6)*1e-9
            step=np.linalg.solve(hessian,grad); beta-=step
            if np.max(np.abs(step))<1e-8: break
        pp_log=1/(1+np.exp(-np.clip(V@beta,-30,30)))
        fold_reports.append({'test_season':season,'train_n':len(tr),'test_n':len(te),
                             'ols_sigma':sigma,'ridge_sigma':ridge_sigma,'ols_coef':coef.tolist()})
        for i,r in enumerate(te):
            if r[6]==0: continue
            base=normal(margin[i]/sigma)
            empirical=(float(np.sum(residual > -margin[i]))+.5)/(len(residual)+1)
            predictions.append({'game_id':r[8],'season':season,'week':int(r[8].split('_')[1]),
                'actual_margin':r[6],'outcome':int(r[6]>0),'margin':float(margin[i]),
                'ols_normal':base,'ols_shrink_065':.5+.65*(base-.5),
                'ols_empirical_residual':empirical,'ridge_normal':normal(float(V[i]@ridge)/ridge_sigma),
                'direct_logistic':float(pp_log[i])})
    methods=['ols_normal','ols_shrink_065','ols_empirical_residual','ridge_normal','direct_logistic']
    metrics=[]
    def metric(rr,key):
        return {'n':len(rr),'brier':statistics.mean((r[key]-r['outcome'])**2 for r in rr),
                'log_loss':statistics.mean(-math.log(max(1e-12,min(1-1e-12,r[key] if r['outcome'] else 1-r[key]))) for r in rr)}
    rng=np.random.default_rng(20260908)
    groups=sorted({(r['season'],r['week']) for r in predictions})
    for method in methods:
        result={'method':method,'pooled':metric(predictions,method),
                'seasons':{str(s):metric([r for r in predictions if r['season']==s],method) for s in (2024,2025)}}
        sums=np.array([sum((r[method]-r['outcome'])**2-(r['ols_normal']-r['outcome'])**2 for r in predictions if (r['season'],r['week'])==g) for g in groups])
        counts=np.array([sum((r['season'],r['week'])==g for r in predictions) for g in groups])
        # Stratify by season, resample week clusters; exploration, not a promotion CI.
        deltas=[]
        for _ in range(2000):
            ids=np.concatenate([rng.choice([i for i,g in enumerate(groups) if g[0]==s],size=sum(g[0]==s for g in groups),replace=True) for s in (2024,2025)])
            deltas.append(float(sums[ids].sum()/counts[ids].sum()))
        result['paired_brier_delta_vs_ols_95pct_week_bootstrap']=np.quantile(deltas,[.025,.975]).tolist()
        metrics.append(result)
    # Historical closing comparison only, with both quote sides before kickoff.
    frames=[wm._read(wm._team_file('americanfootball_nfl','draftkings',str(y))) for y in (2023,2024,2025,2026)]
    df=pd.concat([f for f in frames if f is not None]); df=df[df.source=='closing'].copy()
    lead=(pd.to_datetime(df.commence_time,utc=True)-pd.to_datetime(df.captured_at,utc=True)).dt.total_seconds()/60
    exclude=set(df.loc[lead.isna() | (lead<0) | (lead>60),'event_id'])
    odds=_load_closing_odds(['2023','2024','2025','2026'],'draftkings','closing')
    by_gid={r['game_id']:r for r in predictions}; matched=[]
    for eid,o in odds.items():
        if eid in exclude: continue
        gid,_=schedule.resolve_event(o['home'],o['away'],o['commence_time'],index=games)
        fair,_=fair_two_way(o['ml'].get('home'),o['ml'].get('away'))
        if gid in by_gid and fair is not None:
            matched.append(dict(by_gid[gid],market=fair))
    return {'scope':'Exploratory expanding-season screen; reused historical seasons, inherited injury publication-time limitations, no ROI or deployment claim',
            'fixed_design':'Projected QB, full rest; ridge/logistic alpha=10 on standardized features; empirical residuals train only; no hyperparameter search',
            'folds':fold_reports,'metrics':metrics,
            'clean_close_matched':{method:metric(matched,method) for method in methods+['market']},
            'prediction_rows':predictions}


if __name__=='__main__':
    command=sys.argv[1]
    result={'command':command,'study_date':'2026-09-08','result':globals()[command]()}
    path=ROOT/'notes'/f'improvement_study_20260908_{command}.json'
    path.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    printable=result['result'].copy()
    if 'mirrors' in printable:
        printable['mirror_files']=len(printable.pop('mirrors'))
    printable.pop('prediction_rows',None)
    print(json.dumps(printable,indent=2,allow_nan=False),flush=True)
    print(f'Saved {path.name}',flush=True)
