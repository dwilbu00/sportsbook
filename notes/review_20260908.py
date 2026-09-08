"""Read-only independent-review reproductions. Run from the repository root.

    python notes/review_20260908.py nfl|props|wiring|mlb|totals|benchmark

Uses the existing local mirror; outbound requests are blocked. No model saves,
database connections, odds pulls, or production mutations are performed.
"""
import collections
import datetime as dt
import json
import math
from pathlib import Path
import socket
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def blocked(*args, **kwargs):
    raise RuntimeError("Independent review: outbound network disabled")


socket.socket.connect = blocked
socket.create_connection = blocked
import requests

requests.sessions.Session.request = blocked


def emit(label, value):
    print(label, json.dumps(value, default=str), flush=True)


def nfl():
    import nfl_model as m
    import nfl_data as data
    import nfl_qb_asof as q
    import nfl_schedule as schedule
    from nfl_market_scan import _load_closing_odds
    from r2_sharp import fair_two_way
    from nfl_injury_edge import _solve, _rmse

    seasons = [2023, 2024, 2025]
    rows = m.build_dataset(seasons)
    games = schedule.game_index([str(s) for s in seasons])
    projected, first_rows = [], []
    first = {}
    mismatches = []
    for s in seasons:
        majority = q.game_starters(s)
        for p in q._dropbacks(s):
            first.setdefault(p['game_id'], {}).setdefault(p['posteam'], p['passer_id'])
        names = data.player_week([str(s)]).set_index('player_id')['player_display_name'].to_dict()
        for gid, teams in majority.items():
            for team, pid in teams.items():
                fp = first.get(gid, {}).get(team)
                if fp != pid:
                    mismatches.append((gid, team, names.get(fp, fp), names.get(pid, pid)))
    emit('majority_vs_first_passer', {'n_team_games': len(mismatches), 'examples': mismatches[:12]})
    for r in rows:
        s, gid = r[7:9]
        g = games[gid]
        date, h, a = str(g['gameday'])[:10], g['home_team'], g['away_team']
        qr, _ = q.qb_ratings(s, date)
        pr = q.projected_qb_delta(s, date, h, qr) - q.projected_qb_delta(s, date, a, qr)
        fs = first.get(gid, {})
        fr = q.qb_edge_delta(s, date, h, fs.get(h), qr) - q.qb_edge_delta(s, date, a, fs.get(a), qr)
        projected.append((*r[:2], pr, *r[3:]))
        first_rows.append((*r[:2], fr, *r[3:]))
    emit('sample', {'n': len(rows), 'weeks': dict(collections.Counter(int(r[8].split('_')[1]) for r in rows))})
    preds = {}
    for name, rs in [('majority', rows), ('projected', projected), ('first_passer', first_rows)]:
        preds[name] = {}
        errors = []
        folds = []
        for s in seasons:
            tr, te = [r for r in rs if r[7] != s], [r for r in rs if r[7] == s]
            b = _solve([r[:6] for r in tr], [r[6] for r in tr])
            sigma = _rmse([sum(x*y for x,y in zip(b,r[:6])) for r in tr], [r[6] for r in tr])
            pp = [sum(x*y for x,y in zip(b,r[:6])) for r in te]
            folds.append((s, len(te), _rmse(pp, [r[6] for r in te])))
            for r, p in zip(te, pp):
                preds[name][r[8]] = (p, sigma, r[6])
                errors.append((p-r[6])**2)
        emit('loso_'+name, {'folds': folds, 'pooled_rmse': math.sqrt(sum(errors)/len(errors))})
        tr, te = [r for r in rs if r[7] < 2025], [r for r in rs if r[7] == 2025]
        b = _solve([r[:6] for r in tr], [r[6] for r in tr])
        emit('forward_2025_'+name, {'n': len(te), 'rmse': _rmse([sum(x*y for x,y in zip(b,r[:6])) for r in te], [r[6] for r in te])})
    for requested in ([2023,2024,2025], [2023,2024,2025,2026]):
        odds = _load_closing_odds([str(s) for s in requested], 'draftkings', 'closing')
        scored = []
        for o in odds.values():
            gid, _ = schedule.resolve_event(o['home'],o['away'],o['commence_time'],index=games)
            if gid not in preds['majority']: continue
            sp, mlh, mla = o['spread'].get('home'), o['ml'].get('home'), o['ml'].get('away')
            if not sp or mlh is None or mla is None: continue
            fair, _ = fair_two_way(mlh, mla)
            if fair is None: continue
            p, sig, y = preds['majority'][gid]
            prob = .5*(1+math.erf(p/sig/math.sqrt(2)))
            scored.append((gid,p,sig,y,-sp[0],prob,fair))
        def metrics(rr):
            n=len(rr)
            return {'n':n,'model_rmse':math.sqrt(sum((r[1]-r[3])**2 for r in rr)/n),
                    'close_rmse':math.sqrt(sum((r[4]-r[3])**2 for r in rr)/n),
                    'model_brier':sum((r[5]-(r[3]>0))**2 for r in rr)/n,
                    'market_brier':sum((r[6]-(r[3]>0))**2 for r in rr)/n}
        emit('paired_close_'+str(requested[-1]),metrics(scored))
        if requested[-1]==2025:
            emit('disagree_gt15pct',metrics([r for r in scored if abs(r[5]-r[6])>.15]))
    emit('score_mismatches', [(r[8],r[6],games[r[8]]['home_score']-games[r[8]]['away_score']) for r in rows if r[6] != games[r[8]]['home_score']-games[r[8]]['away_score']])


def props():
    import nfl_props_scan as p
    import nfl_data as d
    seasons=['2023','2024','2025']
    _, _, meta=p.scan(seasons,drill=('player_receptions','OVER','0-1'))
    snaps=d.snap_counts(seasons)
    snapidx={}
    for r in snaps.to_dict('records'):
        k=(p._norm(r['player']),str(r['season']),str(r['week']))
        snapidx.setdefault(k,[]).append(r)
    classified=[]
    for s,w,name,point,seen in meta['drill_dnp']:
        rr=snapidx.get((p._norm(name),s,w),[])
        n=sum(float(r.get('offense_snaps') or 0)+float(r.get('defense_snaps') or 0)+float(r.get('st_snaps') or 0) for r in rr)
        classified.append((s,w,name,n,[(r['team'],r['offense_snaps'],r['st_snaps']) for r in rr]))
    profits=[r[7] for r in meta['drill_rows']]
    nactive=sum(r[3]>0 for r in classified)
    emit('receptions_05',{'as_graded':p._stats(profits),'drops':len(classified),'drop_with_positive_snaps':nactive,
         'roi_counting_only_snap_confirmed_zeros':(sum(profits)-nactive)/(len(profits)+nactive)})
    emit('missing_player_week_snap_audit',classified)
    roster=d.rosters(seasons)
    pbp=d.pbp(seasons)
    target_audit=collections.Counter()
    for season,week,name,_,_ in meta['drill_dnp']:
        matches=roster[(roster.season==int(season)) & (roster.week==int(week)) &
                       (roster.full_name.map(p._norm)==p._norm(name))]
        ids=set(matches.gsis_id.dropna())
        targets=pbp[(pbp.season==int(season)) & (pbp.week==int(week)) &
                    pbp.receiver_player_id.isin(ids)]
        label='no_roster_id' if not ids else ('has_targets' if len(targets) else 'no_targets')
        target_audit[label]+=1
    emit('missing_player_week_target_audit',target_audit)


def wiring():
    import nfl_model as m
    import nfl_epa as e
    import nfl_data as d
    import nfl_qb_asof as q
    import nfl_schedule as s
    import nfl_injury_impact as inj
    import feature_store as fs
    games=s.game_index(['2024'])
    ctx=d.game_context(['2024'])
    gid=next(gid for gid,c in ctx.items() if c['rest_edge_home'] and int(c['week'])>3)
    g=games[gid]; date=str(g['gameday'])[:10]; h,a=g['home_team'],g['away_team']
    features=e.build_matchup_features(e.TEAM_ABBR_TO_NAME[h],e.TEAM_ABBR_TO_NAME[a],date,2024)
    with_gid=m.predict(h,a,date,2024,gid=gid)
    emit('rest_fit_serve',{'gid':gid,'rest':ctx[gid]['rest_edge_home'],'served':features.get('nfl_pred_margin'),'with_gid':with_gid,'expected_difference':m.weights()['w_rest']*ctx[gid]['rest_edge_home']})
    night=next(g for g in games.values() if str(g.get('gametime',''))>='20:00' and int(g['week'])>3)
    gd=str(night['gameday'])[:10]; utc=(dt.date.fromisoformat(gd)+dt.timedelta(days=1)).isoformat()
    emit('utc_week_lookup',{'game':night['game_id'],'local':gd,'utc_date':utc,'local_week':q._week_for(2024,night['home_team'],gd),'utc_week':q._week_for(2024,night['home_team'],utc)})
    qb=[]
    for (team,week),rr in inj._injuries_team_week(2024).items():
        for pid,name,pos,status in rr:
            off,_=inj._prior_share(inj._snap_index(2024),inj._norm(name),team,week)
            if pos=='QB' and off>=inj.REGULAR_SHARE:
                value=abs(sum(v for w,v in inj._epa_index(2024).get(pid,{}).items() if w<week))
                if value: qb.append((team,week,name,value))
    emit('qbs_counted_in_non_qb_injuries',{'n':len(qb),'examples':qb[:8]})
    caches=[]
    for f in Path(fs.CACHE_DIR).glob('americanfootball_nfl*.json'):
        blob=json.loads(f.read_text(encoding='utf-8')); rows=blob.get('rows',[])
        caches.append({'file':f.name,'version':blob.get('version'),'rows':len(rows),'with_model':sum('nfl_pred_margin' in (r[1] or {}) for r in rows)})
    emit('existing_nfl_feature_caches',caches)
    for sport in ('baseball_mlb','americanfootball_nfl'):
        blob=json.loads((ROOT/'calibration'/f'{sport}.json').read_text())
        emit('calibration_'+sport,{k:blob.get(k) for k in ('prob_shrink','market_blend')})


def mlb():
    import book_line_calibration as b
    from unittest.mock import patch
    logs=[{'game_pk':100+i,'game_date':(dt.date(2024,7,2)-dt.timedelta(days=i)).isoformat(),'completed':True,'H':float(i%3)} for i in range(20)]
    row={'game_pk':999,'game_date':'2024-07-03','prop_key':'batter_hits','line':.5}
    out=[]
    with patch.object(b,'_role_matches_gamelog',return_value=True),patch.object(b,'_stat_label_for',return_value='H'):
        b._match_rows_to_gamelog(logs,[row],True,'baseball',out,{'no_game':0})
    emit('explicit_game_pk_mismatch',{'input_pk':999,'output':[(r['game_pk'],r['test_game']['game_pk'],r['game_date'],r['test_game']['game_date']) for r in out]})
    import warehouse_mirror as wm
    for sport in ('baseball_mlb','americanfootball_nfl'):
        counts=[]
        for season in ('2023','2024','2025','2026'):
            name=wm._prop_file(sport,'draftkings',season)
            frame=wm._read(name)
            if frame is not None:
                counts.append((season,len(frame),dict(frame['source'].value_counts()),dict(frame['prop_key'].value_counts())))
        emit('local_props_'+sport,counts)
    import scenario_backtest as scenario
    with patch.object(scenario, '_ALL_PROPS', ('batter_strikeouts',)):
        rows, cov = scenario.scenario_prop_roi('baseball_mlb',['2024','2025','2026'])
    cell=[r for r in rows if r['line']==1.5 and r['side']=='UNDER']
    emit('mlb_survivor_actual_replication', {k:v._asdict() for k,v in scenario.r2_grade.by_key(cell,lambda r:r['season']).items()})
    import calibration_loader as cl
    import tempfile
    with tempfile.TemporaryDirectory(prefix='sportsbook-review-') as tempdir:
        with patch.object(cl,'CALIBRATION_DIR',tempdir):
            live={'props':{'batter_hits':{'method':'C','n_obs':100,'residual_mu':0,'line_methods':{'0.5':{'method':'D'}}}}}
            cand={'props':{'batter_hits':{'method':'C','n_obs':100,'residual_mu':100}}}
            Path(cl.calibration_path('baseball_mlb')).write_text(json.dumps(live))
            Path(cl.candidate_path('baseball_mlb')).write_text(json.dumps(cand))
            emit('candidate_diff_hides_prop_changes',cl.diff_calibration('baseball_mlb'))
    logs=wm.calib_gamelogs_bulk_full('batter',2024)
    frame=wm._read(wm._prop_file('baseball_mlb','draftkings','2024'))
    frame=frame[(frame['prop_key']=='batter_hits') & (frame['source']=='closing') & (frame['direction']=='OVER')]
    examples=[]
    count=0
    for row in frame.to_dict('records'):
        log=logs.get(str(row.get('player_mlb_id')),[])
        if not log or row['game_pk'] in {r.get('game_pk') for r in log}: continue
        row['line']=row['point']
        out=[]
        b._match_rows_to_gamelog(log,[row],True,'baseball',out,{'no_game':0})
        for r in out:
            if r['game_pk'] != r['test_game']['game_pk']:
                count+=1
                if len(examples)<8: examples.append((r['player'],r['game_pk'],r['test_game']['game_pk'],r['game_date'],r['test_game']['game_date']))
    emit('mlb_real_line_helper_mismatches_2024_hits',{'n':count,'examples':examples,'scope':'raw mirror closing OVER lines into join helper; upstream eligibility not applied'})


def totals():
    import nfl_totals as t
    import nfl_schedule as s
    import nfl_epa as e
    from nfl_injury_edge import _solve, _rmse
    seasons=[2023,2024,2025]
    rows=t.build_rows(seasons)
    games=s.game_index([str(y) for y in seasons])
    corrected=[]
    for r in rows:
        g=games[r[8]]
        ratings=e.team_epa(r[7],as_of_date=str(g['gameday'])[:10])
        h,a=ratings[g['home_team']],ratings[g['away_team']]
        total_epa=h['off_epa']+a['off_epa']+h['def_epa']+a['def_epa']
        corrected.append((r[0],total_epa,*r[2:]))
    for name,rs in [('current_subtraction',rows),('corrected_addition',corrected)]:
        errors=[]; folds=[]
        for year in seasons:
            tr,te=[r for r in rs if r[7]!=year],[r for r in rs if r[7]==year]
            b=_solve([r[:6] for r in tr],[r[6] for r in tr])
            preds=[sum(x*y for x,y in zip(b,r[:6])) for r in te]
            errors.extend((p-r[6])**2 for p,r in zip(preds,te))
            folds.append((year,_rmse(preds,[r[6] for r in te])))
        emit('totals_'+name,{'n':len(rs),'pooled_rmse':math.sqrt(sum(errors)/len(errors)),'folds':folds})


def benchmark():
    import pandas as pd
    import warehouse_mirror as wm
    import nfl_model as m
    import nfl_schedule as s
    from nfl_market_scan import _load_closing_odds
    from nfl_injury_edge import _solve, _rmse
    from r2_sharp import fair_two_way
    seasons=[2023,2024,2025]
    frames=[wm._read(wm._team_file('americanfootball_nfl','draftkings',str(y))) for y in (*seasons,2026)]
    df=pd.concat(frames); df=df[df.source=='closing'].copy()
    df['lead']=(pd.to_datetime(df.commence_time,utc=True)-pd.to_datetime(df.captured_at,utc=True)).dt.total_seconds()/60
    post=set(df.loc[df.lead<0,'event_id']); stale=set(df.loc[df.lead>60,'event_id'])
    emit('closing_integrity',{'rows':len(df),'post_kickoff_rows':int((df.lead<0).sum()),'post_kickoff_events':len(post),'over_60min_events':len(stale),'max_lead_min':float(df.lead.max())})
    emit('post_kickoff_examples',df[df.lead<0][['home','away','commence_time','captured_at','lead']].drop_duplicates().to_dict('records'))
    rows=m.build_dataset(seasons); pred={}
    for y in seasons:
        tr,te=[r for r in rows if r[7]!=y],[r for r in rows if r[7]==y]
        b=_solve([r[:6] for r in tr],[r[6] for r in tr])
        sigma=_rmse([sum(x*z for x,z in zip(b,r[:6])) for r in tr],[r[6] for r in tr])
        for r in te: pred[r[8]]=(sum(x*z for x,z in zip(b,r[:6])),sigma,r[6])
    idx=s.game_index([str(y) for y in seasons])
    odds=_load_closing_odds(['2023','2024','2025','2026'],'draftkings','closing')
    for label,exclude in [('all_calendar_files',set()),('reject_post_kickoff',post),('pre_game_within_60min',post|stale)]:
        rr=[]
        for eid,o in odds.items():
            if eid in exclude: continue
            gid,_=s.resolve_event(o['home'],o['away'],o['commence_time'],index=idx)
            if gid not in pred or not o['spread'].get('home'): continue
            fair,_=fair_two_way(o['ml'].get('home'),o['ml'].get('away'))
            if fair is None: continue
            pm,sg,y=pred[gid]; pc=.5*(1+math.erf(pm/sg/math.sqrt(2)))
            rr.append(((pm-y)**2,(-o['spread']['home'][0]-y)**2,(pc-(y>0))**2,(fair-(y>0))**2))
        av=[sum(r[i] for r in rr)/len(rr) for i in range(4)]
        emit('benchmark_'+label,{'n':len(rr),'model_rmse':math.sqrt(av[0]),'close_rmse':math.sqrt(av[1]),'model_brier':av[2],'market_brier':av[3]})


if __name__=='__main__':
    {'nfl':nfl,'props':props,'wiring':wiring,'mlb':mlb,'totals':totals,'benchmark':benchmark}[sys.argv[1]]()
