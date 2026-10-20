import json, sys, datetime
p=sys.argv[1]; c=json.load(open(p))
lst=c['claims'] if isinstance(c,dict) and 'claims' in c else c
v=lst['tb.kappa.fit16'] if isinstance(lst,dict) else [x for x in lst if x.get('id')=='tb.kappa.fit16'][0]
v['recompute']={'fast':{'tol_rel':1e-6},'full':{'tol_rel':1e-6}}
v.setdefault('history',[]).append({'date':datetime.date.today().isoformat(),'change':'recompute tol_rel 1e-9 -> 1e-6',
  'reason':'least-squares fit of band energies differs by 4e-8 relative between macOS (Accelerate) and Linux numpy; printed to one decimal (-0.7)'})
json.dump(c,open(p,'w'),indent=1); print('patched')
