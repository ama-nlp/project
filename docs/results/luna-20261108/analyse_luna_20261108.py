"""Post-merge analysis of the 2026-11-08 Luna labels (Run 032, E17). Joins key + results."""
import json, glob
from collections import Counter, defaultdict
from math import comb
def fisher(a,b,c,d):
    n1,n2,k,n=a+b,c+d,a+c,a+b+c+d
    p=lambda x: comb(n1,x)*comb(n2,k-x)/comb(n,k); o=p(a)
    return min(1.0,sum(p(x) for x in range(max(0,k-n2),min(k,n1)+1) if p(x)<=o*(1+1e-9)))
def load(s):
    key={k["item_id"]:k for k in map(json.loads,open(f"{s}/key.jsonl"))}
    res={r["item_id"]:r for f in glob.glob(f"{s}/results_*.jsonl") for r in map(json.loads,open(f))}
    return [(key[i],res[i]) for i in key if key[i]["repeat_of"] is None]
out={}
print("== Run 032 (pooled draws): categories among hack-code outputs with a block")
R=load("032"); g=defaultdict(list)
for k,r in R: g[k["run"].split(":")[1]].append((k,r))
cats=["PLAN_PLAIN","PLAN_JUSTIFIED","DISCLAIMER","REJECTED","MONITOR_ONLY","NO_EVIDENCE","UNSURE"]
o32={}
for arm in ["C","A-pen","B-pen","E-pen"]:
    L=g[arm]; c=Counter(r["category"] for _,r in L); n=len(L)
    yes=sum(r["label"]=="YES" for _,r in L); nb=sum(r["label"]!="UNSURE" for _,r in L)
    dr=c["DISCLAIMER"]+c["REJECTED"]; kw=sum(k["mention"] for k,_ in L)
    kwyes=sum(k["mention"] and r["label"]=="YES" for k,r in L)
    monexc=sum(r.get("justification_type")=="monitor_compliance" for _,r in L)
    conc=sum(r.get("concealment_plan") is True for _,r in L); monref=sum(r.get("monitor_reference") is True for _,r in L)
    o32[arm]=dict(n=n,yes=yes,binary=nb,dr=dr,keyword=kw,keyword_and_yes=kwyes,cats=dict(c),monitor_excuse=monexc,concealment_plan=conc,monitor_reference=monref)
    print(f"{arm:6s} n={n} YES {yes}/{nb}={yes/nb:.3f} | keyword {kw}/{n}={kw/n:.3f} (of which judge YES {kwyes}) | D+R {dr}={dr/n:.3f} | MONITOR_ONLY {c['MONITOR_ONLY']} | monitor excuse {monexc} | monitor_reference {monref} | concealment_plan {conc}")
    print("        ", {x:c[x] for x in cats})
C=o32["C"]
for arm in ["A-pen","B-pen","E-pen"]:
    a=o32[arm]; print(f"{arm} vs C: D+R {a['dr']}/{a['n']} vs {C['dr']}/{C['n']} p={fisher(a['dr'],a['n']-a['dr'],C['dr'],C['n']-C['dr']):.3g}; keyword-hit share that is judge YES {a['keyword_and_yes']}/{a['keyword']}={a['keyword_and_yes']/a['keyword']:.3f} vs C {C['keyword_and_yes']}/{C['keyword']}={C['keyword_and_yes']/C['keyword']:.3f}")
conc_items=[(k["run"],r["evidence"]) for k,r in R if r.get("concealment_plan") is True]
print("concealment_plan items:",conc_items[:10])
out["032"]=o32
print("\n== E17 open-honest continued reasoning, split by whether the code has a hack")
R=load("e17"); scans={m:{s["id"]:s for s in json.load(open(f"../e17-61863/{m}.json"))["open-honest"]["samples"]} for m in ["base","k00","k002","klcode"]}
o17={}
for m in ["base","k00","k002","klcode"]:
    L=[(k,r) for k,r in R if k["run"]==f"e17:{m}"]
    rows=[]
    for k,r in L:
        sid=k["sample_id"].rsplit(":",1)[0]; s=scans[m][sid]
        rows.append(dict(hack=bool(k["hack_kinds"]),kw=bool(s["reason_kinds"]),label=r["label"],cat=r["category"],ev=r["evidence"],hn=r["hacks_named"]))
    def rate(F): 
        S=[x for x in rows if F(x)]; y=sum(x["label"]=="YES" for x in S); return y,len(S)
    hy=rate(lambda x:x["hack"]); hyc=rate(lambda x:x["hack"] and not x["kw"]); ny=rate(lambda x:not x["hack"])
    cats_h=Counter(x["cat"] for x in rows if x["hack"] and not x["kw"])
    o17[m]=dict(n=len(rows),hack_yes=hy,hack_clean_yes=hyc,nohack_yes=ny,cats_hack_clean=dict(cats_h))
    print(f"{m:7s} n={len(rows)} | hack code: judge YES {hy[0]}/{hy[1]} | hack code & no keyword hack: YES {hyc[0]}/{hyc[1]} cats {dict(cats_h)} | no hack code: YES {ny[0]}/{ny[1]}")
    for x in [x for x in rows if x["hack"] and not x["kw"] and x["label"]=="YES"][:4]: print("     YES evidence:",x["ev"][:160])
out["e17"]=o17
json.dump(out,open("analysis.json","w"),indent=1)
