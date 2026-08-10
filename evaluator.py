"""
Evaluation pipeline for zero-shot LLM extraction of chemical and radiological events.

This script reproduces the entity-level precision, recall, and F1 scores reported
in the manuscript using a 200-article gold-standard dataset annotated by human reviewers.

Author: Damian Honeyman et al.
"""

import argparse, itertools, collections, re
import pandas as pd
from scipy.stats import beta

LABEL_MAP = {
    "chemical substance": "CHEMICAL SUBSTANCE", "radiological substance": "RADIOLOGICAL SUBSTANCE",
    "fatality count": "FATALITY COUNT", "case number": "CASE NUMBER",
    "state": "STATE", "county": "COUNTY", "city": "CITY", "location": "LOCATION",
    "timeframe": "TIMEFRAME", "answer": "Answer", "adverbs": "Adverbs",
    "years": "Years", "months": "Months", "dates": "Dates", "date": "date",
    "dates or days of the week": "Dates or days of the week",
}

def normalize_label(label):
    return LABEL_MAP.get(str(label).strip().lower(), str(label).strip())

def parse(text):
    if pd.isna(text) or str(text).strip() == "":
        return {}
    out, current = {}, None
    for raw in str(text).splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("-"):
            line = line[1:].strip()
        if ":" in line:
            k, v = line.split(":", 1)
            k = normalize_label(k); v = v.strip()
            out.setdefault(k, []); current = k
            if v:
                out[k].append(v)
        elif current is not None:
            out.setdefault(current, []).append(line)
    for k, vs in out.items():
        seen, dedup = set(), []
        for v in vs:
            s = str(v).strip()
            if s.lower() not in seen:
                dedup.append(s); seen.add(s.lower())
        out[k] = dedup
    return out

NULL_TOKEN = "\x00NULL"
NULL_EXACT = {"not mentioned", "not reported", "unknown", "none", "n/a",
              "not applicable", "not specified", "not stated", "not provided"}

MALFORMED = re.compile(r"adverbs of time convey", re.I)

def canon(v):
    s = str(v).strip().lower().rstrip(".")
    if s in NULL_EXACT:
        return NULL_TOKEN
    if re.match(r"^not mentioned\b", s):
        return NULL_TOKEN
    if s.endswith(": not mentioned"):
        return NULL_TOKEN
    return s

def score_set(pred_vals, gold_vals):
    """C2: a null present on both sides is one true negative."""
    ps = {canon(v) for v in pred_vals}
    gs = {canon(v) for v in gold_vals}
    tp = fp = fn = tn = 0
    for v in ps & gs:
        if v == NULL_TOKEN:
            tn += 1
        else:
            tp += 1
    for v in ps - gs:
        fp += 1
    for v in gs - ps:
        fn += 1
    return tp, fp, fn, tn

def cp(x, n, alpha=0.05):
    if n == 0:
        return None, None
    x, n = int(x), int(n)
    lo = 0.0 if x == 0 else float(beta.ppf(alpha / 2, x, n - x + 1))
    hi = 1.0 if x == n else float(beta.ppf(1 - alpha / 2, x + 1, n - x))
    return lo, hi

def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f

ENTITY_FIELDS = {                      
    "Fatality count":          ["FATALITY COUNT"],
    "Case number":             ["CASE NUMBER"],
    "Chemical substance":      ["CHEMICAL SUBSTANCE"],
    "Radiological substance":  ["RADIOLOGICAL SUBSTANCE"],
    "Country":                 ["LOCATION"],
    "State":                   ["STATE"],
    "County":                  ["COUNTY"],
    "City":                    ["CITY"],
}

TEMPORAL_FIELDS = {
    "Date and Timeframe":    ["date"],      
    "Temporal expressions":  ["Dates"],
}

TEMPORAL_DIAGNOSTIC = {
    "[not reported] TIMEFRAME field":  ["TIMEFRAME"],
    "[not reported] Years field":      ["Years"],
    "[not reported] Months field":     ["Months"],
    "[not reported] Adverbs field":    ["Adverbs"],
}

TEMPORAL_POOLED = {
    "[sensitivity] Date and Timeframe pooled (date+TIMEFRAME)":       ["date", "TIMEFRAME"],
    "[sensitivity] Temporal expressions pooled (Y+M+Dates+Adverbs)":  ["Years", "Months", "Dates", "Adverbs"],
}

REPORTED_FIELDS = set(itertools.chain(*ENTITY_FIELDS.values())) | \
                  set(itertools.chain(*TEMPORAL_FIELDS.values())) | \
                  set(itertools.chain(*TEMPORAL_DIAGNOSTIC.values())) | {"Answer"}

def evaluate(path):
    df = pd.read_excel(path)
    n_articles = len(df)
    parsed = [(parse(r["gpt4"]), parse(r["feedback"])) for _, r in df.iterrows()]

    ev = collections.Counter()
    for P, G in parsed:
        p = (P.get("Answer", [""]) or [""])[0].strip().lower()
        g = (G.get("Answer", [""]) or [""])[0].strip().lower()
        ev[(p == "yes", g == "yes")] += 1
    e_tp = ev[(True, True)]; e_tn = ev[(False, False)]
    e_fp = ev[(True, False)]; e_fn = ev[(False, True)]

    results = []
    p, r, f = prf(e_tp, e_fp, e_fn)
    results.append(dict(entity="Event identification", tp=e_tp, fp=e_fp, fn=e_fn, tn=e_tn,
                        precision=p, recall=r, f1=f,
                        p_ci=cp(e_tp, e_tp + e_fp), r_ci=cp(e_tp, e_tp + e_fn),
                        specificity=e_tn / (e_tn + e_fp) if e_tn + e_fp else None,
                        note="document-level binary, n=%d" % n_articles))

    def add(label, fields, note=""):
        t = [0, 0, 0, 0]
        for P, G in parsed:
            pv = [v for v in itertools.chain(*[P.get(k, []) for k in fields]) if not MALFORMED.search(str(v))]
            gv = [v for v in itertools.chain(*[G.get(k, []) for k in fields]) if not MALFORMED.search(str(v))]
            if not pv and not gv:
                continue
            s = score_set(pv, gv)
            for i in range(4):
                t[i] += s[i]
        tp, fp, fn, tn = t
        p, r, f = prf(tp, fp, fn)
        results.append(dict(entity=label, tp=tp, fp=fp, fn=fn, tn=tn,
                            precision=p, recall=r, f1=f,
                            p_ci=cp(tp, tp + fp), r_ci=cp(tp, tp + fn),
                            specificity=None, note=note or "+".join(fields)))

    for label, fields in ENTITY_FIELDS.items():
        add(label, fields)
    for label, fields in TEMPORAL_FIELDS.items():
        add(label, fields)
    for label, fields in TEMPORAL_DIAGNOSTIC.items():
        add(label, fields)
    for label, fields in TEMPORAL_POOLED.items():
        add(label, fields)

    excluded = collections.Counter()
    for P, G in parsed:
        for d in (P, G):
            for k, vs in d.items():
                if k not in REPORTED_FIELDS:
                    excluded[k] += len(vs)

    return pd.DataFrame(results), excluded, n_articles, ev


def fmt(ci):
    return "n/a" if ci is None or ci[0] is None else "%.2f-%.2f" % ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="evaluation_dataset_200_articles.xlsx")
    ap.add_argument("--output", default="chem_llm_eval_results.csv")
    a = ap.parse_args()

    res, excl, n, ev = evaluate(a.input)

    print("Articles evaluated: %d\n" % n)
    print("Event identification 2x2 (document level)")
    print("  pred Yes / gold Yes : %3d   (TP)" % ev[(True, True)])
    print("  pred No  / gold No  : %3d   (TN)" % ev[(False, False)])
    print("  pred Yes / gold No  : %3d   (FP)" % ev[(True, False)])
    print("  pred No  / gold Yes : %3d   (FN)" % ev[(False, True)])
    print("  total               : %3d\n" % sum(ev.values()))

    hdr = "%-58s %5s %4s %4s %5s  %-6s %-12s  %-6s %-12s %-6s" % (
        "Entity", "TP", "FP", "FN", "TN", "Prec", "95% CI", "Rec", "95% CI", "F1")
    print(hdr); print("-" * len(hdr))
    for _, x in res.iterrows():
        print("%-58s %5d %4d %4d %5d  %-6.3f %-12s  %-6.3f %-12s %-6.3f" % (
            x.entity, x.tp, x.fp, x.fn, x.tn,
            x.precision, fmt(x.p_ci), x.recall, fmt(x.r_ci), x.f1))

    print("\nEntity keys present in annotations but NOT reported (%d values):" % sum(excl.values()))
    for k, v in excl.most_common():
        print("  %5d  %s" % (v, k))

    out = res.copy()
    out["precision_ci_low"] = [c[0] if c else None for c in out.p_ci]
    out["precision_ci_high"] = [c[1] if c else None for c in out.p_ci]
    out["recall_ci_low"] = [c[0] if c else None for c in out.r_ci]
    out["recall_ci_high"] = [c[1] if c else None for c in out.r_ci]
    out.drop(columns=["p_ci", "r_ci"]).to_csv(a.output, index=False)
    print("\nSaved:", a.output)


if __name__ == "__main__":
    main()
    main()
