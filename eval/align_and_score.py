"""Exact two-stage alignment, portable and independent of the source repository.

Adapted from chart_encoding/core.py and chart_encoding_v2/shared_fields.py.
Unlike those historical scorers, structure is optimized BEFORE field identity;
every structural tie is searched, and sharing is the primary field objective.
"""
from collections import Counter, defaultdict
from itertools import combinations, permutations
import math

from scipy.optimize import linear_sum_assignment

VERSION = "external_channel_sharing_matched_units_v3"
METRICS = ("variation", "exact_group_structure", "channels", "repetition_slots",
           "repeated_groups", "field_bindings", "encoding", "sharing", "sharing_all_groups", "link_targets")


class SearchLimitError(RuntimeError):
    pass


def rates(tp, predicted, reference):
    return dict(tp=int(tp), predicted=int(predicted), reference=int(reference),
                precision=tp / predicted if predicted else None,
                recall=tp / reference if reference else None,
                f1=2 * tp / (predicted + reference) if predicted + reference else None)


def external_bindings(chart):
    """One view of both legacy repetition fields and channel-specific externals."""
    return chart.get('external_encodings', {'position': chart.get('repetition_position', [])})


def external_slots(chart):
    return {channel: len(value) if isinstance(value, list) else 1
            for channel, value in external_bindings(chart).items()}


def external_overlap(predicted, reference):
    p, r = external_slots(predicted), external_slots(reference)
    return sum(min(p[channel], r[channel]) for channel in p.keys() & r.keys())


def facts(payload):
    return {(c["chart_id"], "encodings", ch, f)
            for c in payload["charts"] for ch, f in c["encodings"].items()} | {
        (c["chart_id"], "repetition_position", "", f)
        for c in payload["charts"] for f in external_bindings(c)['position']} | {
        (c['chart_id'], 'external_encodings', channel, field)
        for c in payload['charts'] for channel, field in external_bindings(c).items()
        if channel != 'position'}


def sharing(edges):
    uses = defaultdict(set)
    for group, scope, channel, field in edges:
        uses[field].add(group)
    # Flat union: two scopes / multiple channels cannot double-count a pair+field.
    return {(a, b, field) for field, groups in uses.items()
            for a, b in combinations(sorted(groups), 2)}


def links(payload):
    return {(c["chart_id"], t) for c in payload["charts"] for t in c.get("link_targets", [])}


def chart_mappings(pc, rc, max_candidates=1000000):
    """All maximum-cardinality same-variation partial injections, including extras."""
    buckets, count = [], 1
    for variation in sorted({c["variation"] for c in pc + rc}):
        pp = sorted(c["chart_id"] for c in pc if c["variation"] == variation)
        rr = sorted(c["chart_id"] for c in rc if c["variation"] == variation)
        count *= math.perm(max(len(pp), len(rr)), min(len(pp), len(rr)))
        buckets.append((pp, rr))
    if max_candidates and count > max_candidates:
        raise SearchLimitError(f"Exact alignment requires {count} group mappings; "
                               f"limit={max_candidates}. Increase --max-candidates or use 0; no approximate score was produced.")

    def generate(index=0, mapping=None):
        mapping = {} if mapping is None else mapping
        if index == len(buckets):
            yield mapping
            return
        pp, rr = buckets[index]
        for perm in permutations(rr if len(pp) <= len(rr) else pp, min(len(pp), len(rr))):
            pairs = zip(pp, perm) if len(pp) <= len(rr) else zip(perm, rr)
            yield from generate(index + 1, {**mapping, **dict(pairs)})
    return generate, count


def structural_score(mapping, pby, rby):
    """Equal weight per native channel and per external binding slot."""
    return sum(len(set(pby[p]["encodings"]) & set(rby[r]["encodings"]))
               + external_overlap(pby[p], rby[r])
               for p, r in mapping.items())


def field_assignment(pe, re, mapping):
    """Exact global injective assignment maximizing (sharing TP, binding TP).

    For fixed groups, each sharing fact contains exactly ONE field. Consequently
    each predicted/reference field pair has an independent integer weight. The
    assignment solver searches the field permutation space exactly, without
    explicitly materializing its factorially many permutations.
    """
    pf, rf = sorted({e[3] for e in pe}), sorted({e[3] for e in re})
    slots = defaultdict(set)
    for g, scope, ch, f in re:
        slots[g, scope, ch].add(f)
    verified_by_pair = defaultdict(set)
    for edge in pe:
        g, scope, ch, f = edge
        for target in slots.get((mapping.get(g), scope, ch), ()):
            verified_by_pair[f, target].add(edge)
    # Binding TP <= min(|pe|, |re|); a unit of sharing always dominates ALL bindings.
    multiplier = min(len(pe), len(re)) + 1
    weights = [[len(sharing(verified_by_pair[p, r])) * multiplier
                + len(verified_by_pair[p, r]) for r in rf] for p in pf]
    fm = {}
    if pf and rf:
        ii, jj = linear_sum_assignment(weights, maximize=True)
        fm = {pf[i]: rf[j] for i, j in zip(ii, jj) if weights[i][j] > 0}
    return fm


def evidence(pe, re, ps, rs, cm, fm):
    verified = {e for e in pe if (cm.get(e[0]), e[1], e[2], fm.get(e[3])) in re}
    recovered_bindings = {(cm[g], scope, ch, fm[f]) for g, scope, ch, f in verified}
    correct_sharing = sharing(verified)
    recovered_sharing = {(*sorted((cm[a], cm[b])), fm[f]) for a, b, f in correct_sharing}
    assert correct_sharing <= ps and recovered_sharing <= rs
    return verified, recovered_bindings, correct_sharing, recovered_sharing


def score(prediction, reference, max_candidates=1000000, trace=None):
    """Inputs must already be validated; trace receives EVERY optimal structure map."""
    pc, rc = prediction["charts"], reference["charts"]
    pby, rby = {c["chart_id"]: c for c in pc}, {c["chart_id"]: c for c in rc}
    generate, total = chart_mappings(pc, rc, max_candidates)
    structural_best = max(structural_score(m, pby, rby) for m in generate())
    pe, re = facts(prediction), facts(reference)
    ps, rs = sharing(pe), sharing(re)
    pl, rl = links(prediction), links(reference)
    best, chosen, ties, eligible, best_sharing_maps = None, None, 0, 0, 0
    for cm in generate():
        if structural_score(cm, pby, rby) != structural_best:
            continue
        eligible += 1
        fm = field_assignment(pe, re, cm)
        verified, recovered, correct, recovered_s = evidence(pe, re, ps, rs, cm, fm)
        correct_links = {(a, b) for a, b in pl if (cm.get(a), cm.get(b)) in rl}
        objective = (len(correct), len(verified), len(correct_links))
        if trace:
            trace(dict(index=eligible, chart_mapping=cm, field_mapping=fm,
                       structure_tp=structural_best, sharing_tp=objective[0],
                       binding_tp=objective[1], link_tp=objective[2]))
        if best is None or objective[0] > best[0]:
            best_sharing_maps = 1
        elif objective[0] == best[0]:
            best_sharing_maps += 1
        if best is None or objective > best:
            best, chosen, ties = objective, (cm, fm), 1
        elif objective == best:
            ties += 1
    cm, fm = chosen
    verified, recovered, correct, recovered_s = evidence(pe, re, ps, rs, cm, fm)
    exact_structure = sum(set(pby[p]["encodings"]) == set(rby[r]["encodings"])
                          and external_slots(pby[p]) == external_slots(rby[r])
                          for p, r in cm.items())
    native_tp = sum(len(set(pby[p]["encodings"]) & set(rby[r]["encodings"])) for p, r in cm.items())
    rep_tp = sum(min(len(external_bindings(pby[p])['position']), len(external_bindings(rby[r])['position'])) for p, r in cm.items())
    repeat_tp = sum(bool(external_bindings(pby[p])['position']) and bool(external_bindings(rby[r])['position']) for p, r in cm.items())
    external_tp = sum(external_overlap(pby[p], rby[r]) for p, r in cm.items())
    correct_links = {(a, b) for a, b in pl if (cm.get(a), cm.get(b)) in rl}
    predicted_counts = Counter(e[0] for e in pe)
    reference_counts = Counter(e[0] for e in re)
    correct_counts = Counter(e[0] for e in verified)
    unit_exact = sum(correct_counts[p] == predicted_counts[p] == reference_counts[r]
                     for p, r in cm.items())
    matched_ref_groups = set(cm.values())
    result = dict(metric_version=VERSION, exact_search=True, total_group_candidates=total,
                  structural_optima=eligible, sharing_optimal_group_maps=best_sharing_maps,
                  final_tied_group_maps=ties, structure_tp=structural_best,
                  chart_mapping=cm, field_mapping=fm,
                  field_ties="One optimal injective field witness per group map; all have the same optimized scores.",
                  unit_exact_match=dict(correct=unit_exact, aligned=len(cm),
                                        rate=unit_exact / len(cm) if cm else None),
                  exact_match=len(cm) == len(pc) == len(rc) and len(verified) == len(pe) == len(re)
                              and len(correct) == len(ps) == len(rs))
    result["metrics"] = {
        "variation": rates(len(cm), len(pc), len(rc)),
        "exact_group_structure": rates(exact_structure, len(pc), len(rc)),
        "channels": rates(native_tp, sum(len(c["encodings"]) for c in pc), sum(len(c["encodings"]) for c in rc)),
        "repetition_slots": rates(rep_tp, sum(len(external_bindings(c)['position']) for c in pc), sum(len(external_bindings(c)['position']) for c in rc)),
        "repeated_groups": rates(repeat_tp, sum(bool(external_bindings(c)['position']) for c in pc), sum(bool(external_bindings(c)['position']) for c in rc)),
        "field_bindings": rates(len(verified), len(pe), len(re)),
        # Structural encoding recovery is conditional on aligned units and
        # independent of field identity; field_bindings checks the global map.
        "encoding": rates(native_tp + external_tp,
            sum(predicted_counts[p] for p in cm), sum(reference_counts[r] for r in cm.values())),
        "sharing": rates(len(correct), sum(a in cm and b in cm for a, b, f in ps),
                         sum(a in matched_ref_groups and b in matched_ref_groups for a, b, f in rs)),
        "sharing_all_groups": rates(len(correct), len(ps), len(rs)),
        "link_targets": rates(len(correct_links), len(pl), len(rl)),
    }
    result["missing_reference_sharing"] = sorted(rs - recovered_s)
    result["extra_or_wrong_prediction_sharing"] = sorted(ps - correct)
    result["recovered_reference_sharing"] = sorted(recovered_s)
    result["missing_bindings"] = sorted(re - recovered)
    result["extra_or_wrong_bindings"] = sorted(pe - verified)
    return result
