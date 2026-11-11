"""RAG racing advisor: TF-IDF retrieval over a small racing knowledge base + rule-based 'SLM' answer composer.

NOTE: no LLM is available offline, so the generator is a deterministic template composer grounded in the retrieved
passages. `Advisor.generate` is the single place to plug a real LLM/SLM (same retrieved passages as context).
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

# id, tags, speed_factor (fraction of max speed, if the passage gives a target), text
KB = [
    ("sharp_corner", {"sharp"}, 0.40, "Sharp corner ahead: brake early and reduce speed to a target speed of 40 percent of maximum before turn-in, then steer smoothly."),
    ("moderate_corner", {"moderate"}, 0.60, "Moderate corner ahead: lift off the throttle and brake lightly to a target speed of 60 percent of maximum speed."),
    ("gentle_bend", {"gentle"}, 0.85, "Gentle bend ahead: stay on the throttle with a target speed of 85 percent of maximum, small steering corrections."),
    ("straight", {"straight"}, 1.00, "Straight road ahead: full throttle, target speed of 100 percent of maximum, keep the steering neutral."),
    ("left_entry", {"left"}, None, "Left corner: turn the steering to the left smoothly and keep the car close to the inside of the bend."),
    ("right_entry", {"right"}, None, "Right corner: turn the steering to the right smoothly and keep the car close to the inside of the bend."),
    ("high_speed_brake", {"high", "corner"}, None, "At high speed before a corner: brake harder and earlier, because the grip limit is exceeded and the car will skid."),
    ("low_speed_exit", {"low"}, None, "At low speed: accelerate progressively and avoid sharp steering input that wastes time."),
    ("skid", {"high", "sharp", "moderate"}, None, "Avoid abrupt steering at high speed; lateral grip is limited and a skid cuts speed."),
    ("track_limits", {"corner", "sharp", "moderate", "gentle"}, None, "Stay within the track limits: leaving the road ends the lap, so never sacrifice safety for speed."),
    ("smooth_inputs", {"gentle", "moderate", "straight"}, None, "Smooth steering and throttle inputs keep the car stable and reduce lap time."),
    # distractors (irrelevant to on-track control)
    ("tyre_pressure", set(), None, "Tyre pressure should be checked in the garage before the race and adjusted for ambient temperature."),
    ("pit_stop", set(), None, "A pit stop strategy depends on tyre wear, fuel load and safety car periods."),
    ("rain_setup", set(), None, "In wet conditions increase wing angle and soften the suspension for better grip."),
    ("fuel_saving", set(), None, "Lift and coast in long braking zones to save fuel during endurance races."),
    ("radio", set(), None, "Radio communication between driver and engineer should be short and clear."),
]
IDS = [k[0] for k in KB]
DOCS = [k[3] for k in KB]
VMAX = 20.0


def situation_text(direction, sharp, speed):
    if direction == "straight":
        return f"straight road ahead at {speed} speed"
    return f"approaching a {sharp} {direction} corner at {speed} speed"


def relevant_ids(direction, sharp, speed):
    """Ground-truth relevant passages for a situation (used only to score retrieval)."""
    want = {"straight"} if direction == "straight" else {sharp, direction, "corner"}
    if speed in ("high", "low"):
        want.add(speed)
    out = set()
    for kid, tags, _, _ in KB:
        if not tags:
            continue
        # a passage is relevant if all its tags apply, or any non-generic tag applies
        if kid in ("sharp_corner", "moderate_corner", "gentle_bend", "straight"):
            ok = tags <= want
        elif kid in ("left_entry", "right_entry"):
            ok = direction in tags
        elif kid == "high_speed_brake":
            ok = direction != "straight" and speed == "high"
        elif kid == "low_speed_exit":
            ok = speed == "low"
        elif kid == "skid":
            ok = direction != "straight" and speed == "high" and sharp in tags
        elif kid == "track_limits":
            ok = direction != "straight"
        else:  # smooth_inputs
            ok = sharp in tags or direction == "straight"
        if ok:
            out.add(kid)
    return out


class Advisor:
    def __init__(self, k=4):
        self.k = k
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2)).fit(DOCS)
        self.M = self.vec.transform(DOCS)

    def retrieve(self, query):
        sims = (self.M @ self.vec.transform([query]).T).toarray().ravel()
        order = np.argsort(-sims)[: self.k]
        return [int(i) for i in order if sims[i] > 0]

    def generate(self, query, passages):
        """Compose advice strictly from retrieved passages. Returns dict with claims for grounding checks."""
        factor, brake = 1.0, 0
        for i in passages:
            if KB[i][2] is not None:
                factor = KB[i][2]
                break
        text = " ".join(DOCS[i] for i in passages)
        brake = int(bool(re.search(r"\bbrake\b", text)))
        advice = " ".join(DOCS[i] for i in passages[:2])
        return dict(advice=advice, target_factor=factor, brake=brake,
                    claims=[f"{int(round(factor * 100))} percent"] if any(KB[i][2] is not None for i in passages) else [])

    def hint(self, direction, sharp, speed):
        q = situation_text(direction, sharp, speed)
        p = self.retrieve(q)
        g = self.generate(q, p)
        return q, p, g


def grounded_rate_and_hallucination(adv, situations):
    """grounded = every numeric claim in the advice appears verbatim in a retrieved passage."""
    ok = tot = 0
    for d, s, v in situations:
        q, p, g = adv.hint(d, s, v)
        text = " ".join(DOCS[i] for i in p)
        for c in g["claims"]:
            tot += 1
            ok += int(c in text)
    return (ok / tot if tot else 1.0)


def evaluate_rag(k=4):
    adv = Advisor(k)
    sits = [(d, s, v) for d in ("left", "straight", "right") for s in ("gentle", "moderate", "sharp") for v in ("low", "mid", "high")
            if not (d == "straight" and s != "gentle")]
    P, R, hits, fact_ok = [], [], 0, 0
    for d, s, v in sits:
        ids = {IDS[i] for i in adv.retrieve(situation_text(d, s, v))}
        rel = relevant_ids(d, s, v)
        P.append(len(ids & rel) / max(len(ids), 1)); R.append(len(ids & rel) / max(len(rel), 1))
        g = adv.hint(d, s, v)[2]
        want = 1.0 if d == "straight" else {"gentle": 0.85, "moderate": 0.6, "sharp": 0.4}[s]
        fact_ok += int(abs(g["target_factor"] - want) < 1e-6)
    gr = grounded_rate_and_hallucination(adv, sits)
    return dict(k=k, retrieval_precision=float(np.mean(P)), retrieval_recall=float(np.mean(R)),
                grounded_answer_rate=gr, hallucination_rate=1 - gr,
                correct_target_speed_rate=fact_ok / len(sits), n_situations=len(sits))


def build_hint_table():
    """Cache hint for every discrete situation: (dir 0/1/2, sharp 0/1/2, speed bin 0/1/2) -> [factor, brake]."""
    adv = Advisor()
    dn, sn, vn = ["left", "straight", "right"], ["gentle", "moderate", "sharp"], ["low", "mid", "high"]
    tab = np.zeros((3, 3, 3, 2), np.float32)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                g = adv.hint(dn[a], sn[b], vn[c])[2]
                tab[a, b, c] = [g["target_factor"], g["brake"]]
    return tab


if __name__ == "__main__":
    import json
    print(json.dumps(evaluate_rag(), indent=1))
    a = Advisor()
    for s in [("left", "sharp", "high"), ("straight", "gentle", "mid")]:
        q, p, g = a.hint(*s)
        print(q, "->", [IDS[i] for i in p], g["target_factor"], g["brake"])
        print("  ", g["advice"])
