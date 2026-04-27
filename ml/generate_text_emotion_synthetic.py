#!/usr/bin/env python3
"""
Generate synthetic (template-based) labeled text for rare emotions.

This is meant to increase class coverage for training/evaluation in a thesis prototype.
It creates diverse English sentences for: anxious, stress, lonely, tired, fear.

Run:
  python ml/generate_text_emotion_synthetic.py --out ml/data/text_emotion_synth.csv --per-class 400
"""
from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


EMOTIONS = ["anxious", "stress", "lonely", "tired", "fear"]


def uniq(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for x in items:
        k = " ".join(x.strip().split()).lower()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(" ".join(x.strip().split()))
    return out


def build_templates() -> dict[str, list[str]]:
    anxious = [
        "I feel anxious about {topic}",
        "I can't stop worrying about {topic}",
        "My mind keeps racing about {topic}",
        "I feel on edge when I think about {topic}",
        "I'm nervous that {event} will go badly",
        "I keep overthinking {topic}",
        "I feel anxious and I can't relax",
        "I feel tense and restless today",
        "I feel anxious and my thoughts won't slow down",
        "I feel uneasy and {symptom}",
        "I feel anxious when {context}",
        "I can't calm down because {topic}",
        "I feel anxious and keep {habit}",
        "I feel anxious and my {body_part} feels {body_feel}",
    ]
    stress = [
        "I'm stressed because of {topic}",
        "I'm overwhelmed with {topic}",
        "I feel under a lot of pressure from {topic}",
        "I feel burned out from {topic}",
        "I'm struggling to keep up with {topic}",
        "I feel stressed and drained after {topic}",
        "I feel stressed and I can't focus on {topic}",
        "I feel stressed even when I try to rest",
        "I feel like I'm drowning in {topic}",
        "I feel stressed and {symptom}",
        "I'm stressed and I keep {habit}",
        "I feel pressure to {goal}",
        "I'm stressed because {event}",
        "I'm overloaded with {topic} and {topic2}",
    ]
    lonely = [
        "I feel lonely even though {context}",
        "I feel isolated and disconnected from people",
        "I don't feel close to anyone right now",
        "I feel like I have no one to talk to",
        "I feel alone with my thoughts",
        "I feel left out and lonely",
        "I miss having someone who understands me",
        "I feel invisible to everyone",
        "I feel lonely and misunderstood",
        "I feel lonely {time_phrase}",
        "I feel alone because {lonely_reason}",
        "I wish I had {support}",
        "I feel disconnected from {people_group}",
        "I feel lonely and I keep {habit}",
    ]
    tired = [
        "I feel exhausted after {topic}",
        "I feel tired all the time lately",
        "I have no energy to do {task}",
        "I feel mentally and physically drained",
        "I feel tired even after sleeping",
        "I feel worn out and unmotivated",
        "I feel tired and I can't concentrate on {topic}",
        "I feel exhausted and overwhelmed",
        "I feel tired and everything feels hard",
        "I'm exhausted because {topic}",
        "I feel tired {time_phrase}",
        "I have no energy to {task2}",
        "I feel drained and my {body_part} feels {body_feel}",
        "I feel tired and I keep {habit}",
    ]
    fear = [
        "I feel scared that {event}",
        "I'm afraid that {event}",
        "I feel terrified about {topic}",
        "I feel frightened and unsafe",
        "I feel scared when {context}",
        "I feel fear and my body feels tense",
        "I feel scared and I want to hide",
        "I feel frightened by my thoughts about {topic}",
        "I feel fearful and on edge",
        "I'm scared because {topic}",
        "I feel afraid when {context}",
        "I feel terrified and my {body_part} feels {body_feel}",
        "I'm scared that {event2}",
        "I feel fear {time_phrase}",
    ]
    return {"anxious": anxious, "stress": stress, "lonely": lonely, "tired": tired, "fear": fear}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("ml/data/text_emotion_synth.csv"))
    p.add_argument("--per-class", type=int, default=400)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rnd = random.Random(args.seed)

    topics = uniq(
        [
            "my exams",
            "my job",
            "my future",
            "money problems",
            "family issues",
            "my relationship",
            "my health",
            "a presentation",
            "an interview",
            "deadlines",
            "school assignments",
            "work expectations",
            "social situations",
            "making mistakes",
            "what people think of me",
            "not meeting my goals",
            "everything happening at once",
        ]
    )
    contexts = uniq(
        [
            "I'm surrounded by people",
            "I'm at home",
            "I try to sleep",
            "I think about tomorrow",
            "I am alone",
            "I am in public",
            "I'm scrolling social media",
            "I have to talk to someone",
            "I open my messages",
        ]
    )
    events = uniq(
        [
            "I will fail",
            "I will mess up",
            "something bad will happen",
            "I will disappoint everyone",
            "I will lose my job",
            "I won't be good enough",
            "I will be judged",
            "I will panic again",
            "I will be alone forever",
        ]
    )
    tasks = uniq(
        [
            "basic chores",
            "my homework",
            "work tasks",
            "getting out of bed",
            "talking to anyone",
            "going outside",
            "making decisions",
            "taking care of myself",
        ]
    )
    tasks2 = uniq(
        [
            "start my day",
            "finish my tasks",
            "cook a meal",
            "clean my room",
            "reply to messages",
            "go to class",
            "go to work",
            "take a shower",
            "exercise",
        ]
    )
    symptoms = uniq(
        [
            "my chest feels tight",
            "my stomach feels upset",
            "I can't sit still",
            "my head hurts",
            "I feel tense",
            "I feel irritable",
            "I keep clenching my jaw",
            "my heart is racing",
            "I feel shaky",
        ]
    )
    habits = uniq(
        [
            "checking my phone",
            "biting my nails",
            "pacing around",
            "replaying conversations in my head",
            "avoiding people",
            "procrastinating",
            "staying in bed",
            "scrolling social media",
            "overthinking everything",
        ]
    )
    goals = uniq(
        [
            "do everything perfectly",
            "meet every deadline",
            "keep everyone happy",
            "be productive all the time",
            "not make any mistakes",
            "handle everything alone",
        ]
    )
    topics2 = uniq(
        [
            "house responsibilities",
            "a difficult conversation",
            "unexpected problems",
            "personal issues",
            "too many commitments",
            "constant notifications",
            "family expectations",
            "health concerns",
        ]
    )
    time_phrases = uniq(
        [
            "most evenings",
            "at night",
            "in the mornings",
            "on weekends",
            "after work",
            "when I'm alone",
            "when I wake up",
            "late at night",
        ]
    )
    lonely_reasons = uniq(
        [
            "my friends are busy",
            "I don't feel understood",
            "I don't fit in",
            "I moved to a new place",
            "I ended a relationship",
            "I can't open up to people",
            "I feel judged",
            "I don't have close friends",
        ]
    )
    supports = uniq(
        [
            "someone to listen to me",
            "a friend to talk to",
            "a real connection",
            "support from people around me",
            "someone who understands",
        ]
    )
    people_groups = uniq(
        [
            "my friends",
            "my family",
            "people around me",
            "my classmates",
            "my coworkers",
        ]
    )
    body_parts = uniq(["chest", "stomach", "hands", "shoulders", "throat", "head"])
    body_feels = uniq(["tight", "heavy", "shaky", "tense", "weak", "numb"])
    events2 = uniq(
        [
            "something bad will happen soon",
            "I will get hurt",
            "I will lose control",
            "I will be rejected",
            "I will be alone",
            "I will fail again",
        ]
    )

    templates = build_templates()
    rows: list[tuple[str, str]] = []

    for emo in EMOTIONS:
        tpls = templates[emo]
        for _ in range(args.per_class * 6):  # oversample then dedupe
            tpl = rnd.choice(tpls)
            s = tpl.format(
                topic=rnd.choice(topics),
                context=rnd.choice(contexts),
                event=rnd.choice(events),
                event2=rnd.choice(events2),
                task=rnd.choice(tasks),
                task2=rnd.choice(tasks2),
                symptom=rnd.choice(symptoms),
                habit=rnd.choice(habits),
                goal=rnd.choice(goals),
                topic2=rnd.choice(topics2),
                time_phrase=rnd.choice(time_phrases),
                lonely_reason=rnd.choice(lonely_reasons),
                support=rnd.choice(supports),
                people_group=rnd.choice(people_groups),
                body_part=rnd.choice(body_parts),
                body_feel=rnd.choice(body_feels),
            )
            rows.append((s, emo))

    # De-dupe within each emotion and trim to per-class
    final_rows: list[tuple[str, str]] = []
    for emo in EMOTIONS:
        emo_texts = uniq([t for (t, e) in rows if e == emo])
        rnd.shuffle(emo_texts)
        emo_texts = emo_texts[: args.per_class]
        final_rows.extend([(t, emo) for t in emo_texts])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "emotion"])
        w.writerows(final_rows)

    print(f"Wrote {len(final_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()

