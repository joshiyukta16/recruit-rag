import glob
import json
import os
from typing import List

from create_parser import create_parser
from recruitrag.factory import create_pipeline


def get_files(path: str) -> List[str]:
    if os.path.isfile(path):
        return [path]
    return sorted(glob.glob(os.path.join(path, "*")))


def print_matches(job, matches) -> None:
    print(f"\n🎯 {job.title or 'Job'}  |  required: {', '.join(job.required_skills) or '-'}  "
          f"|  preferred: {', '.join(job.preferred_skills) or '-'}  |  min years: {job.min_years or '-'}\n")
    for rank, m in enumerate(matches, 1):
        icon = {"Strong": "🟢", "Moderate": "🟡"}.get(m.recommendation, "🔴")
        print(f"{rank}. {icon} {m.name}  —  {m.overall}/100 ({m.recommendation})")
        print(f"   skills {m.skill_score:.0%} | semantic {m.semantic_score:.0%} | "
              f"experience {m.experience_score:.0%} ({m.years_experience} yrs)")
        print(f"   ✔ matched: {', '.join(m.matched_required + m.matched_preferred) or '-'}")
        print(f"   ✘ missing: {', '.join(m.missing_required) or '-'}")
        if m.explanation:
            print(f"   💬 {m.explanation}")
        print()


def run():
    args = create_parser().parse_args()
    pipeline = create_pipeline()

    if args.command == "reset":
        pipeline.reset()

    elif args.command == "add":
        pipeline.ingest(get_files(args.path))

    elif args.command == "profiles":
        for p in pipeline.store.get_profiles().values():
            print(f"• {p.name} <{p.email}> | {p.years_experience} yrs | {', '.join(p.skills)}")

    elif args.command == "query":
        answer = pipeline.ask(args.prompt, anonymize=args.anonymize, top_k=args.top_k)
        print("\n📎 Sources:")
        for h in answer.hits:
            print(f"  - {h.candidate_id} / {h.section}  (score {h.score:.2f})")
        print(f"\n✨ {answer.text}")

    elif args.command == "match":
        with open(args.job, "r", encoding="utf-8") as f:
            job_text = f.read()
        job, matches = pipeline.match_job(job_text, args.top, args.anonymize, args.explain, args.llm_explain)
        if args.json:
            print(json.dumps([m.to_dict() for m in matches], indent=2))
        else:
            print_matches(job, matches)

    elif args.command == "calibrate":
        with open(args.job, "r", encoding="utf-8") as f:
            rows = pipeline.matcher.raw_scores(f.read())
        print("Raw semantic similarity (mean of top-3 chunks):")
        for name, score in rows:
            print(f"  {name:20s} {score:.3f}")
        print(f"Current scaling: {pipeline.matcher.sem_low} -> 0%, {pipeline.matcher.sem_high} -> 100%")

    elif args.command == "evaluate":
        with open(args.eval_file, "r", encoding="utf-8") as f:
            questions = json.load(f)
        r = pipeline.evaluate_retrieval(questions, args.k)
        for row in r["rows"]:
            print(f"{'✅' if row['hit'] else '❌'} {row['question']}\n   expected {row['expected']} | got {row['retrieved']}")
        print(f"\n📊 Hit@{r['k']}: {r['hit_rate']:.0%} | Recall@{r['k']}: {r['recall']:.0%} | "
              f"MRR: {r['mrr']:.2f} ({r['n_questions']} questions)")
        if args.llm_judge:
            results = pipeline.judge_answers(questions)
            print(f"🧑‍⚖️ LLM-judge score: {sum(x.correct for x in results)}/{len(results)}")


def main():
    try:
        run()
    except RuntimeError as exc:      # e.g. Ollama not running: show the hint, not a stack trace
        print(f"\n⚠️  {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
