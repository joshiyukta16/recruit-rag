from typing import List

from .models import CandidateMatch, JobRequirements, SearchHit
from .llm import ask_llm

SYSTEM_PROMPT = """You are RecruitRAG, an assistant that helps recruiters understand candidate resumes.
Rules:
1. Answer ONLY from the <facts> and <context>. If the answer is not there, say so. Never invent facts.
2. <facts> come from parsed resume data and are reliable; prefer them for "who has skill X" questions.
3. Name each candidate you mention, e.g. "Liam Chen: has Kubernetes and Terraform (lead DevOps role)".
4. For "who" or list questions, mention every matching candidate.
5. Never infer, mention or rank by protected attributes (age, gender, ethnicity, religion, nationality, marital status, disability).
6. Answer in 1-4 short sentences or bullets. Do NOT repeat the context or its headers."""

MATCH_PROMPT = """You write a 2-3 sentence hiring-manager summary of how well a candidate fits a job.
Use ONLY the facts given. Do not contradict them. Do not infer protected attributes."""


def build_context(hits: List[SearchHit]) -> str:
    return "\n\n".join(
        f'<chunk candidate_id="{h.candidate_id}" section="{h.section}">\n{h.content}\n</chunk>' for h in hits
    )


def _looks_broken(answer: str) -> bool:
    """Small local models sometimes echo a chunk header or return nothing useful."""
    a = answer.strip()
    return len(a) < 30 or ("| Section:" in a and len(a) < 150)


def extractive_fallback(hits: List[SearchHit], facts: str) -> str:
    if facts:
        return facts
    seen, lines = set(), []
    for h in hits:
        if h.candidate_id not in seen:
            seen.add(h.candidate_id)
            lines.append(f"- {h.candidate_id} ({h.section})")
    return "Most relevant resume sections (the LLM gave no usable answer):\n" + "\n".join(lines[:5])


def template_explanation(job: JobRequirements, m: CandidateMatch) -> str:
    """Deterministic, always-consistent explanation built only from computed facts."""
    total = len(job.required_skills)
    parts = [f"{m.recommendation} fit ({m.overall}/100): matches {len(m.matched_required)}/{total} required skills"
             if total else f"{m.recommendation} fit ({m.overall}/100)"]
    if m.matched_required:
        parts[0] += f" ({', '.join(m.matched_required)})"
    if m.missing_required:
        parts.append(f"missing {', '.join(m.missing_required)}")
    if m.matched_preferred:
        parts.append(f"also has preferred {', '.join(m.matched_preferred)}")
    exp = f"{m.years_experience} yrs experience"
    if job.min_years:
        exp += f" vs {job.min_years:g} required " + ("(meets)" if m.years_experience >= job.min_years else "(below)")
    parts.append(exp)
    return "; ".join(parts) + "."


class LLMResponder:
    def respond(self, query: str, hits: List[SearchHit], facts: str = "") -> str:
        if not hits and not facts:
            return "I couldn't find any resume content relevant to that question."
        user_message = (f"<facts>\n{facts or 'none'}\n</facts>\n<context>\n{build_context(hits)}\n</context>\n"
                        f"<question>\n{query}\n</question>")
        answer = ask_llm(SYSTEM_PROMPT, user_message)
        return extractive_fallback(hits, facts) if _looks_broken(answer) else answer

    def explain_match(self, job: JobRequirements, match: CandidateMatch) -> str:
        return template_explanation(job, match)

    def llm_explain_match(self, job: JobRequirements, match: CandidateMatch) -> str:
        """Optional LLM prose. Small models can contradict the facts - verify before trusting."""
        return ask_llm(MATCH_PROMPT, template_explanation(job, match))
