"""Skill taxonomy + matcher. Canonical name -> list of aliases (all lower-case).

Covers engineering, data/AI, cloud/devops AND recruiting/sales skills, because
staffing platforms place candidates across many functions. Extend freely.
"""
import re
from typing import Dict, List

SKILL_ALIASES: Dict[str, List[str]] = {
    # --- languages
    "Python": ["python"], "Java": ["java"], "JavaScript": ["javascript", "ecmascript"],
    "TypeScript": ["typescript"], "C++": ["c++"], "C#": ["c#"], "Go": ["golang"],
    "SQL": ["sql"], "R": ["r programming", "rstudio"], "Scala": ["scala"], "Kotlin": ["kotlin"],
    # --- web / backend
    "React": ["react", "react.js", "reactjs"], "Angular": ["angular"], "Vue": ["vue", "vue.js"],
    "Node.js": ["node.js", "nodejs", "node js"], "Django": ["django"], "Flask": ["flask"],
    "FastAPI": ["fastapi"], "Spring Boot": ["spring boot", "springboot"],
    "REST APIs": ["rest api", "rest apis", "restful"], "GraphQL": ["graphql"],
    "Microservices": ["microservices", "microservice"], "HTML/CSS": ["html", "css", "html5", "css3"],
    # --- data stores
    "PostgreSQL": ["postgresql", "postgres"], "MySQL": ["mysql"], "MongoDB": ["mongodb"],
    "Redis": ["redis"], "Elasticsearch": ["elasticsearch", "opensearch"],
    "Kafka": ["kafka"], "Spark": ["spark", "pyspark"], "Hadoop": ["hadoop"],
    "Airflow": ["airflow"], "Snowflake": ["snowflake"],
    # --- cloud / devops
    "AWS": ["aws", "amazon web services"], "Azure": ["azure"], "GCP": ["gcp", "google cloud"],
    "Docker": ["docker"], "Kubernetes": ["kubernetes", "k8s"], "Terraform": ["terraform"],
    "CI/CD": ["ci/cd", "ci-cd", "continuous integration"], "Jenkins": ["jenkins"],
    "Git": ["git", "github", "gitlab"], "Linux": ["linux"],
    # --- data science / AI
    "Machine Learning": ["machine learning"], "Deep Learning": ["deep learning"],
    "NLP": ["nlp", "natural language processing"], "Computer Vision": ["computer vision"],
    "LLMs": ["llm", "llms", "large language model", "large language models"],
    "RAG": ["rag", "retrieval-augmented generation", "retrieval augmented generation"],
    "PyTorch": ["pytorch"], "TensorFlow": ["tensorflow"],
    "scikit-learn": ["scikit-learn", "sklearn"], "Pandas": ["pandas"], "NumPy": ["numpy"],
    "Data Analysis": ["data analysis", "data analytics"], "Statistics": ["statistics", "statistical"],
    "Power BI": ["power bi", "powerbi"], "Tableau": ["tableau"], "Excel": ["excel", "ms excel"],
    # --- product / delivery
    "Agile": ["agile", "scrum"], "Jira": ["jira"], "Figma": ["figma"],
    "Project Management": ["project management"], "Stakeholder Management": ["stakeholder management"],
    # --- recruiting / sales
    "Talent Acquisition": ["talent acquisition", "recruitment", "recruiting"],
    "Sourcing": ["sourcing", "candidate sourcing"], "Boolean Search": ["boolean search", "boolean"],
    "ATS": ["ats", "applicant tracking system", "applicant tracking systems"],
    "LinkedIn Recruiter": ["linkedin recruiter"], "Onboarding": ["onboarding"],
    "Salesforce": ["salesforce"], "HubSpot": ["hubspot"], "CRM": ["crm"],
    "Negotiation": ["negotiation", "negotiating"], "Account Management": ["account management"],
    "Lead Generation": ["lead generation"], "Communication": ["communication skills"],
    "Leadership": ["leadership", "team lead", "led a team"],
}

_LB = r"(?<![A-Za-z0-9_+#])"
_LA = r"(?![A-Za-z0-9_+#])"

_PATTERNS = {
    canon: re.compile(
        _LB + "(?:" + "|".join(re.escape(a) for a in sorted(aliases, key=len, reverse=True)) + ")" + _LA,
        re.IGNORECASE,
    )
    for canon, aliases in SKILL_ALIASES.items()
}


def extract_skills(text: str) -> List[str]:
    """Return canonical skills mentioned in `text` (taxonomy order, no duplicates)."""
    return [canon for canon, pat in _PATTERNS.items() if pat.search(text)]
