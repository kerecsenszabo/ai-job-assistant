"""Source-grounded job-description parsing and criterion normalization."""

from __future__ import annotations

import json
import re

from langchain_core.runnables import Runnable

from assistant.cv_tailoring_core import (
    ModelOutputError,
    ParsedJob,
    Requirement,
    RequirementCriterion,
    UnquotedOptionError,
    canonical_text,
    mentions,
    normalized,
    request,
)


SECTION_IMPORTANCE = (
    (
        r"what (?:we['\u2019]re|we are) looking for|requirements|"
        r"(?:required|minimum|essential|basic) qualifications|must[- ]haves?|"
        r"apply,? if you have",
        "required",
    ),
    (
        r"(?:preferred|desirable) qualifications|nice[- ]to[- ]haves?|"
        r"bonus (?:skills|points)|preferred",
        "preferred",
    ),
    (
        r"(?:your |key )?responsibilities|what you['\u2019]ll do|"
        r"what you will do(?: daily)?",
        "responsibility",
    ),
)

JOB_SECTION_HEADINGS = (
    r"(?:the |about the )?(?:position|role)|job description|"
    r"about the job|description|"
    r"you will have an opportunity to|"
    r"why you(?:['\u2019]ll| will) love working here"
)
NON_REQUIREMENT_SECTIONS = (
    r"why you(?:['\u2019]ll| will) love working here|"
    r"(?:employee )?benefits(?: and perks)?|perks|"
    r"what we offer|why join us|about us|company overview|"
    r"our benefits|our hybrid work model"
)


def collapse_repeated_heading(line: str) -> str:
    """Collapse copied page headings without changing the body of a requirement."""
    prefix, separator, rest = line.partition(":")
    words = prefix.split()
    for width in range(1, len(words) // 2 + 1):
        if len(words) % width == 0 and words == words[:width] * (len(words) // width):
            candidate = " ".join(words[:width])
            if is_job_heading(candidate) or (
                separator and width >= 2 and len(words) // width >= 3
            ):
                return candidate + (separator + rest if separator else "")
    return line


def is_job_heading(line: str) -> bool:
    heading = normalized(line).rstrip(":")
    return bool(
        re.fullmatch(NON_REQUIREMENT_SECTIONS, heading)
        or re.fullmatch(JOB_SECTION_HEADINGS, heading)
        or any(re.fullmatch(pattern, heading) for pattern, _ in SECTION_IMPORTANCE)
    )


def prepare_job_description(description: str) -> str:
    """Restore soft-wrapped lines while retaining real section boundaries."""
    lines: list[str] = []
    for raw in description.splitlines():
        line = collapse_repeated_heading(raw.strip())
        if not line:
            continue
        if (
            lines
            and not is_job_heading(line)
            and not is_job_heading(lines[-1])
            and (
                re.match(r"^[a-z(]", line)
                or re.search(
                    r"(?:[,;:]|\b(?:and|or|for|with|across|of|to|the))$", lines[-1]
                )
            )
            and not lines[-1].endswith((".", "!", "?"))
        ):
            lines[-1] += ("" if lines[-1].endswith("-") else " ") + line
        else:
            lines.append(line)
    return "\n".join(lines)


def assessable_job_lines(description: str) -> list[str]:
    """Exclude known non-job sections without discarding unsectioned postings."""
    lines = []
    excluded = False
    prepared = prepare_job_description(description)
    has_description_section = any(
        normalized(part) == "description"
        for raw in prepared.splitlines()
        for part in source_clauses(raw)
    )
    for raw in prepared.splitlines():
        for line in source_clauses(raw):
            heading = normalized(line).rstrip(":")
            if re.fullmatch(NON_REQUIREMENT_SECTIONS, heading) or (
                heading == "about the job" and has_description_section
            ):
                excluded = True
            elif re.fullmatch(JOB_SECTION_HEADINGS, heading) or any(
                re.fullmatch(pattern, heading) for pattern, _ in SECTION_IMPORTANCE
            ):
                excluded = False
            elif not excluded:
                lines.append(line)
    return lines


def source_clauses(line: str) -> list[str]:
    """Separate copied sentences and bullet items before assigning source IDs."""
    line = re.sub(r"^Description(?=We\b)", "Description\n", line)
    return [
        part.strip()
        for part in re.split(
            r"\n|\s*•\s*|(?<=[.!?])(?<!e\.g\.)(?<!i\.e\.)(?<!vs\.)"
            r"(?:\s+|(?=[A-Z]))(?=[A-Z])",
            line,
        )
        if part.strip()
    ]


OPTION_DESCRIPTORS = re.compile(
    r"\b(?:frameworks|orchestration|workflow|applications|fundamentals|"
    r"knowledge|concepts|experience|skills|proficiency|ecosystem|tools|apis)\b"
)
NON_TECHNOLOGY_OPTIONS = re.compile(
    r"(?:and|or|for|with|using|that|the|in|to|of|global|resilient|"
    r"automated|training|tuning|monitoring|versioning|evaluation|"
    r"frameworks?|systems?|pipelines?|planning|forecasting|"
    r"stakeholders?|scalable|scale|build|develop|design|models?|"
    r"practices|best-in-class|multi-metric)",
    re.IGNORECASE,
)
FUNCTION_WORDS = re.compile(
    r"(?:and|or|for|with|using|that|the|in|to|of)", re.IGNORECASE
)
REDUNDANT_DESCRIPTORS = re.compile(
    r"^(?:(?:leading )?ml frameworks|similar (?:tools|apis)|"
    r"working proficiency|ml ecosystem)$"
)
OPTION_SEPARATOR = re.compile(r"^[\s,;/]*(?:(?:and|or|and/or)\s*)?[\s,;/]*$")
GENERAL_OPTIONS = {
    "testing",
    "version control",
    "code reviews",
    "debugging",
    "vector databases",
    "prompt engineering",
    "communication",
}
CAPABILITY_FILLER = (
    r"(?:a|an|the|their|our|your|its|his|her|has|have|had|been|is|are|"
    r"was|were|to|of|in|on)"
)
ROLE_TITLE = re.compile(
    r"(?i)^as an? (?:[\w-]+ ){0,4}"
    r"(engineer|developer|architect|analyst|scientist|manager|consultant|specialist),\s*"
)


def quoted_capability_span(option: str, quote: str) -> str | None:
    """Find short capability phrases with inflection or intervening grammar."""
    words = normalized(option).split()
    if (
        len(words) < 2
        or len(words) > 5
        or option[1:] != option[1:].lower()
        or not all(re.fullmatch(r"[a-z]+", word) for word in words)
    ):
        return None

    def word_pattern(word: str) -> str:
        if word.endswith("es"):
            return re.escape(word[:-2]) + r"(?:e|es|ed|ing)"
        if word.endswith("s"):
            return re.escape(word[:-1]) + r"(?:s)?"
        return re.escape(word)

    pattern = r"\b" + (
        rf"(?:\s+{CAPABILITY_FILLER}){{0,3}}\s+".join(
            word_pattern(word) for word in words
        )
    ) + r"\b"
    match = re.search(pattern, canonical_text(quote), re.IGNORECASE)
    if match and match.group()[1:] == match.group()[1:].lower():
        return match.group()
    return None


def split_criterion(criterion: RequirementCriterion) -> list[RequirementCriterion]:
    """Keep capability clauses outside a named-tool alternative enumeration."""
    if criterion.kind == "general":
        return [criterion]
    if not criterion.options:
        raise ModelOutputError(
            "Technology and language criteria need explicit options."
        )
    quote = normalized(criterion.quote)
    source_quote = canonical_text(criterion.quote)
    role = ROLE_TITLE.match(source_quote)
    named = []
    general = []
    for option in dict.fromkeys(criterion.options):
        if criterion.kind == "technology" and FUNCTION_WORDS.fullmatch(option.strip()):
            continue
        if (
            criterion.kind == "technology"
            and role
            and normalized(option) == role.group(1).casefold()
            and role.end() < len(source_quote)
        ):
            general.append(
                RequirementCriterion(
                    id=criterion.id,
                    text=source_quote[role.end() :],
                    quote=criterion.quote,
                    kind="general",
                )
            )
            continue
        capability_span = (
            quoted_capability_span(option, criterion.quote)
            if criterion.kind == "technology"
            and normalized(option) not in quote
            else None
        )
        if capability_span:
            general.append(
                RequirementCriterion(
                    id=criterion.id,
                    text=capability_span,
                    quote=criterion.quote,
                    kind="general",
                )
            )
            continue
        if (
            OPTION_DESCRIPTORS.search(normalized(option))
            or criterion.kind == "technology"
            and (
                normalized(option) in GENERAL_OPTIONS
                or NON_TECHNOLOGY_OPTIONS.fullmatch(option.strip())
            )
        ):
            if not REDUNDANT_DESCRIPTORS.fullmatch(normalized(option)):
                general.append(
                    RequirementCriterion(
                        id=criterion.id,
                        text=(
                            criterion.text
                            if len(criterion.text.split()) > 1
                            and normalized(criterion.text)
                            in normalized(criterion.quote)
                            and len(criterion.options) == 1
                            else (
                                criterion.quote
                                if len(criterion.options) == 1
                                else option
                            )
                        ),
                        quote=criterion.quote,
                        kind="general",
                    )
                )
        else:
            named.append(option)
    ordered = sorted(named, key=lambda option: quote.find(normalized(option)))
    groups: list[list[str]] = []
    previous_end = None
    for option in ordered:
        position = quote.find(normalized(option))
        if position < 0:
            raise UnquotedOptionError(
                f"A criterion option is absent from its source quote: {option!r} "
                f"in {criterion.quote!r}."
            )
        if previous_end is None or not OPTION_SEPARATOR.fullmatch(
            quote[previous_end:position]
        ):
            groups.append([])
        groups[-1].append(option)
        previous_end = position + len(normalized(option))
    separated = []
    for group_index, group in enumerate(groups):
        start = quote.find(normalized(group[0]))
        end = quote.find(normalized(group[-1])) + len(normalized(group[-1]))
        enumeration = quote[start:end]
        operator = "all"
        if re.search(r"\bor\b", enumeration) or "/" in enumeration:
            operator = "any"
        elif re.search(r"\band\b", enumeration):
            operator = "all"
        elif re.search(r"(?:\be\.g\.,?|\bsuch as)\s*$", quote[:start]):
            operator = "any"
        elif len(group) > 1:
            operator = criterion.operator
        scope_end = (
            quote.find(normalized(groups[group_index + 1][0]))
            if group_index + 1 < len(groups)
            else len(quote)
        )
        if re.match(
            r"\s*,?\s*or\s+(?:similar|equivalent|comparable|other)\b",
            quote[end:scope_end],
        ):
            operator = "any"
        scoped_quote = source_quote[start:scope_end].strip(" ,;")
        global_production = mentions(quote[:start], "production")
        separated.append(
            criterion.model_copy(
                update={
                    "options": group,
                    "operator": operator,
                    "quote": scoped_quote,
                    "production": (
                        criterion.production
                        if len(groups) == 1
                        else global_production
                        or "in production" in normalized(scoped_quote)
                    ),
                    "experience_required": (
                        criterion.experience_required
                        or mentions(criterion.quote, "experience")
                    ),
                }
            )
        )
    if not separated and not general:
        raise ModelOutputError(
            "A technology criterion has no assessable named tools or capabilities."
        )
    return separated + general


def source_importance(description: str, requirement: Requirement) -> str:
    """Use explicit job sections, not the model's guess at a requirement's weight."""
    if requirement.importance == "eligibility":
        return "eligibility"
    quote = normalized(requirement.quote)
    if re.search(r"\b(?:nice.to.have|preferred|optional|a plus)\b", quote):
        return "preferred"
    if re.search(r"\b(?:required|must|mandatory)\b", quote):
        return "required"
    prefix = normalized(description).split(quote, 1)[0]
    last_position = -1
    importance = requirement.importance
    for raw in description.splitlines():
        for line in source_clauses(raw):
            heading = normalized(line).rstrip(":")
            for pattern, category in SECTION_IMPORTANCE:
                if re.fullmatch(pattern, heading):
                    position = prefix.rfind(heading)
                    if position > last_position:
                        last_position = position
                        importance = category
    return importance


def normalize_criteria(requirement: Requirement) -> list[RequirementCriterion]:
    criteria = []
    source_criteria = list(requirement.criteria)
    language_statement = re.search(
        r"\b(?P<proficiency>(?:professional working |full professional |working )?"
        r"(?:proficiency|fluency)) in (?P<language>[a-z]+)\b",
        normalized(requirement.quote),
    )
    if language_statement and not any(
        criterion.kind == "language" for criterion in source_criteria
    ):
        source_criteria.append(
            RequirementCriterion(
                id="language",
                text=language_statement.group(),
                quote=requirement.quote,
                kind="language",
                options=[language_statement.group("language")],
                proficiency=language_statement.group("proficiency"),
            )
        )
    expanded = [
        part for criterion in source_criteria for part in split_criterion(criterion)
    ]
    for index, criterion in enumerate(expanded):
        if normalized(criterion.quote) not in normalized(requirement.quote):
            raise ModelOutputError("A requirement criterion has no valid source quote.")
        if criterion.kind in ("technology", "language") and not criterion.options:
            raise ModelOutputError(
                "Technology and language criteria need explicit options."
            )
        positions = []
        for option in criterion.options:
            if not option.strip():
                raise UnquotedOptionError(
                    "A requirement criterion contains a blank option."
                )
            position = normalized(criterion.quote).find(normalized(option))
            if position < 0:
                raise UnquotedOptionError(
                    f"A criterion option is absent from its source quote: {option!r} "
                    f"in {criterion.quote!r}."
                )
            positions.append((position, position + len(normalized(option))))
        operator = criterion.operator
        if len(positions) > 1:
            start = min(position[0] for position in positions)
            end = max(position[1] for position in positions)
            enumeration = normalized(criterion.quote)[start:end]
            if re.search(r"\bor\b", enumeration):
                operator = "any"
            elif re.search(r"\band\b", enumeration):
                operator = "all"
        production = criterion.production or (
            criterion.kind == "technology"
            and (
                "in production" in normalized(criterion.quote)
                or (
                    sum(item.kind == "technology" for item in expanded) == 1
                    and "in production" in normalized(requirement.quote)
                )
            )
        )
        proficiency = criterion.proficiency
        if criterion.kind == "language" and language_statement:
            proficiency = language_statement.group("proficiency")
        criteria.append(
            criterion.model_copy(
                update={
                    "id": f"{requirement.id}/criterion/{index}",
                    "operator": operator,
                    "production": production,
                    "proficiency": proficiency,
                }
            )
        )
    return criteria


def recover_unquoted_options(
    requirement: Requirement,
) -> tuple[Requirement, list[str]]:
    """Keep only sourced options; assess an optionless clause from its source quote."""
    criteria = []
    warnings = []
    for criterion in requirement.criteria:
        valid = [
            option
            for option in criterion.options
            if option.strip()
            and normalized(option) in normalized(criterion.quote)
        ]
        removed = [option for option in criterion.options if option not in valid]
        if removed:
            warning = (
                f"Job analysis for {requirement.quote!r}: discarded unquoted "
                f"option(s) {removed!r}"
            )
            if not valid:
                warning += "; using the source quote as a general criterion"
                source = canonical_text(criterion.quote)
                role = ROLE_TITLE.match(source)
                criteria.append(
                    criterion.model_copy(
                        update={
                            "text": (
                                source[role.end() :]
                                if role and role.end() < len(source)
                                else source
                            ),
                            "kind": "general",
                            "options": [],
                        }
                    )
                )
            else:
                criteria.append(criterion.model_copy(update={"options": valid}))
            warnings.append(warning + ".")
        else:
            criteria.append(criterion)
    return requirement.model_copy(update={"criteria": criteria}), warnings


def job_description_chunks(description: str, max_chars: int = 800) -> list[str]:
    """Bound extraction output by splitting at source line or word boundaries."""
    description = description.strip()
    if len(description) <= max_chars:
        return [description]
    lines = [line.strip() for line in description.splitlines() if line.strip()]
    if len(lines) > 1:
        chunks = []
        current = ""
        for line in lines:
            for part in job_description_chunks(line, max_chars):
                if current and len(current) + len(part) + 1 > max_chars:
                    chunks.append(current)
                    current = ""
                current += ("\n" if current else "") + part
        if current:
            chunks.append(current)
        return chunks
    chunks = []
    remaining = description
    while len(remaining) > max_chars:
        boundary = remaining.rfind("\n", 0, max_chars + 1)
        if boundary <= 0:
            spaces = list(re.finditer(r"\s", remaining[: max_chars + 1]))
            if spaces:
                boundary = spaces[-1].start()
            else:
                next_space = re.search(r"\s", remaining[max_chars:])
                if next_space is None:
                    break
                boundary = max_chars + next_space.start()
        chunks.append(remaining[:boundary].strip())
        remaining = remaining[boundary:].strip()
    if remaining or not chunks:
        chunks.append(remaining)
    return chunks


def source_requirement_quotes(response: str, sources: dict[str, str]) -> str:
    """Resolve compact source keys to quotes without asking the model to copy them."""
    try:
        job = json.loads(response)
    except json.JSONDecodeError:
        return response
    if not isinstance(job, dict):
        return response
    requirements = job.get("requirements")
    if not isinstance(requirements, dict):
        return response
    resolved = []
    for source_id, requirement in requirements.items():
        if source_id not in sources:
            return response
        if requirement is None:
            continue
        if not isinstance(requirement, dict):
            return response
        requirement["id"] = source_id
        requirement["quote"] = sources[source_id]
        requirement.setdefault("text", sources[source_id])
        criteria = requirement.get("criteria")
        if isinstance(criteria, list):
            for index, criterion in enumerate(criteria):
                if isinstance(criterion, dict):
                    criterion["id"] = f"{source_id}/criterion/{index}"
                    criterion["quote"] = sources[source_id]
                    options = criterion.get("options")
                    if isinstance(options, list):
                        criterion["options"] = [
                            (
                                source_option(option, sources[source_id])
                                if isinstance(option, str)
                                else option
                            )
                            for option in options
                        ]
        resolved.append(requirement)
    job["requirements"] = resolved
    job.pop("warnings", None)
    return json.dumps(job, ensure_ascii=True)


def source_option(option: str, quote: str) -> str:
    """Resolve unambiguous separator and apostrophe formatting to source text."""

    def key(value: str) -> str:
        return normalized(
            value.replace("\u2019", "'").replace("-", " ").replace("_", " ")
        )

    if normalized(option) in normalized(quote):
        return option
    words = list(re.finditer(r"\S+", quote))
    matches = set()
    for start in range(len(words)):
        for end in range(start, len(words)):
            candidate = quote[words[start].start() : words[end].end()].strip(
                "()[]{}.,;:"
            )
            if key(candidate) == key(option):
                matches.add(candidate)
    return matches.pop() if len(matches) == 1 else option


def source_option_pattern(quote: str) -> str:
    """Allow only contiguous source-word spans, including multiword tool names."""
    words = list(re.finditer(r"(?:\.[A-Za-z]|[A-Za-z0-9])[A-Za-z0-9+.#/-]*", quote))
    alternatives = []
    suffix = ""
    for index in range(len(words) - 1, -1, -1):
        word = words[index]
        end = word.end()
        while end > word.start() and quote[end - 1] in ".-/":
            end -= 1
        literal = re.escape(quote[word.start() : end])
        if suffix:
            separator = re.escape(quote[end : words[index + 1].start()])
            suffix = literal + "(?:" + separator + suffix + ")?"
        else:
            suffix = literal
        alternatives.append(suffix)
    return "^(?:" + "|".join(alternatives) + ")$" if alternatives else r"^\b\B$"


def job_extraction_schema(sources: dict[str, str]) -> dict:
    schema = ParsedJob.model_json_schema()
    del schema["properties"]["warnings"]
    requirement = schema["$defs"]["Requirement"]
    criterion = schema["$defs"]["RequirementCriterion"]
    for definition in (requirement, criterion):
        for field in ("id", "quote"):
            del definition["properties"][field]
            definition["required"].remove(field)
    requirement["required"].append("criteria")
    del requirement["properties"]["text"]
    requirement["required"].remove("text")
    requirement["properties"]["criteria"]["minItems"] = 1
    criterion["required"].extend(["kind", "options"])
    source_schemas = {}
    for source_id, quote in sources.items():
        word_count = max(1, len(quote.split()))
        option_pattern = source_option_pattern(quote)
        criteria = {
            "oneOf": [
                {
                    **criterion,
                    "properties": {
                        **criterion["properties"],
                        "kind": {"const": kind},
                        "options": {
                            **criterion["properties"]["options"],
                            "items": {
                                "type": "string",
                                "pattern": option_pattern,
                            },
                            **(
                                {"maxItems": 0}
                                if kind == "general"
                                else {"minItems": 1, "maxItems": word_count}
                            ),
                        },
                    },
                }
                for kind in ("general", "technology", "language")
            ],
        }
        source_schemas[source_id] = {
            "anyOf": [
                {
                    **requirement,
                    "properties": {
                        **requirement["properties"],
                        "criteria": {
                            **requirement["properties"]["criteria"],
                            "maxItems": min(12, word_count),
                            "items": criteria,
                        },
                    },
                    "description": f"Extract requirements ONLY from this line: {quote}",
                },
                {"type": "null"},
            ],
        }
    schema["properties"]["requirements"] = {
        "type": "object",
        "properties": source_schemas,
        "additionalProperties": False,
    }
    return schema


def parse_job(llm: Runnable, description: str) -> ParsedJob:
    instruction = (
        "Extract assessable job requirements from the supplied source lines. "
        "Return a requirements object keyed by source ID. Use null for lines that contain "
        "no assessable qualification, responsibility, or eligibility constraint. "
        "Do not extract job titles, section headings, company descriptions, "
        "or employee benefits. Do not invent requirements for these lines. "
        "Return at most one requirement per source ID, using criteria to cover "
        "its atomic assessable clauses. Do not output requirement text, IDs or quotes inside "
        "requirements or criteria; these are attached from the source IDs. "
        "Deduplicate equivalent "
        "requirements; do not count repeated wording twice. Distinguish explicit "
        "must-haves (required), nice-to-haves (preferred), responsibilities, and "
        "explicit eligibility constraints. Do not invent must-haves. "
        "Preserve source section priority: ALL bullets under 'What we're looking "
        "for' or required qualifications are required, including English; only "
        "explicit nice-to-haves are preferred. Include responsibilities too. "
        "For each requirement extract criteria covering EVERY assessable clause. "
        "Criteria inherit their parent's source line. Use "
        "kind=technology only for explicit named tools, products, libraries or "
        "model families, with options "
        "containing only exact names from that quote. 'A, B, or C' means "
        "operator=any, NOT all; 'A and B' means all. Examples introduced by "
        "'e.g.' or 'such as' are alternatives unless explicitly all required. "
        "Set production=true when production use is required. Use kind=language "
        "for a named natural language and preserve the requested proficiency "
        "phrase. Communication skills are a separate general criterion; they "
        "cannot establish language proficiency. General criteria cover other "
        "clauses, such as seniority, testing, CI/CD, or ownership. Do not turn "
        "a single alternative list into separately required technologies. "
        "Keep text concise. Extract each source clause only once; do not repeat "
        "requirements or criteria to fill the response. Never split a clause "
        "into individual words or treat ordinary words as technology names. "
        "Training, monitoring, forecasting, planning, pipelines and adjectives "
        "are capabilities, not technology names. Keeping up with changing "
        "technology and sharing ideas are general criteria, not technology "
        "options. Do not paraphrase source wording in options. Only named "
        "technologies may have technology options. Job titles such as engineer "
        "are not technologies; extract the actual responsibilities instead. "
        "Never extract standalone "
        "function words (and, with, using) as criteria. One general criterion "
        "can cover a complete clause; do not fill every available schema slot. "
        "Return no more than twelve meaningful criteria per source line. "
        "Stop after the last "
        "requirement. Options must be exact substrings of the corresponding "
        "source line. General criteria should have empty options. "
        "If there are no assessable requirements return an empty object."
    )
    prepared = prepare_job_description(description)
    sectioned_lines = set()
    in_section = False
    for raw in prepared.splitlines():
        for line in source_clauses(raw):
            heading = normalized(line).rstrip(":")
            if re.fullmatch(NON_REQUIREMENT_SECTIONS, heading):
                in_section = False
            elif any(re.fullmatch(pattern, heading) for pattern, _ in SECTION_IMPORTANCE):
                in_section = True
            elif re.fullmatch(JOB_SECTION_HEADINGS, heading):
                in_section = False
            elif in_section:
                sectioned_lines.add(normalized(line))
    seen = set()
    unique = []
    requirements = []
    warnings = []
    for chunk in job_description_chunks("\n".join(assessable_job_lines(description))):
        if not chunk:
            continue
        sources = {
            f"source/{index}": line.strip()
            for index, line in enumerate(chunk.splitlines())
            if line.strip()
        }
        feedback = ""
        for attempt in range(3):
            try:
                job = request(
                    llm,
                    ParsedJob,
                    instruction + feedback,
                    {"source_lines": sources},
                    json_schema=job_extraction_schema(sources),
                    transform_response=lambda response: source_requirement_quotes(
                        response, sources
                    ),
                    stage="job_parsing",
                )
                keyed = not job.requirements or all(
                    item.id.startswith("source/") for item in job.requirements
                )
                missing = [
                    quote
                    for quote in sources.values()
                    if keyed
                    and normalized(quote) in sectioned_lines
                    and not any(
                        normalized(item.quote) in normalized(quote)
                        for item in job.requirements
                    )
                ]
                if missing:
                    raise ModelOutputError(
                        "Missing assessable lines under a requirements or "
                        f"responsibilities heading: {missing!r}."
                    )
                cleaned = []
                for requirement in job.requirements:
                    criteria = [
                        item
                        for item in requirement.criteria
                        if not (
                            item.kind == "general"
                            and FUNCTION_WORDS.fullmatch(item.text.strip())
                        )
                    ]
                    if not criteria and requirement.criteria:
                        raise ModelOutputError(
                            "A source line has no assessable criteria after removing function words."
                        )
                    if (
                        len(criteria) > 12
                        or sum(
                            item.kind == "general" and len(item.text.split()) == 1
                            for item in criteria
                        )
                        >= 3
                    ):
                        criteria = [
                            item for item in criteria if item.kind != "general"
                        ] + [
                            RequirementCriterion(
                                id=f"{requirement.id}/general",
                                text=requirement.quote,
                                quote=requirement.quote,
                            )
                        ]
                    if len(criteria) > 12:
                        raise ModelOutputError(
                            "A source line has too many technology or language criteria."
                        )
                    requirement = requirement.model_copy(update={"criteria": criteria})
                    try:
                        normalize_criteria(requirement)
                    except UnquotedOptionError:
                        if attempt != 2:
                            raise
                        requirement, recovered = recover_unquoted_options(requirement)
                        normalize_criteria(requirement)
                        warnings.extend(recovered)
                    cleaned.append(requirement)
                job = job.model_copy(update={"requirements": cleaned})
                break
            except ModelOutputError as exc:
                if attempt == 2:
                    raise
                feedback = (
                    "\nThe previous extraction failed validation: "
                    f"{exc}\nRegenerate the extraction using only the supplied "
                    "source lines. Do not invent options. Include all assessable "
                    "lines under labeled requirements or responsibilities sections. "
                    "Use null only for non-assessable lines."
                )
        for requirement in job.requirements:
            if normalized(requirement.quote) not in normalized(chunk):
                raise ValueError("A parsed job requirement has no valid source quote.")
        requirements.extend(job.requirements)
    for requirement in requirements:
        if normalized(requirement.quote) not in normalized(prepared):
            raise ValueError("A parsed job requirement has no valid source quote.")
        key = normalized(requirement.text)
        if key not in seen:
            seen.add(key)
            requirement = requirement.model_copy(
                update={
                    "id": f"requirement/{len(unique)}",
                    "importance": source_importance(prepared, requirement),
                }
            )
            requirement = requirement.model_copy(
                update={
                    "criteria": normalize_criteria(requirement),
                }
            )
            if not requirement.criteria:
                raise ModelOutputError(
                    f"{requirement.id} has no assessable criteria in the parsed job."
                )
            unique.append(requirement)
    return ParsedJob(requirements=unique, warnings=warnings)

