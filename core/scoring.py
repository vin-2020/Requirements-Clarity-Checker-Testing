# core/scoring.py

def calculate_clarity_score(total_reqs, flagged_or_issues, rule_engine=None):
    """
    Clarity score = % of requirements with zero issues.

    Supports both:
      - (total_reqs, flagged_reqs)
      - (total_reqs, issue_counts_dict, rule_engine)   # rule_engine ignored here
    """
    if total_reqs == 0:
        return 100

    if isinstance(flagged_or_issues, int):
        flagged_reqs = max(0, min(total_reqs, flagged_or_issues))
    elif isinstance(flagged_or_issues, dict):
        counts = list(flagged_or_issues.values())
        flagged_reqs = max(counts) if counts else 0   # lower-bound unique flags
    else:
        flagged_reqs = 0

    clear_reqs = total_reqs - flagged_reqs
    return int((clear_reqs / total_reqs) * 100)
