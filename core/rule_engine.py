# core/rule_engine.py
import json
import re
from typing import List, Dict, Any


class RuleEngine:
    def __init__(self, rule_filepath: str = "data/default_rules.json"):
        """
        Initializes the Rule Engine by loading a JSON rule file.
        """
        # remember where we tried to load from (for diagnostics)
        self._rule_filepath = rule_filepath
        self.rules: Dict[str, Any] = {}

        try:
            with open(rule_filepath, "r", encoding="utf-8") as f:
                self.rules = json.load(f)
            print(f"Successfully loaded rules: {self.rules.get('name')}")
        except FileNotFoundError:
            print(f"ERROR: Rule file not found at {rule_filepath}. Using empty rules.")
            self.rules = {}
        except json.JSONDecodeError:
            print(f"ERROR: Could not parse JSON in {rule_filepath}. Using empty rules.")
            self.rules = {}

    # -------------------- existing getters (unchanged) --------------------

    def get_ambiguity_words(self) -> List[str]:
        """Returns the list of ambiguous words from the loaded rules."""
        return self.rules.get("rules", {}).get("ambiguity", {}).get("words", []) or []

    def get_penalty(self, issue_type: str) -> int:
        """Returns the penalty score for a given issue type."""
        try:
            return int(self.rules.get("rules", {}).get(issue_type, {}).get("penalty", 0))
        except Exception:
            return 0

    def is_check_enabled(self, check_name: str) -> bool:
        """Checks if a specific analysis is enabled in the rules."""
        return bool(self.rules.get("rules", {}).get(check_name, {}).get("enabled", False))

    # -------------------- NEW: unified ambiguity checker --------------------

    def check_ambiguity(self, text: str) -> List[str]:
        """
        Returns a list of ambiguity findings for `text`.
        - Includes classic word hits (from rules.ambiguity.words)
        - Plus specific issues:
            * Non-binding modal (use 'shall')
            * No measurable criterion (weak verb but no numbers/units/timing)
            * Alert without trigger/condition
        NOTE: This keeps the analyzer tab unchanged — it already expects
        a list of strings and renders chips / word cloud from it.
        """
        findings: List[str] = []
        t = text or ""
        rules_root = self.rules.get("rules", {}) if isinstance(self.rules, dict) else {}

        # 1) Classic ambiguous word hits (preserve original tokens for word cloud)
        amb_cfg = rules_root.get("ambiguity", {}) or {}
        if amb_cfg.get("enabled", False):
            words = amb_cfg.get("words", []) or []
            if words:
                sorted_words = sorted(set(words), key=len, reverse=True)
                amb_re = re.compile(r"\b(" + "|".join(map(re.escape, sorted_words)) + r")\b", re.IGNORECASE)
                for m in amb_re.finditer(t):
                    findings.append(m.group(1).lower())

        # 2) Non-binding modal (rules.binding_modal)
        bm_cfg = rules_root.get("binding_modal", {}) or {}
        if bm_cfg.get("enabled", False):
            modals = bm_cfg.get("non_binding_words", []) or []
            if modals:
                bm_re = re.compile(r"\b(" + "|".join(map(re.escape, modals)) + r")\b", re.IGNORECASE)
                if bm_re.search(t):
                    findings.append("Non-binding modal (use 'shall' instead)")

        # 3) Measurability (weak verbs + no number/unit)
        meas_cfg = rules_root.get("measurability", {}) or {}
        if meas_cfg.get("enabled", False):
            weak_verbs = meas_cfg.get("weak_verbs", []) or []
            num_unit_pat = meas_cfg.get(
                "number_unit_regex",
                r"\b\d+(?:\.\d+)?\s*(ms|s|min|h|%|m|km|ft|nm|Hz|kHz|MHz|GHz|°C|C|K|V|A|W|g|kg|MB|GB|dB|bps|kbps|Mbps|ppm)\b",
            )
            if weak_verbs:
                weak_re = re.compile(r"\b(" + "|".join(map(re.escape, weak_verbs)) + r")\b", re.IGNORECASE)
                unit_re = re.compile(num_unit_pat, re.IGNORECASE)
                if weak_re.search(t) and not unit_re.search(t):
                    findings.append("No measurable criterion (add number/unit/timing)")

        # 4) Alert triggers (alert words present but no trigger/condition)
        alert_cfg = rules_root.get("alert_triggers", {}) or {}
        if alert_cfg.get("enabled", False):
            alert_words = alert_cfg.get("alert_words", []) or []
            trigger_words = alert_cfg.get("trigger_words", []) or []
            alert_re = re.compile(r"\b(" + "|".join(map(re.escape, alert_words)) + r")\b", re.IGNORECASE) if alert_words else None
            trig_re = re.compile(r"\b(" + "|".join(map(re.escape, trigger_words)) + r")\b", re.IGNORECASE) if trigger_words else None

            if alert_re and alert_re.search(t):
                if not trig_re or not trig_re.search(t):
                    findings.append("Alert without trigger/condition (add when/if/upon/within/after/…)")

        return _dedupe_preserve_order(findings)

    # -------------------- Diagnostics helper --------------------

    def debug_summary(self) -> Dict[str, Any]:
        r = self.rules.get("rules", {}) if isinstance(self.rules, dict) else {}
        return {
            "loaded": bool(self.rules),
            "source": self._rule_filepath,
            "sections": list(r.keys()),
            "enabled": {k: bool((r.get(k) or {}).get("enabled", False)) for k in r},
            "ambiguity_words_count": len((r.get("ambiguity") or {}).get("words", []) or []),
        }


# -------------------- small utility --------------------

def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            out.append(x)
            seen.add(x)
    return out
