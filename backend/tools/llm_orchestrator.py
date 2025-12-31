# llm_orchestrator.py
"""
LLM Orchestrator utilities: robust JSON extraction and deterministic file/column matching.

Intended usage:
 - orchestrator = LLMOrchestrator(llm_client)   # llm_client optional
 - parsed_intent = orchestrator.understand_user_intent(user_message, context)
 - chosen_files, meta = orchestrator.choose_files(parsed_intent.get("parameters", {}), available_files)
 - col_map = orchestrator.choose_columns(requested_x, requested_y, df_columns)
"""

import json
import re
import logging
from typing import Dict, Any, List, Tuple, Optional

# fuzzy
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ = True
except Exception:
    import difflib
    RAPIDFUZZ = False

logger = logging.getLogger("llm_orchestrator")
logger.setLevel(logging.INFO)


def extract_json_from_text(text: str) -> Optional[Dict[str, Any]]:
    """
    Robustly extract a JSON object from text returned by an LLM.
    Returns dict or None.
    """
    if not text or not isinstance(text, str):
        return None
    s = text.strip()

    # 1) try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # 2) try code fence extraction
    fence_patterns = [r'```json\s*(\{.*?\})\s*```', r'```\s*(\{.*?\})\s*```']
    for pat in fence_patterns:
        m = re.search(pat, s, flags=re.S)
        if m:
            candidate = m.group(1)
            try:
                return json.loads(candidate)
            except Exception:
                # attempt tolerant cleanup
                cleaned = re.sub(r",\s*}", "}", candidate)
                cleaned = re.sub(r",\s*]", "]", cleaned)
                try:
                    return json.loads(cleaned)
                except Exception:
                    continue

    # 3) find first balanced {...} (heuristic)
    braces = re.findall(r'\{(?:[^{}]|\n)*\}', s)
    for b in braces:
        try:
            return json.loads(b)
        except Exception:
            try:
                b2 = re.sub(r",\s*}", "}", b)
                b2 = re.sub(r",\s*]", "]", b2)
                return json.loads(b2)
            except Exception:
                continue

    return None


def normalize_name(name: str) -> str:
    if not name:
        return ""
    n = re.sub(r'\.[^.]+$', '', name)   # strip extension
    n = re.sub(r'[^0-9a-zA-Z]+', ' ', n)
    return ' '.join(n.lower().strip().split())


def fuzzy_score(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if RAPIDFUZZ:
        return float(fuzz.token_set_ratio(a, b))
    else:
        return float(difflib.SequenceMatcher(None, a, b).ratio() * 100.0)


def match_filename_single_token(user_token: str, filenames: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    token = normalize_name(user_token)
    if not token:
        return []
    scores = []
    for fname in filenames:
        norm = normalize_name(fname)
        if token in norm.split():
            conf = 98.0
        elif token in norm:
            conf = 90.0
        else:
            conf = fuzzy_score(token, norm)
        scores.append((fname, conf))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[:top_k]


def match_column_token(user_col: str, df_columns: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    user_norm = normalize_name(user_col)
    candidates = []
    for c in df_columns:
        cn = normalize_name(c)
        if user_norm == cn:
            candidates.append((c, 100.0))
            continue
        if user_norm in cn or cn in user_norm:
            candidates.append((c, 90.0))
            continue
        score = fuzzy_score(user_norm, cn)
        candidates.append((c, float(score)))
    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


class LLMOrchestrator:
    def __init__(self, llm_client: Any = None):
        self.llm = llm_client

    def understand_user_intent(self, message: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Query LLM for intent. Returns parsed dict or a safe fallback.
        'context' is optional and used only for the prompt.
        """
        fallback = {"action": "chat", "confidence": 0.5, "parameters": {}}
        if not self.llm:
            # heuristic fallback for simple patterns
            ml = message.lower()
            if any(w in ml for w in ["list", "uploaded", "files"]):
                return {"action": "list_files", "confidence": 0.8, "parameters": {}}
            if "plot" in ml or "heatmap" in ml or "chart" in ml:
                # try to parse "plot X vs Y"
                m = re.search(r"plot\s+(.+?)\s+vs\s+(.+)", ml)
                if m:
                    return {"action": "generate_chart", "confidence": 0.6,
                            "parameters": {"x_axis": m.group(1).strip(), "y_axis": m.group(2).strip()}}
                return {"action": "generate_chart", "confidence": 0.5, "parameters": {}}
            return fallback

        # build a succinct prompt asking explicitly for compact JSON only
        prompt = (
            "You are an assistant that returns a compact JSON describing the user's intent.\n"
            "Return only JSON. Example schema:\n"
            '{"action":"generate_chart","confidence":0.0,"parameters":{"x_axis":"...","y_axis":"...","file_filter_keyword":"..."} }\n\n'
            f"User message: {message}\n"
            f"Available files: {context.get('files', [])}\n"
        )

        try:
            # calling shape is left generic to accommodate different LLM clients
            # Expectation: self.llm.chat.completions.create or self.llm.create or similar
            raw_text = None
            try:
                resp = self.llm.chat.completions.create(
                    model="llama3.2",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                    max_tokens=400
                )
                # normalize common response shapes
                try:
                    raw_text = resp.choices[0].message.content.strip()
                except Exception:
                    try:
                        raw_text = resp.choices[0].text.strip()
                    except Exception:
                        raw_text = str(resp)
            except Exception:
                # fallback to generic call shapes
                try:
                    resp = self.llm.create(prompt)
                    raw_text = getattr(resp, "text", str(resp))
                except Exception as e:
                    logger.exception("LLM call failed: %s", e)
                    return fallback

            parsed = extract_json_from_text(raw_text)
            if parsed and isinstance(parsed, dict):
                parsed.setdefault("confidence", parsed.get("confidence", 0.5))
                parsed.setdefault("parameters", parsed.get("parameters", {}))
                parsed["raw_model_text"] = raw_text
                return parsed
            else:
                logger.warning("Could not parse JSON from LLM response. Raw start: %s", (raw_text or "")[:200])
                return {"action": "chat", "confidence": 0.5, "parameters": {}, "raw_model_text": raw_text}
        except Exception as e:
            logger.exception("understand_user_intent failed: %s", e)
            return fallback

    def get_chat_completion(self, messages: List[Dict[str, str]], system_message: Optional[str] = None, temperature: float = 0.7, max_tokens: int = 500) -> Optional[str]:
        """
        Generic wrapper for chat completion with error handling.
        """
        if not self.llm:
            logger.warning("LLM client not available for chat completion")
            return None

        try:
            # Prepare messages
            final_messages = []
            if system_message:
                final_messages.append({"role": "system", "content": system_message})
            final_messages.extend(messages)

            # Call LLM
            # We try standard OpenAI/Ollama client shape first
            try:
                resp = self.llm.chat.completions.create(
                    model="llama3.2",
                    messages=final_messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                return resp.choices[0].message.content.strip()
            except AttributeError:
                # Fallback for other client types (e.g. simple text completion)
                # This is a best-effort fallback
                prompt = "\n".join([f"{m['role']}: {m['content']}" for m in final_messages])
                resp = self.llm.create(prompt)
                return getattr(resp, "text", str(resp))

        except Exception as e:
            logger.exception("get_chat_completion failed: %s", e)
            return None

    def choose_files(self, intent_parameters: Dict[str, Any], available_files: List[str]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Deterministic file selection:
         - specific_files explicitly requested (validated)
         - file_filter_keyword fuzzy match
         - files_to_use: 'all' or 'first'
         - infer from axis hints
         - fallback: if single upload => that file
        Returns (chosen_files, metadata)
        """
        params = intent_parameters or {}
        meta = {"method": None, "details": None}
        if not available_files:
            meta["method"] = "no_files"
            return [], meta

        # 1) specific files
        specific = params.get("specific_files") or params.get("files")
        if specific:
            valid = []
            invalid = []
            for s in specific:
                if s in available_files:
                    valid.append(s)
                else:
                    matches = match_filename_single_token(s, available_files, top_k=1)
                    if matches and matches[0][1] >= 90.0:
                        valid.append(matches[0][0])
                    else:
                        invalid.append(s)
            meta["method"] = "specific_files"
            meta["details"] = {"valid": valid, "invalid": invalid}
            if valid:
                return valid, meta

        # 2) file_filter_keyword
        keyword = params.get("file_filter_keyword")
        if keyword:
            matches = match_filename_single_token(keyword, available_files, top_k=len(available_files))
            selected = [m[0] for m in matches if m[1] >= 60.0]
            meta["method"] = "keyword_filter"
            meta["details"] = {"keyword": keyword, "matches": matches[:5]}
            if selected:
                return selected, meta

        # 3) files_to_use
        files_to_use = params.get("files_to_use")
        if files_to_use == "all":
            meta["method"] = "all"
            return list(available_files), meta
        if files_to_use == "first":
            meta["method"] = "first"
            return [available_files[0]], meta

        # 4) infer from axis hints (short tokens)
        for key in ("y_axis", "x_axis", "file_hint", "chart_name"):
            v = params.get(key)
            if v and isinstance(v, str) and len(v.split()) <= 4:
                matches = match_filename_single_token(v, available_files, top_k=3)
                if matches and matches[0][1] >= 70.0:
                    meta["method"] = "inferred_from_axis"
                    meta["details"] = {"hint_key": key, "hint_value": v, "matches": matches}
                    return [matches[0][0]], meta

        # 5) fallback single upload
        if len(available_files) == 1:
            meta["method"] = "single_upload_fallback"
            return [available_files[0]], meta

        meta["method"] = "ambiguous"
        return [], meta

    def choose_columns(self, requested_x: Optional[str], requested_y: Optional[str], df_columns: List[str]) -> Dict[str, Any]:
        """
        Map requested axis names to dataframe columns. Returns a dict:
        { "x": selected_x_or_None, "y": selected_y_or_None, "meta": {...} }
        """
        res = {"x": None, "y": None, "meta": {}}
        if requested_x:
            cx = match_column_token(requested_x, df_columns, top_k=3)
            if cx:
                res["x"] = cx[0][0]
                res["meta"]["x_candidates"] = cx
        if requested_y:
            cy = match_column_token(requested_y, df_columns, top_k=5)
            if cy:
                res["y"] = cy[0][0]
                res["meta"]["y_candidates"] = cy
        return res
