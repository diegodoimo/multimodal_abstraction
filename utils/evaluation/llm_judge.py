"""LLM judge of lmms-eval's MMBench scoring: credentials and concurrent judge calls.

``ensure_llm_judge_env`` makes sure OPENAI_API_KEY, OPENAI_API_URL and MODEL_VERSION are set: exported
variables are used as they are, then a ``.env`` file at the repository root is read (python-dotenv);
without OPENAI_API_KEY, a token is requested from the ORFEO cluster's OAuth endpoint (ORFEO_* variables).
``install_parallel_mmbench_judge`` runs the judge calls concurrently with unchanged scoring.
"""

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path


def _load_project_env() -> None:
    """Load ``.env`` from the repository root without overriding exported variables."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    project_root = Path(__file__).resolve().parents[2]  # repository root
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)


def _acquire_orfeo_jwt() -> str | None:
    """Access token from the ORFEO OAuth endpoint, or None."""
    auth_url = os.environ.get("ORFEO_AUTH_URL")
    client_id = os.environ.get("ORFEO_CLIENT_ID")
    username = os.environ.get("ORFEO_USERNAME")
    password = os.environ.get("ORFEO_CLIENT_SECRET")

    if not all([auth_url, client_id, username, password]):
        return None

    data = urllib.parse.urlencode(
        {
            "grant_type": "password",
            "client_id": client_id,
            "username": username,
            "password": password,
            "scope": "profile",
        }
    ).encode()

    try:
        req = urllib.request.Request(
            auth_url,
            data=data,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read())
        token = body.get("access_token")
        if token and token != "null":
            return token
    except Exception:
        pass

    return None


def ensure_llm_judge_env() -> bool:
    """Set the judge's environment variables (module docstring); True when a key is available."""
    _load_project_env()
    if os.environ.get("OPENAI_API_KEY"):
        return True

    token = _acquire_orfeo_jwt()
    if not token:
        return False

    os.environ["OPENAI_API_KEY"] = token

    orfeo_url = os.environ.get("ORFEO_LLM_URL")
    if orfeo_url and not os.environ.get("OPENAI_API_URL"):
        os.environ["OPENAI_API_URL"] = orfeo_url

    orfeo_model = os.environ.get("ORFEO_MODEL")
    if orfeo_model and not os.environ.get("MODEL_VERSION"):
        os.environ["MODEL_VERSION"] = orfeo_model

    return True


def install_parallel_mmbench_judge(max_workers: int | None = None) -> None:
    """Run the MMBench LLM-judge calls concurrently instead of one at a time.

    lmms-eval's ``MMBench_Evaluator.eval_result`` scores the items sequentially and calls the judge
    (one HTTP request, ~10 s on our endpoint) for every answer its rule-based parser cannot read.
    Verbose models trigger it for most answers, which makes MMBench the slowest part of a benchmark
    or intervention run. This patch pre-computes those judge verdicts with a thread pool and lets the
    original (unchanged) scoring loop pick them up from a cache, so the result is the same, only
    faster. ``max_workers`` defaults to ``MMBENCH_JUDGE_WORKERS`` (8).
    """
    from concurrent.futures import ThreadPoolExecutor

    import pandas as pd
    from lmms_eval.tasks.mmbench.mmbench_evals import MMBench_Evaluator

    if getattr(MMBench_Evaluator, "_parallel_judge_installed", False):
        return
    workers = max_workers or int(os.environ.get("MMBENCH_JUDGE_WORKERS", "8"))
    original_eval_result = MMBench_Evaluator.eval_result
    original_extract = MMBench_Evaluator.extract_answer_from_item

    def extract_answer_from_item(self, item):
        cache = getattr(self, "_judge_cache", None)
        if cache is not None and int(item["index"]) in cache:
            return cache[int(item["index"])]
        return original_extract(self, item)

    def eval_result(self, results, eval_method):
        # Same preprocessing as the original: string predictions, lower-case column names (except A-D).
        data = pd.DataFrame(results)
        data["prediction"] = [str(x) for x in data["prediction"]]
        for k in list(data.keys()):
            data[k.lower() if k not in "ABCD" else k] = data.pop(k)
        rows = [data.iloc[i] for i in range(len(data))]
        pending = [row for row in rows if not self.prefetch_answer(row)]
        self._judge_cache = {}
        if pending:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                verdicts = pool.map(lambda row: original_extract(self, row), pending)
                for row, verdict in zip(pending, verdicts):
                    self._judge_cache[int(row["index"])] = verdict
        try:
            return original_eval_result(self, results, eval_method)
        finally:
            self._judge_cache = None

    MMBench_Evaluator.extract_answer_from_item = extract_answer_from_item
    MMBench_Evaluator.eval_result = eval_result
    MMBench_Evaluator._parallel_judge_installed = True
