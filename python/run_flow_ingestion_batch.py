from __future__ import annotations

import os
import logging

from run_pipeline import create_model_run_id, maybe_run_flow_ingestion, setup_logging


def main() -> int:
    setup_logging()
    enabled = str(os.environ.get("ENABLE_FLOW_INGESTION", "")).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        logging.info("ENABLE_FLOW_INGESTION is disabled -> skip flow ingestion batch")
        return 0
    run_id = create_model_run_id()
    logging.info("Created flow ingestion batch run_id=%s", run_id)
    maybe_run_flow_ingestion(run_id)
    logging.info("Flow ingestion batch completed run_id=%s", run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
