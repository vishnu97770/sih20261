from __future__ import annotations

import logging
import os
from io import BytesIO
import mimetypes
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from ..config import settings
from ..services import analytics, document_service, insights, notifications, rag, report
from ..services.anomaly import detect_anomalies
from ..services.data_service import get_dataframe, get_filter_options, get_session, has_data, remove_dataset, set_session_from_dataframe, upload_dataset
from ..services.forecast import forecast, train_forecast_model
from .auth import get_current_user

logger = logging.getLogger(__name__)

# Every route below requires a valid bearer token - this is the platform's
# actual production/geological data, not public content.
router = APIRouter(prefix="/api", dependencies=[Depends(get_current_user)])

DATASET_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class AskRequest(BaseModel):
    question: str = Field(min_length=1)


class ReportRequest(BaseModel):
    mine: str | None = None
    report_type: str | None = None


class ForecastRequest(BaseModel):
    horizon: int = Field(default=3, ge=1, le=12)
    year: list[int] | None = None
    mine: list[str] | None = None
    mineral: list[str] | None = None
    state: list[str] | None = None
    district: list[str] | None = None


def _filters_from_query(
    year: list[int] | None = Query(default=None),
    mine: list[str] | None = Query(default=None),
    mineral: list[str] | None = Query(default=None),
    state: list[str] | None = Query(default=None),
    district: list[str] | None = Query(default=None),
) -> dict[str, Any]:
    filters: dict[str, Any] = {}
    if year:
        filters["year"] = year
    if mine:
        filters["mine"] = mine
    if mineral:
        filters["mineral"] = mineral
    if state:
        filters["state"] = state
    if district:
        filters["district"] = district
    return filters


@router.get("/health")
def health():
    session = get_session()
    return {
        "status": "ok",
        "has_data": has_data(),
        "session_id": session.session_id,
        # Boolean only - never the key itself or its length in an HTTP response.
        # Lets you confirm Render picked up GROQ_API_KEY without checking logs.
        "groq_configured": bool(settings.groq_api_key),
    }


@router.get("/session")
def session_state():
    session = get_session()
    return {
        "session": session.as_metadata(),
        "filters": get_filter_options(),
        "quality": session.quality,
    }


@router.get("/assistant/suggestions")
def suggestions():
    return {
        "suggestions": [
            "Summarize this dataset",
            "Why did production fall in the latest year?",
            "Which mine had the highest production?",
            "Show major anomalies",
            "What does the forecast look like?",
            "Generate an executive summary",
        ]
    }


@router.post("/ask")
def ask(req: AskRequest):
    try:
        return rag.answer_question(req.question)
    except Exception as exc:
        logger.exception("Assistant failed to answer question")
        raise HTTPException(
            status_code=500,
            detail="I couldn't process that request right now. Please try again.",
        ) from exc


@router.get("/notifications")
def get_notifications():
    return {"notifications": notifications.list_notifications()}


@router.post("/notifications/read")
def read_notifications():
    notifications.mark_all_read()
    return {"ok": True}


@router.post("/upload")
async def upload(file: UploadFile = File(...)):
    """Accepts any file type. Production datasets (CSV/XLSX/XLS) feed the analytics
    pipeline; every other file (PDF, PNG, JPG, DOCX, TXT, or anything else) is
    parsed for real text where possible via the document service."""
    filename = file.filename or "upload"
    suffix = Path(filename).suffix.lower()

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is too large. Maximum allowed size is {settings.max_upload_mb} MB.",
        )

    if suffix not in DATASET_EXTENSIONS:
        try:
            result = document_service.ingest_document(content, filename)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Unexpected error while processing uploaded document %s", filename)
            raise HTTPException(
                status_code=400,
                detail=f"Could not process the uploaded document ({exc.__class__.__name__}: {exc}).",
            ) from exc
        message = f"{filename} processed - {result['chunk_count']} passages extracted from {result['pages']} page(s)."
        if result["ocr_unavailable_pages"]:
            message += f" {result['ocr_unavailable_pages']} page(s) looked scanned but OCR is not installed on this server, so no text could be extracted from them."
        notifications.add_notification("document", f"{filename} is ready to cite in the AI Mining Assistant.")
        return {"ok": True, "kind": "document", "message": message, **result}

    try:
        safe_name = f"{Path(filename).stem}_{os.getpid()}_{Path(filename).suffix.lstrip('.')}"
        dest = Path(settings.upload_dir) / safe_name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    except OSError as exc:
        logger.exception("Could not save uploaded file %s to %s", filename, settings.upload_dir)
        raise HTTPException(
            status_code=500,
            detail="Could not save the uploaded file on the server. Check that the upload directory is writable.",
        ) from exc

    try:
        result = upload_dataset(content, filename)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Unexpected error while parsing uploaded file %s", filename)
        raise HTTPException(
            status_code=400,
            detail=f"Could not process the uploaded file ({exc.__class__.__name__}: {exc}).",
        ) from exc

    rows = result.get("quality", {}).get("rows")
    notifications.add_notification("dataset", f"Dataset uploaded: {filename} ({rows} rows processed).")

    try:
        anomaly_pack = detect_anomalies(get_dataframe())
        primary = anomaly_pack.get("primary")
        if primary and primary.get("severity") in ("HIGH", "CRITICAL"):
            notifications.add_notification(
                "anomaly",
                f"A {primary['severity']} anomaly was flagged for {primary['year']} ({primary.get('deviation_pct')}% deviation).",
            )
    except Exception:
        logger.exception("Could not check for anomalies to notify about after upload")

    try:
        kpi_pack = analytics.kpis()
        achievement = kpi_pack.get("target_achievement_pct")
        if achievement is not None and achievement < 80:
            notifications.add_notification(
                "target",
                f"Target achievement is {achievement}%, below the 80% threshold.",
            )
    except Exception:
        logger.exception("Could not check target achievement to notify about after upload")

    try:
        train_forecast_model()
    except Exception as exc:
        logger.exception("Forecast training failed after uploading %s", filename)
        result["forecast_warning"] = (
            f"Dataset uploaded, but the forecast model could not be trained ({exc.__class__.__name__}: {exc})."
        )

    return {"ok": True, "kind": "dataset", **result}


@router.post("/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    return await upload(file)


@router.get("/documents/{doc_id}/file")
def get_document_file(doc_id: str):
    try:
        path, original_name = document_service.get_document_file(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    media_type, _ = mimetypes.guess_type(original_name)
    return FileResponse(path, media_type=media_type or "application/octet-stream", filename=original_name)


@router.delete("/documents/{doc_id}")
def delete_document(doc_id: str):
    try:
        return document_service.remove_document(doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/dataset")
def delete_dataset():
    return remove_dataset()


@router.post("/demo/load")
def load_demo_dataset():
    demo = pd.DataFrame(
        [
            {"year": 2019, "mine": "Mine A", "mineral": "Iron Ore", "state": "Odisha", "district": "Keonjhar", "production": 720, "target": 780},
            {"year": 2020, "mine": "Mine A", "mineral": "Iron Ore", "state": "Odisha", "district": "Keonjhar", "production": 760, "target": 800},
            {"year": 2021, "mine": "Mine A", "mineral": "Iron Ore", "state": "Odisha", "district": "Keonjhar", "production": 830, "target": 840},
            {"year": 2022, "mine": "Mine A", "mineral": "Iron Ore", "state": "Odisha", "district": "Keonjhar", "production": 905, "target": 880},
            {"year": 2023, "mine": "Mine A", "mineral": "Iron Ore", "state": "Odisha", "district": "Keonjhar", "production": 940, "target": 920},
            {"year": 2024, "mine": "Mine A", "mineral": "Iron Ore", "state": "Odisha", "district": "Keonjhar", "production": 615, "target": 980},
            {"year": 2025, "mine": "Mine A", "mineral": "Iron Ore", "state": "Odisha", "district": "Keonjhar", "production": 860, "target": 1010},
            {"year": 2021, "mine": "Mine B", "mineral": "Manganese", "state": "Jharkhand", "district": "Singhbhum", "production": 220, "target": 240},
            {"year": 2022, "mine": "Mine B", "mineral": "Manganese", "state": "Jharkhand", "district": "Singhbhum", "production": 250, "target": 255},
            {"year": 2023, "mine": "Mine B", "mineral": "Manganese", "state": "Jharkhand", "district": "Singhbhum", "production": 245, "target": 260},
            {"year": 2024, "mine": "Mine B", "mineral": "Manganese", "state": "Jharkhand", "district": "Singhbhum", "production": 270, "target": 270},
            {"year": 2025, "mine": "Mine B", "mineral": "Manganese", "state": "Jharkhand", "district": "Singhbhum", "production": 292, "target": 285},
        ]
    )
    result = set_session_from_dataframe(demo, filename="demo_mining_dataset.csv")
    train_forecast_model()
    return {
        "ok": True,
        "message": "Demo dataset loaded. Replace it with your government dataset when ready.",
        **result,
    }


@router.get("/production")
def production(
    year: list[int] | None = Query(default=None),
    mine: list[str] | None = Query(default=None),
    mineral: list[str] | None = Query(default=None),
    state: list[str] | None = Query(default=None),
    district: list[str] | None = Query(default=None),
):
    return analytics.production_series(
        _filters_from_query(year=year, mine=mine, mineral=mineral, state=state, district=district)
    )


@router.get("/kpis")
@router.get("/production/kpis")
def kpis(
    year: list[int] | None = Query(default=None),
    mine: list[str] | None = Query(default=None),
    mineral: list[str] | None = Query(default=None),
    state: list[str] | None = Query(default=None),
    district: list[str] | None = Query(default=None),
):
    return analytics.kpis(_filters_from_query(year=year, mine=mine, mineral=mineral, state=state, district=district))


@router.get("/quality")
def quality():
    return analytics.data_quality()


@router.get("/filters")
def filters():
    return analytics.filters_meta()


@router.get("/anomalies")
def anomalies(
    year: list[int] | None = Query(default=None),
    mine: list[str] | None = Query(default=None),
    mineral: list[str] | None = Query(default=None),
    state: list[str] | None = Query(default=None),
    district: list[str] | None = Query(default=None),
):
    df = get_dataframe(_filters_from_query(year=year, mine=mine, mineral=mineral, state=state, district=district))
    return detect_anomalies(df)


@router.get("/forecast")
def forecast_get(
    horizon: int = Query(default=3, ge=1, le=12),
    year: list[int] | None = Query(default=None),
    mine: list[str] | None = Query(default=None),
    mineral: list[str] | None = Query(default=None),
    state: list[str] | None = Query(default=None),
    district: list[str] | None = Query(default=None),
):
    return forecast(horizon, _filters_from_query(year=year, mine=mine, mineral=mineral, state=state, district=district))


@router.post("/forecast")
def forecast_post(payload: ForecastRequest):
    filters = {
        "year": payload.year,
        "mine": payload.mine,
        "mineral": payload.mineral,
        "state": payload.state,
        "district": payload.district,
    }
    filters = {k: v for k, v in filters.items() if v}
    return forecast(payload.horizon, filters)


@router.post("/train-models")
def retrain_models():
    artifact = train_forecast_model(force=True)
    if artifact.get("status") == "insufficient_data":
        raise HTTPException(status_code=400, detail=artifact["message"])
    notifications.add_notification("forecast", f"Forecast model retrained ({artifact.get('model_name', 'unknown model')}).")
    return {"ok": True, "artifact": artifact}


@router.post("/report")
def create_report(payload: ReportRequest | None = Body(default=None)):
    payload = payload or ReportRequest()
    data = report.build_report_data(mine=payload.mine, report_type=payload.report_type)
    notifications.add_notification("report", f"Report ready for {payload.mine or 'all mines'}.")
    return data


@router.get("/report/pdf")
def report_pdf(
    mine: str | None = Query(default=None),
    report_type: str | None = Query(default=None),
):
    path = report.generate_pdf(mine=mine, report_type=report_type)
    return FileResponse(path, media_type="application/pdf", filename="mine_intelligence_report.pdf")


@router.get("/insights/wordcloud")
def wordcloud():
    return insights.build_wordcloud()


@router.get("/documents")
def documents():
    session = get_session()
    entries: list[dict[str, Any]] = []

    if has_data():
        entries.append(
            {
                "id": session.session_id[:8],
                "name": session.source_name or "Uploaded dataset",
                "type": "Spreadsheet",
                "status": "Processed",
                "records": session.row_count,
                "date": session.uploaded_at[:10] if session.uploaded_at else None,
                "quality": session.quality,
            }
        )

    entries.extend(document_service.list_documents())
    return {"documents": entries, "tesseract_available": document_service.tesseract_available()}
