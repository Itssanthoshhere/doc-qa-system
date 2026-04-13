"""
api/routers/documents.py — Document management endpoints.
"""
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, UploadFile, HTTPException, BackgroundTasks
from pydantic import BaseModel

from core.config import settings
from core.pipeline import DocSagePipeline
from core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

# In-memory doc registry (replace with DB in production)
_documents: dict[str, dict] = {}


class DocumentResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    total_sections: Optional[int] = None
    total_chunks: Optional[int] = None
    file_size_kb: Optional[float] = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


async def _ingest_background(file_path: Path, doc_id: str) -> None:
    """Background ingestion task."""
    try:
        _documents[doc_id]["status"] = "processing"
        pipeline = DocSagePipeline.get()
        pipeline.ingest_document(file_path, doc_id)
        _documents[doc_id]["status"] = "ready"
        logger.info("background_ingestion_complete", doc_id=doc_id)
    except Exception as e:
        _documents[doc_id]["status"] = "failed"
        _documents[doc_id]["error"] = str(e)
        logger.error("background_ingestion_failed", doc_id=doc_id, error=str(e))


@router.post("/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload a document (PDF, DOCX, TXT, MD) for indexing.
    Returns immediately; ingestion happens in background.
    Poll /documents/{doc_id} to check status.
    """
    allowed_types = {".pdf", ".docx", ".doc", ".txt", ".md"}
    suffix = Path(file.filename).suffix.lower()
    if suffix not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{suffix}' not supported. Allowed: {allowed_types}",
        )

    file_size = 0
    doc_id = str(uuid.uuid4())
    save_path = settings.documents_dir / f"{doc_id}{suffix}"

    try:
        with open(save_path, "wb") as f:
            while chunk := await file.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > settings.max_upload_size_mb * 1024 * 1024:
                    save_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"File exceeds {settings.max_upload_size_mb}MB limit",
                    )
                f.write(chunk)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    _documents[doc_id] = {
        "doc_id": doc_id,
        "filename": file.filename,
        "status": "queued",
        "file_path": str(save_path),
        "file_size_kb": file_size / 1024,
    }

    background_tasks.add_task(_ingest_background, save_path, doc_id)

    logger.info("document_uploaded", doc_id=doc_id, filename=file.filename)
    return DocumentResponse(
        doc_id=doc_id,
        filename=file.filename,
        status="queued",
        file_size_kb=round(file_size / 1024, 1),
    )


@router.get("/{doc_id}", response_model=DocumentResponse)
async def get_document(doc_id: str):
    """Get document status and metadata."""
    doc = _documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    return DocumentResponse(**{k: v for k, v in doc.items() if k != "file_path" and k != "error"})


@router.get("/", response_model=DocumentListResponse)
async def list_documents():
    """List all documents."""
    docs = [
        DocumentResponse(**{k: v for k, v in d.items() if k not in ("file_path", "error")})
        for d in _documents.values()
    ]
    return DocumentListResponse(documents=docs, total=len(docs))


@router.delete("/{doc_id}")
async def delete_document(doc_id: str):
    """Delete a document and remove it from the index."""
    doc = _documents.get(doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    pipeline = DocSagePipeline.get()
    removed_chunks = pipeline.remove_document(doc_id)

    # Remove file
    file_path = Path(doc.get("file_path", ""))
    if file_path.exists():
        file_path.unlink()

    del _documents[doc_id]

    return {"deleted": doc_id, "chunks_removed": removed_chunks}