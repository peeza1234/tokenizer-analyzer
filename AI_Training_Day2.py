from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
 
app = FastAPI(title="PDF Upload Microservice", version="1.0.0")
 
 
class PDFUploadResponse(BaseModel):
    filename: str
    status: str
    bytes_count: int
 
 
@app.post("/upload-pdf", response_model=PDFUploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> PDFUploadResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: '{file.filename}'. Only .pdf files are accepted.",
        )
 
    contents = await file.read()
    bytes_count = len(contents)
    await file.seek(0)
 
    return PDFUploadResponse(
        filename=file.filename,
        status="success",
        bytes_count=bytes_count,
    )
 
 
@app.get("/")
async def root():
    return {"message": "PDF Upload Microservice is running. Visit /docs to test the /upload-pdf endpoint."}