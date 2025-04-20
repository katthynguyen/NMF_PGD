# api/main.py
from fastapi import FastAPI
from api.routes import router as face_router
from config.settings import get_settings
import uvicorn

settings = get_settings()
app = FastAPI(title="Face Recognition Service")

app.include_router(face_router, prefix="/face", tags=["Feature Face"])


@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT)