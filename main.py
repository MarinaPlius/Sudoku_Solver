from fastapi import FastAPI, status, UploadFile, File, HTTPException
import io
import numpy as np
import cv2
from starlette.responses import StreamingResponse

from image import Image

app = FastAPI(title="Sudoku Solver API")

@app.get("/")
def root():
	"""home page redirects to docs"""
	return "http://localhost:8000/docs"

@app.get("/health", status_code=status.HTTP_200_OK)
def check_health():
	"""checks status of API"""
	return "status is Ok!"

@app.post("/get_solution")
async def predict(file: UploadFile = File(...)):

	# validate file
    filename = file.filename
    fileExtension = filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not fileExtension:
        raise HTTPException(status_code=415, detail="Unsupported file format!")

    # transform raw image into csv
    image_stream = io.BytesIO(file.file.read())
    image_stream.seek(0)
    file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # create an instance of Image
    image_original = Image(image)

    # Save it in a folder within the server
    cv2.imwrite(f'images_uploaded/{filename}', image)
    file_image = open(f'images_uploaded/{filename}', mode="rb")
    
    return StreamingResponse(file_image, media_type="image/jpeg")



