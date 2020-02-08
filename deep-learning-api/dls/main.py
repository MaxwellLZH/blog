from fastapi import FastAPI, File, UploadFile, Body, Query
from starlette.responses import HTMLResponse, FileResponse, StreamingResponse
import requests
import io


import sentiment_analysis_plain_lstm
import object_detection
from cat import get_random_cat_image



def save_pil_image_to_bytes(img):
	out = io.BytesIO()
	img.save(out, format='PNG')
	out.seek(0)
	return out


app = FastAPI()


@app.get('/health')
def health_check():
	return HTMLResponse("<h1>Hello world!</h1>")


@app.get('/cat')
def cat():
	img = get_random_cat_image()
	return StreamingResponse(save_pil_image_to_bytes(img), media_type="image/png")


@app.get('/joke')
def programming_joke():
	headers = {
		'user-agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.130 Safari/537.36'
	}
	resp = requests.get('https://official-joke-api.appspot.com/jokes/programming/random', headers=headers).json()[0]
	return resp['setup'] + ' ' + resp['punchline']


@app.get('/sentiment/')
async def sentiment_analysis_service(s: str):
	score = sentiment_analysis_plain_lstm.make_prediction(s)
	return {'score': float(score)}


@app.post('/detection/')
async def object_detection_service(img: UploadFile = File(...),
								   min_score: float = Query(0.2, gt=0.0, lt=1.0, alias='min-score')):
	img = object_detection.make_prediction(img, min_score)
	return StreamingResponse(save_pil_image_to_bytes(img), media_type="image/png")
	# return {'file_name': img.filename, 'image': object_detection.make_prediction(img)}
