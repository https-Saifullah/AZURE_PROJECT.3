import os
import base64
from flask import Flask, request, jsonify, send_from_directory
import requests
from dotenv import load_dotenv
 
load_dotenv()
 
app = Flask(__name__, static_folder="static")
 
VISION_KEY = os.environ.get("VISION_KEY", "")
VISION_ENDPOINT = os.environ.get("VISION_ENDPOINT", "").rstrip("/")
 
# Frontend checkbox value -> which API family handles it.
# Image Analysis 4.0 (imageanalysis:analyze) supports: Caption, Tags, Read (among others).
# Faces / Landmarks / Brands are NOT available in 4.0 and require the legacy v3.2 /analyze endpoint.
V4_FEATURE_MAP = {
    "caption": "Caption",
    "tags": "Tags",
    "ocr": "Read",
}
V32_VISUAL_FEATURE_MAP = {
    "faces": "Faces",
    "brands": "Brands",
}
# Landmarks come back via details=Landmarks, not visualFeatures
V32_DETAILS_FEATURES = {"landmarks"}
 
 
@app.route("/")
def index():
    return send_from_directory("static", "index.html")
 
 
def _prepare_request(image_url, image_base64):
    """Returns (headers_extra, json_body, data_body) for the given image source."""
    if image_url:
        return {"Content-Type": "application/json"}, {"url": image_url}, None
    if image_base64:
        b64 = image_base64
        if "," in b64:
            b64 = b64.split(",", 1)[1]
        image_bytes = base64.b64decode(b64)
        return {"Content-Type": "application/octet-stream"}, None, image_bytes
    return None, None, None
 
 
def call_v4_analyze(image_url, image_base64, features):
    """Calls Image Analysis 4.0 for caption / tags / read (ocr)."""
    features_param = ",".join(features)
    api_url = (
        f"{VISION_ENDPOINT}/computervision/imageanalysis:analyze"
        f"?api-version=2023-10-01&features={features_param}"
    )
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    extra_headers, json_body, data_body = _prepare_request(image_url, image_base64)
    headers.update(extra_headers)
    resp = requests.post(
        api_url, headers=headers, json=json_body, data=data_body, timeout=20
    )
    return resp
 
 
def call_v32_analyze(image_url, image_base64, visual_features, details):
    """Calls legacy Computer Vision v3.2 /analyze for faces / brands / landmarks."""
    params = []
    if visual_features:
        params.append(f"visualFeatures={','.join(visual_features)}")
    if details:
        params.append(f"details={','.join(details)}")
    query = "&".join(params)
    api_url = f"{VISION_ENDPOINT}/vision/v3.2/analyze?{query}"
    headers = {"Ocp-Apim-Subscription-Key": VISION_KEY}
    extra_headers, json_body, data_body = _prepare_request(image_url, image_base64)
    headers.update(extra_headers)
    resp = requests.post(
        api_url, headers=headers, json=json_body, data=data_body, timeout=20
    )
    return resp
 
 
@app.route("/analyze", methods=["POST"])
def analyze():
    if not VISION_KEY or not VISION_ENDPOINT:
        return jsonify({
            "error": "Azure Vision not configured. Set VISION_KEY and "
                     "VISION_ENDPOINT in App Service -> Configuration -> "
                     "Application settings, then restart the app."
        }), 500
 
    data = request.get_json(silent=True) or {}
    image_url = data.get("url")
    image_base64 = data.get("image_base64")
    requested_features = data.get("features") or ["caption", "tags", "ocr"]
 
    if not image_url and not image_base64:
        return jsonify({"error": "No image URL or image data provided"}), 400
 
    v4_features = [
        V4_FEATURE_MAP[f] for f in requested_features if f in V4_FEATURE_MAP
    ]
    v32_visual_features = [
        V32_VISUAL_FEATURE_MAP[f]
        for f in requested_features
        if f in V32_VISUAL_FEATURE_MAP
    ]
    v32_details = ["Landmarks"] if any(
        f in V32_DETAILS_FEATURES for f in requested_features
    ) else []
 
    result = {}
    errors = []
 
    # --- Image Analysis 4.0: caption / tags / ocr ---
    if v4_features:
        try:
            resp = call_v4_analyze(image_url, image_base64, v4_features)
            try:
                body = resp.json()
            except ValueError:
                body = {"error": resp.text or "Unexpected response from Azure AI Vision (v4)"}
            if resp.ok:
                if "captionResult" in body:
                    result["captionResult"] = body["captionResult"]
                if "tagsResult" in body:
                    result["tagsResult"] = body["tagsResult"]
                if "readResult" in body:
                    result["readResult"] = body["readResult"]
            else:
                errors.append({"api": "imageanalysis-v4", "status": resp.status_code, "detail": body})
        except requests.RequestException as exc:
            errors.append({"api": "imageanalysis-v4", "error": f"Request failed: {exc}"})
 
    # --- Legacy v3.2: faces / brands / landmarks ---
    if v32_visual_features or v32_details:
        try:
            resp = call_v32_analyze(image_url, image_base64, v32_visual_features, v32_details)
            try:
                body = resp.json()
            except ValueError:
                body = {"error": resp.text or "Unexpected response from Azure AI Vision (v3.2)"}
            if resp.ok:
                if "faces" in body:
                    result["faceResult"] = {"values": body.get("faces", [])}
                if "brands" in body:
                    result["brandResult"] = {"values": body.get("brands", [])}
                if v32_details:
                    landmarks = []
                    for category in body.get("categories", []):
                        detail = category.get("detail", {}) or {}
                        landmarks.extend(detail.get("landmarks", []))
                    result["landmarkResult"] = {"values": landmarks}
            else:
                errors.append({"api": "analyze-v3.2", "status": resp.status_code, "detail": body})
        except requests.RequestException as exc:
            errors.append({"api": "analyze-v3.2", "error": f"Request failed: {exc}"})
 
    if errors and not result:
        # Every call that was attempted failed outright.
        return jsonify({"error": "Azure AI Vision request(s) failed", "details": errors}), 502
 
    if errors:
        # Partial success: return what we got, plus a note about what failed.
        result["_warnings"] = errors
 
    return jsonify(result), 200
 
 
@app.route("/health")
def health():
    return jsonify({"status": "ok", "configured": bool(VISION_KEY and VISION_ENDPOINT)})
 
 
if __name__ == "__main__":
    app.run(debug=True)

