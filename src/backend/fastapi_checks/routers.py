from fastapi import APIRouter

from src.backend.fastapi_checks.config.env import REPUTATION_API_KEY
from src.backend.fastapi_checks.utils import reputation_check, model_predict_risk, extract_features_from_rep_data


router = APIRouter(tags=["check"])

@router.post("/check")
async def check(request: list[dict]):
    reputation_data = await reputation_check(entyties=request, REPUTATION_API_KEY=REPUTATION_API_KEY)
    data = extract_features_from_rep_data(reputation_data)
    prediction = model_predict_risk(data)
    return prediction