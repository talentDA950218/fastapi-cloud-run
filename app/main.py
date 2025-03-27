from typing import List
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from utility.simulation import generate_report
import uvicorn
import os

app = FastAPI()

class SimulationRequest(BaseModel):
    client_info: object
    scenarios: List[dict]

@app.get("/")
async def read_root():
    return {"Running": "Successfully"}

@app.post("/api/simulation")
async def run_simulation(request: SimulationRequest):
    try:
        # Validate scenarios
        if not request.scenarios:
            raise HTTPException(status_code=400, detail="Scenarios must be a non-empty array")
            
        report_data = generate_report(request.client_info, request.scenarios)
        return report_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__ == "__main__":
    os.environ["PORT"] = "8000"
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ["PORT"]))
