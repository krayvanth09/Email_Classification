from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from email_classifier import EmailClassifier
from pii_masker import PIIMasker

# Define Pydantic models for input/output
class EmailRequest(BaseModel):
    input_email_body: str

class MaskedEntity(BaseModel):
    position: List[int]
    classification: str
    entity: str

class EmailResponse(BaseModel):
    input_email_body: str
    list_of_masked_entities: List[MaskedEntity]
    masked_email: str
    category_of_the_email: str

# Initialize FastAPI app
app = FastAPI()

# Load components
classifier = EmailClassifier()
pii_masker = PIIMasker()

@app.post("/classify", response_model=EmailResponse)
async def classify_email(request: EmailRequest):
    try:
        email_body = request.input_email_body.strip()
        if not email_body:
            raise HTTPException(status_code=400, detail="Email body cannot be empty")

        # Step 1: Mask PII
        masked_result = pii_masker.mask_pii(email_body)
        masked_email = masked_result['masked_text']
        entities = masked_result['entities']

        # Step 2: Classify masked email
        category = classifier.classify(masked_email)

        # Step 3: Format masked entities
        masked_entities = [
            MaskedEntity(
                position=[e['start'], e['end']],
                classification=e['label'],
                entity=e['text']
            ) for e in entities
        ]

        return EmailResponse(
            input_email_body=email_body,
            list_of_masked_entities=masked_entities,
            masked_email=masked_email,
            category_of_the_email=category
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
