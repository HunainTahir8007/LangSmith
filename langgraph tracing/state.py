import operator
from typing import TypedDict, Annotated, List

class EssayState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add] 
    avg_score: float
