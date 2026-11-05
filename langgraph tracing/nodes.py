from langgraph.graph import StateGraph


@traceable(name="evaluate_language_fn", tags=["dimension:language"], metadata={"dimension": "language"})
def evaluate_language(state: EssayState):
    prompt = (
        "Evaluate the language quality of the following essay and provide feedback "
        "and assign a score out of 10.\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"language_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_analysis_fn")
def evaluate_analysis(state: EssayState):
    prompt = (
        "Evaluate the depth of analysis of the following essay and provide feedback "
        "and assign a score out of 10.\n" + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"analysis_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="evaluate_thought_fn")
def evaluate_thought(state: EssayState):
    prompt = (
        "Evaluate the clarity of thought of the following essay and provide feedback "
        "and assign a score out of 10." + state["essay"]
    )
    out = structured_model.invoke(prompt)
    return {"clarity_feedback": out.feedback, "individual_scores": [out.score]}

@traceable(name="final_evaluation_fn")
def final_evaluation(state: EssayState):
    prompt = (
        "Based on the following feedback, create a summarized overall feedback.\n"
        f"Language feedback: {state.get('language_feedback','')}\n"
        f"Depth of analysis feedback: {state.get('analysis_feedback','')}\n"
        f"Clarity of thought feedback: {state.get('clarity_feedback','')}\n"
    )
    overall = model.invoke(prompt).content
    scores = state.get("individual_scores", []) 
    avg = (sum(scores) / len(scores)) if scores else 0.0
    return {"overall_feedback": overall, "avg_score": avg}
graph = StateGraph(EssayState)