import operator
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langsmith import traceable
from langchain_openai import ChatOpenAI


load_dotenv()
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)


class EvaluationSchema(BaseModel):
    feedback: str = Field(description="Detailed feedback for the essay")
    score: int = Field(description="Score out of 10", gt=0, lt=10)

structured_model = model.with_structured_output(EvaluationSchema)

essay = """Nature is the ultimate foundation of life on Earth, encompassing everything from vast forests and towering mountains to serene oceans and intricate ecosystems. It provides the essential resources necessary for human survival, including clean air, water, food, and medicine. Beyond its material utility, nature offers an overwhelming sense of beauty and tranquility that has inspired artists, poets, and thinkers throughout human history. Observing the delicate balance within natural habitats reveals the extraordinary complexity and interconnectedness of all living organisms.

Every element of the natural world plays a crucial role in maintaining ecological harmony. Trees and plants absorb carbon dioxide while releasing oxygen, regulating the planet's climate and supporting breathing life. Rivers and lakes sustain agricultural systems and nurture diverse aquatic species, while wildlife maintains food chains that prevent ecological collapses. However, modern industrialization, urbanization, and excessive consumption have severely disrupted these delicate systems. Deforestation, pollution, and climate change now pose unprecedented threats to natural habitats across the globe.

Protecting nature is no longer an option but an urgent responsibility for future generations. Conserving biodiversity, reducing waste, and transitioning toward sustainable practices are vital steps in healing our planet. Reconnecting with nature also benefits human well-being, reducing stress and fostering a deeper appreciation for the environment. By respecting and preserving the natural world, humanity ensures its own continued existence and preserves the beauty of Earth for generations to come.
"""
class EssayState(TypedDict, total=False):
    essay: str
    language_feedback: str
    analysis_feedback: str
    clarity_feedback: str
    overall_feedback: str
    individual_scores: Annotated[List[int], operator.add] 
    avg_score: float

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

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")
graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")
graph.add_edge("final_evaluation", END)

workflow = graph.compile()
result = workflow.invoke(
        {"essay": essay},
        config={
            "run_name": "evaluate_upsc_essay",  #
            "tags": ["essay", "langgraph", "evaluation"],
            "metadata": {
                "essay_length": len(essay),
                "model": "gpt-4o-mini",
                "dimensions": ["language", "analysis", "clarity"],
            },
        },
    )

print("------------- Evaluation Result of each agent-------------")
print("Language feedback:\n", result.get("language_feedback", ""), "\n")
print("Analysis feedback:\n", result.get("analysis_feedback", ""), "\n")
print("Clarity feedback:\n", result.get("clarity_feedback", ""), "\n")
print("Overall feedback:\n", result.get("overall_feedback", ""), "\n")
print("Individual scores:", result.get("individual_scores", []))
print("Average score:", result.get("avg_score", 0.0))