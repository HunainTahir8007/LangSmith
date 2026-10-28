
essay = """Nature is the ultimate foundation of life on Earth, encompassing everything from vast forests and towering mountains to serene oceans and intricate ecosystems. It provides the essential resources necessary for human survival, including clean air, water, food, and medicine. Beyond its material utility, nature offers an overwhelming sense of beauty and tranquility that has inspired artists, poets, and thinkers throughout human history. Observing the delicate balance within natural habitats reveals the extraordinary complexity and interconnectedness of all living organisms.

Every element of the natural world plays a crucial role in maintaining ecological harmony. Trees and plants absorb carbon dioxide while releasing oxygen, regulating the planet's climate and supporting breathing life. Rivers and lakes sustain agricultural systems and nurture diverse aquatic species, while wildlife maintains food chains that prevent ecological collapses. However, modern industrialization, urbanization, and excessive consumption have severely disrupted these delicate systems. Deforestation, pollution, and climate change now pose unprecedented threats to natural habitats across the globe.

Protectin"""
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