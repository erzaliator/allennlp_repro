
def load_heirarchy(type='overview'):
    if type=='overview':
        heirarchy = {
            "antithesis": "Presentational", 
            "background": "Presentational", 
            "concession": "Presentational", 
            "enablement": "Presentational",
            "evidence": "Presentational", 
            "justify": "Presentational", 
            "motivation": "Presentational", 
            "preparation": "Presentational", 
            "restatement": "Presentational", 
            "summary": "Presentational", 

            "circumstance": "SubjectMatter", 
            "condition": "SubjectMatter", 
            "elaboration": "SubjectMatter", 
            "e-elaboration": "SubjectMatter", 
            "evaluation": "SubjectMatter", 
            "evaluation-n": "SubjectMatter", 
            "evaluation-s": "SubjectMatter", 
            "interpretation": "SubjectMatter", 
            "means": "SubjectMatter", 
            "non-volitional Cause": "SubjectMatter", 
            "non-volitional Result": "SubjectMatter", 
            "otherwise	": "SubjectMatter", 
            "purpose": "SubjectMatter", 
            "solutionhood": "SubjectMatter", 
            "unconditional": "SubjectMatter", 
            "unless": "SubjectMatter", 
            "volitional Cause": "SubjectMatter", 
            "volitional Result": "SubjectMatter",
            "cause": "SubjectMatter",
            "reason": "SubjectMatter",
            "result": "SubjectMatter",

            "conjunction": "Multinuclear",
            "contrast": "Multinuclear",
            "disjunction": "Multinuclear",
            "joint": "Multinuclear",
            "list": "Multinuclear",
            "sequence": "Multinuclear"}
    return heirarchy #USE RST THEORY PAPER HEIRARCHY
    #refactor for deu. the model definition featureful bert layers

def parent(label, heirarchy):
    return heirarchy[label]