import os
import joblib
import m2cgen as m2c

WORKSPACE = '/home/godkiller/Documents/tata'
MODEL_REG_PATH = os.path.join(WORKSPACE, 'models/trained_model_reg.pkl')
MODEL_CLF_PATH = os.path.join(WORKSPACE, 'models/trained_model_clf.pkl')

def main():
    print("Loading models...")
    reg_model = joblib.load(MODEL_REG_PATH)
    clf_model = joblib.load(MODEL_CLF_PATH)
    
    # 1. Export Regression Model
    print("Transpiling regression model...")
    import json
    reg_config = json.loads(reg_model.get_booster().save_config())
    reg_base_score = float(reg_config['learner']['learner_model_param']['base_score'].strip('[]'))
    print(f"  Detected reg base_score: {reg_base_score}")
    reg_model.base_score = reg_base_score
    reg_code = m2c.export_to_c(reg_model)
    # Post-process to rename default score function
    reg_code = reg_code.replace("double score(", "double predict_knee_cycle(")
    
    reg_header = """#ifndef MODEL_REG_H
#define MODEL_REG_H

double predict_knee_cycle(double * input);

#endif
"""

    # 2. Export Classification Model
    print("Transpiling classification model...")
    clf_config = json.loads(clf_model.get_booster().save_config())
    clf_base_score = float(clf_config['learner']['learner_model_param']['base_score'].strip('[]'))
    print(f"  Detected clf base_score: {clf_base_score}")
    clf_model.base_score = clf_base_score
    clf_code = m2c.export_to_c(clf_model)
    # Post-process to rename functions and avoid clashes
    clf_code = clf_code.replace("void score(", "void predict_knee_early(")
    clf_code = clf_code.replace("double sigmoid(", "static double sigmoid(")
    
    clf_header = """#ifndef MODEL_CLF_H
#define MODEL_CLF_H

void predict_knee_early(double * input, double * output);

#endif
"""

    # Destination directories
    dirs = [
        os.path.join(WORKSPACE, 'edge', 'host_test'),
        os.path.join(WORKSPACE, 'edge', 'esp32_project', 'main')
    ]
    
    for d in dirs:
        os.makedirs(d, exist_ok=True)
        
        # Write regression C files
        with open(os.path.join(d, 'model_reg.c'), 'w') as f:
            f.write(reg_code)
        with open(os.path.join(d, 'model_reg.h'), 'w') as f:
            f.write(reg_header)
            
        # Write classification C files
        with open(os.path.join(d, 'model_clf.c'), 'w') as f:
            f.write(clf_code)
        with open(os.path.join(d, 'model_clf.h'), 'w') as f:
            f.write(clf_header)
            
    print("✅ C files successfully exported and copied to all locations!")

if __name__ == '__main__':
    main()
