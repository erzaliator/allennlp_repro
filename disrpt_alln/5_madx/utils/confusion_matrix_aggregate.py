from collections import Counter
import json
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.colors import ListedColormap

class ConfusionMatrixPlotter():
    def __init__(self, cmap=None):
        if cmap==None:
            cmap = 'Blues'
        self.cmap = ListedColormap(sns.color_palette(cmap))

    def save_cm(self, y_true, y_pred, save_fig_path, title=''):
        classes = list(set(y_true) | set(y_pred))
        classes_dict = {}
        for i in range(len(classes)):
            classes_dict[classes[i]] = i
        classes = classes_dict
        print(classes)

        y_true = [classes[y] for y in y_true]
        y_pred = [classes[y] for y in y_pred]

        cm = confusion_matrix(y_true, y_pred)
        cm = cm/3
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
        disp.plot(xticks_rotation="vertical", cmap=self.cmap)

        for labels in disp.text_.ravel():
            labels.set_fontsize(6)

        plt.title(title)
        plt.tight_layout()

        plt.savefig(save_fig_path)
        

# if __name__=='main':

foldername = '../../4_adapter/run1/'
filenames = ['labelweighing_1__best_latest.json', 'labelweighing_2__best_latest.json', 'labelweighing_3__best_latest.json']
filenames = ['../../4_adapter/run1/Baseline_graph_2__best_latest.json', '../../4_adapter/run1/Baseline_graph_3__best_latest.json', '../../4_adapter/run1/Baseline_graph_4__best_latest.json', ]

y_pred = []
y_true = []
for filename in filenames:
    filename = foldername+filename
    json_idx = json.load(open(filename, 'r'))
    for i in json_idx['pred_label']: y_pred.append(i)
    for i in json_idx['gold_label']: y_true.append(i)

    # y_true = ['evidence', 'list', 'list', 'list', 'list', 'list', 'concession', 'interpretation', 'background', 'evaluation-s', 'conjunction', 'evaluation-n', 'evidence', 'antithesis', 'evidence', 'list', 'condition', 'reason', 'evidence', 'reason', 'conjunction', 'background', 'sequence', 'interpretation', 'antithesis', 'contrast', 'cause', 'means', 'list', 'reason', 'e-elaboration', 'interpretation', 'conjunction', 'evaluation-s', 'antithesis', 'background', 'e-elaboration', 'antithesis', 'reason', 'joint', 'cause', 'joint', 'reason', 'list', 'antithesis', 'reason', 'antithesis', 'list', 'antithesis', 'list', 'antithesis', 'list', 'antithesis', 'list', 'concession', 'summary', 'interpretation', 'circumstance', 'background', 'evaluation-s', 'joint', 'antithesis', 'reason', 'joint', 'elaboration', 'condition', 'reason', 'joint', 'joint', 'concession', 'evidence', 'contrast', 'evaluation-s', 'concession', 'evidence', 'evaluation-s', 'condition', 'background', 'circumstance', 'circumstance', 'e-elaboration', 'circumstance', 'e-elaboration', 'evidence', 'joint', 'joint', 'reason', 'elaboration', 'joint', 'reason', 'joint', 'condition', 'reason', 'reason', 'condition', 'reason', 'contrast', 'interpretation', 'sequence', 'e-elaboration', 'concession', 'reason', 'reason', 'sequence', 'sequence', 'joint', 'interpretation', 'concession', 'joint', 'interpretation', 'concession', 'evaluation-s', 'evaluation-s', 'circumstance', 'sequence', 'sequence', 'background', 'evaluation-s', 'antithesis', 'e-elaboration', 'evidence', 'joint', 'reason', 'joint', 'evaluation-s', 'joint', 'reason', 'circumstance', 'background', 'antithesis', 'concession', 'background', 'joint', 'reason', 'reason', 'reason', 'e-elaboration', 'circumstance', 'preparation', 'sequence', 'interpretation', 'circumstance', 'concession', 'evaluation-s', 'e-elaboration', 'background', 'joint', 'joint', 'concession', 'reason', 'evaluation-s', 'list', 'evaluation-s', 'concession', 'reason', 'contrast', 'evaluation-s', 'background', 'list', 'concession', 'background', 'reason', 'evidence', 'purpose', 'joint', 'elaboration', 'contrast', 'reason', 'circumstance', 'elaboration', 'evaluation-s', 'background', 'joint', 'evaluation-s', 'e-elaboration', 'joint', 'reason', 'conjunction', 'contrast', 'condition', 'elaboration', 'list', 'list', 'reason', 'circumstance', 'reason', 'reason', 'e-elaboration', 'circumstance', 'list', 'list', 'antithesis', 'condition', 'list', 'list', 'list', 'reason', 'restatement', 'joint', 'joint', 'joint', 'means', 'antithesis', 'background', 'reason', 'condition', 'conjunction', 'solutionhood', 'conjunction', 'preparation', 'contrast', 'elaboration', 'list', 'list', 'preparation', 'evaluation-n', 'elaboration', 'evaluation-s', 'contrast', 'circumstance', 'antithesis', 'list', 'elaboration', 'list', 'list', 'interpretation', 'interpretation', 'elaboration', 'background', 'background', 'joint', 'concession', 'antithesis', 'reason', 'antithesis', 'reason', 'purpose', 'preparation', 'antithesis', 'evidence', 'e-elaboration', 'interpretation', 'background', 'joint', 'reason', 'conjunction', 'background', 'circumstance', 'circumstance', 'joint', 'joint', 'elaboration', 'condition', 'interpretation', 'circumstance', 'joint', 'purpose', 'evaluation-s', 'evaluation-n', 'reason']
    # y_pred = ['joint', 'reason', 'reason', 'concession', 'concession', 'reason', 'elaboration', 'reason', 'condition', 'interpretation', 'conjunction', 'elaboration', 'reason', 'list', 'reason', 'list', 'condition', 'reason', 'reason', 'reason', 'conjunction', 'condition', 'condition', 'reason', 'reason', 'reason', 'conjunction', 'conjunction', 'joint', 'reason', 'elaboration', 'reason', 'conjunction', 'elaboration', 'background', 'background', 'elaboration', 'reason', 'condition', 'reason', 'elaboration', 'elaboration', 'reason', 'reason', 'list', 'reason', 'list', 'conjunction', 'condition', 'condition', 'conjunction', 'conjunction', 'conjunction', 'conjunction', 'joint', 'reason', 'reason', 'reason', 'reason', 'reason', 'reason', 'interpretation', 'list', 'reason', 'reason', 'condition', 'interpretation', 'reason', 'reason', 'reason', 'elaboration', 'concession', 'reason', 'interpretation', 'elaboration', 'interpretation', 'joint', 'condition', 'condition', 'condition', 'elaboration', 'condition', 'list', 'reason', 'reason', 'reason', 'interpretation', 'list', 'reason', 'reason', 'interpretation', 'condition', 'reason', 'interpretation', 'condition', 'interpretation', 'interpretation', 'reason', 'interpretation', 'e-elaboration', 'joint', 'cause', 'reason', 'conjunction', 'joint', 'joint', 'condition', 'condition', 'condition', 'elaboration', 'reason', 'elaboration', 'background', 'elaboration', 'list', 'list', 'concession', 'interpretation', 'reason', 'interpretation', 'reason', 'concession', 'reason', 'condition', 'interpretation', 'concession', 'reason', 'reason', 'reason', 'e-elaboration', 'elaboration', 'reason', 'reason', 'elaboration', 'elaboration', 'interpretation', 'elaboration', 'reason', 'elaboration', 'elaboration', 'condition', 'interpretation', 'reason', 'reason', 'conjunction', 'reason', 'background', 'reason', 'joint', 'interpretation', 'condition', 'condition', 'joint', 'interpretation', 'reason', 'reason', 'reason', 'reason', 'concession', 'reason', 'reason', 'reason', 'reason', 'conjunction', 'interpretation', 'conjunction', 'reason', 'list', 'condition', 'condition', 'joint', 'reason', 'reason', 'reason', 'e-elaboration', 'reason', 'reason', 'conjunction', 'concession', 'condition', 'reason', 'reason', 'conjunction', 'reason', 'reason', 'reason', 'reason', 'e-elaboration', 'reason', 'condition', 'list', 'reason', 'condition', 'condition', 'condition', 'conjunction', 'condition', 'condition', 'reason', 'reason', 'interpretation', 'interpretation', 'reason', 'interpretation', 'reason', 'condition', 'condition', 'interpretation', 'interpretation', 'interpretation', 'reason', 'e-elaboration', 'interpretation', 'conjunction', 'reason', 'interpretation', 'interpretation', 'reason', 'conjunction', 'conjunction', 'conjunction', 'conjunction', 'elaboration', 'joint', 'interpretation', 'elaboration', 'concession', 'joint', 'reason', 'reason', 'reason', 'condition', 'condition', 'condition', 'conjunction', 'conjunction', 'condition', 'reason', 'condition', 'condition', 'e-elaboration', 'list', 'conjunction', 'reason', 'reason', 'conjunction', 'conjunction', 'conjunction', 'conjunction', 'list', 'interpretation', 'reason', 'condition', 'result', 'interpretation', 'list', 'conjunction', 'joint', 'interpretation', 'interpretation']
save_fig_path = 'CM.png'
cmplotter = ConfusionMatrixPlotter()
cmplotter.save_cm(y_true, y_pred, save_fig_path, 'Best latest')