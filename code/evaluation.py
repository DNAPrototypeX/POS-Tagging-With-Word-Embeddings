from pathlib import Path
from flair.models import SequenceTagger
from flair.data import Sentence
import flair.datasets
import os

corpus = flair.datasets.UD_NORWEGIAN()

root_dir = Path('resources/taggers')

for model_path in root_dir.glob('*/final-model.pt'):
    print(f"--- Loading model from: {model_path} ---")
    
    try:
        # Load the tagger
        tagger = SequenceTagger.load(model_path)
        
        #os.makedir()
        result = tagger.evaluate(corpus.test,
                                 gold_label_type='upos',
                                 mini_batch_size=16,
                                 out_path=f"{model_path.parent}/transfer_evaluation.txt")
        
        print(result.detailed_results)
    except Exception as e:
        print(f"Failed to load {model_path}: {e}")
