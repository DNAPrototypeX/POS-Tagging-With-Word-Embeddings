import flair.datasets
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus = flair.datasets.UD_NORWEGIAN()
corpus.downsample(0.05)

label_type = 'upos'

label_dict = corpus.make_label_dictionary(label_type=label_type)



tagger = SequenceTagger.load("resources/taggers/t5-upos-adam/final-model.pt")

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune('resources/taggers/t5-transfer-upos-adam',
                  learning_rate=5.0e-3,
                  mini_batch_size=16,
                  max_epochs=5
              )
