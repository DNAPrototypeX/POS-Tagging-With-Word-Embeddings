import flair.datasets
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus = flair.datasets.UD_NORWEGIAN()
corpus.downsample(0.05)

label_type = 'upos'

label_dict = corpus.make_label_dictionary(label_type=label_type)



tagger = SequenceTagger.load("resources/taggers/xlnet-upos-adam/final-model.pt")

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune('resources/taggers/xlnet-transfer-upos-adam',
                  learning_rate=5.0e-4,
                  mini_batch_size=16,
                  max_epochs=5
              )
