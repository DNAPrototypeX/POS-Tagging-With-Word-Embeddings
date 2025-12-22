from flair.datasets import UD_NORWEGIAN
from flair.embeddings import WordEmbeddings, StackedEmbeddings, FlairEmbeddings, TransformerWordEmbeddings
from flair.models import SequenceTagger
from flair.trainers import ModelTrainer

corpus = UD_NORWEGIAN()
corpus.downsample(0.05)

label_type = 'upos'

label_dict = corpus.make_label_dictionary(label_type=label_type)
print(label_dict)

embedding_types = [
    WordEmbeddings('en')
]

embeddings = StackedEmbeddings(embeddings=embedding_types)

tagger = SequenceTagger(hidden_size=256,
                        embeddings=embeddings,
                        tag_dictionary=label_dict,
                        tag_type=label_type,
                        use_crf=False,
                        use_rnn=False,
                        reproject_embeddings=False)

trainer = ModelTrainer(tagger, corpus)

trainer.fine_tune('resources/taggers/fasttext-norwegian-upos-adam',
              learning_rate=0.1,
              mini_batch_size=16,
              )
